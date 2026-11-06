"""lens/tensor.py — our own tensors and our own autograd.

DESIGN CHOICE THAT FIXES THE OLD ENGINE'S BUG
---------------------------------------------
The previous engine gave every op a `backward` CLOSURE that captured the output tensor. That
creates a reference cycle: output -> closure -> output. Python needs the cycle collector to
reclaim it, and in a long training run that leak is real.

This engine has no closures at all. Each tensor records only:
    _parents : the input tensors (a tuple)
    _op      : a string naming the operation
    _saved   : the small data the backward needs (e.g. the sigmoid output)
and backward is dispatched through OP_BACKWARD[op](out, parents, saved, grad), where the
gradient is PASSED IN as an argument. Nothing captures anything, so nothing cycles. The graph
is released the moment the output is.

BROADCASTING
------------
Deliberately minimal. Only two shapes are allowed: identical shapes, and (N, M) + (M,) which is
the bias add. Anything else raises with both shapes in the message. This removes an entire class
of silent shape bugs at the cost of not supporting a general broadcast engine we do not need.

NUMERICS
--------
float32 by default, matching the kernels. The gradient gate runs in float64 so central
differences are meaningful, and Tensor accepts whatever dtype it is handed.

CORRECTION (2026-10-10): the sentence above was FALSE until today. The constructor cast EVERY
input to float32 unconditionally, so `gradcheck_vision.py` -- which builds float64 tensors and
compares against central differences with a 1e-6 bar -- could never pass: eps=1e-6 is below the
float32 resolution of the accumulated sums, so the "numeric" gradient came back quantised in steps
of 2^-22/(2*eps) = 0.119 and differed from the analytic gradient by up to 1.0. The engine's
gradients were correct all along; the verification was impossible. `set_dtype("float64")` now makes
the docstring true: float32 stays the training default (so nothing about training changes), and the
gradient gate can run in genuine float64.
"""
import numpy as np

from lens import _capi as C
from lens import backend

OPS = {}

# The working dtype. None means "the backend's default" (float32), which is what training uses.
# set_dtype("float64") switches the whole engine for the duration of a gradient check, then
# set_dtype(None) restores float32. Never leave it in float64 for training: the C kernels are
# float32 and float64 doubles memory traffic for no accuracy we need.
_DTYPE = None


def set_dtype(dtype):
    """Select the working dtype for tensors built from here on. None = backend default (float32)."""
    global _DTYPE
    _DTYPE = None if dtype is None else np.dtype(dtype)
    return current_dtype()


def current_dtype():
    """The dtype new tensors will use: the override if set, else the backend default (float32)."""
    return backend.xp.float32 if _DTYPE is None else _DTYPE

# ---------------------------------------------------------------- device
# "cpu" runs everything on our C kernels. "gpu" sends the COMPUTE-BOUND ops (matmul,
# matmul_abt, fused softmax+CE) to our CUDA kernels and leaves the elementwise ops on the CPU.
#
# Why not everything: an elementwise add is memory-bound. Shipping 1 MB to the card and back
# costs more time than the add itself takes on the CPU, so a "fully GPU" version of those ops
# would be SLOWER. The two matmuls and the fused loss are where the FLOPs are -- that is the
# ~90% -- so those are what move. This is stated rather than glossed: elementwise is on the CPU.
DEVICE = "cpu"


def set_device(name):
    """Switch the array module. 'cpu' = NumPy, 'gpu' = CuPy (arrays stay on the card).

    Kept as a thin wrapper over `lens.backend` so existing callers keep working. The old version
    of this function dispatched individual ops to hand-written CUDA kernels while the arrays
    stayed on the host -- which measured 0.69x, SLOWER than the CPU, because every op paid a
    transfer. The device now follows the array, not the op.
    """
    if name not in ("cpu", "gpu"):
        raise ValueError(f"device must be 'cpu' or 'gpu', got {name!r}")
    backend.use_gpu(name == "gpu")
    return backend.device()


def device():
    return backend.device()


def register(name):
    def deco(fn):
        OPS[name] = fn
        return fn
    return deco


class Tensor:
    __slots__ = ("data", "grad", "requires_grad", "name", "_parents", "_op", "_saved")

    def __init__(self, data, requires_grad=False, name=None, _parents=(), _op=None, _saved=None):
        # Duck-typed rather than isinstance-checked: the array module can be numpy or cupy, and
        # `backend.xp` is a module, not a type. Anything with a dtype is already an array.
        target = current_dtype()
        if not hasattr(data, "dtype"):
            data = backend.xp.asarray(data, dtype=target)
        if data.dtype != target:
            data = data.astype(target)
        # Place it on whichever device the backend is using. Without this the parameters stay
        # host arrays after set_device("gpu") and the first cupy op fails with the confusing
        # "Unsupported type <class 'numpy.ndarray'>" -- which is exactly what happened when the
        # device switch was first wired up.
        self.data = backend.to_dev(data)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self.name = name
        self._parents = _parents
        self._op = _op
        self._saved = _saved

    # ---------------------------------------------------------------- introspection
    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return (f"Tensor(shape={self.shape}, name={self.name!r}, "
                f"grad={'yes' if self.grad is not None else 'no'})")

    # ---------------------------------------------------------------- backward
    def backward(self):
        """Reverse-mode over the recorded graph. Iterative, no recursion limit issues."""
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))

        self.grad = backend.xp.ones_like(self.data)
        for node in reversed(topo):
            if node._op is None or node.grad is None:
                continue
            OPS[node._op](node, node._parents, node._saved, node.grad)

    def zero_grad(self):
        self.grad = None


def _accumulate(t, g):
    """Add a gradient into a tensor, creating it on first use. Never in-place on t.grad."""
    if not t.requires_grad or g is None:
        return
    t.grad = g if t.grad is None else t.grad + g


def _unbroadcast(g, shape):
    """Undo a bias-style broadcast: (N, M) gradient back to (M,)."""
    if g.shape == shape:
        return g
    if len(shape) == 1 and len(g.shape) == 2 and g.shape[1] == shape[0]:
        return g.sum(axis=0)
    raise ValueError(f"cannot unbroadcast {g.shape} to {shape}")


def _check_same(a, b, op):
    if a.shape != b.shape:
        raise ValueError(f"{op}: shapes must match, got {a.shape} and {b.shape}")


# ---------------------------------------------------------------- ops
@register("add")
def _bw_add(out, parents, saved, grad):
    a, b = parents
    _accumulate(a, _unbroadcast(grad, a.shape))
    _accumulate(b, _unbroadcast(grad, b.shape))


@register("sub")
def _bw_sub(out, parents, saved, grad):
    a, b = parents
    _accumulate(a, _unbroadcast(grad, a.shape))
    _accumulate(b, _unbroadcast(-grad, b.shape))


@register("mul")
def _bw_mul(out, parents, saved, grad):
    a, b = parents
    _accumulate(a, _unbroadcast(grad * b.data, a.shape))
    _accumulate(b, _unbroadcast(grad * a.data, b.shape))


@register("matmul")
def _bw_matmul(out, parents, saved, grad):
    """For out = a @ b with a (m,k) and b (k,n):
         dA = grad @ b^T      -> (m,n) @ (n,k) = (m,k)
         dB = a^T @ grad      -> (k,m) @ (m,n) = (k,n)

    Written as plain array expressions so it runs on whichever module `backend.xp` is. The
    hand-written kernel version needed two different kernels (matmul_abt and matmul_atb) and a
    GPU dispatch that got the second one wrong -- it called the A@B^T kernel where A^T@B was
    needed and failed with the shapes not matching. The expression cannot get that wrong.
    """
    a, b = parents
    if a.requires_grad:
        _accumulate(a, grad @ b.data.T)
    if b.requires_grad:
        _accumulate(b, a.data.T @ grad)


@register("relu")
def _bw_relu(out, parents, saved, grad):
    x = parents[0]
    _accumulate(x, grad * (x.data > 0))


@register("sigmoid")
def _bw_sigmoid(out, parents, saved, grad):
    s = saved["out"]
    _accumulate(parents[0], grad * s * (1.0 - s))


def _scalar(x):
    """A true Python float from a 0-d, (1,), or scalar-valued array, on either device."""
    a = backend.to_cpu(x)
    if a.size != 1:
        raise ValueError(f"expected a scalar gradient, got shape {a.shape}")
    return float(a.reshape(-1)[0])


@register("sum")
def _bw_sum(out, parents, saved, grad):
    x = parents[0]
    _accumulate(x, backend.xp.full_like(x.data, _scalar(grad)))


@register("softmax_ce")
def _bw_softmax_ce(out, parents, saved, grad):
    """Fused: the gradient of the mean CE w.r.t. the logits, already divided by rows."""
    logits = parents[0]
    if not logits.requires_grad:
        return
    g = saved["dlogits"] * _scalar(grad)
    _accumulate(logits, g)


# ---------------------------------------------------------------- forward
def add(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    if a.shape != b.shape:
        if len(b.shape) == 1 and len(a.shape) == 2 and a.shape[1] == b.shape[0]:
            pass                                   # bias add, allowed
        else:
            raise ValueError(f"add: shapes {a.shape} and {b.shape} are not supported "
                             "(only equal shapes and (N,M)+(M,))")
    out = Tensor(a.data + b.data, a.requires_grad or b.requires_grad,
                 _parents=(a, b), _op="add")
    return out


def sub(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    _check_same(a, b, "sub")
    return Tensor(a.data - b.data, a.requires_grad or b.requires_grad, _parents=(a, b), _op="sub")


def mul(a, b):
    a = a if isinstance(a, Tensor) else Tensor(a)
    b = b if isinstance(b, Tensor) else Tensor(b)
    _check_same(a, b, "mul")
    return Tensor(a.data * b.data, a.requires_grad or b.requires_grad, _parents=(a, b), _op="mul")


def transpose(t):
    """Swap the last two axes. The gradient of a transpose is the same transpose.

    Added 2026-10-21 for the TIED direct path: the answer layer's readout must use the model's own
    embedding matrix, so the gradient has to flow back into that matrix. Without this op the tied
    path would be a dead end that trains nothing.
    """
    return Tensor(backend.xp.swapaxes(t.data, -1, -2), t.requires_grad,
                  _parents=(t,), _op="transpose")


@register("transpose")
def _bw_transpose(out, parents, saved, grad):
    _accumulate(parents[0], backend.xp.swapaxes(grad, -1, -2))


def matmul(a, b):
    if a.data.ndim != 2 or b.data.ndim != 2:
        raise ValueError(f"matmul: both inputs must be 2D, got {a.shape} and {b.shape}")
    return Tensor(a.data @ b.data, a.requires_grad or b.requires_grad,
                  _parents=(a, b), _op="matmul")


@register("bmm3")
def _bw_bmm3(out, parents, saved, grad):
    """Batched 3D matmul backward. out[n,a,b] = sum_j a[n,a,j] b'[n,j,b] where b' = b (plain) or
    b^T over the last two dims (`trans_b`), i.e. b keys (N,B,J) with scores a @ b^T.
      plain:  out[n,a,b] = Σ_j a[n,a,j] b[n,j,b]  -> dA = g·b  (einsum "nab,njb->naj")
                                                  -> dB = aᵀ·g  (einsum "nab,naj->njb")
      trans:  out[n,a,b] = Σ_j a[n,a,j] b[n,b,j]  -> dA = g·b  (einsum "nab,nbj->naj")
                                                  -> dB = gᵀ·a  (einsum "nab,naj->nbj")
    """
    a, b = parents
    xp = backend.xp
    if saved["trans_b"]:
        ga = xp.matmul(grad, b.data)                                  # g @ b
        gb = xp.matmul(xp.swapaxes(grad, -1, -2), a.data)             # g^T @ a
    else:
        ga = xp.matmul(grad, xp.swapaxes(b.data, -1, -2))             # g @ b^T
        gb = xp.matmul(xp.swapaxes(a.data, -1, -2), grad)             # a^T @ g
    _accumulate(a, ga)
    _accumulate(b, gb)


def bmm3(a, b, trans_b=False):
    """Batched matmul over the last two dims (numpy/cupy einsum). One op, exact backward."""
    if a.data.ndim != 3 or b.data.ndim != 3:
        raise ValueError(f"bmm3: both inputs must be 3D (N,*,*) got {a.shape} and {b.shape}")
    xp = backend.xp
    # SPEED FIX (measured): einsum patterns like "nab,naj->nbj" (used in the bmm3 BACKWARD) do not
    # map to a batched GEMM, so CuPy/NumPy take a slow path -- attention ran 8x slower with it
    # (4.5 min/epoch vs 33 s for the identical model without attention). Explicit batched matmuls
    # with a transpose hit cuBLAS: identical arithmetic, verified equal to the einsum version
    # (max abs diff printed by run_verify_bmm3.sh) before this replaced it.
    if trans_b:
        out = xp.matmul(a.data, xp.swapaxes(b.data, -1, -2))
    else:
        out = xp.matmul(a.data, b.data)
    return Tensor(out, a.requires_grad or b.requires_grad, _parents=(a, b),
                  _op="bmm3", _saved={"trans_b": bool(trans_b)})


@register("softmax_rows")
def _bw_softmax_rows(out, parents, saved, grad):
    """Row softmax over the last axis. dX = P * (g - sum(P*g, axis=-1, keepdims))."""
    x = parents[0]
    p = saved["p"]
    _accumulate(x, p * (grad - (p * grad).sum(axis=-1, keepdims=True)))


def softmax_rows(x):
    """Numerically stable softmax over the LAST axis, probabilities kept for the backward."""
    xp = backend.xp
    m = xp.max(x.data, axis=-1, keepdims=True)
    e = xp.exp(x.data - m)
    p = e / e.sum(axis=-1, keepdims=True)
    return Tensor(p, x.requires_grad, _parents=(x,), _op="softmax_rows", _saved={"p": p})


@register("reshape")
def _bw_reshape(out, parents, saved, grad):
    """The gradient of a reshape is the inverse reshape. Nothing else changes."""
    x = parents[0]
    _accumulate(x, backend.xp.reshape(grad, saved["in_shape"]))


def reshape(x, shape):
    """A pure reinterpretation of the same numbers: (B*P, D) <-> (B, P*D).

    Needed by the image front-end: patches are projected as a flat (B*P, pd) matrix with ONE shared
    projection, then reinterpreted as (B, P*D) so the concept layer sees one row per image -- the
    same shape it sees for a text window of 16 words.
    """
    return Tensor(x.data.reshape(shape), x.requires_grad, _parents=(x,), _op="reshape",
                  _saved={"in_shape": tuple(x.data.shape)})


@register("concat")
def _bw_concat(out, parents, saved, grad):
    """Split the gradient back along the concatenation axis, in the same order."""
    axis, shapes = saved["axis"], saved["shapes"]
    idx = 0
    for p, s in zip(parents, shapes):
        sl = [slice(None)] * grad.ndim
        sl[axis] = slice(idx, idx + s)
        _accumulate(p, grad[tuple(sl)])
        idx += s


def concat(parts, axis=1):
    """Join tensors along an axis. Needed by the VL model: image patch features and caption token
    features are concatenated and fed to the SAME concept layer, which is what makes it one network
    rather than two."""
    parts = list(parts)
    shapes = [p.data.shape[axis] for p in parts]
    return Tensor(backend.xp.concatenate([p.data for p in parts], axis=axis),
                  any(p.requires_grad for p in parts), _parents=tuple(parts), _op="concat",
                  _saved={"axis": axis, "shapes": shapes})


def relu(x):
    return Tensor(backend.xp.maximum(x.data, 0), x.requires_grad, _parents=(x,), _op="relu")


def sigmoid(x):
    """Numerically stable: np.exp overflows for large negative inputs, which warns and returns inf
    even though the sigmoid itself is perfectly well defined there. Clamping the exponent keeps
    the value correct and the run quiet -- an overflow warning in a training loop trains people to
    ignore warnings."""
    z = backend.xp.clip(-x.data, -60.0, 60.0)
    s = 1.0 / (1.0 + backend.xp.exp(z))
    return Tensor(s, x.requires_grad, _parents=(x,), _op="sigmoid", _saved={"out": s})


@register("embedding")
def _bw_embedding(out, parents, saved, grad):
    """Scatter the window gradient back into the embedding rows that were looked up.

    This is an OP, not a closure. The old engine attached a closure to the output tensor to do
    exactly this, which is what created the reference cycles the L1 gate exists to prevent. A
    first attempt at this model re-introduced the closure out of habit and crashed immediately
    with "'Tensor' object has no attribute '_backward'" -- the new engine has no such attribute
    by design, so the mistake cannot survive.
    """
    emb = parents[0]
    if not emb.requires_grad or grad is None:
        return
    ids, W, D = saved["ids"], saved["W"], saved["D"]
    g = grad.reshape(len(ids), W, D)
    acc = backend.xp.zeros_like(emb.data)
    backend.xp.add.at(acc, ids, g)
    _accumulate(emb, acc)


def embedding(emb, ids, window, d):
    """Look up one embedding row per token and flatten the window. Returns (N, window*d).

    The ids stay on the HOST -- they are integer indices, not numbers to compute with, and cupy
    accepts a numpy integer array for fancy indexing.
    """
    ids = np.asarray(ids)
    E = emb.data[ids]
    return Tensor(E.reshape(len(ids), -1), emb.requires_grad, _parents=(emb,),
                  _op="embedding", _saved={"ids": ids, "W": window, "D": d})


def sum_all(x):
    v = x.data.sum()
    return Tensor(v, x.requires_grad, _parents=(x,), _op="sum")


def softmax_ce(logits, labels, class_weights=None, sample_weights=None):
    """Fused softmax + cross-entropy, computed on whichever device the logits live on.

    `class_weights` (optional) is indexed BY LABEL: w[lab] re-weights each sample by its label's
    weight. That is the existing behaviour and is left untouched.

    `sample_weights` (optional) is indexed BY SAMPLE: one value per row, already in the order of
    `logits`. This exists for ASSISTANT-ONLY chat training, where each target token either belongs to
    the assistant's turn (weight 1) or the human's (weight 0). It is deliberately a SEPARATE argument
    from class_weights: class_weights means "this LABEL matters more", sample_weights means "this
    POSITION matters more", and collapsing the two would silently change the medical runs that
    already depend on the label semantics.

    Numerically stable (max subtraction) and written as array expressions, so there is no
    separate CPU and GPU implementation to drift apart. The old hand-written pair had a sign
    error in the loss that the L0 gate caught, and the GPU version of it was a separate file.

    `class_weights` (optional) re-weights each sample by its LABEL's weight, normalised by the mean
    weight so the loss scale stays comparable to the unweighted version:

        L = (1/N) * sum_i (w_{y_i} / mean(w)) * (-log p_{i,y_i})
        dL/dz_ij = (1/N) * (w_{i} / mean(w)) * (p_ij - delta_{ij})

    WHY IT EXISTS: the medical VQA run collapsed onto the answer "no", which was 41.41% of the test
    answers -- exactly the majority baseline, to the digit. The model found the free lunch and
    stopped. Weighting by inverse label frequency takes the free lunch away.
    """
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    x = logits.data
    rows = x.shape[0]
    z = x - x.max(axis=1, keepdims=True)
    e = backend.xp.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    idx = backend.xp.arange(rows)
    lab = backend.to_dev(labels)

    if class_weights is None and sample_weights is None:
        w_sample = None
        loss = float(backend.to_cpu(-backend.xp.log(probs[idx, lab] + 1e-12).mean()))
        dlog = probs
        dlog[idx, lab] -= 1.0
        dlog /= float(rows)
    else:
        if sample_weights is not None:
            # per-POSITION weights (already in row order). Normalised by the mean so the loss scale
            # stays comparable to the unweighted run, exactly as class_weights does.
            w_sample = backend.to_dev(np.asarray(sample_weights, dtype=current_dtype()))
        else:
            # per-LABEL weights: unchanged existing behaviour
            w = backend.to_dev(np.asarray(class_weights, dtype=current_dtype()))
            w_sample = w[lab]
        w_sample = w_sample / backend.xp.maximum(w_sample.mean(), 1e-12)
        loss = float(backend.to_cpu(-(w_sample * backend.xp.log(probs[idx, lab] + 1e-12)).mean()))
        dlog = probs * w_sample[:, None]
        dlog[idx, lab] -= w_sample
        dlog /= float(rows)

    return Tensor(loss, logits.requires_grad, _parents=(logits,), _op="softmax_ce",
                  _saved={"dlogits": dlog, "probs": probs, "w_sample": w_sample})
