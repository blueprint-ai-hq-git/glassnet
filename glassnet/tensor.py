"""glassnet.tensor — our own tiny autograd engine, written from scratch.

A Tensor wraps an array plus the graph that made it. Every operation records
how to push gradients backwards; `backward()` walks the graph once and fills
each tensor's `.grad`. Nothing here comes from a deep-learning library:
+, *, @, relu, sigmoid and the losses are all coded by hand.

ARRAY BACKEND: all maths goes through the module-level `xp`, which is NumPy
by default and can be switched to CuPy (a raw GPU array library — allowed by
the from-scratch rules; it is an array backend, not a DL framework) with
`set_backend("cupy")`. No op changes; only which array library executes it.
"""
from __future__ import annotations

import numpy as np

xp = np                      # active array backend (numpy | cupy)
_DEFAULT_DTYPE = np.float32


def set_backend(name="numpy"):
    """Switch the array backend for every op in this module.
    'cupy'/'gpu' -> GPU via CuPy; anything else -> NumPy."""
    global xp
    if str(name).lower() in ("cupy", "gpu"):
        import cupy as cp
        xp = cp
    else:
        xp = np
    return xp


def backend_name():
    return "cupy" if xp is not np else "numpy"


def set_default_dtype(dtype):
    """Default dtype for brand-new constants (training stays float32;
    gradient checks switch to float64 for exactness, then switch back)."""
    global _DEFAULT_DTYPE
    _DEFAULT_DTYPE = dtype


def _is_array(x):
    if isinstance(x, np.ndarray):
        return True
    return xp is not np and isinstance(x, xp.ndarray)


def _as_array(data, dtype=None):
    # numpy FIRST: with the cupy backend a host array must be moved to the
    # GPU. (Checking _is_array() first would match numpy arrays and return
    # them untouched, making the transfer below dead code — a bug the GPU
    # verification caught: every tensor silently stayed on the CPU.)
    if isinstance(data, np.ndarray):
        if xp is np:
            return data if dtype is None else data.astype(dtype)
        return xp.asarray(data, dtype=dtype if dtype is not None else data.dtype)
    if _is_array(data):
        return data if dtype is None else data.astype(dtype)
    return xp.asarray(data, dtype=_DEFAULT_DTYPE if dtype is None else dtype)


def as_backend(x, dtype=None):
    """Accept a host array, a backend array, or a python scalar; return a
    backend array of the requested dtype."""
    if isinstance(x, np.ndarray) and xp is not np:
        return xp.asarray(x, dtype=dtype if dtype is not None else x.dtype)
    if _is_array(x):
        return x if (dtype is None or x.dtype == dtype) else x.astype(dtype)
    return _as_array(x, dtype)


def to_backend(arr):
    """Move a host (NumPy) array onto the active backend, once."""
    if xp is np or _is_array(arr):
        return arr
    return xp.asarray(arr)

def _sum_to(g, shape):
    """Reduce gradient g so it matches the broadcasted input shape `shape`."""
    if not _is_array(g):
        g = xp.asarray(g)
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def _accum(t, g):
    if not t.requires_grad:
        return
    if t.grad is None:
        t.grad = g
    else:
        t.grad = t.grad + g


class Tensor:
    __slots__ = ("data", "requires_grad", "grad", "_prev", "_backward", "name")

    def __init__(self, data, requires_grad=False, _prev=(), _backward=None, name=None):
        self.data = _as_array(data)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._prev = set(_prev)
        self._backward = _backward
        self.name = name

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        tag = f" name={self.name}" if self.name else ""
        return (f"Tensor(shape={tuple(self.data.shape)}, dtype={self.data.dtype}, "
                f"grad={'set' if self.grad is not None else 'None'}{tag})")

    # ------------------------------------------------------------- autograd
    def backward(self):
        topo, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            for p in t._prev:
                visit(p)
            topo.append(t)

        visit(self)
        self.grad = xp.ones_like(self.data)
        for t in reversed(topo):
            if t._backward is not None:
                t._backward()

    # ------------------------------------------------------------- sugar
    def __add__(self, other):  return add(self, other)
    def __radd__(self, other): return add(other, self)
    def __mul__(self, other):  return mul(self, other)
    def __rmul__(self, other): return mul(other, self)
    def __matmul__(self, other): return matmul(self, other)
    def __neg__(self): return mul(self, -1.0)


def _coerce(v):
    if isinstance(v, Tensor):
        return v
    return Tensor(_as_array(v), requires_grad=False)


def add(a, b):
    a, b = _coerce(a), _coerce(b)
    out = Tensor(a.data + b.data,
                 requires_grad=a.requires_grad or b.requires_grad,
                 _prev=(a, b), name="add")

    def bw():
        if out.grad is None:
            return
        _accum(a, _sum_to(out.grad, a.shape))
        _accum(b, _sum_to(out.grad, b.shape))

    out._backward = bw
    return out


def mul(a, b):
    a, b = _coerce(a), _coerce(b)
    out = Tensor(a.data * b.data,
                 requires_grad=a.requires_grad or b.requires_grad,
                 _prev=(a, b), name="mul")

    def bw():
        if out.grad is None:
            return
        _accum(a, _sum_to(out.grad * b.data, a.shape))
        _accum(b, _sum_to(out.grad * a.data, b.shape))

    out._backward = bw
    return out


def matmul(a, b):
    a, b = _coerce(a), _coerce(b)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul: only 2-D @ 2-D supported (batches x features)")
    out = Tensor(a.data @ b.data,
                 requires_grad=a.requires_grad or b.requires_grad,
                 _prev=(a, b), name="matmul")

    def bw():
        if out.grad is None:
            return
        _accum(a, out.grad @ b.data.T)
        _accum(b, a.data.T @ out.grad)

    out._backward = bw
    return out


def relu(x):
    x = _coerce(x)
    out = Tensor(xp.maximum(x.data, 0.0), requires_grad=x.requires_grad,
                 _prev=(x,), name="relu")

    def bw():
        if out.grad is None:
            return
        _accum(x, out.grad * (x.data > 0))

    out._backward = bw
    return out


def sigmoid(x):
    x = _coerce(x)
    s = 1.0 / (1.0 + xp.exp(-x.data))
    out = Tensor(s, requires_grad=x.requires_grad, _prev=(x,), name="sigmoid")

    def bw():
        if out.grad is None:
            return
        _accum(x, out.grad * (s * (1.0 - s)))

    out._backward = bw
    return out


def abs_(x):
    """Elementwise |x| (for the L1 sparsity penalty)."""
    x = _coerce(x)
    out = Tensor(xp.abs(x.data), requires_grad=x.requires_grad,
                 _prev=(x,), name="abs")

    def bw():
        if out.grad is None:
            return
        _accum(x, out.grad * xp.sign(x.data))

    out._backward = bw
    return out


def sum_all(x):
    """Sum of every element (scalar output)."""
    x = _coerce(x)
    out = Tensor(xp.sum(x.data), requires_grad=x.requires_grad,
                 _prev=(x,), name="sum_all")

    def bw():
        if out.grad is None:
            return
        _accum(x, out.grad * xp.ones_like(x.data))

    out._backward = bw
    return out


def take_cols(x, cols):
    """Select columns (axis=1) of a 2-D tensor.

    Backward scatters the gradient back into the selected columns and leaves the
    rest at zero, so a loss applied to a SUBSET of concept columns trains only
    those columns and cannot touch the others.
    """
    x = _coerce(x)
    cols = np.asarray(cols, dtype=np.int64)
    out = Tensor(x.data[:, cols], requires_grad=x.requires_grad,
                 _prev=(x,), name="take_cols")

    def bw():
        if out.grad is None:
            return
        g = xp.zeros_like(x.data)
        g[:, cols] = out.grad
        _accum(x, g)

    out._backward = bw
    return out


def sum_axis(x, axis):
    """Sum along one axis, keeping the other dims (N,W,D) -> (N,D)."""
    x = _coerce(x)
    out = Tensor(xp.sum(x.data, axis=axis), requires_grad=x.requires_grad,
                 _prev=(x,), name="sum_axis")

    def bw():
        if out.grad is None:
            return
        g = xp.expand_dims(out.grad, axis)
        _accum(x, xp.broadcast_to(g, x.data.shape).copy())

    out._backward = bw
    return out


def embedding(E, idx):
    """Gather rows of an (V,D) embedding matrix: out[i,j] = E[idx[i,j]].
    idx: integer array, any shape. Forward is a flat gather; backward
    scatter-adds the output gradient into the gathered rows (this is our
    own op — written from scratch like every other op)."""
    E = _coerce(E)
    if isinstance(idx, Tensor):      # accept either a Tensor or a raw array
        idx = idx.data
    idx = as_backend(idx)
    out = Tensor(E.data[idx], requires_grad=E.requires_grad,
                 _prev=(E,), name="embedding")

    def bw():
        if out.grad is None:
            return
        g = xp.zeros_like(E.data)
        if xp is np:
            np.add.at(g, idx, out.grad)
        else:
            # cupy's ufunc.at is very slow; scatter_add is the fast path
            import cupyx
            cupyx.scatter_add(g, idx.ravel().astype(xp.int32),
                              out.grad.reshape(-1, E.data.shape[1]))
        _accum(E, g)

    out._backward = bw
    return out
