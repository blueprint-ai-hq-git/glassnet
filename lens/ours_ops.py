"""TWO OPS OUR ENGINE WAS MISSING, which is why real multi-head attention could never be built.

Every honest attention needs to treat a flat vector as (window x dim) and to stack per-step states. Our engine
only had 2-D mats, so yesterday's attention could not be built properly. These two ops add that ability, written
by hand and gradient-checked numerically before anything uses them:

  reshape(x, shape)     - same numbers, new arrangement. Gradient: reshape the gradient back.
  stack(tensors, axis)  - glue equal-shaped tensors into a new axis. Gradient: slice the gradient back out.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # project root (this file is in lens/)
import numpy as np
import lens.tensor as T

def reshape(x, shape):
    old = x.data.shape
    y = np.ascontiguousarray(x.data).reshape(shape).astype(np.float32)
    return T.Tensor(y, x.requires_grad, _parents=(x,), _op="reshape", _saved={"old": old})

def _bw_reshape(out, parents, saved, grad):
    x = parents[0]
    g = np.ascontiguousarray(grad).reshape(saved["old"]).astype(np.float32)
    x.grad = g if x.grad is None else x.grad + g

def stack(ts, axis=0):
    """Glue tensors that share a shape into a new axis (default: new leading axis)."""
    ts = [t if isinstance(t, T.Tensor) else T.Tensor(t) for t in ts]
    shape = ts[0].data.shape
    for t in ts:
        assert t.data.shape == shape, f"stack needs equal shapes, got {t.data.shape} and {shape}"
    y = np.stack([t.data for t in ts], axis=axis).astype(np.float32)
    return T.Tensor(y, any(t.requires_grad for t in ts), _parents=tuple(ts), _op="stack",
                     _saved={"axis": axis, "n": len(ts)})

def _bw_stack(out, parents, saved, grad):
    axis, n = saved["axis"], saved["n"]
    for i, t in enumerate(parents):
        if not t.requires_grad:
            continue
        g = np.take(grad, i, axis=axis).astype(np.float32)
        t.grad = g if t.grad is None else t.grad + g

T.OPS["reshape"] = _bw_reshape
T.OPS["stack"] = _bw_stack

# ---------------- GRADIENT CHECK, before anything is allowed to use them ----------------
# (the engine has no sum op, so the scalar loss comes from softmax_ce on a 2-D view - the same loss the model
#  trains with, which makes this check the one that matters)
from lens.tensor import softmax_ce
rng = np.random.default_rng(0)
print("OP            max_rel_error   verdict", flush=True)

def check(name, build, xshape, ncls, eps=1e-3, samples=8):
    x = T.Tensor((rng.standard_normal(xshape) * 1.3).astype(np.float32), requires_grad=True)
    labels = (np.arange(ncls) % 7).astype(np.int64)
    loss = softmax_ce(build(x), labels)
    loss.backward()
    an = x.grad.copy()
    worst = 0.0
    for _ in range(samples):
        idx = tuple(int(rng.integers(0, s)) for s in xshape)
        old = float(x.data[idx])
        x.data[idx] = old + eps
        lp = float(softmax_ce(build(x), labels).data)
        x.data[idx] = old - eps
        lm = float(softmax_ce(build(x), labels).data)
        x.data[idx] = old
        num = (lp - lm) / (2 * eps)
        ana = float(an[idx])
        worst = max(worst, abs(num - ana) / (abs(num) + abs(ana) + 1e-9))
    ok = worst < 0.02
    print(f"{name:<13} {worst:>10.6f}   {'PASS' if ok else 'FAIL'}", flush=True)
    return ok

# reshape: flat (8,32) -> (8,4,8) -> back to a 2-D view (32,8) for the loss
r1 = check("reshape", lambda x: reshape(reshape(x, (8, 4, 8)), (32, 8)), (8, 32), 32)
# stack: two copies glued on a new axis, flattened for the loss
r2 = check("stack", lambda x: reshape(stack([x, x], axis=0), (12, 16)), (6, 16), 12)
# the combination attention needs: flat -> window x dim, then stack states
r3 = check("combo", lambda x: reshape(stack([reshape(x, (8, 4, 8)), reshape(x, (8, 4, 8))], axis=0), (64, 8)), (8, 32), 64)
print(f"RESULT: reshape={'PASS' if r1 else 'FAIL'} stack={'PASS' if r2 else 'FAIL'} combo={'PASS' if r3 else 'FAIL'}", flush=True)
