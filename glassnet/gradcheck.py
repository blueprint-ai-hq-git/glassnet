"""glassnet.gradcheck — compare our hand-written backward pass against
central finite differences.

The forward graph is rebuilt from scratch for EVERY (+eps / -eps) probe.
Reusing cached activations would leave the numeric gradient at ~0 and pass
a broken backprop — that exact bug burned this project before.
"""
import numpy as np

from .tensor import xp


def numeric_grads(build_loss, params, eps=1e-6):
    """build_loss() must rebuild the graph every call and return a scalar-loss
    Tensor; only its .data is read here. Backend-aware: the forward runs on
    whichever backend the params live on, so this checks the GPU backward too."""
    grads = {}
    for p in params:
        g = xp.zeros_like(p.data)
        for idx in np.ndindex(p.data.shape):
            # float(...) is essential: on CuPy, p.data[idx] can be a 0-dim
            # VIEW, so writing p.data[idx] = orig + eps would mutate `orig`
            # too and the -eps probe would read the +eps value, halving the
            # numeric gradient (rel err exactly 1/3 on every check).
            orig = float(p.data[idx])
            p.data[idx] = orig + eps
            lp = float(build_loss().data)
            p.data[idx] = orig - eps
            lm = float(build_loss().data)
            p.data[idx] = orig
            g[idx] = (lp - lm) / (2.0 * eps)
        grads[id(p)] = g
    return grads


def analytic_grads(build_loss, params):
    loss = build_loss()
    loss.backward()
    return {id(p): p.grad.copy() for p in params}


def check(build_loss, params, eps=1e-6, abs_tol=1e-6, rel_tol=1e-4, name="net"):
    """Run the gate on one graph. Returns a dict with worst abs/rel error."""
    an = analytic_grads(build_loss, params)
    nu = numeric_grads(build_loss, params, eps=eps)
    worst_abs, worst_rel, worst_param = 0.0, 0.0, None
    for p in params:
        a, n = an[id(p)], nu[id(p)]
        abs_err = xp.abs(a - n)
        rel_err = abs_err / (xp.abs(a) + xp.abs(n) + 1e-9)
        ia, ir = float(abs_err.max()), float(rel_err.max())
        if ia > worst_abs:
            worst_abs = ia
        if ir > worst_rel:
            worst_rel = ir
            worst_param = p.name
    ok = worst_abs < abs_tol and worst_rel < rel_tol
    return {"name": name, "ok": ok, "worst_abs": worst_abs,
            "worst_rel": worst_rel, "worst_param": worst_param}