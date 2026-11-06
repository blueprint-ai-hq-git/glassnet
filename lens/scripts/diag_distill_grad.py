"""diag_distill_grad.py — is the 2.3e-2 gradcheck miss a MATH error or float32 noise?

Method: the same loss in PURE float64 numpy (no engine), finite differences against the analytic
formula. If float64 FD matches the formula to <1e-8, the math is right and the engine's 2.3e-2 is
float32 rounding at eps=1e-2 — then the correct fix is a float64 gradcheck or larger eps, NOT a
code change to the op. If float64 FD also misses, the formula (value or gradient) is wrong.
"""
import numpy as np

rng = np.random.default_rng(0)
B, V = 8, 12
z = rng.normal(size=(B, V))
q = rng.random((B, V))
q /= q.sum(axis=1, keepdims=True)


def loss_val(z, q, tau):
    zt = z / tau
    zt = zt - zt.max(axis=1, keepdims=True)
    p = np.exp(zt)
    p /= p.sum(axis=1, keepdims=True)
    eps = 1e-9
    qq = np.clip(q, eps, None)
    return tau * tau * (qq * (np.log(qq) - np.log(np.clip(p, eps, None)))).sum(axis=1).mean()


def grad_formula(z, q, tau):
    zt = z / tau
    zt = zt - zt.max(axis=1, keepdims=True)
    p = np.exp(zt)
    p /= p.sum(axis=1, keepdims=True)
    return tau * (p - q) / B


for tau in (1.0, 2.0):
    g = grad_formula(z, q, tau)
    worst = 0.0
    for eps in (1e-2, 1e-4, 1e-6):
        w = 0.0
        for _ in range(20):
            i, j = rng.integers(0, B), rng.integers(0, V)
            zp = z.copy(); zp[i, j] += eps
            zm = z.copy(); zm[i, j] -= eps
            num = (loss_val(zp, q, tau) - loss_val(zm, q, tau)) / (2 * eps)
            ana = g[i, j]
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            w = max(w, rel)
        print(f"  tau={tau}  float64 eps={eps:g}:  worst rel diff {w:.3e}")
        if eps == 1e-4:
            worst = w
print()
print("If the float64 rows hit ~1e-9 or better at eps 1e-4/1e-6, the op MATH IS CORRECT and")
print("the engine gradcheck missed only because float32 + eps=1e-2 rounds through softmax twice.")
