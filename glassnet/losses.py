"""glassnet.losses — loss functions, coded by hand (no library).

All maths goes through the active array backend (`xp` from .tensor), so the
same code runs on NumPy or CuPy with no change.
"""
import numpy as np

from .tensor import (Tensor, _accum, abs_, sum_all, mul, xp, as_backend)


def mse(pred, target):
    """Mean squared error. target: array broadcastable to pred."""
    t = as_backend(target, pred.data.dtype)
    r = pred.data - t
    out = Tensor(xp.mean(r * r), requires_grad=pred.requires_grad,
                 _prev=(pred,), name="mse")

    def bw():
        if out.grad is None:
            return
        _accum(pred, out.grad * (2.0 * r / r.size))

    out._backward = bw
    return out


def cross_entropy(logits, labels):
    """Mean cross-entropy of softmax(logits) vs integer class labels.
    logits: Tensor (N, C). labels: int array (N,).
    Numerically stable (max-subtraction); backward = (softmax - onehot)/N."""
    n = logits.data.shape[0]
    lbl = as_backend(labels, np.int64)
    if lbl.shape != (n,):
        raise ValueError(f"labels shape {lbl.shape} != ({n},)")
    m = logits.data.max(axis=1, keepdims=True)
    z = logits.data - m
    ex = xp.exp(z)
    probs = ex / ex.sum(axis=1, keepdims=True)
    rows = xp.arange(n)
    p = probs[rows, lbl]
    out = Tensor(xp.mean(-xp.log(p + 1e-12)),
                 requires_grad=logits.requires_grad,
                 _prev=(logits,), name="cross_entropy")

    def bw():
        if out.grad is None:
            return
        d = probs.copy()
        d[rows, lbl] -= 1.0
        _accum(logits, out.grad * (d / n))

    out._backward = bw
    return out


def bce_logits(logits, target, w=None):
    """Binary cross-entropy over sigmoid(logits), (optionally per-concept
    weighted) mean over entries. `w` is a (K,) weight per concept column."""
    z = logits.data
    s = 1.0 / (1.0 + xp.exp(-z))
    s = xp.clip(s, 1e-7, 1.0 - 1e-7)
    t = as_backend(target, z.dtype)
    el = -(t * xp.log(s) + (1.0 - t) * xp.log(1.0 - s))
    wv = None
    if w is not None:
        wv = as_backend(w, z.dtype)[None, :]
        el = el * wv
    out = Tensor(xp.mean(el),
                 requires_grad=logits.requires_grad,
                 _prev=(logits,), name="bce_logits")

    def bw():
        if out.grad is None:
            return
        g = out.grad * ((s - t) / s.size)
        if wv is not None:
            g = g * wv
        _accum(logits, g)

    out._backward = bw
    return out


def l1_sum(w):
    """SUM of |w| — the sparsity-penalty driver. Gradient = sign(w) per
    connection (not scaled by tensor size), so every weak connection feels
    the same constant pull toward zero and the task gradient must hold the
    useful ones up. Stronger than a mean-based penalty at the same lambda."""
    return sum_all(abs_(w))
