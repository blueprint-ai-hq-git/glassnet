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


def concept_exclusivity(h, c):
    """Penalise a concept neuron for tracking a DIFFERENT concept's label.

    WHY: on the shipped 80.78% model only 9 of 14 concept neurons meant what
    their label said (tools/check_concept_honesty.py). Correlation between the
    dishonest neurons' own labels and the labels they actually tracked was
    mostly LOW (r = +0.274 to +0.521), so the layer learned the wrong thing
    rather than being handed indistinguishable labels. Plain per-concept BCE
    does not care: a neuron may fit someone else's label as long as its own
    error is small. This term makes caring explicit.

    WHAT IT COMPUTES: the off-diagonal squared correlations between every
    neuron activation column and every concept TARGET column, so a neuron is
    charged for matching any label except its own. Both sides are centred
    first, so a neuron that merely copies a base rate is not charged.

    HONEST LIMIT: the derivative of the column normalisers is dropped (the
    penalty is differentiated as if the scales were constants). That changes
    the step size, not the direction of descent, and the term is scaled by a
    weight the caller sets; the gate, not this comment, decides whether it
    worked.
    """
    H = h.data
    C = as_backend(c, H.dtype)
    n = H.shape[0]
    Hc = H - xp.mean(H, axis=0, keepdims=True)
    Cc = C - xp.mean(C, axis=0, keepdims=True)
    hn = Hc / (xp.sqrt(xp.sum(Hc * Hc, axis=0, keepdims=True)) + 1e-8)
    cn = Cc / (xp.sqrt(xp.sum(Cc * Cc, axis=0, keepdims=True)) + 1e-8)
    R = hn.T @ cn                                   # (K, K) correlations
    off = R - xp.diag(xp.diag(R))                   # zero the diagonal: own label is free
    out = Tensor(xp.mean(off * off), requires_grad=h.requires_grad, _prev=(h,), name="concept_excl")

    def bw():
        if out.grad is None:
            return
        g = out.grad * (2.0 / (R.size)) * (hn @ off.T)
        _accum(h, g)

    out._backward = bw
    return out

