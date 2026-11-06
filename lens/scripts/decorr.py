"""decorr.py — an ORTHOGONALITY (rivalry) penalty for the concept layer.

WHY THIS TERM
The strict audit's criterion is: for every concept neuron, |corr(neuron, its own formula)| minus
|corr(neuron, the best OTHER formula)| > 0.10. A concept pair whose neurons track each other can
never satisfy that, however well each tracks its own formula. This term pushes the concept
ACTIVATIONS apart: it penalises the squared off-diagonal of their covariance — exactly the rivalry
the audit measures.

DEVICE-CORRECT (v2): the first version computed the penalty with numpy while the activations lived
on the GPU, so the backward died on a device mismatch mid-run (both K21k runs). Everything here now
goes through the engine's own array module (`B.xp`), like the soft-KL op does.

DERIVATION (written out, then checked numerically before use)
    Hc = h - mean(h)                       centred activations, (N, K)
    C  = (1/N) Hc^T Hc                     (K, K) covariance
    L  = sum_{i != j} C_ij^2
    dL/dHc = (4/N) Hc @ (C - diag(C))      the diagonal is excluded from the sum
    dL/dh  = dL/dHc - mean(dL/dHc)         centring passes the gradient back
"""
from lens import backend as B

import lens.tensor as T


def _bw_decorr(out, parents, saved, grad):
    h = parents[0]
    if not h.requires_grad:
        return
    Hc, Coff, n = saved["Hc"], saved["Coff"], saved["n"]
    gHc = B.xp.asarray(4.0 / n, dtype=Hc.dtype) * (Hc @ Coff)
    gh = gHc - gHc.mean(axis=0, keepdims=True)
    T._accumulate(h, gh * T._scalar(grad))


T.OPS["decorr"] = _bw_decorr


def decorr_term(h):
    """Penalty node from a concept-activation tensor h (N, K). Scalar node + its value."""
    hd = h.data
    N = int(hd.shape[0])
    Hc = hd - hd.mean(axis=0, keepdims=True)
    C = (Hc.T @ Hc) / max(N, 1)
    K = int(C.shape[0])
    eye = B.xp.eye(K, dtype=C.dtype)
    Coff = C * (B.xp.asarray(1.0, dtype=C.dtype) - eye)      # zero the diagonal, on-device
    val = float(B.to_cpu((Coff ** 2).sum()))
    node = T.Tensor(B.to_dev(B.xp.asarray(val, dtype=T.current_dtype())),
                    requires_grad=h.requires_grad, _parents=(h,), _op="decorr",
                    _saved={"Hc": Hc, "Coff": Coff, "n": N})
    return node, val
