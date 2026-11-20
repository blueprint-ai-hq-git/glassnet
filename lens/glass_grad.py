#!/usr/bin/env python3
"""glass_grad.py - backward pass for GlassBlock, certified against finite differences.

The rule this file exists to satisfy (project rule, unchanged since Phase 1): a gradient is not trusted until it agrees with a
numerical gradient. Target here is stricter than the rest of the project because this block is small: worst relative error < 1e-6
in float64, and the check prints EVERY tensor's error, not just the worst one, so a single bad tensor cannot hide behind an average.

Everything is written out longhand for a reader: no autograd, no library. The chain rule is applied by hand, one operation at a time,
in the exact reverse order of the forward pass.
"""
import numpy as np

from lens.glass_block import layernorm, softmax_causal, split_heads, merge_heads


# ----------------------------------------------------------------------------------------------------------------------------
# backward helpers, each the exact inverse of one forward step
# ----------------------------------------------------------------------------------------------------------------------------

def ln_backward(dout, x, scale, eps=1e-5):
    """Backward through LayerNorm. Returns (dx, dscale, dshift).

    The forward is  xhat = (x - mu)/sqrt(var+eps);  y = xhat*scale + shift.  With D features and sigma = sqrt(var+eps):
        dx = (1/(D*sigma)) * (D*dxhat - sum(dxhat) - xhat * sum(dxhat*xhat))
    where dxhat = dout*scale. That closed form is used instead of differentiating the mean/variance separately because it is exact.
    """
    D = x.shape[-1]
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    sigma = np.sqrt(var + eps)
    xhat = (x - mu) / sigma
    dxhat = dout * scale
    dscale = (dout * xhat).sum(axis=0)
    dshift = dout.sum(axis=0)
    dx = (1.0 / (D * sigma)) * (D * dxhat - dxhat.sum(axis=-1, keepdims=True)
                                - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True))
    return dx, dscale, dshift


def softmax_backward(dw, weights):
    """Backward through a softmax whose output is already known: ds = w * (dw - sum(dw*w))."""
    return weights * (dw - (dw * weights).sum(axis=-1, keepdims=True))


def split_heads_backward(dsplit, T, D):
    """(H,T,dh) gradient -> (T,D) gradient, the exact inverse of split_heads."""
    return dsplit.transpose(1, 0, 2).reshape(T, D)


def merge_heads_backward(dmerged, T, D, n_heads):
    """(T,D) gradient -> (H,T,dh) gradient, the exact inverse of merge_heads."""
    dh = D // n_heads
    return dmerged.reshape(T, n_heads, dh).transpose(1, 0, 2)


# ----------------------------------------------------------------------------------------------------------------------------
# forward recording everything needed, then backward
# ----------------------------------------------------------------------------------------------------------------------------

def forward_record(x, p, n_heads, norm=True):
    """Same computation as GlassBlock.forward (no cache), but stores every intermediate so backward can use it."""
    T, D = x.shape
    dh = D // n_heads
    rec = {"x": x, "T": T, "D": D, "dh": dh, "norm": norm}

    n1 = layernorm(x, p["ln1_scale"], p["ln1_shift"]) if norm else x
    q, k, v = n1 @ p["Wq"], n1 @ p["Wk"], n1 @ p["Wv"]
    qh, kh, vh = split_heads(q, n_heads, T), split_heads(k, n_heads, T), split_heads(v, n_heads, T)
    scores = np.einsum("htd,hSd->htS", qh, kh) / np.sqrt(dh)
    weights = softmax_causal(scores)
    ctx = np.einsum("htS,hSd->htd", weights, vh)
    attn = merge_heads(ctx, T, D) @ p["Wo"]
    x1 = x + attn
    n2 = layernorm(x1, p["ln2_scale"], p["ln2_shift"]) if norm else x1
    a = n2 @ p["W1"]
    relu_a = np.maximum(a, 0.0)
    mlp = relu_a @ p["W2"]
    out = x1 + mlp
    rec.update({"n1": n1, "q": q, "k": k, "v": v, "qh": qh, "kh": kh, "vh": vh,
                "scores": scores, "weights": weights, "ctx": ctx, "attn": attn,
                "x1": x1, "n2": n2, "a": a, "relu_a": relu_a, "mlp": mlp})
    return out, rec


def backward_record(dout, rec, p, n_heads):
    """Reverse of forward_record. Returns (dx, grads dict) with a gradient for every named parameter."""
    T, D, dh = rec["T"], rec["D"], rec["dh"]
    g = {}

    # out = x1 + mlp
    dx1 = dout.copy()
    dmlp = dout

    # mlp = relu(a) @ W2
    g["W2"] = rec["relu_a"].T @ dmlp
    drelu_a = dmlp @ p["W2"].T
    da = drelu_a * (rec["a"] > 0)

    # a = n2 @ W1
    g["W1"] = rec["n2"].T @ da
    dn2 = da @ p["W1"].T

    # n2 = LN(x1)
    if rec["norm"]:
        dx1_ln, g["ln2_scale"], g["ln2_shift"] = ln_backward(dn2, rec["x1"], p["ln2_scale"])
        dx1 = dx1 + dx1_ln
    else:
        g["ln2_scale"] = np.zeros_like(p["ln2_scale"])
        g["ln2_shift"] = np.zeros_like(p["ln2_shift"])
        dx1 = dx1 + dn2

    # x1 = x + attn
    dx = dx1.copy()
    dattn = dx1

    # attn = merge(ctx) @ Wo
    ctx_merged = merge_heads(rec["ctx"], T, D)
    g["Wo"] = ctx_merged.T @ dattn
    dctx_merged = dattn @ p["Wo"].T
    dctx = merge_heads_backward(dctx_merged, T, D, n_heads)

    # ctx = weights @ v  (per head)
    dweights = np.einsum("htd,hSd->htS", dctx, rec["vh"])
    dvh = np.einsum("htS,htd->hSd", rec["weights"], dctx)

    # weights = softmax(scores)
    dscores = softmax_backward(dweights, rec["weights"])

    # scores = q k^T / sqrt(dh)
    dqh = np.einsum("htS,hSd->htd", dscores, rec["kh"]) / np.sqrt(dh)
    dkh = np.einsum("htS,htd->hSd", dscores, rec["qh"]) / np.sqrt(dh)

    dq = split_heads_backward(dqh, T, D)
    dk = split_heads_backward(dkh, T, D)
    dv = split_heads_backward(dvh, T, D)

    # q,k,v = n1 @ W*
    g["Wq"] = rec["n1"].T @ dq
    g["Wk"] = rec["n1"].T @ dk
    g["Wv"] = rec["n1"].T @ dv
    dn1 = dq @ p["Wq"].T + dk @ p["Wk"].T + dv @ p["Wv"].T

    # n1 = LN(x)
    if rec["norm"]:
        dx_ln, g["ln1_scale"], g["ln1_shift"] = ln_backward(dn1, rec["x"], p["ln1_scale"])
        dx = dx + dx_ln
    else:
        g["ln1_scale"] = np.zeros_like(p["ln1_scale"])
        g["ln1_shift"] = np.zeros_like(p["ln1_shift"])
        dx = dx + dn1

    return dx, g


# ----------------------------------------------------------------------------------------------------------------------------
# the certification
# ----------------------------------------------------------------------------------------------------------------------------

def make_params(D, n_heads, d_mlp, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    s = 1.0 / np.sqrt(D)
    p = {
        "Wq": rng.standard_normal((D, D)) * s, "Wk": rng.standard_normal((D, D)) * s,
        "Wv": rng.standard_normal((D, D)) * s, "Wo": rng.standard_normal((D, D)) * s,
        "W1": rng.standard_normal((D, d_mlp)) * s, "W2": rng.standard_normal((d_mlp, D)) * s,
        "ln1_scale": rng.standard_normal(D) * 0.1 + 1.0, "ln1_shift": rng.standard_normal(D) * 0.1,
        "ln2_scale": rng.standard_normal(D) * 0.1 + 1.0, "ln2_shift": rng.standard_normal(D) * 0.1,
    }
    return {k: v.astype(dtype) for k, v in p.items()}


def loss_and_grads(x, p, R, n_heads, norm=True):
    """Scalar loss = sum(out * R). Analytic gradient via backward_record."""
    out, rec = forward_record(x, p, n_heads, norm=norm)
    loss = float((out * R).sum())
    dout = R.copy()
    dx, g = backward_record(dout, rec, p, n_heads)
    return loss, dx, g


def numerical_grad_check(eps=1e-6, verbose=True):
    """Finite-difference check of EVERY parameter and of the input gradient, in float64."""
    D, H, d_mlp, T = 6, 2, 10, 3
    p = make_params(D, H, d_mlp, seed=3)
    rng = np.random.default_rng(11)
    x = rng.standard_normal((T, D))
    R = rng.standard_normal((T, D))

    for norm in (True, False):
        loss, dx, g = loss_and_grads(x, p, R, H, norm=norm)
        print("=" * 100)
        print("GRADIENT CHECK  (d_model=%d, heads=%d, d_mlp=%d, T=%d, layernorm=%s, dtype=float64)"
              % (D, H, d_mlp, T, norm))
        print("  loss = %.12f" % loss)
        worst_rel, worst_abs, worst_name = 0.0, 0.0, ""
        rows = []

        def report(name, analytic, numeric):
            nonlocal worst_rel, worst_abs, worst_name
            a = np.asarray(analytic, dtype=np.float64)
            n = np.asarray(numeric, dtype=np.float64)
            if a.shape != n.shape:
                raise SystemExit("shape mismatch on %s: %s vs %s" % (name, a.shape, n.shape))
            denom = np.maximum(np.abs(a) + np.abs(n), 1e-12)
            rel = float(np.max(np.abs(a - n) / denom))
            ab = float(np.max(np.abs(a - n)))
            rows.append((name, rel, ab))
            if rel > worst_rel:
                worst_rel, worst_abs, worst_name = rel, ab, name

        # every named parameter
        for name in p:
            num = np.zeros_like(p[name])
            it = np.nditer(p[name], flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                orig = p[name][idx]
                p[name][idx] = orig + eps
                lp = float((forward_record(x, p, H, norm=norm)[0] * R).sum())
                p[name][idx] = orig - eps
                lm = float((forward_record(x, p, H, norm=norm)[0] * R).sum())
                p[name][idx] = orig
                num[idx] = (lp - lm) / (2 * eps)
            report(name, g[name], num)

        # the input gradient too: without it a wrong dx silently breaks anything stacked above this block
        numdx = np.zeros_like(x)
        it = np.nditer(x, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            orig = x[idx]
            x[idx] = orig + eps
            lp = float((forward_records_safe(x, p, R, H, norm)[0] * R).sum())
            x[idx] = orig - eps
            lm = float((forward_records_safe(x, p, R, H, norm)[0] * R).sum())
            x[idx] = orig
            numdx[idx] = (lp - lm) / (2 * eps)
        report("dInput", dx, numdx)

        print("  %-12s %14s %14s" % ("tensor", "relative err", "absolute err"))
        for name, rel, ab in rows:
            flag = "" if rel < 1e-6 else "   <<< OVER TOLERANCE"
            print("  %-12s %14.3e %14.3e%s" % (name, rel, ab, flag))
        ok = worst_rel < 1e-6
        print("  WORST: %s  relative %.3e  absolute %.3e   -> %s"
              % (worst_name, worst_rel, worst_abs, "PASS" if ok else "FAIL"))
        if not ok:
            return False
    return True


def forward_records_safe(x, p, R, H, norm):
    return forward_record(x, p, H, norm=norm)


if __name__ == "__main__":
    ok = numerical_grad_check()
    print("\nVERDICT: %s" % ("every named gradient agrees with finite differences - the block can be trained"
                             if ok else "GRADIENTS DO NOT AGREE - do not train this block"))
    raise SystemExit(0 if ok else 1)
