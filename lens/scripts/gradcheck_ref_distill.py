"""gradcheck_ref_distill.py — verify distill_kl END-TO-END against a float64 reference.

WHY A DIFFERENT METHOD THAN THE FAILED PROBE: the engine casts every Tensor to float32
(tensor.py L79-80), so the loss VALUE itself carries float32 rounding (~1e-7 on a 0.2-scale loss).
Finite-differencing that value can never have a signal smaller than its own rounding, and both eps
attempts proved it (2.3e-2 relative at eps=1e-2; exactly 1.0 at eps=1e-4 where the signal dropped
below the floor). That is a property of the PROBE, not the op.

THE PRECEDENT THIS FOLLOWS: gradcheck_wce.py verified softmax_ce(class_weights) by comparing the
ENGINE's gradient against a hand-built NUMPY REFERENCE (bit-identical loss 0.00e+00, gradient
1.49e-08). Same idea here, one level deeper:

  REFERENCE forward (float64 numpy, mirrors the engine graph exactly):
      Pp   = patches(X)                     # (B,P,49)
      z    = Pp.reshape(-1,49) @ fW + fb    # (B*P,D)
      zf   = reshape to (B, P*D)
      cl   = zf @ cW + cb ;  h = sigmoid(cl)
      cxt  = cap_emb[S] flattened to (B, W*D)
      inp  = concat([h, cxt])
      logits = inp @ oW + ob
      L    = tau^2 * KL(qt || softmax(logits/tau))

  FD: perturb ONE reference parameter entry by eps=1e-3, recompute L (float64 — no rounding floor),
  central difference -> NUMERIC gradient.
  ENGINE: build the same graph with Tensors, run distill_kl, backward -> ANALYTIC gradients on the
  actual parameter Tensors.
  COMPARE per-entry, pass = rel<1e-2 OR abs<1e-5.

This checks the OP MATH *and* the WIRING: distill gradients must reach cap_emb rows (embedding
backward), concept/front (through sigmoid and the two reshapes), and concat splitting — not just
the last Dense.
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens.tensor as T                                  # noqa: E402
import train_seetalk                                     # noqa: E402
from lens import backend                                 # noqa: E402
from teacher_soft import distill_kl                      # noqa: E402


def ref_logits(Pn, S, fW, fb, cW, cb, emb, oW, ob, P, D, W):
    B = Pn.shape[0]
    z = Pn.reshape(-1, Pn.shape[-1]) @ fW + fb
    zf = z.reshape(B, P * D)
    h = 1.0 / (1.0 + np.exp(-np.clip(zf @ cW + cb, -60, 60)))
    cxt = np.concatenate([emb[S[:, w]] for w in range(W)], axis=1)
    return np.concatenate([h, cxt], axis=1) @ oW + ob, h, zf


def main():
    T.set_device("cpu")
    rng = np.random.default_rng(0)
    data = train_seetalk.build_caption_data(n=400, seed=0)
    concepts, _ = train_seetalk.image_concepts.build(data["Xtr"][:300], min_support=0.02,
                                                     verbose=False)
    net = train_seetalk.SeeTalkNet(concepts, len(data["vocab"]), d=16, window=data["W"], seed=1)
    B = 16
    X = data["Xtr"][:B].astype(np.float64)
    S = data["Str"][:B]
    V = len(data["vocab"])
    qt = rng.random((B, V))
    qt /= qt.sum(axis=1, keepdims=True)

    def getf64():
        return {k: backend.to_cpu(v.data).astype(np.float64) for k, v in {
            "fW": net.front.proj.W, "fb": net.front.proj.b,
            "cW": net.concept.W, "cb": net.concept.b,
            "oW": net.cap_out.W, "ob": net.cap_out.b, "emb": net.cap_emb}.items()}

    Pn = net.front.patches(X.astype(np.float32))          # (B,P,49)
    P, D, W, tau = net.P, net.D, net.W, 2.0

    def ref_loss(pr):
        lg, _, _ = ref_logits(Pn, S, pr["fW"], pr["fb"], pr["cW"], pr["cb"], pr["emb"],
                              pr["oW"], pr["ob"], P, D, W, )
        zt = lg / tau
        zt -= zt.max(axis=1, keepdims=True)
        p = np.exp(zt)
        p /= p.sum(axis=1, keepdims=True)
        q = np.clip(qt, 1e-12, None)
        q = q / q.sum(axis=1, keepdims=True)
        return float(tau * tau * (q * (np.log(q) - np.log(np.clip(p, 1e-12, None)))
                                  ).sum(axis=1).mean())

    def engine_grads():
        for p in net.parameters():
            p.grad = None
        out = net.forward(X.astype(np.float32), S)
        L = distill_kl(out["logits"], qt.astype(np.float32), tau=tau)
        L.backward()
        return {k: backend.to_cpu(v.grad.data).astype(np.float64) for k, v in {
            "fW": net.front.proj.W, "fb": net.front.proj.b,
            "cW": net.concept.W, "cb": net.concept.b,
            "oW": net.cap_out.W, "ob": net.cap_out.b, "emb": net.cap_emb}.items()}

    pr = getf64()
    g_eng = engine_grads()
    eps = 1e-3
    print("=" * 92)
    print("distill_kl END-TO-END gradcheck: float64 reference FD vs engine analytic gradients")
    print(f"  tau={tau}, B={B}, V={V}, eps={eps:g};  pass = rel<1e-2 OR abs<1e-5")
    print("=" * 92)
    all_ok = True
    for pname, arr in pr.items():
        entries = rng.choice(arr.size, size=min(6, arr.size), replace=False).ravel()
        worst_rel = worst_abs = 0.0
        for i in entries:
            a = np.unravel_index(i, arr.shape)
            orig = arr[a]
            pp = dict(pr); pp[pname] = arr.copy(); pp[pname][a] = orig + eps
            pm = dict(pr); pm[pname] = arr.copy(); pm[pname][a] = orig - eps
            num = (ref_loss(pp) - ref_loss(pm)) / (2 * eps)
            ana = g_eng[pname][a]
            rel = abs(num - ana) / max(abs(num), abs(ana), 1e-12)
            ab = abs(num - ana)
            worst_rel = max(worst_rel, rel)
            worst_abs = max(worst_abs, ab)
        ok = worst_rel < 1e-2 or worst_abs < 1e-5
        all_ok = all_ok and ok
        filled = "grads present" if np.isfinite(g_eng[pname]).all() else "NONFINITE GRAD"
        nonzero = abs(g_eng[pname]).sum() > 0
        print(f"  {pname:<4} {str(arr.shape):>10}  worst rel {worst_rel:.3e}  "
              f"worst abs {worst_abs:.3e}  {filled}, {'flows' if nonzero else 'ZERO!'} -> "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    print(f"  OVERALL: {'PASS — op math AND wiring verified on every parameter' if all_ok else 'FAIL'}")
    print("=" * 92)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
