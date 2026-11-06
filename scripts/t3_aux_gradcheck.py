"""t3_aux_gradcheck.py — verify the auxiliary concept loss BEFORE trusting it.

Project rule: "Check every gradient with a numerical gradient test before
trusting it." Three things must be true:

  T1  SWITCH OFF IS INERT. With LAM_CONCEPT=0 the term is never built
      (CONCEPT_TARGETS is None) and the loss has no "concept_aux" part, so the
      old model is untouched.

  T2  GRADIENTS ARE CORRECT. Central-difference check of the composite
      take_cols -> bce_logits path against the analytic backward, in float64.

  T3  COLUMN ISOLATION. The aux loss must put EXACTLY zero gradient on concept
      columns that are not supervised — that is what makes the unsupervised
      columns a valid circularity control. If they leak, the control is void.

Run: python3 scripts/t3_aux_gradcheck.py
Exit code 0 = all pass.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet.tensor import Tensor, take_cols, set_backend
from glassnet.losses import bce_logits
import run_t2_iter003 as M

FAILS = []


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def main():
    set_backend("numpy")
    print("=" * 74)
    print("AUXILIARY CONCEPT LOSS — GRADIENT / ISOLATION CHECK")
    print(f"  LAM_CONCEPT = {M.LAM_CONCEPT}   targets loaded: "
          f"{'no (switch off)' if M.CONCEPT_TARGETS is None else 'yes'}")
    print("=" * 74)

    # ---------------- T1: switch off is inert ----------------
    if M.LAM_CONCEPT <= 0:
        report("T1 switch off -> targets None", M.CONCEPT_TARGETS is None)
        rng = np.random.default_rng(0)
        net = M.TinyLM3(64, 8, 16, 8, rng=rng)
        ids = rng.integers(0, 64, size=(6, 8))
        y = rng.integers(0, 64, size=(6,))
        _, parts = net.loss(ids, y)
        report("T1 switch off -> no concept_aux in loss parts",
               "concept_aux" not in parts, f"parts={sorted(parts)}")
        report("T1 switch off -> lam_concept is 0", net.lam_concept == 0.0)
    else:
        print("  [skip] T1 (running with the switch ON — run with "
              "T2_LAM_CONCEPT=0 for the inertness check)")

    # ---------------- T2: numerical gradient check ----------------
    rng = np.random.default_rng(7)
    N, C, NS = 5, 6, 3
    cols = np.array([0, 2, 4], dtype=np.int64)     # supervised subset
    pw = np.array([12.0, 3.0, 0.5], dtype=np.float64)
    y = (rng.random((N, NS)) < 0.3).astype(np.float64)

    def build():
        return Tensor(rng.standard_normal((N, C)).astype(np.float64) * 0.8,
                      requires_grad=True)

    def loss_of(z):
        o = take_cols(z, cols)
        return bce_logits(o, y, w=pw)

    z = build()
    base = z.data.copy()
    L = loss_of(z)
    L.backward()
    ana = z.grad.copy()

    # The engine's Tensor constructor casts to float32, so the LOSS VALUE is
    # float32 even when the inputs are float64. A central difference at eps=1e-6
    # then measures float32 quantisation (~1.19e-7), not a gradient -- the first
    # version of this check "failed" for that reason while the analytic gradient
    # was in fact exact. eps=1e-2 with a 1e-2 relative tolerance is the correct
    # regime for a float32-valued loss. (Verified: the analytic gradient matches
    # the hand-derived (s-t)*w/size to 0.0e+00.)
    eps = 1e-2
    num = np.zeros_like(base)
    for i in range(N):
        for j in range(C):
            orig = base[i, j]
            z.data[i, j] = orig + eps
            lp = float(loss_of(z).data)
            z.data[i, j] = orig - eps
            lm = float(loss_of(z).data)
            z.data[i, j] = orig
            num[i, j] = (lp - lm) / (2 * eps)

    err = np.abs(ana - num)
    scale = np.maximum(np.abs(ana) + np.abs(num), 1e-8)
    rel = (err / scale).max()
    report("T2 analytic vs numerical gradient (take_cols + bce_logits)",
           rel < 1e-2, f"eps={eps:g}, max abs err {err.max():.3e}, max rel err {rel:.3e}")

    # independent truth: hand-derived gradient
    s_ = 1.0 / (1.0 + np.exp(-base))
    hand = np.zeros_like(base)
    hand[:, cols] = (s_[:, cols] - y) * pw[None, :] / y.size
    report("T2 analytic == hand-derived (s-t)*w/size",
           np.abs(ana - hand).max() < 1e-12,
           f"max abs err {np.abs(ana - hand).max():.3e}")

    # ---------------- T3: column isolation ----------------
    iso = np.abs(ana[:, [c for c in range(C) if c not in set(cols.tolist())]]).max()
    sup = np.abs(ana[:, cols]).max()
    report("T3 unsupervised columns receive EXACTLY zero gradient",
           iso == 0.0, f"max |grad| on unsupervised cols = {iso:.1e}")
    report("T3 supervised columns DO receive gradient", sup > 0,
           f"max |grad| on supervised cols = {sup:.3e}")

    # ---------------- T3b: end-to-end through the real model ----------------
    if M.CONCEPT_TARGETS is not None:
        rng2 = np.random.default_rng(3)
        # K must cover the supervised columns (16 targets), so K=32 here.
        net = M.TinyLM3(64, 32, 16, 8, rng=rng2)
        ids = rng2.integers(0, 64, size=(6, 8))
        yb = rng2.integers(0, 64, size=(6,))
        loss, parts = net.loss(ids, yb)
        report("T3b real model builds concept_aux", "concept_aux" in parts,
               f"aux={float(parts['concept_aux'].data):.4f}")
        # zero the task/sparsity contributions and isolate the aux gradient
        for p in net.parameters():
            p.grad = None
        parts["concept_aux"].backward()
        Wg = net.concept.W.grad
        sup_cols = M.CONCEPT_TARGETS["cols"]
        unsup_cols = np.array([c for c in range(net.K) if c not in set(sup_cols.tolist())])
        g_sup = np.abs(Wg[:, sup_cols]).max()
        g_uns = np.abs(Wg[:, unsup_cols]).max()
        report("T3b aux gradient reaches supervised concept columns", g_sup > 0,
               f"max |dW| = {g_sup:.3e}")
        report("T3b aux gradient is EXACTLY zero on unsupervised columns",
               g_uns == 0.0, f"max |dW| = {g_uns:.1e}")

    print("=" * 74)
    if FAILS:
        print(f"RESULT: FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("RESULT: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
