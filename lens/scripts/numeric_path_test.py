#!/usr/bin/env python3
"""numeric_path_test.py - DOES THE NUMERIC PATH MAKE A RULE LEARNABLE INSIDE THE REAL ARCHITECTURE?

PLAN.txt item (d). Predictions written BEFORE this run: lens/runs/NUMERIC-PATH-PREDICTIONS.txt (P29a-e).
Output: lens/runs/NUMERIC-PATH.txt

THE TEST (the arbiter the whole project uses): train on addition pairs with operands 0..9, then score on
pairs with operands 10..19, which no memorised table can answer. 100 train pairs, 100 unseen pairs.

  A   presence concepts + token classifier        = the architecture as it is today (control)
  B   concepts + MAGNITUDE channel + numeric head = the change (numeric=True)
  B0  arm B with the magnitude channel ZEROED     = control on the channel itself
  C   magnitude channel -> numeric head only      = harness check (reproduces the probe's lever)

The magnitude channel is the number a token denotes, scaled by 40, so a linear head can express a+b as
weights 1 and 1 on the two operand positions. It is an INPUT, not something the model must recover from a
token id it has never seen.

Usage: python3 lens/scripts/numeric_path_test.py
"""
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
os.chdir(ROOT)

import lens  # noqa: E402
from lens import backend, tensor as T  # noqa: E402
from lens.concepts import Concept  # noqa: E402
from lens.model import Dense  # noqa: E402  (kept for readers: the numeric head is one affine layer)
from lens.numeric import NumericHead, magnitude_channel, num_mse, token_value  # noqa: E402

W = 4
SCALE = 40.0
V = 42                                   # 0..38 = the numbers, 39 = '+', 40 = '=', 41 = '<pad>'
TOK = {i: str(i) for i in range(39)}
TOK[39], TOK[40], TOK[41] = "+", "=", "<pad>"
STEPS = int(os.environ.get("NUMERIC_STEPS", "4000"))
LR = float(os.environ.get("NUMERIC_LR", "0.05"))
LAM_CONCEPT = 0.5
LAM_NUM = 1.0


def corpus(lo, hi):
    """Windows [a, '+', b, '='] -> the answer token, the answer as a quantity, the magnitude channel and the
    numeric mask (1 where the token is NOT a number, i.e. where a learned readout is allowed)."""
    pairs = [(a, b) for a in range(lo, hi) for b in range(lo, hi)]
    ids = np.asarray([[a, 39, b, 40] for a, b in pairs], dtype=np.int64)
    ytok = np.asarray([a + b for a, b in pairs], dtype=np.int32)
    ynum = (ytok.astype(np.float32) / SCALE).reshape(-1, 1)
    mag, numeric = magnitude_channel(ids, TOK, scale=SCALE)
    return pairs, ids, ytok, ynum, mag, 1.0 - numeric


def sgd(net, ids, ytok, ynum, mag, mask, steps=STEPS, lr=LR, lam_num=LAM_NUM, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        for p in net.parameters():
            p.grad = None
        loss, parts = net.loss(ids, ytok, mag=mag, ynum=ynum, lam_num=lam_num, magmask=mask)
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                p.data = p.data - lr * p.grad
    return parts


def exact_tok(net, ids, ytok, mag=None, mask=None):
    out = net.forward_arrays(ids, mag=mag, magmask=mask)
    return float((np.argmax(out["logits"], axis=1) == ytok).mean())


def quantity(net, ids, mag, mask):
    out = net.forward_arrays(ids, mag=mag, magmask=mask)
    return np.asarray(out["num"]).reshape(-1) * SCALE


def make_net(seed, numeric):
    rng = np.random.default_rng(seed)
    pairs, ids, ytok, ynum, mag, mask = corpus(0, 10)
    concepts = []
    for d in range(10):
        support = float((ids == d).any(axis=1).mean())
        concepts.append(Concept(f"digit_{d}", [d], support=support))
    net = lens.Net(vocab=V, concepts=concepts, d=32, window=W, seed=seed, numeric=numeric)
    return net, concepts


def main():
    tr = corpus(0, 10)
    ex = corpus(10, 20)
    ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr = tr[1], tr[2], tr[3], tr[4], tr[5]
    ids_ex, ytok_ex, ynum_ex, mag_ex, mask_ex = ex[1], ex[2], ex[3], ex[4], ex[5]
    mag_ex_zero = np.zeros_like(mag_ex)

    print("=" * 100)
    print("THE NUMERIC / MAGNITUDE PATH INSIDE THE REAL ARCHITECTURE - does a RULE become learnable?")
    print("=" * 100)
    print(f"train: 100 pairs, operands 0..9      unseen: 100 pairs, operands 10..19 (no table can answer)")
    print(f"window [a, '+', b, '='] W={W}  vocab {V}  d=32  steps {STEPS}  lr {LR}  plain SGD (as train_chat.py)")
    print(f"predictions: lens/runs/NUMERIC-PATH-PREDICTIONS.txt (P29a-e, written before this run)")
    print()
    print(f"  {'arm':44s} {'seed':>4s} {'train tok':>9s} {'unseen tok':>10s} {'unseen exact':>12s} "
          f"{'MAE':>7s}")
    rows = {}
    for seed in (0, 1, 2):
        # ---- A: presence concepts + token classifier (the architecture today)
        netA, _ = make_net(seed, numeric=False)
        sgd(netA, ids_tr, ytok_tr, None, None, None, lam_num=0.0, seed=seed)
        a_tr = exact_tok(netA, ids_tr, ytok_tr)
        a_ex = exact_tok(netA, ids_ex, ytok_ex)
        rows.setdefault("A presence + classifier", []).append((a_tr, a_ex, None, 1))
        print(f"  {'A presence concepts + token classifier':44s} {seed:>4d} {a_tr*100:>8.1f}% "
              f"{a_ex*100:>9.1f}% {'n/a':>12s} {'n/a':>7s}")

        # ---- B: concepts + magnitude channel + numeric head
        netB, _ = make_net(seed, numeric=True)
        parts = sgd(netB, ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, seed=seed)
        b_tr_tok = exact_tok(netB, ids_tr, ytok_tr, mag=mag_tr, mask=mask_tr)
        b_ex_tok = exact_tok(netB, ids_ex, ytok_ex, mag=mag_ex, mask=mask_ex)
        pred_ex = quantity(netB, ids_ex, mag_ex, mask_ex)
        b_ex_exact = float((np.round(pred_ex) == ytok_ex).mean())
        b_ex_mae = float(np.abs(pred_ex - ytok_ex).mean())
        pred_tr = quantity(netB, ids_tr, mag_tr, mask_tr)
        b_tr_exact = float((np.round(pred_tr) == ytok_tr).mean())
        rows.setdefault("B concepts + magnitudes + num head", []).append(
            (b_tr_exact, b_ex_exact, b_ex_mae, int(bool(np.isfinite(pred_ex).all()))))
        print(f"  {'B concepts + MAGNITUDES + numeric head':44s} {seed:>4d} {b_tr_tok*100:>8.1f}% "
              f"{b_ex_tok*100:>9.1f}% {b_ex_exact*100:>11.1f}% {b_ex_mae:>7.3f}   "
              f"(num head; token cls seen {b_tr_exact*100:.0f}% unseen {b_ex_tok*100:.0f}%)")

        # ---- B0: same model, magnitude channel ZEROED at test time (the numbers are simply not there)
        pred0 = quantity(netB, ids_ex, mag_ex_zero, mask_ex)
        b0_exact = float((np.round(pred0) == ytok_ex).mean())
        b0_mae = float(np.abs(pred0 - ytok_ex).mean())
        rows.setdefault("B0 channel zeroed at test", []).append(
            (None, b0_exact, b0_mae, int(bool(np.isfinite(pred0).all()))))
        print(f"  {'B0 same model, magnitude channel ZEROED':44s} {seed:>4d} {'-':>9s} {'-':>10s} "
              f"{b0_exact*100:>11.1f}% {b0_mae:>7.3f}")

        # ---- C: magnitude channel -> numeric head only (harness check, no embedding path)
        rng = np.random.default_rng(seed)
        head = NumericHead(W, rng, name="num_c")
        q_tr = T.Tensor(backend.to_dev(mag_tr))
        q_ex = T.Tensor(backend.to_dev(mag_ex))
        for _ in range(STEPS):
            for p in head.parameters():
                p.grad = None
            num_mse(head(q_tr), ynum_tr).backward()
            for p in head.parameters():
                if p.grad is not None:
                    p.data = p.data - LR * p.grad
        c_tr = np.asarray(backend.to_cpu(head(q_tr).data)).reshape(-1) * SCALE
        c_ex = np.asarray(backend.to_cpu(head(q_ex).data)).reshape(-1) * SCALE
        c_ex_exact = float((np.round(c_ex) == ytok_ex).mean())
        c_ex_mae = float(np.abs(c_ex - ytok_ex).mean())
        rows.setdefault("C magnitudes -> num head only", []).append(
            (float((np.round(c_tr) == ytok_tr).mean()), c_ex_exact, c_ex_mae,
             int(bool(np.isfinite(c_ex).all()))))
        print(f"  {'C magnitude channel -> numeric head only':44s} {seed:>4d} "
              f"{rows['C magnitudes -> num head only'][-1][0]*100:>8.1f}% {'n/a':>10s} "
              f"{c_ex_exact*100:>11.1f}% {c_ex_mae:>7.3f}")

    print()
    print("-" * 100)
    print("  MEANS OVER 3 SEEDS (spread = max-min). A seed whose predictions came out non-finite is EXCLUDED")
    print("  from its mean and counted in 'seeds finite' - silently averaging a divergence away would hide it.")
    print("  A diverged seed still counts as 0% exact, because a NaN answer is not a right answer.")
    means = {}
    for k, v in rows.items():
        arr = np.asarray([[x if x is not None else np.nan for x in row] for row in v], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            means[k] = np.nanmean(arr, axis=0)
        finite = int(arr[:, 3].sum())
        print(f"    {k:44s} unseen exact {means[k][1]*100:6.1f}%   "
              f"(spread {(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]))*100:4.1f} pts)   "
              f"MAE {means[k][2] if not np.isnan(means[k][2]) else float('nan'):.3f}   "
              f"seeds finite {finite}/3")

    a_unseen = means["A presence + classifier"][1]
    b_unseen = means["B concepts + magnitudes + num head"][1]
    b0_unseen = means["B0 channel zeroed at test"][1]
    c_unseen = means["C magnitudes -> num head only"][1]
    gate = (a_unseen <= 0.05) and (b_unseen >= 0.90) and (b0_unseen <= 0.10) and (c_unseen >= 0.95)
    print()
    print(f"  P29a arm A unseen <= 5%          : {a_unseen*100:6.1f}%   "
          f"{'PASS' if a_unseen <= 0.05 else 'FAIL'}")
    print(f"  P29b arm C unseen >= 95% (harness): {c_unseen*100:6.1f}%   "
          f"{'PASS' if c_unseen >= 0.95 else 'FAIL'}")
    print(f"  P29c arm B unseen >= 90%          : {b_unseen*100:6.1f}%   "
          f"{'PASS' if b_unseen >= 0.90 else 'FAIL'}")
    print(f"  P29d arm B0 (zeroed) <= 10%       : {b0_unseen*100:6.1f}%   "
          f"{'PASS' if b0_unseen <= 0.10 else 'FAIL'}")
    print()
    print(f"  NUMERIC PATH GATE: {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  A FAIL here is a real answer: the numeric path does not make the rule learnable inside")
        print("  this architecture at these settings, and the arithmetic claim must not be made.")
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
