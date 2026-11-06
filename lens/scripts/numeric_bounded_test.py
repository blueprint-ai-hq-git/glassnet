#!/usr/bin/env python3
"""numeric_bounded_test.py - PLAN.txt item (d3b): the BOUNDED magnitude representation.

Predictions written BEFORE this script existed and before any run:
lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44a-i). Output: lens/runs/NUMERIC-BOUNDED.txt (appended, never
overwritten).

WHAT IS BEING TESTED, AND WHAT IS NOT
  The arithmetic exactness gate (P29a-d) is the arbiter this project uses: train on addition with operands
  0..9, score on operands 10..19, which no memorised table can answer. lens/scripts/numeric_path_test.py is
  IMPORTED UNCHANGED - the way numeric_path_opt_test.py imports it - so the ONLY difference between this file
  and the stored NUMERIC-PATH.txt is the REPRESENTATION of the numbers:

      unbounded (shipped):  q_t = v_t / 40                     target = (a+b) / 40
      bounded   (this test): q_t = v_t / d,  d = max(1, max_j |v_j|)   target = (a+b) / d

  with the readout inverting the bounded form by multiplying by d - a FIXED, declared transform, exactly the
  role SCALE plays in the shipped form. The divisor comes from the window's own numbers, which are the input,
  so nothing is fitted on train data and no train/val leakage is possible.

  NOT TESTED HERE, AND NOT CLAIMED: anything about chat accuracy. The numeric term is still OFF over chat
  text (lens/runs/NUMERIC-CHAT-READOUT.txt: a linear readout of the numbers in a chat window carries no
  held-out signal about the next token's quantity, P43l FAIL). This file only answers (d3b)'s own bar:
  does the bounded representation move the ARITHMETIC numbers?

Usage: NUMERIC_LR=0.2 python3 lens/scripts/numeric_bounded_test.py
"""
import datetime
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.chdir(ROOT)

import numeric_path_test as npt                                        # noqa: E402  (the harness, IMPORTED)
from lens import backend, tensor as T                                  # noqa: E402
from lens.numeric import NumericHead, magnitude_channel, num_mse, window_divisor   # noqa: E402

W = npt.W
SCALE = npt.SCALE
V = npt.V
TOK = npt.TOK
LR = float(os.environ.get("NUMERIC_LR", "0.2"))
STEPS = int(os.environ.get("NUMERIC_STEPS", "4000"))
LAM_NUM = npt.LAM_NUM
OUT = os.path.join(ROOT, "lens", "runs", "NUMERIC-BOUNDED.txt")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def corpus_bounded(lo, hi):
    """[a,'+',b,'='] -> answer token, answer as a quantity BOUNDED by the window's own divisor, the bounded
    magnitude channel, the numeric mask (1 where the token is NOT a number) and the divisor itself."""
    pairs = [(a, b) for a in range(lo, hi) for b in range(lo, hi)]
    ids = np.asarray([[a, 39, b, 40] for a, b in pairs], dtype=np.int64)
    ytok = np.asarray([a + b for a, b in pairs], dtype=np.int32)
    raw, numeric = magnitude_channel(ids, TOK, scale=1.0)               # ORIGINAL units, unbounded
    d = window_divisor(raw)                                            # (N,1), float32, from the window itself
    mag = raw / d
    ynum = ytok.astype(np.float32).reshape(-1, 1) / d
    return pairs, ids, ytok, ynum, mag, (1.0 - numeric), d


def quantity(net, ids, mag, mask, d):
    """The head's answer in ORIGINAL units: multiply by the divisor, the declared inverse of the bounded form."""
    out = net.forward_arrays(ids, mag=mag, magmask=mask)
    return (np.asarray(out["num"]).reshape(-1, 1) * d).reshape(-1)


def main():
    say("=" * 100)
    say("PLAN (d3b) THE BOUNDED MAGNITUDE REPRESENTATION ON THE ARITHMETIC EXACTNESS GATE")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    say("predictions written first: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44a-i)")
    say("harness: lens/scripts/numeric_path_test.py IMPORTED UNCHANGED - only the representation differs")
    say("=" * 100)
    say(f"bounded: q_t = v_t / d,  d = max(1, max_j |v_j|) over the window's own numbers   target = (a+b)/d")
    say(f"readout: (head output) * d      train 100 pairs (0..9)   unseen 100 pairs (10..19)   lr {LR}   "
        f"steps {STEPS}")
    say("")

    tr = corpus_bounded(0, 10)
    ex = corpus_bounded(10, 20)
    ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, d_tr = tr[1], tr[2], tr[3], tr[4], tr[5], tr[6]
    ids_ex, ytok_ex, ynum_ex, mag_ex, mask_ex, d_ex = ex[1], ex[2], ex[3], ex[4], ex[5], ex[6]

    # ---- the declared inverse, checked in exact arithmetic before any model exists
    exact_pred = (mag_tr[:, 0] + mag_tr[:, 2]) * d_tr.reshape(-1)        # weights 1 and 1 on the operands
    inv_err = float(np.max(np.abs(exact_pred - ytok_tr)))
    say(f"  declared inverse, weights 1 and 1 on the two operand positions, in float32: max |pred - (a+b)| = "
        f"{inv_err:.3e}   -> {'EXACT' if inv_err == 0.0 else 'NOT EXACT'}")
    say(f"  divisor on the train windows: min {float(d_tr.min()):.1f}  max {float(d_tr.max()):.1f}  "
        f"windows with d > 1: {int((d_tr.reshape(-1) > 1).sum())}/{len(d_tr)}")
    say("")

    rows = {}
    say(f"  {'arm':44s} {'seed':>4s} {'train tok':>9s} {'unseen tok':>10s} {'unseen exact':>12s} {'MAE':>8s}")
    for seed in (0, 1, 2):
        # ---- A: the architecture as it is today (control; must stay at 0% on unseen)
        netA, _ = npt.make_net(seed, numeric=False)
        npt.sgd(netA, ids_tr, ytok_tr, None, None, None, lam_num=0.0, seed=seed)
        a_tr = npt.exact_tok(netA, ids_tr, ytok_tr)
        a_ex = npt.exact_tok(netA, ids_ex, ytok_ex)
        rows.setdefault("A presence + classifier", []).append((a_tr, a_ex, None, 1))
        say(f"  {'A presence concepts + token classifier':44s} {seed:>4d} {a_tr*100:>8.1f}% "
            f"{a_ex*100:>9.1f}% {'n/a':>12s} {'n/a':>8s}")

        # ---- B: bounded magnitudes + numeric head inside the real architecture
        netB, _ = npt.make_net(seed, numeric=True)
        npt.sgd(netB, ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr, seed=seed)
        b_tr_tok = npt.exact_tok(netB, ids_tr, ytok_tr, mag=mag_tr, mask=mask_tr)
        b_ex_tok = npt.exact_tok(netB, ids_ex, ytok_ex, mag=mag_ex, mask=mask_ex)
        pred_tr = quantity(netB, ids_tr, mag_tr, mask_tr, d_tr)
        pred_ex = quantity(netB, ids_ex, mag_ex, mask_ex, d_ex)
        b_tr_exact = float((np.round(pred_tr) == ytok_tr).mean())
        b_ex_exact = float((np.round(pred_ex) == ytok_ex).mean())
        b_ex_mae = float(np.abs(pred_ex - ytok_ex).mean())
        rows.setdefault("B bounded magnitudes + num head", []).append(
            (b_tr_exact, b_ex_exact, b_ex_mae, int(bool(np.isfinite(pred_ex).all()))))
        say(f"  {'B concepts + BOUNDED magnitudes + numeric head':44s} {seed:>4d} {b_tr_tok*100:>8.1f}% "
            f"{b_ex_tok*100:>9.1f}% {b_ex_exact*100:>11.1f}% {b_ex_mae:>8.3f}   "
            f"(num head; token cls seen {b_tr_exact*100:.0f}% unseen {b_ex_tok*100:.0f}%)")

        # ---- B0: the same model, channel ZEROED at test time
        pred0 = quantity(netB, ids_ex, np.zeros_like(mag_ex), mask_ex, d_ex)
        b0_exact = float((np.round(pred0) == ytok_ex).mean())
        b0_mae = float(np.abs(pred0 - ytok_ex).mean())
        rows.setdefault("B0 channel zeroed at test", []).append(
            (None, b0_exact, b0_mae, int(bool(np.isfinite(pred0).all()))))
        say(f"  {'B0 same model, bounded channel ZEROED':44s} {seed:>4d} {'-':>9s} {'-':>10s} "
            f"{b0_exact*100:>11.1f}% {b0_mae:>8.3f}")

        # ---- C: bounded channel -> numeric head only (harness check, no embedding path)
        rng = np.random.default_rng(seed)
        head = NumericHead(W, rng, name="num_cb")
        q_tr = T.Tensor(backend.to_dev(mag_tr))
        q_ex = T.Tensor(backend.to_dev(mag_ex))
        for _ in range(STEPS):
            for p in head.parameters():
                p.grad = None
            num_mse(head(q_tr), ynum_tr).backward()
            for p in head.parameters():
                if p.grad is not None:
                    p.data = p.data - LR * p.grad
        c_tr = (np.asarray(backend.to_cpu(head(q_tr).data)).reshape(-1, 1) * d_tr).reshape(-1)
        c_ex = (np.asarray(backend.to_cpu(head(q_ex).data)).reshape(-1, 1) * d_ex).reshape(-1)
        c_tr_exact = float((np.round(c_tr) == ytok_tr).mean())
        c_ex_exact = float((np.round(c_ex) == ytok_ex).mean())
        c_ex_mae = float(np.abs(c_ex - ytok_ex).mean())
        rows.setdefault("C bounded mags -> num head only", []).append(
            (c_tr_exact, c_ex_exact, c_ex_mae, int(bool(np.isfinite(c_ex).all()))))
        say(f"  {'C bounded magnitude channel -> numeric head':44s} {seed:>4d} {c_tr_exact*100:>8.1f}% "
            f"{'n/a':>10s} {c_ex_exact*100:>11.1f}% {c_ex_mae:>8.3f}")

    say("")
    say("-" * 100)
    say("  MEANS OVER 3 SEEDS (spread = max-min). A seed whose numbers came out non-finite is EXCLUDED from its")
    say("  mean and counted in 'seeds finite'; it still counts as 0% exact, because a NaN answer is not right.")
    means = {}
    for k, v in rows.items():
        arr = np.asarray([[x if x is not None else np.nan for x in row] for row in v], dtype=np.float64)
        with np.errstate(invalid="ignore"):
            means[k] = np.nanmean(arr, axis=0)
        finite = int(arr[:, 3].sum())
        say(f"    {k:44s} unseen exact {means[k][1]*100:6.1f}%   "
            f"(spread {(np.nanmax(arr[:, 1]) - np.nanmin(arr[:, 1]))*100:4.1f} pts)   "
            f"MAE {means[k][2]:.4f}   seeds finite {finite}/3")

    a_unseen = means["A presence + classifier"][1]
    b_unseen = means["B bounded magnitudes + num head"][1]
    b0_unseen = means["B0 channel zeroed at test"][1]
    c_unseen = means["C bounded mags -> num head only"][1]
    b_mae = means["B bounded magnitudes + num head"][2]

    p44b = c_unseen >= 0.95
    p44c = b_unseen >= 0.90
    p44d = b0_unseen <= 0.10
    say("")
    say("  VERDICTS (bars declared in lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt before this ran)")
    say(f"  P44b bounded arm C unseen >= 95%      : {c_unseen*100:6.1f}%   "
        f"{'PASS' if p44b else 'FAIL'}")
    say(f"  P44c bounded arm B unseen >= 90%      : {b_unseen*100:6.1f}%   "
        f"{'PASS' if p44c else 'FAIL'}")
    say(f"  P44d bounded arm B0 (zeroed) <= 10%   : {b0_unseen*100:6.1f}%   "
        f"{'PASS' if p44d else 'FAIL'}")
    say(f"  (context, no bar here: arm A unseen {a_unseen*100:.1f}%, arm B MAE in original units {b_mae:.4f})")
    say(f"  BOUNDED ARITHMETIC GATE: {'PASS' if (p44b and p44c and p44d) else 'FAIL'}")
    say("")
    say("  NOT CLAIMED: nothing about chat accuracy. The numeric term stays OFF over chat text (P43l FAIL,")
    say("  lens/runs/NUMERIC-CHAT-READOUT.txt). This file answers (d3b)'s own bar only: the bounded")
    say("  representation does not move the arithmetic numbers.")
    say("=== DONE " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ===")

    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended to {OUT}")
    return 0 if (p44b and p44c and p44d) else 1


if __name__ == "__main__":
    sys.exit(main())
