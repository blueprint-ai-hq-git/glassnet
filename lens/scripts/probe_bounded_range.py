#!/usr/bin/env python3
"""probe_bounded_range.py - PLAN.txt item (d3b): is the bounded representation actually bounded, on chat text?

Predictions written BEFORE this run: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44e). Read-only over the
corpus; no training, no GPU, no writes to the corpus. Output: lens/runs/NUMERIC-BOUNDED-RANGE.txt.

WHAT IT MEASURES, on windows of the CLEAN chat corpus (which=0, seeded, held-out split untouched):
  * the channel's and the target's maximum absolute value, bounded (v/d) against unbounded (v/scale), and
    whether every value is finite;
  * how often the divisor is actually different from 1 (a representation that is inert is not a fix);
  * the log-compression alternative's arithmetic defect, so the choice made in lens/numeric.py is a measured
    fact and not a preference: max over a,b in 0..9 of |log1p(a+b) - (log1p(a) + log1p(b))| and the factor by
    which that would miss the answer after inverting.

Usage: python3 lens/scripts/probe_bounded_range.py

A BUG OF MINE, FIXED IN THIS FILE BEFORE IT RAN TO COMPLETION (recorded, not hidden): the first version divided
the target array by the (N,1) divisor while chat_data.make_windows returns Y as (N,) - the single next token -
so numpy broadcast (N,) against (N,1) and tried to allocate a (100,000 x 100,000) array: "Unable to allocate
37.3 GiB", exit 1. The divisor is now reshaped to (N,). The channel (N,W) was never affected.
"""
import datetime
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.chdir(ROOT)
D_DIR = os.environ.get("CHAT_DATA_DIR") or os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D_DIR

import chat_data                                                       # noqa: E402
from lens.numeric import magnitude_channel, numeric_targets, window_divisor, _parse_decimal   # noqa: E402

W = 32
SCALE = 100.0                       # the scale the chat readout used (lens/scripts/numeric_chat_readout.py)
N = int(os.environ.get("BOUNDED_WINDOWS", "100000"))
OUT = os.path.join(ROOT, "lens", "runs", "NUMERIC-BOUNDED-RANGE.txt")
JSON_OUT = os.path.join(ROOT, "lens", "runs", "numeric_bounded_range.json")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def main():
    say("=" * 100)
    say("IS THE BOUNDED MAGNITUDE REPRESENTATION ACTUALLY BOUNDED? (chat corpus, closed form, no training)")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {D_DIR}")
    say("predictions written first: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44e)")
    say("=" * 100)
    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    X, Y = chat_data.make_windows(ids, bounds, split, W, which=0, max_windows=N, seed=0)
    say(f"windows: {len(Y):,} drawn from the clean corpus which=0 (seeded), split untouched")

    vvals = np.zeros(len(vocab), dtype=np.float32)
    vmask = np.zeros(len(vocab), dtype=np.float32)
    for i, tok in enumerate(vocab):
        v = _parse_decimal(tok)
        if v is not None:
            vvals[i], vmask[i] = v, 1.0

    id_to_token = {i: t for i, t in enumerate(vocab)}
    chk_v, chk_m = magnitude_channel(X[:2000], id_to_token, scale=1.0)
    f_v, f_m = vvals[np.asarray(X[:2000])], vmask[np.asarray(X[:2000])]
    assert np.array_equal(np.asarray(chk_v), f_v) and np.array_equal(np.asarray(chk_m), f_m), \
        "the fast lookup disagrees with magnitude_channel - do not trust this run"
    say("   fast value lookup checked against lens.numeric.magnitude_channel on 2,000 windows: identical")

    raw = vvals[np.asarray(X)]                      # ORIGINAL units, unbounded
    nummask = vmask[np.asarray(X)]                  # 1 where the token IS a number
    divisor = window_divisor(raw)
    q_bounded = raw / divisor
    q_unb = raw / np.float32(SCALE)
    tgt_raw, tgt_mask = numeric_targets(vocab, Y, scale=1.0)     # Y is (N,): the single next token
    tgt_bounded = tgt_raw / divisor.reshape(-1)                  # the SAME divisor as the window's channel

    say("")
    say("  THE CHANNEL (values at the window's positions)")
    say(f"    bounded   q = v/d      max |q| {float(np.max(np.abs(q_bounded))):>12.6f}   "
        f"finite {bool(np.isfinite(q_bounded).all())}")
    say(f"    unbounded q = v/100    max |q| {float(np.max(np.abs(q_unb))):>12.6f}   "
        f"finite {bool(np.isfinite(q_unb).all())}")
    say(f"    unbounded q = v (x1)   max |q| {float(np.max(np.abs(raw))):>12.6f}   "
        f"finite {bool(np.isfinite(raw).all())}")
    say("")
    say("  THE NUMERIC TARGET (the value of the token the classifier is trained to predict)")
    say(f"    bounded   t = v/d      max |t| {float(np.max(np.abs(tgt_bounded))):>12.6f}   "
        f"finite {bool(np.isfinite(tgt_bounded).all())}   rows with a target {int((tgt_mask > 0.5).sum()):,}"
        f"/{len(Y):,}")
    say(f"    unbounded t = v        max |t| {float(np.max(np.abs(tgt_raw))):>12.6f}   "
        f"finite {bool(np.isfinite(tgt_raw).all())}")
    say("")
    say("  IS THE DIVISOR DOING ANYTHING? (an inert representation is not a fix)")
    say(f"    windows with d > 1: {int((divisor.reshape(-1) > 1).sum()):,}/{len(Y):,}   "
        f"d min {float(divisor.min()):.1f}  d max {float(divisor.max()):.1f}  "
        f"d median {float(np.median(divisor)):.1f}")
    say(f"    windows with at least one number: {int((nummask.sum(axis=1) > 0).sum()):,}/{len(Y):,}")
    say("")
    say("  THE LOG-COMPRESSION ALTERNATIVE, MEASURED (why lens/numeric.py divides instead of taking a log)")
    worst = 0.0
    worst_ab = (0, 0)
    worst_factor = 1.0
    for a in range(10):
        for b in range(10):
            lhs = float(np.log1p(a + b))
            rhs = float(np.log1p(a) + np.log1p(b))
            if abs(lhs - rhs) > worst:
                worst, worst_ab = abs(lhs - rhs), (a, b)
            ans_true = a + b
            ans_log = float(np.expm1(rhs))
            if ans_true > 0:
                worst_factor = max(worst_factor, abs(ans_log - ans_true) / ans_true)
    say(f"    max |log1p(a+b) - (log1p(a)+log1p(b))| over a,b in 0..9: {worst:.6f}  at a={worst_ab[0]}, "
        f"b={worst_ab[1]}")
    say(f"    the answer a+b it would produce after inverting: up to {worst_factor:.2f}x too large")
    say("    -> a linear head over log-compressed magnitudes cannot express a sum, so the per-window divisor")
    say("       is used instead: it keeps every linear combination linear.")

    p44e_bound = (float(np.max(np.abs(q_bounded))) <= 1.0 and float(np.max(np.abs(tgt_bounded))) <= 1.0
                  and bool(np.isfinite(q_bounded).all()) and bool(np.isfinite(tgt_bounded).all()))
    p44e_nonvac_target = float(np.max(np.abs(tgt_raw))) > 1.0
    p44e_nonvac_channel = float(np.max(np.abs(q_unb))) > 1.0
    say("")
    say("  VERDICTS (bars declared before this ran)")
    say(f"  P44e bounded: max|q| {float(np.max(np.abs(q_bounded))):.6f} <= 1.0 and max|t| "
        f"{float(np.max(np.abs(tgt_bounded))):.6f} <= 1.0, all finite   -> "
        f"{'PASS' if p44e_bound else 'FAIL'}")
    say(f"       non-vacuity, TARGET: unbounded max|t| {float(np.max(np.abs(tgt_raw))):.1f} > 1.0   -> "
        f"{'PASS' if p44e_nonvac_target else 'FAIL'}")
    say(f"       non-vacuity, CHANNEL at scale 100: unbounded max|q| {float(np.max(np.abs(q_unb))):.4f} > 1.0"
        f"   -> {'PASS' if p44e_nonvac_channel else 'FAIL'}   (at this scale the raw channel is already inside"
        f" 1.0)")
    say(f"  P44e OVERALL: {'PASS' if (p44e_bound and p44e_nonvac_target) else 'FAIL'}")
    say("")
    say("  NOT CLAIMED: nothing about accuracy, and nothing about chat quality. This file measures the RANGE of")
    say("  the representation only. Whether a bounded channel makes the numeric term trainable on text is a")
    say("  different question, answered by lens/runs/NUMERIC-CHAT-READOUT-BOUNDED.txt (P44f).")
    say("=== DONE " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ===")

    json.dump({"when": datetime.datetime.now().isoformat(), "corpus": D_DIR, "windows": int(len(Y)),
               "scale": SCALE,
               "bounded": {"channel_max_abs": float(np.max(np.abs(q_bounded))),
                           "target_max_abs": float(np.max(np.abs(tgt_bounded)))},
               "unbounded": {"channel_max_abs_at_scale100": float(np.max(np.abs(q_unb))),
                             "channel_max_abs_original": float(np.max(np.abs(raw))),
                             "target_max_abs_original": float(np.max(np.abs(tgt_raw)))},
               "divisor": {"windows_d_gt_1": int((divisor.reshape(-1) > 1).sum()),
                           "min": float(divisor.min()), "max": float(divisor.max()),
                           "median": float(np.median(divisor))},
               "target_rows": int((tgt_mask > 0.5).sum()),
               "log_compression": {"max_abs_identity_error": worst, "at": list(worst_ab),
                                   "max_answer_factor": worst_factor},
               "verdicts": {"P44e_bounded": bool(p44e_bound),
                            "P44e_nonvacuous_target": bool(p44e_nonvac_target),
                            "P44e_nonvacuous_channel": bool(p44e_nonvac_channel)}},
              open(JSON_OUT, "w"), indent=1)

    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended to {OUT}\njson -> {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
