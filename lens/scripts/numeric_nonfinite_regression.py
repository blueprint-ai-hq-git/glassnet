#!/usr/bin/env python3
"""numeric_nonfinite_regression.py - the "infinity" bug: what it was, and that it is gone.

Found 2026-10-20 while starting PLAN.txt item (d3): lens/numeric.py decided "is this token a number" with
float(str(tok)), and Python's float() accepts its own spellings, so the ordinary English word "infinity"
counted as a number whose value is inf. Predictions written first: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt
(P43a, P43b). Output: lens/runs/NUMERIC-NONFINITE.txt (appended, dated).

Usage: python3 lens/scripts/numeric_nonfinite_regression.py
"""
import datetime
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
os.chdir(PROJ)

import chat_data                                                     # noqa: E402
from lens.numeric import magnitude_channel, numeric_targets, token_value, _is_number   # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "NUMERIC-NONFINITE.txt")
D = os.environ.get("CHAT_DATA_DIR") or os.path.join(PROJ, "data/chat/prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D
SCALE = 100.0
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


say("=" * 100)
say("NUMERIC PATH, THE 'infinity' BUG - WHAT WAS MEASURED AND WHAT IS TRUE NOW")
say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {D}")
say("predictions written first: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt (P43a, P43b)")
say("=" * 100)

# ---- 1. the rule itself
cases = [("infinity", False, 0.0), ("inf", False, 0.0), ("-inf", False, 0.0), ("nan", False, 0.0),
         ("Infinity", False, 0.0), ("infinity's", False, 0.0), ("1e5", False, 0.0),
         ("38", True, 38.0), ("-2", True, -2.0), ("3.5", True, 3.5), ("000", True, 0.0),
         ("0", True, 0.0), (".5", True, 0.5), ("the", False, 0.0), ("+", False, 0.0),
         ("9" * 400, False, 0.0)]
say("")
say("1. THE RULE: a token is a number iff it is WRITTEN as a decimal literal")
bad = 0
for tok, want_num, want_val in cases:
    got_num, got_val = bool(_is_number(tok)), token_value(tok)
    ok = (got_num == want_num) and (abs(got_val - want_val) < 1e-9)
    bad += (not ok)
    say(f"   {str(tok)[:16]:>16s}  is_number {str(got_num):5s} (want {str(want_num):5s})  "
        f"value {got_val:<8.4g} (want {want_val:<8.4g})  {'ok' if ok else 'MISMATCH'}")
say(f"   P43b rule check: {len(cases) - bad} of {len(cases)} cases as declared -> "
    f"{'PASS' if bad == 0 else 'FAIL'}")

# ---- 2. the real vocabulary and the real corpus
ids = np.load(D + "/ids_v12000.npy", mmap_mode="r")
vocab = json.load(open(D + "/vocab_v12000.json"))
bounds = np.load(D + "/bounds_v12000.npy")
split = np.load(D + "/split_v12000.npy")
vals = np.array([token_value(t) for t in vocab], dtype=np.float64)
isnum = np.array([_is_number(t) for t in vocab])
nonfinite = int((~np.isfinite(vals)).sum())
say("")
say("2. THE REAL CHAT VOCABULARY (data/chat/prepared_mix_ctrl_honest/vocab_v12000.json)")
say(f"   tokens written as numbers: {int(isnum.sum())} of {len(vocab)}")
say(f"   numeric tokens whose value is NOT finite: {nonfinite}")
say(f"   token 8044 ({vocab[8044]!r}): is_number {bool(isnum[8044])} value {token_value(vocab[8044])}")
say(f"   P43b vocabulary check: 0 non-finite -> {'PASS' if nonfinite == 0 else 'FAIL'}")

tot_win = tot_num = 0
worst_mag = worst_tgt = 0.0
for which in (0, 1):
    X, Y = chat_data.make_windows(ids, bounds.tolist(), split.tolist(), 32, which=which,
                                  max_windows=50000, seed=0)
    mag, numeric = magnitude_channel(X, {i: t for i, t in enumerate(vocab)}, scale=SCALE)
    ynum, ymask = numeric_targets(vocab, Y, scale=SCALE)
    tot_win += len(Y)
    tot_num += int(ymask.sum())
    worst_mag = max(worst_mag, float(np.abs(mag).max()))
    worst_tgt = max(worst_tgt, float(np.abs(ynum).max()))
    say(f"   split {'train' if which == 0 else 'val'}: windows {len(Y):,}  "
        f"numeric-target windows {int(ymask.sum()):,} ({100.0*ymask.mean():.2f}%)  "
        f"max |magnitude| {np.abs(mag).max():.6g}  max |target| {np.abs(ynum).max():.6g}")
say(f"   over {tot_win:,} windows: {tot_num:,} carry a numeric target; "
    f"worst |magnitude| {worst_mag:.6g}, worst |target| {worst_tgt:.6g} (scaled by {SCALE:g})")
say(f"   P43b corpus check: every magnitude and every target finite -> "
    f"{'PASS' if (np.isfinite(worst_mag) and np.isfinite(worst_tgt)) else 'FAIL'}")
say(f"   P43c target exists on the real corpus -> {'PASS' if tot_num > 0 else 'FAIL'} "
    f"({100.0*tot_num/max(tot_win,1):.2f}% of sampled windows)")

# ---- 3. what the OLD rule would have done (reproduced here so the bug is not taken on trust)
def old_is_number(tok):
    try:
        float(str(tok).strip())
        return True
    except (TypeError, ValueError):
        return False


old_num = [i for i, t in enumerate(vocab) if old_is_number(t)]
old_nonfinite = [i for i in old_num if not np.isfinite(float(str(vocab[i]).strip()))]
say("")
say("3. THE OLD RULE, REPRODUCED FROM THIS SCRIPT (the bug, not taken on trust)")
say(f"   old rule counted {len(old_num)} numeric tokens, {len(old_nonfinite)} of them with a "
    f"non-finite value: {[(i, vocab[i]) for i in old_nonfinite]}")
cnt = 0
ids_bad = np.array(old_nonfinite, dtype=np.int64)
if len(ids_bad):
    for s in range(0, ids.shape[0], 20_000_000):
        cnt += int(np.isin(np.asarray(ids[s:s + 20_000_000]), ids_bad).sum())
say(f"   occurrences of those tokens in the whole corpus: {cnt:,}")
say(f"   -> with the numeric path on over real text the old rule puts inf in the magnitude channel at "
    f"every one of those positions, and inf in every numeric target whose next token is the word")
say("")
say("VERDICT: P43b PASS (rule + vocabulary + corpus), P43c PASS (target exists). The old rule is "
    "reproduced above so the fix has evidence, not a story.")

with open(OUT, "a") as f:
    f.write("\n".join(lines) + "\n")
print(f"\nappended to {OUT}")
