#!/usr/bin/env python3
"""VAL SPLIT BY SOURCE GROUP — does the held-out column measure dialogue, or maths/code/textbook text?

WHY: lens/runs/CORPUS-MIX-AUDIT.txt (run 2026-10-25 15:54, its own file) counts the TRAINING side of the
merged corpus: of the 197,204,306 assistant-turn tokens the loss actually trains on, 168,754,291 = 85.57%
come from the kn2 group (metamath, cosmopedia textbooks/math, hendrycks_math, OpenR1-Math, gutenberg,
magicoder/codefeedback/opencode) and 28,450,015 = 14.43% from the ext_honest group (hh-rlhf, oasst1,
ultrachat, orca-math = the actual human dialogue). The project's headline honest number (held-out
next-token accuracy over 58,191 assistant-by-position targets) is computed on the HELD-OUT windows of the
same merged corpus, so the obvious question is: what population does that number measure? If the val
windows are dominated by the same 85% of maths/textbook/code text, then the number and the talk test are
measuring two different things, and every "it is 35% accurate and still does not talk" sentence in this
project needs that qualifier.

WHAT IT MEASURES, EXACTLY, AND WHAT IT DOES NOT:
  * it counts, per source group, the number of CONVERSATIONS flagged val in split_v12000.npy and the
    TOKENS those conversations hold (read from bounds_v12000.npy + ids_v12000.npy, mem-mapped, chunked);
  * it does NOT reconstruct the trainer's window sampler, so it does NOT print a window-level share. The
    "if windows are drawn in proportion to val tokens" line is an ASSUMPTION and is labelled as one.
PREDICTIONS, WRITTEN BEFORE THE FIRST RUN (P106a-c)
  P106a split_v12000.npy takes at least 2 distinct values and the val flag is one of them; if only one value
        exists the file is refused (exit 3) rather than guessed at.
  P106b the val conversations are present in BOTH groups (the meta file names val_conversations for both:
        1,343 ext and 11,584 kn2), so neither group's held-out answer is invisible to the val set.
  P106c the kn2 group holds 80-90% of the val TOKENS (matching its 86.75% of all tokens). Falsifier: below
        70%, which would say the split was deliberately rebalanced toward dialogue and the numbers are
        more chat-weighted than the training mix suggests.
CPU only. No model, no GPU, no corpus loader.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest")
META = json.load(open(os.path.join(D, "meta_v12000.json"), "r"))
BOUNDS = np.load(os.path.join(D, "bounds_v12000.npy"), mmap_mode="r")
SPLIT = np.load(os.path.join(D, "split_v12000.npy"), mmap_mode="r")
IDS_LEN = np.load(os.path.join(D, "ids_v12000.npy"), mmap_mode="r").shape[0]

N_EXT = META["per_source_corpus"]["data/chat/prepared_ext_honest"]["conversations"]
N_CONV = len(BOUNDS)
N_SPLIT = len(SPLIT)          # 740,665: bounds carries one extra end sentinel (740,666 rows)
N_WALK = min(N_CONV, N_SPLIT)


def main():
    print("=" * 100)
    print("VAL SPLIT BY SOURCE GROUP (CPU only) — which population the held-out column measures")
    print("=" * 100)
    vals = Counter(SPLIT[:].tolist())
    print(f"  split_v12000.npy values and counts: {dict(vals)}  ({len(SPLIT):,} conversations; "
          f"bounds_v12000.npy has {len(BOUNDS):,} rows = one end sentinel more)")
    if len(vals) < 2:
        print("  P106a REFUSED (exit 3): fewer than two distinct split values — refusing to guess which is val.")
        return 3
    print("  P106a PASS")

    # per-group conversation and token totals for every split value, so the reading does not depend on
    # knowing which value is 'val'
    per = {}
    for g in (0, 1):
        per[g] = {v: {"conv": 0, "tok": 0} for v in vals}
    for c in range(N_WALK):
        g = 0 if c < N_EXT else 1
        v = int(SPLIT[c])
        start = int(BOUNDS[c])
        end = int(BOUNDS[c + 1]) if c + 1 < N_CONV else IDS_LEN
        per[g][v]["conv"] += 1
        per[g][v]["tok"] += max(0, end - start)

    print("\n  by group x split value (conversations / tokens):")
    for g, name in ((0, "ext_honest  (HUMAN DIALOGUE: hh-rlhf, oasst1, ultrachat, orca-math)"),
                    (1, "kn2         (maths, textbooks, gutenberg, code)")):
        print(f"    {name}")
        for v in sorted(vals):
            d = per[g][v]
            print(f"       split={v}:  {d['conv']:>9,} conv   {d['tok']:>13,} tok")

    print("\n  the meta file's own val_conversations counts, for comparison (NOT recomputed here):")
    for src, dd in META["per_source_corpus"].items():
        print(f"    {src}: {dd['val_conversations']:,} val conversations of {dd['conversations']:,}")

    print("\n  ASSUMPTION-DEPENDENT LINE (labelled): if the trainer's val windows are drawn in proportion to")
    print("  val tokens, then the held-out column's population share is:")
    for v in sorted(vals):
        t_ext = per[0][v]["tok"]
        t_kn2 = per[1][v]["tok"]
        tot = t_ext + t_kn2
        if tot == 0:
            continue
        share_kn2 = 100.0 * t_kn2 / tot
        print(f"    split={v}:  ext {100.0 * t_ext / tot:6.2f}%  |  kn2 {share_kn2:6.2f}%"
              f"   -> P106c for this split value: {'PASS' if 80.0 <= share_kn2 <= 90.0 else 'FAIL'}"
              f"{'  (<70 FALSIFIER FIRED)' if share_kn2 < 70.0 else ''}")
        print(f"                   P106b: both groups present in this split -> "
              f"{'PASS' if t_ext > 0 and t_kn2 > 0 else 'FAIL'}")

    print("\n  WHAT THIS DOES NOT SAY: it does not say the held-out number is wrong, and it does not")
    print("  reconstruct the window sampler. It says what the corpus is made of, which is the qualifier")
    print("  every 'held-out 35.26%' sentence in this project has been missing.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
