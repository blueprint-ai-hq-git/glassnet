"""record_ladder_bug.py — the wrong-subject gate bug (verdict-check class #4) + re-verify plan.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = f"""
---

## 2026-10-08 late — LADDER GATE BUG: image_cls was scoring the PLAIN CONTROL, not the readable arm

**The fourth verdict-check class bug: right number, WRONG SUBJECT.** `final_test()` takes the
LAST `TEST x%` line in train_image's output. The summary prints `readable ... TEST a%` then
`plain ... TEST b%` — so the ladder's image_cls gate has been reading **plain's number** at every
rung. It went unnoticed because at most rungs plain also clears the 2x-chance bar (at small the
two arms are nearly identical: readable 65.76 vs plain 65.81 — coincidence, not correctness).
Consequence stated plainly: **yesterday's ladder-big `image_cls 86.42%` headline was the plain
control's score, not the readable model's.** The readable-model claims that stand are the ones
parsed by label from train_image's own logs (82.5-86.4 line) and the ktest runs (84.36 real arm) —
those never used final_test.

Fix: `readable_test()` parses `readable <n> params TEST <x>%` BY LABEL. Ladder rule "the gate must
score the arm it names" added to the rung docstring.

Re-verification after fix (in flight): full big rung, all 7 capabilities, pre-registered:
  p1 image_cls readable > 2x chance  p2 readable-plain delta stays small  p3 the other six arms
  repeat yesterday's PASS (nothing else changed — any other failure is a real new break).
  yesterday's big JSON is preserved untouched; ladder_big.json will be overwritten ONLY by the
  new run's honest output, with the old numbers kept in this log entry.

RUNG FACT (not a bug): tiny text_lm FAILS at 6/7 (ppl 8339 vs bar 7000) — 2,000 windows can't
clear a vocab-3500 bar in 2 epochs; small rung PASSES it (1162). The bar was NOT moved; the tiny
rung is simply information-only for text_lm. Recorded, not hidden.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
# keep yesterday's big numbers visible next to the fix
old = json.load(open(os.path.join(P, "lens", "runs", "ladder", "ladder_big.json")))
print("prior big rung image_cls line (plain-parsed):",
      [r["checks"] for r in old if r["capability"] == "image_cls"])
