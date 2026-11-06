"""record_target_i_final.py — the trade-off number from Target I's full verdict.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 21:10 — TARGET I, FINAL VERDICT: THE SPARSITY PENALTY IS A REAL TRADE-OFF KNOB

The full verdict adds one number I had not recorded, and it changes what the result MEANS.

```
I-FAILING  pronoun_subject      0/3 at lam_act=0   FAIL  (auroc [0.5044, 0.5087, 0.5068])
I-FAILING  is_a_dialogue_line   0/3 at lam_act=0   FAIL  (auroc [0.5337, 0.5272, 0.5479])
I-READABLE 70.4% of concepts pass at lam_act=0     PASS
I-COVERAGE PASS (100.0000% x3)
I-CONTROL  control reproduces 7/9                  PASS
I-CONTROL  same two concepts fail in control       PASS
I-NOWORSE  readable 76.724 vs matched normal 69.968 -> delta +6.756
```

**The hypothesis is dead -- removing the activation penalty does NOT rescue the two concepts.**
But the perplexity line is the interesting part:

| | readable ppl | normal ppl | gap |
|---|---|---|---|
| Target G2 (lam_act=0.2) | 79.569 / 82.057 / 84.845 | 67.108 / 69.968 / 70.367 | **+12.46 / +12.09 / +14.48** |
| **Target I (lam_act=0.0)** | **76.724** | **69.968** | **+6.756** |

**Removing the activation sparsity penalty HALVES the readable model's perplexity cost** -- from
roughly +12 to +6.76. So the penalty accounts for about half the gap between the readable model
and the matched normal.

**But it is not free.** At `lam_act=0` the concept pass rate falls (70.4% vs the control's 77.8%)
and two ADDITIONAL concepts break. So:

**The sparsity penalty buys concepts at the cost of perplexity.** That is a genuine trade-off
knob, not a bug and not a fix. Turning it off makes the model better at the language and worse at
being readable. Turning it on does the reverse.

**What this means for the rebuild:** the activation sparsity penalty is an explicit, documented
switch in `lens/model.py` (`lam_act`, default 0.2), and the trade-off above is the reason it
exists as a knob rather than a constant. The remaining question -- WHY those two concepts fail --
is now the rebuild's to answer, in a codebase where the concept loss is provably active and
provably reaching the layer, which the old code could not demonstrate.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

# add the trade-off to the ledger's blocking section
p = os.path.join(PROJ, "TARGET-LEDGER.md")
txt = open(p).read()
anchor = "## WHAT \"MOVING DOWN TO THE NEXT ONE\" MEANS"
add = """## TARGET I, FINAL — THE SPARSITY PENALTY IS A TRADE-OFF, NOT A FIX

The full verdict confirmed the hypothesis is dead (both failing concepts still fail at
`lam_act=0`), but produced one number that matters more:

```
                 readable ppl      normal ppl     gap
G2 (lam_act=0.2) 79.6 / 82.1 / 84.8   67.1 / 70.0 / 70.4    +12.46 / +12.09 / +14.48
I  (lam_act=0.0) 76.724              69.968                +6.756
```

**Removing the activation sparsity penalty HALVES the perplexity cost** -- from ~+12 to +6.76.
It is the largest single lever found on the blocker. **But it is not free:** the concept pass rate
drops from 77.8% to 70.4% and two additional concepts break.

**So the penalty buys readability at the cost of perplexity.** It is a knob, and it is now an
explicit documented switch in the new engine (`lam_act`), not a hidden constant.

"""
assert anchor in txt
txt = txt.replace(anchor, add + anchor, 1)
open(p, "w").write(txt)
print("TARGET-LEDGER.md updated with the trade-off")
