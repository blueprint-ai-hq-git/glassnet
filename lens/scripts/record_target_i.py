"""record_target_i.py — record Target I's verdict: the sparsity hypothesis is dead.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 19:25 — TARGET I: HYPOTHESIS DEAD. THE SPARSITY PENALTY IS NOT THE CAUSE.

```
lam_act=0.0 (the one change)   seed 0: 7/9  ppl 73.718  failing: pronoun_subject, is_a_dialogue_line
                               seed 1: 6/9  ppl 79.217  failing: + is_a_negation
                               seed 2: 6/9  ppl 77.236  failing: + pronoun_object
lam_act=0.2 (CONTROL)          seed 0: 7/9  ppl 78.915  failing: pronoun_subject, is_a_dialogue_line
                               seed 1: 7/9  ppl 79.724  failing: pronoun_subject, is_a_dialogue_line
```

**Removing the activation sparsity penalty does not rescue the two concepts -- it makes the pass
count slightly WORSE (6/9 twice against the control's 7/9).** The control reproduced Target G2's
7/9 with the same two failing, so the comparison is valid.

**Predictions scored -- 4 of 6 wrong:**
1. `pronoun_subject` passes at lam_act=0 -- **WRONG**
2. `is_a_dialogue_line` passes at lam_act=0 -- **WRONG**
3. the other seven still pass -- **WRONG** (two more break: is_a_negation, pronoun_object)
4. pass count rises 7/9 -> 9/9 -- **WRONG**
5. perplexity gets worse at lam_act=0 -- **WRONG** (it is slightly better: 73.7-79.2 vs 78.9-79.7)
6. firing rates rise -- not the deciding factor either way

**What is now RULED OUT as the cause of those two failures:**
* the targets -- all nine are linearly readable from the window at AUROC 1.0000
* redundancy -- max Jaccard overlap 0.133, mean 0.021-0.065
* the sparsity penalty -- removing it makes the result worse, not better
* scale -- they fail at D=32, D=64 and D=128

**Still standing as the likely area:** how the concept layer is *trained* rather than what it is
penalised with -- in particular the balanced-BCE weighting (`pos_weight = (1-s)/s`), which is 3.0
for a 25%-support concept and 32.0 for a 3%-support one. Whether that weight is actually applied in
the old loss is worth checking. **This is now a finding to carry into the rebuild rather than a
target to chase in the old code**, because the old code is being replaced.

**This is the third wrong prediction of the day** (Target G's teacher, G2's pass-rate trend, and
now this). Every one was caught by a measurement, not by care. That is the pattern, and it is
recorded rather than smoothed over.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
