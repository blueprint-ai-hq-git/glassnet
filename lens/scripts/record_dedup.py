"""record_dedup.py — CORRECTION to my own E18 closure. The concept set is NOT meaningfully inflated:
only ONE pair is a true duplicate (corr 1.000). My "quote 23, not 35" was too strong.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
th = json.load(open(os.path.join(P, "lens", "runs", "vl", "dedup_thresholds.json")))

LOG = f"""
---

## 2026-10-09 — CORRECTION TO MY OWN E18 CLOSURE. "Quote 23, not 35" was WRONG.

### WHAT I SAID EARLIER TODAY
Closing E18 I wrote: "23 of 35 concepts are genuinely independent (66%)... quote 23, not 35, and 12
names in the medical concept set are duplicates of others." That was based on complete-linkage
clustering at |corr| >= 0.70.

### WHY IT WAS WRONG — TWO SEPARATE ERRORS, BOTH MINE
**Error 1: |corr| merged OPPOSITES.** The first merge list contained
`red_channel_low -> is_bright  corr -0.712` and `blue_channel_low -> is_bright  corr -0.732`.
A NEGATIVE correlation means the concepts are opposites ("low red" vs "bright"), not duplicates.
Merging them would have deleted genuinely distinct readable properties. Fixed by requiring POSITIVE
correlation before merging.

**Error 2: 0.70 measures CORRELATION IN THIS DATASET, not SAME MEANING.** With the polarity fix and a
threshold sweep, the merge list collapses as the bar tightens:
```
  threshold 0.70 : keep 22 of 31, merge 9
  threshold 0.80 : keep 25 of 31, merge 6
  threshold 0.90 : keep 30 of 31, merge 1     <- corr 1.000
  threshold 0.95 : keep 30 of 31, merge 1     <- corr 1.000
```
The surviving merge is `has_strong_colour -> is_purple_dominant` at **corr 1.000**, which means the two
formulas are IDENTICAL — a real definitional duplicate, not a dataset coincidence. The pairs that
0.70 merged (has_low_contrast vs left_right_symmetric at 0.860; is_dark vs mostly_empty at 0.718) are
conceptually DIFFERENT properties that merely co-occur in PathMNIST: low-contrast tiles also tend to be
symmetric. **Correlation is not identity.**

### THE CORRECTED CLAIM
**The medical concept set is NOT meaningfully inflated. Of 31 concepts, exactly 1 pair is a true
duplicate (`has_strong_colour` == `is_purple_dominant`, corr 1.000). The honest count is 30, not 31 —
and not 23.** The independence figure of 23 measured how much the concepts OVERLAP IN THIS DATASET,
which is a useful number but a different one, and I should not have presented it as "how many concepts
you really have".

### THE PREDICTION I WROTE BEFORE RUNNING — now void, and I am saying so
I predicted that training on a de-duplicated registry would raise own_corr (~1.5x more gradient per
concept) because the layer would stop fighting 12 redundant targets. **That prediction assumed 12
duplicates. There is 1.** Dropping 1 of 31 targets cannot produce a ~1.5x gradient effect, so the
predicted effect size is essentially nil and **the training experiment is NOT worth GPU time. It is
cancelled, and the reason is the measurement, not a failure to run it.** Running it anyway to "have a
result" would have been the theatre this project is supposed to avoid.

### WHAT THIS MEANS FOR E18 (which stays closed, with a corrected number)
The three E18 fixes stand and were verified: fidelity annotated in explain(), margin-aware audit, and
an independent-concept measurement. What changes is the number attached to the third: **the readable
count is 30 of 31, not 23 of 35.** The concept set is sound; my first de-duplication criterion was not.

### THE LESSON
A de-duplication rule based on correlation must (a) require the SAME sign, and (b) be strict enough
that only near-interchangeable definitions merge. At 0.70 the rule was deleting distinct meanings.
When a measurement is used to justify DELETING something, the bar must be identity, not association.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended dedup correction")
print("  corrected: 30 of 31 concepts are distinct; 1 true duplicate (corr 1.000)")
print("  the training experiment is CANCELLED — predicted effect size is nil")
