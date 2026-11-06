"""record_independence.py — E18's de-duplication piece. My first effective-rank metric was WRONG;
the corrected numbers are coherent and the concept-count claim survives with a caveat.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "vl", "effrank_dbg.json")))
ind = json.load(open(os.path.join(P, "lens", "runs", "vl", "concept_independence.json")))

LOG = f"""
---

## 2026-10-09 — CONCEPT INDEPENDENCE: 23 of 35 concepts are genuinely distinct (i3 PASS), and my
## first effective-rank metric was WRONG.

### THE QUESTION (E18's remaining piece)
The layer advertises 35 named concepts, but 34 pairs correlate above 0.70. Do those names describe
the same properties, meaning the readable count is inflated?

### RESULT: the count mostly HOLDS
```
  mlow_lc0.15   advertised 35   INDEPENDENT 23 (66%)   effective rank {d['eff_rank_C']:.1f} (corrected)
  mlow_lc0.00   advertised 35   INDEPENDENT 23 (66%)
  i3  independent >= 50% of advertised -> PASS
```
Complete-linkage clustering at |corr| >= 0.70 (conservative: a cluster only forms if EVERY member
pairs above threshold with every other) groups the redundant names:
```
  size 4   is_bright          <- red_channel_low, green_channel_low, blue_channel_low
  size 4   is_pink_dominant   <- is_purple_dominant, has_strong_colour, colour_contrast_high
  size 3   has_low_contrast   <- left_right_symmetric, top_bottom_symmetric
  size 3   red_channel_high   <- green_channel_high, blue_channel_high
  size 2   mostly_empty       <- is_thin_structure
  size 2   is_green_dominant  <- blue_exceeds_green
  ...plus 17 singletons
```
So **12 of the 35 names are duplicative** and the honest independent count is **23**. That is the
number to quote; 35 is the advertised label count.

### MY FIRST METRIC WAS WRONG — caught by an internal contradiction, not by luck
I reported "effective rank 2.0" alongside "23 independent clusters". Those cannot both be true: a
2-dimensional set would have nearly all pairs correlated, giving 1-3 clusters. So I checked the
computation instead of the number:
```
  eigenvalues of C (correlation matrix)  trace 35.00  -> effective rank {d['eff_rank_C']:.2f}   <- CORRECT
  eigenvalues of C @ C.T                 trace 160.84 -> effective rank {d['eff_rank_CCT']:.2f}   <- what I reported
  singular values of standardised T                     -> effective rank {d['eff_rank_sv']:.2f}
```
`C @ C.T` SQUARES the spectrum, and its trace is ||C||_F^2 (160.84), not n. The participation ratio
computed from it is not the effective rank of the correlation matrix. **The correct effective rank is
{d['eff_rank_C']:.2f} of 35** — the concept set spans about 8 substantial directions, not 2.

### THE CROSS-CHECK THAT SETTLES IT
```
  pairs with |corr| >= 0.70 : {d['pairs_above_070']} of 595
  mean |corr| off-diagonal  : 0.222
```
Only 35 of 595 pairs exceed the threshold and the mean off-diagonal correlation is 0.22. A rank-2 set
would show nearly all pairs above threshold. So the clustering result (23 independent) is the right
picture and the rank-2 figure was a computation error.

### E18 STATUS: CLOSED
All three pieces done: (1) fidelity annotated on every concept name in explain(); (2) the audit now
reports the own-vs-best-other margin, not just the weak 0.55 AUROC bar; (3) the independent-concept
count is measured and reported (23, not 35) with the duplicate names listed.
Remaining honest caveat, recorded rather than smoothed over: **quote 23, not 35**, and 12 names in
the medical concept set are duplicates of others.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended independence + metric-error record")
print(f"  i3 PASS: 23/35 independent (66%)")
print(f"  MY BUG: reported eff rank 1.97 from C@C.T; correct value {d['eff_rank_C']:.2f}")
