"""record_blocker.py — the rebuilt engine against the blocker: one replication, one refutation.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 02:00 — THE REBUILT ENGINE vs THE BLOCKER. THE GAP REPLICATES; THE "LEVER" DOES NOT.

All three arms are the SAME architecture with the SAME parameter count, differing in exactly one
setting each. 60 epochs, V=1500, d=256, window 8, tiny_shakespeare, 417,707 params, GPU.

```
arm        params    val ppl   val acc   concepts   coverage
readable   417,707    90.085    29.73%      7/9      100.00%
lam0       417,707    88.933    29.58%      7/9      100.00%
normal     417,707    77.804    28.43%      0/9      100.00%

GAP vs the matched normal:
  readable   90.085 - 77.804 = +12.281
  lam0       88.933 - 77.804 = +11.129
```

### 1. THE BLOCKER REPLICATES -- the measurement is now trustworthy

```
rebuilt engine, readable arm   +12.281
old codebase (Target G2)       +12.46 / +12.09 / +14.48
G2's bars                       5.81 /  6.90 /  5.77
```

**+12.281 against +12.46 on a completely rebuilt engine.** The readable model really does cost
about 12 perplexity points against a matched normal. This is a replication, not a new claim, and
it is the first time the number comes from code where the concept loss is provably active and
`explain()` asserts its own exactness.

### 2. THE "LEVER" DOES NOT HOLD -- Target I's +6.756 is NOT CONFIRMED

Target I reported that setting the activation sparsity penalty to 0 HALVES the gap: +6.756 against
G2's ~+12. Here, with the same one change and nothing else, the gap is **+11.129** -- barely below
the readable arm's +12.281 and nowhere near +6.756.

**So Target I's headline number should be treated as not confirmed**, and the earlier statement
that the penalty "accounts for about half the gap" is WITHDRAWN. The most likely explanation is
that the two runs are not comparable: G2's matched normal sat at ppl ~69.968 while this run's
normal sits at 77.804, so the arms were trained under different conditions (batch, learning rate,
or data pipeline). Whatever the cause, **the honest position is that the lever did not reproduce,
not that the old number was right.**

**Also visible in the same table:** `lam0` costs accuracy too -- 29.58% against the readable arm's
29.73%. So the change is not free even on the metric it was supposed to help.

### 3. THE CONCEPT AUDIT REPLICATES EXACTLY

```
pronoun_subject            AUROC 0.5443   fires 0.273   FAIL
is_a_dialogue_line         AUROC 0.5066   fires 0.331   FAIL
pronoun_object             AUROC 1.0000   fires 0.147   PASS
is_a_negation              AUROC 0.6527   fires 0.164   PASS
is_archaic_address         AUROC 1.0000   fires 0.084   PASS
contains_a_wh_word         AUROC 1.0000   fires 0.047   PASS
mentions_speech            AUROC 1.0000   fires 0.045   PASS
mentions_giving            AUROC 1.0000   fires 0.023   PASS
mentions_knowing           AUROC 1.0000   fires 0.045   PASS
```

**7/9 pass, and the same two fail** -- `pronoun_subject` and `is_a_dialogue_line`, exactly the two
that failed in G2, D, E and Target I. Five concepts are perfect (AUROC 1.0000). The firing floor
is doing its job: at 2 epochs every concept had AUROC 0.82-0.98 but fired at 0.00-0.01 and was
correctly failed; by 60 epochs they fire at 0.023-0.33 and seven pass.

**So the two failures are structural, not a scale or training-length artefact.** They have now
failed on the old engine (D, E, G2, I) and on the rebuilt engine, with the same AUROC near 0.50 --
which is what "detects nothing" looks like.

### WHAT THIS CHANGES

* The blocker is confirmed on trustworthy code. **+12.3 is the number to beat.**
* Target I's lever is withdrawn as a fix. The search for what causes the gap continues.
* The concept layer is reproducible: 7/9, same two failures, on new code.

Artifacts: `lens/runs/blocker/{readable,lam0,normal}.npz`, `{arm}.jsonl`, `{arm}_summary.json`,
`verdict.json`, `blocker_run.log`.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "TARGET-LEDGER.md")
txt = open(p).read()
add = """## TARGET I, THE LEVER -- WITHDRAWN ON RE-MEASUREMENT (2026-10-07)

Target I reported that `lam_act=0` halves the readable model's perplexity gap (+6.756 vs G2's
~+12), and that finding was recorded as "the largest single lever found on the blocker".

**Re-measured on the rebuilt engine, with the same one change and nothing else: the gap is
+11.129, not +6.756.** The readable arm on the same run measures +12.281, replicating G2's +12.46.

So:
* **the blocker replicates** (+12.281 vs +12.46) -- the measurement is trustworthy;
* **the lever does NOT replicate** (+11.129 vs +6.756) -- **the "halves the gap" claim is
  WITHDRAWN**;
* `lam0` also costs accuracy (29.58% vs 29.73%), so it is not free on any metric.

Likely cause of the discrepancy: the two runs are not comparable -- G2's matched normal sat at ppl
69.968 while the rebuilt run's normal sits at 77.804, so batch/lr/data differed. The honest
position is that the lever did not reproduce. The search for the cause of the gap continues.

"""
anchor = "## WHAT \"MOVING DOWN TO THE NEXT ONE\" MEANS"
assert anchor in txt
txt = txt.replace(anchor, add + anchor, 1)
open(p, "w").write(txt)
print("TARGET-LEDGER.md updated: the lever is withdrawn")
