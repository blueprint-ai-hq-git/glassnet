"""build_review09.py — GLASSNET-REVIEW-09.md: the Meter-03 correction.

REVIEW-08 told the reviewer that the audit half cleared the 50% bar on a "genuine
non-circular gain" of 45.8%. Meter-03 overturned that. This file supersedes those
claims, states the correction up front, and embeds the real code from disk with hashes.
"""
import hashlib
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "GLASSNET-REVIEW-09.md")


def read(p):
    return open(os.path.join(PROJ, p)).read().splitlines()


def whole(p):
    return "\n".join(read(p)).rstrip()


def sha(p):
    return hashlib.sha256(open(os.path.join(PROJ, p), "rb").read()).hexdigest()[:16]


def nl(p):
    return len(read(p))


parts = []
A = parts.append

A("""# GLASSNET — REVIEW 09

**Date:** 2026-10-04 (late evening) · **Machine:** idle
**Supersedes `GLASSNET-REVIEW-08.md`** on the points listed in §1. Read this one.
**Bars:** unchanged all day (AUROC 0.55, contribution 0.25, firing floor 2%, 50% pass,
lie probe caught). **Test set:** untouched. Every number comes from a log on disk.

---

## 1. CORRECTION to REVIEW-08 — read this first

REVIEW-08 told you: *"the audit half now CLEARS the 50% bar on a genuine non-circular
gain"*, with the honest half at **45.8%**. **That claim is wrong and is withdrawn.**

An owner decision added a pre-registered audit (Meter-03) plus two controls. Re-scored
against targets the columns were **never trained on**, the same non-circular half is:

```
45.8%   (Meter-02, names derived from each column's own firings)
 2.1%   (Meter-03, pre-registered targets)   <- the number that survives
```

The 45.8% was an artifact of the audit's own naming: the namer chose each concept's name
**from that column's firings**, so the name matched the firings by construction. Nothing
was faked; the measurement simply could not fail.

**What survives from REVIEW-08:** the Meter-02 firing floor (it removed 31–32 points of
false passes and caught dead neurons in the rungs), the re-audit of every earlier result,
the autograd reference-cycle bug and its fix, and the list of my own mistakes.

**What REVIEW-08 got wrong, besides the 45.8%:** it described the 50% bar as met. Under
the corrected audit the bar is not met on the half that matters.

---

## 2. Meter-03 — one change: where the target comes from

Everything else is identical to Meter-02 (same length-residualised AUROC, same logit-gain
contribution formula, same thresholds, same firing floor). Only the target changes:

| | Meter-02 | Meter-03 |
|---|---|---|
| target for column c | word-set parsed from the **namer's name** for that column | the **pre-registered target** frozen before training (single word, or cluster members) |

The owner also required two controls, both pre-registered before running:

1. **Random-column control** — concept layer at random init, never trained; report the
   **free pass rate**, and report trained results as the **excess** above it.
2. **Name-swap probe** — audit a trained column against another column's target; the
   audit must reject it.

And a claim rule fixed **before** results: Target A may be claimed only if the lie probe
is **caught** AND the **unsupervised** excess over the random control clears 50%.

---

## 3. Results (2% firing floor, re-score of existing checkpoints)

```
WORDS targets (v1 checkpoints)
model                          SUP     rand   excess |  UNSUP    rand   excess | best-m  swap-rej
baseline lc=0 (unsupervised)   0.0%    0.0%   +0.0  |   0.0%    0.0%   +0.0  |   0.0%     99.7%
single words lc=0.1           31.2%    0.0%  +31.2  |   0.0%    0.0%   +0.0  |   8.3%    100.0%
single words lc=0.5           31.2%    0.0%  +31.2  |   0.0%    0.0%   +0.0  |  20.8%    100.0%
single words lc=1.0           31.2%    0.0%  +31.2  |   0.0%    0.0%   +0.0  |  16.7%    100.0%

CLUSTERS targets (v2 checkpoints)
clusters lc=0.1               62.5%    0.0%  +62.5  |   0.0%    0.0%   +0.0  |  10.4%    100.0%
clusters lc=0.5               62.5%    0.0%  +62.5  |   2.1%    0.0%   +2.1  |  16.7%    100.0%
clusters lc=1.0               62.5%    0.0%  +62.5  |   2.1%    0.0%   +2.1  |  18.8%    100.0%
```

`SUP` = supervised columns vs their own pre-registered target. `UNSUP` = unsupervised
columns vs a pre-registered target (fixed pairing; `best-m` = best of all 16, an upper
bound). `rand` = random-column control.

## 4. Control 1 — the random-column control: 0.0% free pass rate

Concept layer at random init, **never trained**, embedding and answer layer from the
checkpoint, **10 draws × 3 seeds = 30 audits**, and the columns were **live** (32.0/32
above the floor) in every draw. **Nothing passed.**

**Consequence: there are no free passes to subtract.** Every pass in §3 is real. This is
the cleanest result of the day: the audit does not hand passes to random columns, so the
old "dead column named to match its own firings" failure mode is genuinely closed.

## 5. Control 2 — name-swap: 100% rejected, decomposed honestly

```
words   lc=1.0: all pairs 100.0% (n=720) | LIVE columns only 100.0% (n=225)
clusters lc=0.5: all pairs 100.0% (n=720) | LIVE columns only 100.0% (n=450)
among live wrong pairs: contribution median +0.0000 / -0.0033 (bar +0.25)
                        AUROC median 0.437 (clusters)
diagonal AUROC median: 1.000 (clusters)
```

I pre-registered that a 100% rejection rate would be **suspicious**, so it is decomposed
rather than presented as a triumph:

* **Most raw rejections come from the firing floor** — `firing=495` of 720 pairs for
  words. The floor is a **column** property: a dead column fails every pairing, including
  its own, and says nothing about target-matching. Reporting 100% alone would mislead.
* **Restricted to live columns it is still 100%**, driven by the **contribution clause**:
  a wrong pairing has median contribution ≈ 0.0000 against a +0.25 bar, because a column
  has no logit pathway for another target's words.
* So the control passes, but its power comes from the contribution clause, not AUROC.

## 6. The lie probe is now CAUGHT — the point of the whole exercise

```
                     own target   auroc    contrib   firing   Meter-03   Meter-02 said
supervised lc=1.0    The          0.7633   +0.0166   0.3638   CAUGHT     CAUGHT (floor only)
baseline   lc=0      The          0.5088   +0.0017   0.4462   CAUGHT     MISSED
```

* Both caught **while live** (firing 36% / 45%, far above the 2% floor) — the floor is not
  doing the catching, unlike Meter-02's supervised catch (`caught_by_floor_only = True`).
* Both fail **even against their best of all 16 targets**, so the catch is not an unlucky
  pairing.
* Mechanism: a column never trained on a target has no logit pathway for its words
  (contribution ≈ 0) and no rank information about it (AUROC at chance).

**This is the first time the lie probe has been caught honestly**, and it is caught by the
clauses that are supposed to do the work.

## 7. Verdict

**TARGET A = NOT MET.**

* lie probe caught: **YES** — this half of Target A is now genuinely satisfied;
* unsupervised excess over the random control: **+0.0% to +2.1%** against a 50% bar: **NO**.

**The finding, stated plainly: the concept layer detects pre-registered targets only where
it was explicitly trained. Supervision works (62.5% on clusters, against a 0.0% random
control). Generalisation to untaught targets does not.**

## 8. Diagnostics, not hidden

* Word target `KING` has **0 validation positives**, so its AUROC is undefined and it
  fails automatically. That is not evidence about the model.
* Word columns are often not live (19–21 of 32), so many fail on the floor before
  target-matching is considered.
* Several word targets are thin in validation (`My` 0.78%, `What` 0.87%, `they` 0.88%).
* The supervised half is audited against the target it was trained on, so it measures
  whether supervision worked — not whether the concepts are real.

## 9. Predictions written before running (all recorded, wrong ones included)

| # | prediction | result |
|---|---|---|
| 1 | supervised 85–100% | **NOT CONFIRMED** — 31.2% (words) / 62.5% (clusters) |
| 2 | unsupervised 5–20% | **NOT CONFIRMED** — 0.0–2.1%, even lower |
| 3 | random control 10–30% | **NOT CONFIRMED** — 0.0% |
| 4 | name-swap 70–95% | 100%, decomposed in §5 |
| 5 | lie probe caught | **CONFIRMED** |
| 6 | Target A not met | **CONFIRMED** |

## 10. Next step proposed, NOT run

**Option B — fix the names before training.** Pre-register every concept's name and train
the layer to detect them, so no name is ever derived from a column's firings. This removes
the circularity at its source instead of auditing around it.

* Why it is worth trying: supervision works (62.5% vs a 0.0% random control), so
  extending pre-registration from 16 to all 32 columns is a natural step.
* Why it may not be enough: the untaught half was ~0%. Naming and training more columns
  may raise the supervised number without producing concepts that generalise. The honest
  test is whether **untaught** targets start passing — and today they do not.
* Cost: one training run per setting, 3 seeds, no test touch.
* **Not started. Awaiting the owner.**

## 11. Code, as it actually is on disk

### 11.1 The Meter-03 audit — `scripts/t3_audit_m3.py`
""")
A("```python\n" + whole("scripts/t3_audit_m3.py") + "\n```\n")

A("### 11.2 The Meter-03 lie probe — `scripts/t3_m3_probe.py`\n")
A("```python\n" + whole("scripts/t3_m3_probe.py") + "\n```\n")

A("### 11.3 The name-swap clause decomposition — `scripts/t3_m3_clauses.py`\n")
A("```python\n" + whole("scripts/t3_m3_clauses.py") + "\n```\n")

A("### 11.4 The diagnostics that qualify the numbers — `scripts/t3_m3_diag.py`\n")
A("```python\n" + whole("scripts/t3_m3_diag.py") + "\n```\n")

A("""---

## 12. Verification

`scripts/verify_claims.py` re-derives every reported figure from the raw JSON instead of
trusting any summary. It specifically guards the three things that could make §6 a false
claim: that the probe is caught, that it is **live** (so the catch is not a floor
artifact), and that it is caught **even on its best target** (so the catch is not an
unlucky pairing). It also asserts the random control really is 0%.

**50 checks passed, 0 mismatches.** (One earlier false positive was my own tolerance, and
one was an inverted condition in my own check; both fixed and the reasons recorded.)

## 13. Files and hashes

| file | lines | sha256[:16] |
|---|---|---|
""")
for f in ["scripts/t3_audit_m3.py", "scripts/t3_m3_probe.py", "scripts/t3_m3_clauses.py",
          "scripts/t3_m3_diag.py", "scripts/verify_claims.py", "METER-RULE-03.md",
          "METER-03-RESULTS.md", "METER-RULE-02.md", "REAUDIT-METER-02.md",
          "CONCEPT-TARGETS-02.md", "LOG.md", "STATUS.md"]:
    if os.path.exists(os.path.join(PROJ, f)):
        A(f"| `{f}` | {nl(f)} | `{sha(f)}` |")
    else:
        A(f"| `{f}` | (missing) | |")

A("""
### Where the numbers live

| number | file |
|---|---|
| Meter-03 audits | `runs/t3/meter3_words.json`, `runs/t3/meter3_clusters.json` |
| lie probe | `runs/t3/meter3_lieprobe.json` |
| name-swap decomposition | `runs/t3/meter3_clauses.json` |
| target/live diagnostics | `runs/t3/meter3_diag.json` |
| Meter-02 lie records (preserved) | `runs/t3/audit_K32_lr*_pa_*.json.meter02.bak` |
| full run log | `runs/t3/meter3.log` |

---

*Prepared by the working session. Hypotheses are labelled as such; predictions were
written before their runs and the wrong ones are reported alongside the right ones.*
""")

open(OUT, "w").write("\n".join(parts) + "\n")
print(f"wrote {OUT}")
print(f"  {nl('GLASSNET-REVIEW-09.md')} lines, sha256[:16] {sha('GLASSNET-REVIEW-09.md')}")
