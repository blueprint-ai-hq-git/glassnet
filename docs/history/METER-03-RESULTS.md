# METER-03 RESULTS — pre-registered-target audit + two controls

**Date:** 2026-10-04 · **Rule:** `METER-RULE-03.md`, pre-registered BEFORE running
**Bars:** UNCHANGED (AUROC 0.55, contribution 0.25, firing floor 2%, 50% pass, lie probe caught)
**Method:** re-score of existing checkpoints. No model retrained, re-tuned or re-selected.
**Test set:** untouched.

---

## 0. Headline

**The lie probe is now CAUGHT — Option A worked.** Under Meter-02 a frozen column passed
with AUROC 0.80. Audited against a pre-registered target it was never trained on, the
same kind of column is now caught **while live**, on its own merits:

```
supervised lc=1.0: own target (The)  auroc 0.7633  contrib +0.0166  firing 0.3638 -> CAUGHT
baseline   lc=0:   own target (The)  auroc 0.5088  contrib +0.0017  firing 0.4462 -> CAUGHT
```

Neither catch depends on the firing floor (firing 36% / 45%, far above the 2% bar), and
the probe fails even against its **best** of all 16 targets — so the catch is not an
unlucky pairing.

**But Target A is NOT MET.** The pre-registered rule requires the lie probe caught **AND**
the unsupervised excess over the random control to clear 50%. The unsupervised excess is
**+0.0% to +2.1%**. The lie-probe half is now satisfied; the other half is nowhere near.

**And the 45.8% figure is OVERTURNED.** That was Meter-02's non-circular number — the one
I flagged as the honest gain. Re-scored against pre-registered targets those columns were
never trained on, it is **0.0–2.1%**.

---

## 1. Results — Meter-03 (2% firing floor)

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

`SUP` = supervised columns vs their OWN pre-registered target. `UNSUP` = unsupervised
columns vs a pre-registered target (fixed pairing `c mod 16`; `best-m` is the best of all
16, an upper bound). `rand` = the random-column control's free pass rate.

## 2. Control 1 — random-column control: **0.0% free pass rate**

Concept layer at random init, never trained, embedding/answer from the checkpoint,
**10 draws × 3 seeds**, and the columns were **live** (32.0/32 firing above the floor in
every draw). Nothing passed: **0.0%** overall, on both halves.

**Consequence: there are no free passes to subtract.** Every pass reported above is a
real pass. This is the cleanest result of the day — the audit is not handing out passes
to random columns, and the earlier "dead column named to match its own firings" failure
mode is genuinely closed.

## 3. Control 2 — name-swap probe: 100% rejected, and here is the honest breakdown

```
words   lc=1.0: all pairs 100.0% rejected (n=720) | LIVE columns only 100.0% (n=225)
clusters lc=0.5: all pairs 100.0% rejected (n=720) | LIVE columns only 100.0% (n=450)
among live wrong pairs: contribution median +0.0000 (words) / -0.0033 (clusters)
                        AUROC median nan (words) / 0.437 (clusters)
diagonal AUROC median: 1.000 (clusters)
```

I pre-registered that a 100% rejection rate would be **suspicious**, so I decomposed it
rather than reporting it as a triumph:

* **Most raw rejections come from the firing floor, not from target mismatch.** The floor
  is a *column* property: a dead column fails every pairing including its own. Raw clause
  counts were `firing=495, both=192, contrib_only=33` (words) — i.e. the floor accounts
  for 69% of the rejections and says nothing about target-matching. **Reporting 100%
  without this would have been misleading.**
* **Restricted to LIVE columns, rejection is still 100%** (n=225 / n=450) — so the control
  does work. The mechanism is the **contribution clause**: a wrong pairing has median
  contribution ≈ 0.0000 against a +0.25 bar, because a column has no logit pathway for
  another target's words. Off-diagonal AUROC median 0.437 (below chance) for clusters.
* Verdict: **the name-swap control passes**, but its power comes from the contribution
  clause, not from AUROC.

## 4. Lie probe (Meter-03)

```
                     own target   auroc    contrib   firing   verdict   Meter-02 said
supervised lc=1.0    The          0.7633   +0.0166   0.3638   CAUGHT    CAUGHT (floor only)
baseline   lc=0      The          0.5088   +0.0017   0.4462   CAUGHT    MISSED
```

* Both caught **live** — the firing floor did not do the catching (contrast Meter-02, where
  the supervised catch was `caught_by_floor_only = True`).
* Both fail even against their **best** target of 16, so the catch is robust to the pairing.
* Clause failures: supervised = contribution only; baseline = AUROC and contribution.

## 5. Claims this changes (flagged)

| earlier claim | Meter-03 result | status |
|---|---|---|
| **"Unsupervised half 45.8% — the honest non-circular gain"** | **0.0–2.1%** | **OVERTURNED.** It was an artifact of the namer deriving names from each column's own firings. |
| "Meter-02 supervised 62.5% (clusters)" | 62.5% (clusters), 31.2% (words) | unchanged for clusters, lower for words |
| "Lie probe MISSED at baseline (auroc 0.5591)" | **CAUGHT** | **IMPROVED** — the probe now fails as it should |
| "Random control unknown / passes could be free" | **0.0% free pass rate** | **RESOLVED** — no free passes |
| "T2 supervised 71.9% above the bar" (Meter-01) | 39.6% (Meter-02) / 31.2% (Meter-03, words) | already overturned; further reduced |

## 6. Diagnostics that qualify the numbers (not hidden)

* **Word target `KING` has 0 validation positives**, so its AUROC is undefined and it fails
  automatically. One of the 16 word targets is structurally unauditable; its failure is not
  evidence about the model.
* Several word targets have thin validation support (`My` 0.78%, `What` 0.87%, `they` 0.88%,
  `The` 1.15%, `more` 1.18%).
* **Live columns are few in the word runs** (19–21 of 32, vs 24–26 in the cluster runs), so
  many word columns fail on the floor before target-matching is even considered.
* The supervised half is audited against the target it was TRAINED on, so it measures
  whether supervision worked — not whether the concepts are real. Only the unsupervised
  half bears on the latter, and it is ~0%.

## 7. Pre-registered predictions (METER-RULE-03.md §7)

| # | prediction | result |
|---|---|---|
| 1 | supervised 85–100% | **NOT CONFIRMED** — 31.2% (words), 62.5% (clusters) |
| 2 | unsupervised 5–20% | **NOT CONFIRMED** — 0.0–2.1%, even LOWER |
| 3 | random control 10–30% | **NOT CONFIRMED** — 0.0% |
| 4 | name-swap rejection 70–95% | **100%** — decomposed in §3, not a clean triumph |
| 5 | lie probe CAUGHT | **CONFIRMED** |
| 6 | Target A NOT MET | **CONFIRMED** |

Predictions 1–3 were all wrong, in both directions: I over-estimated the supervised pass
rate, the random free pass rate, and the unsupervised pass rate.

## 8. Verdict

**TARGET A = NOT MET.**

* Lie probe caught: **YES** (new — this half of Target A is now genuinely satisfied).
* Unsupervised excess over the random control: **+0.0% to +2.1%**, against a 50% bar: **NO**.

The audit is now sound in the two ways it was broken: it no longer passes frozen columns,
and it hands out no free passes to random ones. What it shows is that the concept layer's
detection of pre-registered targets exists only where it was explicitly trained —
**supervision works, generalisation to untaught targets does not.** The earlier 45.8%
"non-circular gain" does not survive.

## 9. Option B — proposed, NOT run

Per the owner's instruction, Option B is a **design experiment proposal only**:

**Fix the names before training.** Pre-register every concept's name and train the layer
to detect them, so no name is ever derived from a column's firings. This removes the
circularity at its source rather than auditing around it.

* Why it is worth trying: Meter-03 shows supervision works (62.5% on clusters, against a
  0.0% random control), so pre-registering *all* columns' targets is a natural extension
  from 16 supervised to 32.
* Why it may not be enough: the unsupervised half was ~0%, so simply naming more columns
  and training them may raise the supervised number without producing concepts that
  generalise. The honest test is whether untaught targets start passing — and on today's
  evidence they do not.
* Cost: a training run per setting, 3 seeds, no test touch.
* **Not started. Awaiting the owner.**

## 10. Not done (deliberately)

* **No bar was changed**, and the claim rule was fixed in advance.
* **The engine's reference cycles remain unfixed** — a separate, later change.
* **No extrapolation**; everything is measured on surviving checkpoints.
* **Targets B/F/D/E were not started.**

## 11. Files

| what | where |
|---|---|
| the frozen rule | `METER-RULE-03.md` |
| audit results | `runs/t3/meter3_words.json`, `runs/t3/meter3_clusters.json` |
| lie probe | `runs/t3/meter3_lieprobe.json`, `scripts/t3_m3_probe.py` |
| name-swap decomposition | `runs/t3/meter3_clauses.json` |
| target/live diagnostics | `runs/t3/meter3_diag.json` |
| Meter-02 lie records (preserved) | `runs/t3/audit_K32_lr*_pa_*.json.meter02.bak` |
| full log | `runs/t3/meter3.log` |
