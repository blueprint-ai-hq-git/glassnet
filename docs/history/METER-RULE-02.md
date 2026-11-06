# METER-RULE-02 — AUDIT METER, REVISION 2 (firing-rate floor + live lie probe)

**PRE-REGISTERED AND FROZEN: 2026-10-04, before any re-audit was run.**
**Authority:** owner decision "A + C, as a rule change for the meter only."
**Bars: UNCHANGED** — AUROC ≥ 0.55, contribution ≥ 0.25, 50% pass, lie probe caught.

This file is written **before** the code change and **before** any re-audit. Nothing
below may be edited after results are seen. If a number in section 6 is wrong, it gets
a correction note appended — not silently rewritten.

---

## 1. Why this revision exists

The lie probe is the audit's integrity check: it must catch a concept column that
passes without detecting anything. On 2026-10-04 it was **MISSED** under supervision
where it had been CAUGHT at baseline. Diagnosis (`scripts/diag_lie_probe.py`):

* The frozen column's weights never move (freeze verified in both training loops).
* But the auxiliary loss trains the **embeddings**, which inflates their norm
  (84.7 → 121.6) and drives the frozen column's fixed random projection into
  saturation: it became **98.5% dead**, firing on 117 of 10,124 validation windows.
* The namer then picked a word-set present in *exactly* those windows
  (`pos_frac` 0.0115 == `frac>0.5` 0.0115), giving AUROC 1.0000 **by construction**.

So a nearly dead column can be named to match its own few firings and then passes.
Measured on the existing checkpoints, this was not confined to the probe: applying a
2% firing floor drops the supervised columns from **93.8% → 31.2%** pass, i.e. below
the unsupervised 47.9%.

---

## 2. THE CHANGE (one change only)

A concept column counts as PASSING if and only if **all three** hold, evaluated on
VALIDATION:

1. length-residualized AUROC ≥ **0.55** *(unchanged)*
2. mean logit contribution over fired windows ≥ **0.25** *(unchanged)*
3. **firing rate ≥ FLOOR** *(NEW)*

**Definition of firing rate (fixed here):** the fraction of validation windows in
which the concept column's activation exceeds **0.5** — i.e. `mean(h[:, k] > 0.5)`
over the validation split, where `h = sigmoid(concept_layer(x))`.

**Primary floor: FLOOR = 0.02 (2% of validation windows).**
**Sensitivity floors reported alongside: 0.01 (1%) and 0.05 (5%).**
The 2% figure is the verdict; 1% and 5% are reported as sensitivity only and may not
be substituted for it after seeing results.

---

## 3. THE LIVE LIE PROBE (owner condition 2)

Old probe: the column was frozen **from initialisation**, so it could die and then
pass by being dead.

New probe (fixed here): the probe column is a **LIVE** column.

1. Train the probe network **normally** (no freeze) until the probe column's
   validation firing rate first reaches **≥ 2%** (the primary floor). Record that
   epoch.
2. **Then** freeze the column's gradients (`W.grad[:, k] = 0`, `b.grad[k] = 0`) so it
   receives no further training.
3. Continue training the rest of the network to the same stopping rule as every other
   run.
4. Audit as usual, and additionally record the probe column's firing rate **at audit
   time**.

**Verdict rule (fixed here):**

* If the probe column is flagged by the meter → **CAUGHT**.
* If it is not flagged → **MISSED**, which is a failure of the audit.
* The firing rate at audit time is reported either way. If the column died after
  freezing, that is stated explicitly, because then the *floor* did the catching and
  the AUROC/contribution clauses were not the thing under test.
* A probe that never reaches 2% firing while unfrozen is reported as **INCONCLUSIVE**
  (not as a pass), with the firing rate it did reach.

---

## 4. SCOPE OF THE RE-AUDIT (owner condition 3)

Every earlier concept-audit result is re-scored under Meter-02. The recorded audit
files do **not** contain firing rates, so firing rates are recomputed from the saved
checkpoints by re-running each concept layer's forward pass on that rung's validation
split. No model is retrained for this; no tuning is done.

| # | result | meter | artifacts | re-auditable? |
|---|---|---|---|---|
| 1 | T2 baseline (lam_concept=0) | AUROC + contribution | `runs/t3/audit_K32_lr10_pa_w20.json` + checkpoints | yes |
| 2 | T2 supervised (lc 0.1 / 0.5 / 1.0) | AUROC + contribution | `audit_K32_lr*_pa_lc*.json` + checkpoints | yes |
| 3 | T2 earlier sweeps (K 8/16/24/32, namer, support, lam_act, lam_w) | AUROC + contribution | `runs/t3/audit_*.json` + 96 checkpoints | yes, where a checkpoint exists |
| 4 | Fashion-MNIST (rung 1 / phase 2) | fidelity AUROC (+ mean act) | `runs/phase2/audit.md`, `final_seed0.npz` | yes — firing recomputed from the checkpoint |
| 5 | PathMNIST (rung 2, loop iters 1–14) | fidelity AUROC | `runs/loop/iter*/audit.md` + `final_seed*.npz` | yes, where a checkpoint exists |
| 6 | Banking77 (rung 3, loop iters 15–18) | fidelity AUROC | `runs/loop/iter018/audit.md` + `final_seed*.npz` | yes — firing recomputed from the checkpoint |

**If a result has no surviving checkpoint, it is reported as NOT RE-AUDITABLE.** It is
never estimated, inferred, or left out silently.

**Old vs new reported side by side** for every row, with the number of columns whose
verdict changed, and any earlier claim the new meter overturns is flagged explicitly.

### Note on rung-1/rung-2/rung-3 meters

Those rungs use a *fidelity AUROC* meter (each neuron tested against its own named
concept on held-out data) rather than the T2 AUROC+contribution meter. The firing
floor applies to them the same way (a column must fire on ≥2% of validation windows),
but their pass rule is otherwise theirs, not T2's. Where a rung's audit already
recorded a positive-rate or skew flag (Banking77 records `pos` and flags
`SKEWED-TARGET`), that is reported alongside rather than replaced.

---

## 5. THE HONEST COST — which LEGITIMATE rare concepts this floor excludes

The floor cannot tell "rare because the concept is junk" from "rare because the
phenomenon is rare". It excludes both. Concretely, from `CONCEPT-TARGETS.md`
(TRAIN-only supports):

| target word | train_pos_frac | below the 2% floor? |
|---|---|---|
| `And` | 0.0705 | no |
| `he` | 0.0464 | no |
| `The` | 0.0337 | no |
| `To` | 0.0319 | no |
| `him` | 0.0292 | no |
| `That` | 0.0246 | no |
| `But` | 0.0210 | no |
| `KING` | 0.0204 | no |
| `good` | 0.0198 | **yes** |
| `For` | 0.0183 | **yes** |
| `My` | 0.0176 | **yes** |
| `they` | 0.0174 | **yes** |
| `I'll` | 0.0166 | **yes** |
| `more` | 0.0155 | **yes** |
| `As` | 0.0151 | **yes** |
| `What` | 0.0149 | **yes** |

**8 of the 16 supervised targets sit below the 2% floor.** A concept that is a
*perfect, precise* detector of `What` — firing on exactly the 1.49% of windows that
contain it, AUROC 1.0 — would be **excluded** by this floor. That is a real cost, it is
accepted deliberately, and it is recorded here so it cannot be quietly forgotten later.

The same applies to any rung with genuinely rare concepts (e.g. a rare Banking77
intent family). At the 5% sensitivity floor the exclusion is broader still.

**Rationale for accepting the cost:** a concept that fires on ~1% of windows gives an
AUROC that is dominated by a handful of windows, so its AUROC is not evidence of
detection regardless of whether the phenomenon is rare. The floor trades a small
number of genuine rare detectors for the removal of a class of false passes.

---

## 6. PRE-REGISTERED PREDICTIONS (written before running)

1. **Live lie probe:** reaches ≥2% firing while unfrozen; at audit time it is
   **CAUGHT**, and — the stronger claim — caught by the AUROC/contribution clauses
   (AUROC < 0.55) rather than only by the floor.
2. **T2 supervised (lc=1.0, lr=0.5):** pass fraction falls from 71.9% to **≈39.6%**
   at the 2% floor (already measured on existing checkpoints; the re-audit should
   reproduce this). Supervised half falls from 93.8% to **≈31.2%**, i.e. *below* the
   unsupervised half.
3. **T2 baseline (lam_concept=0):** barely moves — 29.2% → **≈29.2%**, because its
   passing columns already had a median support of 29%.
4. **Fashion-MNIST (rung 1):** moves little. Its recorded mean activations are mostly
   healthy (0.13–0.77), so few neurons should be near-dead. Prediction: **≤2 of 14
   neurons change verdict**.
5. **Banking77 (rung 3):** already FAILED its own audit (12 of 16 broken, and its own
   `SKEWED-TARGET` flags). The floor should not rescue it; prediction: pass fraction
   **stays 0 or near 0**.
6. **The meter change lowers scores everywhere it bites.** If any rung's pass fraction
   *rises* under the floor, that indicates a bug in my implementation, not a finding.

---

## 7. What is NOT changed

* AUROC bar 0.55, contribution bar 0.25, 50% pass bar, lie probe caught — **unchanged**.
* The validation/test discipline: **validation only**; the test set is not touched by
  any of this work.
* The frozen selection rule (`SELECTION-RULE.md`) — unchanged.
* The supervised/unsupervised split reporting — unchanged, and still required.
* No model is retrained or re-tuned to satisfy the new meter. Re-scoring only.

## 8. After the re-audit (owner condition 4)

Only then continue Target A: **richer concept targets (word clusters rather than
single words)**, keeping the supervised/unsupervised split. No test-set touch. If
Target A is met, **do not claim a win** — Target B comes next.

**Stopping conditions unchanged:** if something looks broken or unclear, stop and ask.
