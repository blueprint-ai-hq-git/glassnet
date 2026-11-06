# METER-RULE-03 — PRE-REGISTERED TARGET-SOURCE AUDIT + TWO CONTROLS

**PRE-REGISTERED: 2026-10-04, before any Meter-03 audit is run.**
**Authority:** owner DECISION — "Option A now, plus two pre-registered controls."
**Bars: UNCHANGED** — AUROC 0.55, contribution 0.25, firing floor 2% of validation
windows, 50% pass, lie probe caught. This file may not be edited after seeing results.
**Test set: untouched.**

---

## 0. The problem this fixes

Meter-02 asked whether a column's name describes its firings — but the namer DERIVED
that name from the column's own firings, and the answer layer learns to use whatever
columns exist. So any live column passed by construction: a column frozen after one
epoch passed with AUROC 0.80. The measured consequence: **the meter could not tell a
trained concept from a frozen one.**

## 1. Meter-03 — the ONE change: where the target comes from

Everything else is byte-identical to Meter-02 (same residualisation, same contribution
formula, same thresholds, same floor). Only the target changes:

| | Meter-02 | **Meter-03** |
|---|---|---|
| target for column c | word-set parsed from the namer's name for column c | **the PRE-REGISTERED target frozen before training** (single word, or cluster member list) |

Concretely, for column `c` with pre-registered target `T_c`:

* **fidelity** = AUROC of the column's activation, residualised on window length
  (`L = log1p(#non-pad tokens)`), against `1[window contains any member of T_c]`.
  Identical statistic to Meter-02 (`_auroc`), same `lstsq` residualisation.
* **contribution** = mean over fired windows of the summed logit gain
  `Σ_{w ∈ T_c} Σ_{p ∈ positions of w} block_p · emb_w`, with `block_p` the `(W,D)`
  reshape of `concept.W[:, c]`. Identical formula to Meter-02, target words only.
* **firing** = fraction of validation windows with activation > 0.5.
* **pass** ⟺ `fidelity ≥ 0.55` AND `contribution ≥ 0.25` AND `firing ≥ 0.02`.

**Supervised columns (0..15)** are audited against `T_c` — the target they were trained
on. **Unsupervised columns (16..K−1) have no pre-registered target by construction**,
so they are audited against all 16 pre-registered targets; their pass rate is a
control-style number (they were never trained on any of them).

## 2. Lie probe under Meter-03

The probe column (column 2) trains normally until its validation firing reaches 2%,
then its gradients are frozen, so it is **never trained on any target**. It is then
audited **against target 2 — a target it was never trained on**. Verdict rule
unchanged: the probe must be **CAUGHT** (flagged BROKEN).

Because the probe checkpoint was not previously saved, the probe's own control run is
re-run with **unchanged settings** and its checkpoint saved. This is the control, not a
retrain: no model is retrained to fit the new meter.

## 3. Control 1 — random-column control (the free pass rate)

The concept layer is replaced by **random initialisation and never trained** (all K
columns frozen at random init); the embedding and answer layer are taken from the
corresponding trained checkpoint. Each draw is audited exactly like a trained model.
Drawn **10 times** (seeds 0..9) so the free pass rate is a distribution, not one sample.

* Report the free pass rate at the 2% floor, overall and split supervised/unsupervised.
* Report how many of the 32 columns are **live** (firing ≥ 2%) in each draw, since a
  dead column cannot pass and would flatter the control.
* **Trained results are reported as the EXCESS above this rate**: `excess = trained −
  random`, for both halves.

## 4. Control 2 — name-swap probe

For every trained column `c`, audit it against **another column's** pre-registered
target `T_d`, `d ≠ c`. A correct audit must **reject** the swapped pairing. Report:

* the **rejection rate** over all off-diagonal pairs (fraction flagged BROKEN),
* the **diagonal pass rate** for comparison,
* the **gap** between them.

## 5. Re-score only

Every trained model is an existing checkpoint, re-scored. No model is retrained, re-tuned
or re-selected for Meter-03. The only new computation is (a) the probe's own control run
and (b) the random-column control, which involves **no training at all**.

## 6. The Target A claim rule (fixed in advance)

Target A may be claimed **only if BOTH**:

1. the lie probe is **CAUGHT**, and
2. the **excess over the random control clears 50%** on the **UNSUPERVISED half**.

Why the unsupervised half governs: the supervised columns are audited against the very
target they were trained on, so their pass rate measures whether supervision worked, not
whether the concepts are real. The unsupervised half is the only half whose targets were
never used in training. Both halves are reported; only the unsupervised excess can
support the claim. **If the supervised excess clears 50% and the unsupervised one does
not, the verdict is NOT MET**, and I will say so.

## 7. Predictions (written before running)

1. **Supervised columns vs their true target: 85–100%** (higher than Meter-02's 62.5%,
   because the audit now asks exactly what they were trained on).
2. **Unsupervised half vs pre-registered targets: 5–20%** — far BELOW Meter-02's 45.8%,
   because those columns were never trained on these targets. **This is the prediction
   that would overturn the 45.8% figure**, and it is the main reason I am writing this
   before running.
3. **Random-column control free pass rate: 10–30%.** The AUROC clause can pass by
   chance on correlated targets; the contribution clause should be harder to reach.
4. **Name-swap rejection: 70–95%** — high but NOT 100%, because word and cluster targets
   overlap in the text (a column trained on `And` also fires on windows containing
   other frequent words). A rejection rate of 100% would be suspicious, not good.
5. **Lie probe: CAUGHT** (a frozen column should not detect a specific word or cluster).
6. **Target A: NOT MET**, because the unsupervised excess will be well below 50%.

## 8. Honest caveats, stated in advance

* **Correlated targets** make the name-swap probe and the random control imperfect: the
  text is small (1501-word vocab, 8-token windows), so several targets co-occur. A
  swapped or random column can therefore score above chance on a target it never saw.
  This inflates the free pass rate and lowers the rejection rate; it does NOT create a
  false pass for the trained columns' own targets, which is what the claim rests on.
* **The unsupervised pass rate is not a "concepts work" number** under Meter-03; it is a
  control-style number by construction. It is reported as excess over the random control
  and labelled as such wherever it appears.
* **The 45.8% figure from Meter-02 will change meaning** — it was measured against
  namer-derived names. Its Meter-03 replacement will be reported side by side, and any
  earlier claim resting on it will be flagged.
* **No extrapolation.** Everything is measured on surviving checkpoints.

## 9. Outputs

`runs/t3/meter3_<model>.json` per model, plus a side-by-side report
`METER-03-RESULTS.md` and the standard LOG/STATUS entries.
