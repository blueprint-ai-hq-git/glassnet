# GLASSNET — MILESTONE GATE RULE (T2, written BEFORE any test touch)

**Date:** 2026-10-03
**Status:** thresholds computed from VALIDATION ONLY. The test set is untouched
(last touch #6). This file is frozen before the test touch so no threshold can be
moved after seeing test numbers.

---

## 1. THE RULE (reviewer's wording, verbatim)

> "WIN" = readable beats normal on BOTH perplexity and accuracy by more than 2x
> pooled std AND at least 50% of concepts pass the audit with the lie probe
> caught. Otherwise "no win."

## 2. EXACT OPERATIONAL DEFINITIONS (fixed before the touch)

1. **Config selection.** Each model uses the lr that maximises its **validation
   accuracy** over the item-(2) grid {0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0} at
   K=64. (Pre-registered rule. See §5 for the ambiguity this creates.)
2. **n = 5 seeds** per model, same data, same shuffle, same batch, same
   stopping rule (patience 15, max 96, no early stop before epoch 30).
3. **Accuracy margin** = mean(readable acc) − mean(normal acc).
4. **Perplexity margin** = mean(normal ppl) − mean(readable ppl) (positive =
   readable better).
5. **Pooled std** (per metric) = `sqrt( (std_readable² + std_normal²) / 2 )`,
   population std, n=5 each.
6. **Concept clause** = ≥ 50% of the K=64 concepts pass the standing audit
   (residualized AUROC ≥ 0.55 **AND** logit name-contribution ≥ 0.25), i.e.
   **≥ 32 of 64**.
7. **Lie clause** = the frozen-neuron probe is CAUGHT (flagged BROKEN or AUROC
   < 0.55).
8. **WIN requires ALL FOUR clauses.** Anything else is **"no win."**
9. **One test touch.** Train on train only; validation for all selection; test
   evaluated exactly once and the touch logged in TEST-USE.md.
10. **Never call parity a win.**

## 3. THE EXACT NUMBERS (validation, 5 seeds per model)

### Selected configs
| model | selected lr | accuracy | perplexity |
|---|---|---|---|
| readable | **0.5** | 30.87 ± 0.08 | 60.85 ± 2.82 |
| normal | **0.2** | 30.61 ± 0.12 | 63.47 ± 1.65 |

### Clause-by-clause thresholds — PRIMARY (pre-registered selection)

| clause | margin | pooled std | threshold (2× pooled) | result |
|---|---|---|---|---|
| accuracy | **+0.26 pts** | 0.101 | **0.202 pts** | **PASS** |
| perplexity | **+2.62** | 2.308 | **4.616** | **FAIL** |
| concepts ≥ 32/64 | measured **16.3/64 = 25.5%** | — | **32/64** | **FAIL** |
| lie probe caught | caught (col 2, AUROC 0.614) | — | caught | **PASS** |

### Sensitivity — the readable at its other grid optima

| readable lr | acc margin vs 0.202 / 0.181 / 0.189 | ppl margin vs its 2× pooled | margins |
|---|---|---|---|
| 0.5 (primary) | +0.26 vs 0.202 | +2.62 vs 4.616 | acc PASS, **ppl FAIL** |
| 1.0 | +0.23 vs 0.181 | +5.48 vs 2.721 | both PASS |
| 0.3 | +0.23 vs 0.189 | +3.25 vs 2.636 | both PASS |

(readable 5-seed sets: lr 0.5 = 30.87 ± 0.08 / 60.85 ± 2.82; lr 1.0 =
30.84 ± 0.05 / 57.98 ± 0.99; lr 0.3 = 30.85 ± 0.07 / 60.22 ± 0.87.
normal lr 0.2 = 30.61 ± 0.12 / 63.47 ± 1.65; normal lr 0.1 =
30.56 ± 0.10 / 65.09 ± 2.31.)

## 4. PRE-COMPUTED VERDICT — **NO WIN**

| selection | accuracy | perplexity | concepts | lie | verdict |
|---|---|---|---|---|---|
| primary (readable lr 0.5) | PASS | **FAIL** | **FAIL** | PASS | **NO WIN** |
| sensitivity (readable lr 1.0) | PASS | PASS | **FAIL** | PASS | **NO WIN** |
| sensitivity (readable lr 0.3) | PASS | PASS | **FAIL** | PASS | **NO WIN** |

**The verdict is "NO WIN" under every selection, because the concept clause
fails (25.5% vs the 50% bar).** The accuracy and perplexity margins cannot
rescue it — the rule requires all four clauses.

**Consequence: the test touch cannot change this verdict.** A "no win" is
already determined by validation. The test touch would only record test-set
numbers for a milestone that has already failed its readability clause.

## 5. FLAGGED PROBLEM: THE SELECTION RULE IS NOT COHERENT FOR A JOINT GATE

The readable's validation accuracy is **flat** across lr 0.3 / 0.5 / 1.0
(30.85 / 30.87 / 30.84, stds 0.05–0.08 — all inside the noise). Taking the
argmax of a flat noisy surface is arbitrary, and it lands on lr 0.5, which has
**the worst perplexity of the three**. So a rule that selects on accuracy alone
can hand the model its worst config on the other metric the gate also requires.

Under the primary rule the perplexity clause therefore FAILS (+2.62 vs 4.616);
under either alternative readable config BOTH margin clauses PASS. **The
selection rule, not the model, decides the margin outcome.**

This is flagged, not resolved by choosing the flattering config. The honest
fix is to pre-register a **joint** selection rule (e.g. select on validation
accuracy AND perplexity together, or fix the lr per model before the gate and
never re-select), and that must be done **before** the next test touch.

## 6. WHAT THE DATA ACTUALLY SAYS

- **Readable vs matched normal on margins:** a small, consistent edge on
  accuracy (+0.23 to +0.26 pts) and on perplexity (+2.6 to +5.5), but the
  perplexity edge's significance depends entirely on which readable lr the
  selection rule hands it.
- **Readability:** FAILS. 25.5% of concepts pass against a 50% bar, and the
  binding constraint is **contribution** (only 30% of neurons contribute
  ≥ 0.25 to the logits, while 86% pass the fidelity AUROC). The concepts mean
  roughly what their names say; the answer layer does not use them enough.
- **So the honest one-liner: the model is slightly better than the control and
  not readable enough to claim a win. "NO WIN."**
