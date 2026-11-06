# METER-06 — FIRING FLOOR FOR GRADED CONCEPTS. PRE-REGISTERED BEFORE RE-SCORING.

Date: 2026-10-07 17:20. **Written before any re-scoring is run.**

## WHY, AND WHAT IT IS NOT

FIX-03 scored **1/9** under the existing bars. **That verdict stands: FIX-03 FAILED.** Nothing here
changes it, and the run is not being reinterpreted.

But the per-concept numbers show a mismatch that is worth stating precisely:

```
pronoun_object        AUROC 1.0000   fires 0.011   FAIL
is_a_negation         AUROC 1.0000   fires 0.006   FAIL
contains_a_wh_word    AUROC 1.0000   fires 0.001   FAIL
mentions_speech       AUROC 1.0000   fires 0.000   FAIL
```

**Seven concepts rank perfectly and all are failed by the firing floor.** The floor asks "is the
neuron's output above 0.5?" That question is correct for a BINARY concept, whose target is 0 or 1.
It is the wrong question for a GRADED concept, whose target is the FRACTION of the window the
concept occupies -- typically 1/8 = 0.125, and often 0. A graded neuron that ranks perfectly and
outputs 0.01-0.10 everywhere is doing its job; it will never cross 0.5, and requiring it to is
requiring it to be a different kind of neuron.

**This is a measurement defect, not a model result.** It is being fixed the way the project
already fixed one before: **pre-register the meter, then re-score existing checkpoints. No
retraining.**

## THE METER (frozen now)

For a graded concept, "fires" becomes **support-matched top-q coverage**:

1. Take the neuron's output over the evaluation windows.
2. Take the declared support `s` of the concept (measured from TRAIN text, as always).
3. **"fires" = the fraction of windows whose neuron output is in the TOP `s` quantile.**
4. A concept that means what its name says should have a high overlap between "top-q by neuron"
   and "the windows that actually contain the concept".

**AUROC is unchanged** and remains the primary fidelity measure: AUROC >= 0.55.

**The firing requirement becomes: the top-s quantile overlap must be at least 2x the rate that
random selection would give.** Random selection gives `s` (the support) by construction, so the bar
is **overlap >= 2 * s**, capped at 1.0. This is deliberately a WEAK bar: it asks only that the
neuron's most-active windows are twice as likely to be the true ones, which is the least a concept
neuron should be able to manage.

**Why 2x and not something tighter:** a tighter bar could be chosen later and look like tuning to
the result. 2x is fixed here, before any re-scoring, and applies to every concept equally.

## THE REPORTING RULE

Every re-scored result is reported with **BOTH meters side by side**, labeled:

* **METER-05 (binary firing, `output > 0.5`)** -- the original, unchanged.
* **METER-06 (graded, support-matched top-q)** -- this one.

**No result may be quoted with only METER-06.** A concept that passes the new meter and fails the
old one is described exactly that way, because the difference is the finding: it means the neuron
ranks correctly but does not saturate.

## WHAT IS RE-SCORED

The checkpoints already on disk. **No retraining, no test-set touch, nothing overwritten.** Every
re-scored output goes to a new file with `meter06` in the name.

## THE HONEST FRAMING

Even if METER-06 passes everything, **the blocker is not solved.** The perplexity gap is +12.281 to
+38.626 across seven arms against a bar of ~6. A concept audit passing under a corrected meter says
the concepts are real; it says nothing about the gap. Those are separate questions and will be
reported as separate questions.
