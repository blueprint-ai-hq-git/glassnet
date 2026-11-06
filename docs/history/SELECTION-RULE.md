# GLASSNET — JOINT SELECTION RULE (frozen 2026-10-04 06:20 IST)

**Frozen before any run in this milestone. Not to be changed after seeing
results.** Applies to every comparison from here on.

## Why this exists

At K=64 the readable's validation accuracy was **flat** across lr 0.3/0.5/1.0
(30.85/30.87/30.84 — all inside the noise), so "max validation accuracy" was an
arbitrary argmax on a flat surface, and it happened to hand the readable the
config with its **worst perplexity**. That decided the margin outcome. The gate
is a **joint** condition on accuracy AND perplexity, so the selection rule must
be joint too.

## THE RULE

1. **Scope.** Selecting each model's learning rate (and any other tuned knob)
   inside a comparison group.
2. **Budget.** Both models get an **identical** budget: the same lr list, the
   same seeds, the same epoch/patience settings, the same data.
3. **Primary criterion: validation perplexity.** For each model independently,
   select the config with the **lowest mean validation perplexity**
   (`exp(mean CE)` over the validation split). Lower is better.
4. **Tie-break 1 (accuracy).** If two configs' mean validation perplexities are
   within **1.0% relative** of each other, prefer the one with the **higher mean
   validation accuracy**.
5. **Tie-break 2 (determinism).** If accuracy is also within **0.05 pts**, prefer
   the **lower learning rate**.
6. **Selection data.** Validation only. The test split is never used for
   selection.
7. **Reporting.** Every comparison reports the full tuning table (all configs,
   both models), the selection made, and the tie-break that fired — so the
   selection is auditable and cannot be quietly re-picked.

## FIXED BARS (this milestone — never lowered)

| bar | value |
|---|---|
| concept fidelity AUROC | **≥ 0.55** |
| concept logit contribution | **≥ 0.25** |
| concepts passing | **≥ 50%** |
| lie probe | **caught** |

## TARGET A (this milestone)

**≥ 50% of concepts pass the audit with the lie probe caught, AND accuracy and
perplexity are no worse than the matched normal model.**

## TARGET B (only after A passes)

Readable beats the equally tuned normal on **perplexity AND accuracy by more
than 2× pooled std**, 5 seeds. Only then consider a single test-set touch, with
the rule written first.

## What is NOT allowed

- Changing this rule, the bars, or a gate rule after seeing results.
- "Passed in spirit."
- Calling parity a win.
- Using the test set for any selection.
- Quietly re-picking a config because a different one looked better.
