# Phase 2 rule layer: built, measured, and it FAILS against the right bar (2026-11-11)

The morning brief's middle layer now exists: 36 named concepts -> 96 rules wired to exactly 3 concepts each -> answer,
with an L1 penalty on the rule weights and pruning. This is the first time the project has had it, and this is what it
measures.

## The gate numbers, with the bar that matters
```
BASELINE    : always predicting token id 0 scores 5.42% on the same held-out windows
WITH rules  : held-out next-token accuracy 5.42%    concept accuracy 82.9% mean over 36 concepts
WITHOUT rules: 3.55%  (ablation: concepts straight to the answer, no rule layer)
connections per rule: 3 before pruning, 2 after
rules a human must read: 96, each naming its 2 concepts
concept honesty: mean 82.9%, weakest "window contains 'of'" at 56.1%
```

**The rule model does not beat "always output the most common token".** It ties it exactly at 5.42%. My prediction was
"within 2 points of the ablation" and it passed that - which is precisely how a bad gate hides a null result, because the
ablation itself scores 3.55%, *below* a constant. Comparing two weak models and declaring the winner progress is the
mistake this project keeps a rule against, and it happened again here until the trivial baseline was computed.

## Why it cannot learn anything yet
Mean pooling over the 32-token window destroys word order. A next-token predictor with no order information has nothing
to condition on except the frequencies it already knows, so no amount of rule structure can recover what the input no
longer contains. The rule layer is not the problem; the input to it was.

## What is genuinely established by this run
- The three-layer structure works mechanically and is readable: 96 rules, each wired to 2 concepts after pruning, each
  one printable as "rule #k fires when concepts A and B are on".
- Concept supervision works: 82.9% mean accuracy against computed labels, with the weak ones being the most frequent
  tokens ("of" 56.1%) - the same pattern that makes frequent-token concepts hard, and it is reported rather than hidden.
- Two optimiser settings were needed to see any signal at all: lr 0.30 diverged (loss 6.3455 -> 5.8448 -> 5.9712 ->
  6.2648) while lr 0.03 fell steadily (5.7051 -> 5.5408 -> 4.9077 at step 1,500).

## The fix now running, with its prediction
Feed the rule layer the **last-token embedding** so order is available, keep everything else identical. Prediction: beats
the printed trivial baseline by at least 1 point, with the same readability numbers. The baseline is now printed inside
the script itself, so this particular mistake cannot pass silently again.
