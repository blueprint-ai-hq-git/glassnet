# Concept selection by data: the dead concept was an empty label, not a broken neuron (2026-11-12)

The explain() run on the best model reported a defect: `shape: last token is capitalised` never fired and sat at exactly
**50.0%** honesty, so it was written up as a dead neuron. Measuring the label against the data gives the real cause.

```
facet concept                          prevalence (train)   train probe (balanced)
shape: last token is punctuation            27.36%                99.4%
shape: last token is a digit                 2.71%                98.4%
shape: last token is capitalised             0.00%                75.0%   <- label occurs ZERO times
shape: last token is 6+ chars               19.65%                81.5%
content: window has '$'                     17.37%                88.5%
content: window has '{'                     12.86%                92.5%
content: window has backslash               12.02%                88.1%
content: window has <unk>                   67.72%                77.7%
position: chat tag in last 4                 1.32%                74.5%
position: chat tag in last 8                 2.66%                66.0%
earlier: token at -2 is punct               27.19%                66.6%
earlier: token at -3 is punct               27.04%                63.6%
```

**The label occurs in 0 of 20,000 training windows** - the corpus vocabulary is lowercased, so a capitalised token does not
exist in it. A neuron asked to predict an always-false label can only ever output 0, and balanced accuracy for a constant
predictor on an empty class is exactly 50.0%. The "dead neuron" was a **concept-set error**: a label with no positive
examples was allowed into the model for six arms.

Nothing in the loss curve showed this. Only checking each label against the data did.

## The fix, and what it bought
`FACET_SEL=1` now measures every label's prevalence on TRAINING windows before the model is built and drops any label with
no positive examples. It dropped exactly one, as predicted.

```
config                                    accuracy   honesty (mean / min)   median per-rule loss   worst rule   lift-corr
12 concepts, answer-row L1 (previous best)  15.78%      74.8% / 50.0%          0.500 pts           -0.550       +0.263
11 concepts, impossible label dropped       15.22%      79.5% / 61.8%          0.487 pts           -0.125       +0.184 (s+0.433)
```

Predictions: "exactly 1 concept dropped" **passed**; "no concept at exactly 50.0%" **passed** (min is now 61.8%); "mean
honesty >= 76.5%" **passed** (79.5%); "accuracy within 0.5 points of 15.78%" **missed by 0.06** (15.22%).

Four things improved from removing one impossible label:
1. **No concept is dead any more** - worst concept 61.8% instead of exactly 50.0%.
2. **Mean concept honesty rose 74.8% -> 79.5%** (+4.7 points) with the same training recipe.
3. **The harmful rule mostly healed**: its per-rule loss went from **-0.550 to -0.125**, so deleting the worst rule now gains
   only an eighth of a point instead of half a point.
4. **Fewer implausible rules among the heaviest**: 3 of 10 instead of 4 of 10, with spearman +0.433 between rule
   plausibility and rule necessity.

One thing got slightly worse: accuracy 15.78% -> 15.22% (-0.56 points), and the readability price rose 1.65 -> 1.90 points.
The accuracy move is inside single-seed noise, but it is recorded as a move, not explained away.

## Why the 11-concept model is the better deliverable
The 12-concept model scored half a point higher only by carrying a concept that cannot be true. A model whose every label
occurs in the data, whose every rule is load-bearing, and whose rules are plausible more often, is the honest artifact even
when its accuracy is marginally lower.

## The rule this adds
**A concept label must be verified to occur in the training data before it may enter the model** - the same check that was
already applied to token concepts, now applied to every concept set, and enforced in code (`FACET_SEL`) rather than
remembered. This is the third time a measurement defect, not a modelling one, produced a wrong number: the inflated plain
accuracy, the silently-unwired mask, and now an empty label reported as a dead neuron.

## Analysis-tool fix made in the same run
Both analysis tools compared the checkpoint's concept list to the full facet set **positionally**, so a legitimate subset
(FACET_SEL dropped one) was refused as "not a facet model". They now map concepts **by name**. A tool that refuses a valid
checkpoint is safer than one that silently mislabels it, but it still cost a run to discover, and positional comparison of a
set is the wrong data structure.
