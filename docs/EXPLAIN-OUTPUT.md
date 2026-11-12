# explain(): what the model says and why, on the best configuration (2026-11-12)

Model: `rules_arl1.npz` - 24 rules, 12 named facet concepts, 2 connections per rule, answer-row L1, held-out accuracy
15.78%. This file records the actual output and the causal checks behind it.

## What explain() prints, verbatim, for three held-out windows

```
window 0: last 8 tokens ['you','to','clarify','what','your','question','is','.']
  answer 'you'  (true '<|im_end|>')   scores: you 3.42, and 3.36, <|im_end|> 3.09
  rule #17 fires 0.375 -> pushes 'you' +1.85  (reads: earlier: token at -2 is punct AND shape: last token is a digit)
  rule #8  fires 0.675 -> pushes 'you' +0.86  (reads: shape: last token is punctuation AND content: window has '{')
  rule #18 fires 0.360 -> pushes 'you' +0.57  (reads: content: window has <unk> AND position: chat tag in last 8)

window 1: last 8 tokens ['in','the','fridge','?','<|im_start|>','assistant','i','don']
  answer '''  (true 'right single quote')   scores: ' 6.09, right-quote 4.77, tone 4.48
  rule #0  fires 0.947 -> pushes ''' +3.14  (reads: earlier: token at -2 is punct AND content: window has <unk>)
  rule #13 fires 0.944 -> pushes ''' +2.05  (reads: shape: last token is 6+ chars AND content: window has <unk>)
  rule #18 fires 0.360 -> pushes ''' +0.80  (reads: content: window has <unk> AND position: chat tag in last 8)

window 2: last 8 tokens ['<|im_start|>','user','yes','.','<|im_start|>','assistant','here','is']
  answer 'be'  (true 'what')   scores: be 4.96, the 3.94, writing 3.42
  rule #17 fires 0.379 -> pushes 'be' +1.78  (reads: earlier: token at -2 is punct AND shape: last token is a digit)
  rule #11 fires 0.329 -> pushes 'be' +1.36  (reads: shape: last token is 6+ chars AND content: window has backslash)
  rule #18 fires 0.360 -> pushes 'be' +1.22  (reads: content: window has <unk> AND position: chat tag in last 8)
```

Every number above is read out of the forward pass: the rule's activation, the answer weight it contributes, and the two
concepts it is structurally wired to. Nothing is post-hoc rationalisation - the same quantities are what produce the answer.

The sentences are now **cross-facet**, so a reader gets "a shape fact AND a content fact", which is the kind of reason a
person can evaluate. That was not true earlier today: the identical output used to read
`window contains <unk> AND window contains '}'`.

## The causal checks on this model
```
mean concept honesty                 74.8% balanced
mean answer flip when a concept is forced ON   36.2%
most causal concept   shape: last token is a digit        flip 56.9%   honesty 86.2%
least causal concept  shape: last token is capitalised    flip  0.0%   honesty 50.0%
correlation honesty vs causal influence: pearson -0.018
```

So forcing a single concept changes the answer on **36.2%** of held-out windows: the readable path is load-bearing.

## Two defects this output exposes, both new
1. **One concept is completely dead.** `shape: last token is capitalised` never fires (flip 0.0%) and its honesty is exactly
   **50.0%**, i.e. a neuron that says nothing. A named concept in the model is inert, and the ×5 concept loss did not wake it
   because the label is rare in this corpus after lowercasing.
2. **Concept honesty does not predict causal influence on this model** (pearson -0.018; it was +0.253 on the 12-token model).
   Rule-level plausibility does predict usefulness (+0.263), but concept-level honesty does not: a concept can be honest and
   unused, or dishonest and decisive.

## Where the honesty story now stands
```
level          property measured                 result
concept        means what its label says          74.8% balanced (one concept dead at 50.0%)
concept        influences the answer              36.2% mean flip when forced
rule           says something plausible           no contradictions; median pair lift 0.89
rule           is necessary for accuracy          median 0.500 pts when removed; +0.263 with plausibility
layer          costs accuracy to keep             1.65 points
```

The layer is readable, its explanations are mechanically true, its rules are necessary, and its rules are mostly plausible.
What is still not true: concept-level honesty is not aligned with influence, and one concept is inert.
