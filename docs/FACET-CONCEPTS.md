# Facet concepts: the best concept set so far, but the model still prefers impossible rules (2026-11-12)

The pair-concept result said the defect was structural: concepts about the same token are mutually exclusive, so
2-concept AND-rules are contradictions or tautologies. That points at a specific fix - build the concept set out of
near-independent **facets**, so a rule can join two things that genuinely co-occur.

12 concepts, three per facet:
```
shape:    last token is punctuation | last token is a digit | last token is capitalised | last token is 6+ chars
content:  window has '$' | window has '{' | window has backslash | window has <unk>
position: chat tag in last 4 tokens | chat tag in last 8 tokens
earlier:  token at -2 is punctuation | token at -3 is punctuation
```

```
concept set                accuracy   honesty   ablation   price     median lift   median |PMI|   independent rules
single-token (12)           15.28%     84.0%     17.75%    2.47 pp      0.98         0.58 bits       23 of 96
pair/positional (12)        15.40%     74.8%     17.65%    2.25 pp      0.72         2.18 bits       10 of 96
FACET (12, 4 facets)        16.38%     77.1%     18.30%    1.92 pp      0.82         0.93 bits       19 of 96
trivial baseline             5.42%       -          -         -            -             -              -
```

**The facet set is the best concept set measured: 16.38%, +1.10 points over single tokens and +0.98 over pairs, with the
cheapest readability price of the three (1.92 points).** Predictions: accuracy within 1.5 points of 15.28% **passed**;
median rule lift **>= 1.2** **failed** (0.82); independent rules **<= 15** **failed** (19); honesty **>= 80%** **failed**
(77.1%).

## What improved and what did not
Improved: accuracy, and cross-facet rules now appear among the heavyweights - the model's top rule is
`position: chat tag in last 4 AND earlier: token at -3 is punct`, which is a sentence a person can actually read.
Improved: median |PMI| settled at 0.93 bits instead of the pair set's contradictory 2.18.

Not improved: **the model still picks impossible rules when they are available.** Its second-heaviest rule is
`shape: last token is punctuation AND shape: last token is 6+ chars` - lift **0.00**, influence **74.3**. Wiring is still
random among all 66 pairs, so same-facet contradictions remain on the menu and the training objective still likes them,
because a contradiction is a rare, sharp, high-precision pattern: it fires almost never and is nearly always right when it
does.

The influence-vs-lift correlation stayed negative (-0.119). Across three concept sets (single tokens, pairs, facets) the
model has preferred high-influence low-lift rules every time.

## The arm running now
Cross-facet wiring: a rule may only join two concepts from **different** facets, which makes the impossible pairs
unavailable rather than discouraged. On the facet set there are 52 legal pairs, so the constraint does not starve the
layer the way the lift-based mask did (16 of 66).

Prediction: median rule lift **>= 1.2**; **zero** rules joining logically impossible pairs; accuracy within 1 point of
16.38%; at most 15 of 96 rules joining independent concepts.

## Run hygiene
The first facet arm is **void**: the concept-branch chain let the token-presence branch overwrite the facet set, so the run
trained 12 token concepts under a facet label. The tell was the checkpoint's own names (`window contains 'of'`), not the
loss curve. A guard now runs before every training job and refuses to start unless the concept names are distinct, the
count matches `C`, and the label builder produces exactly `C` columns - printed as a "concept set verified" line. That
guard is what makes the accepted run trustworthy.
