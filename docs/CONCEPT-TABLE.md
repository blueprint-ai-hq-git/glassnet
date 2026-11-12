# The full 36-concept table: what makes a concept honest (2026-11-12)

Phase 2's fourth gate item asks whether each concept neuron really fires on its named concept. Until now only the five
weakest were ever printed. `tools/concept_table.py` produces the whole table from the saved checkpoint without retraining,
and adds a cross-firing test: for every neuron, which OTHER concept's label it predicts better than its own.

## The table, weakest first (balanced accuracy; a dead neuron scores 50%)

```
concept                              balanced  preval%  recall+
window contains '`'                     48.0%    10.2%    28.6%
last token is '?'                       49.1%     0.1%     0.0%
window contains 'is'                    49.7%    24.1%    22.8%
last two tokens are identical           50.0%     2.1%     0.0%
window contains 'for'                   50.8%    18.3%    15.3%
window contains '*'                     51.0%    10.0%     5.5%
window contains 'x'                     51.5%    12.6%    10.8%
window contains '''                     54.0%    15.2%    42.4%
window contains <|im_end|>              54.6%     1.6%    13.8%
window contains '2'                     55.9%    14.2%    37.8%
window contains '-'                     56.3%    26.9%    34.6%
window contains 'to'                    56.6%    36.6%    38.3%
window contains 'a'                     57.5%    35.0%    25.0%
window contains '.'                     57.5%    76.8%    72.4%
window contains 'and'                   58.7%    38.5%    35.7%
window contains ','                     59.5%    70.2%    76.2%
window contains '<unk>'                 62.4%    67.5%    74.3%
window contains ':'                     62.9%    29.9%    36.4%
window contains 'of'                    64.0%    43.9%    62.9%
window contains 'in'                    64.0%    24.5%    42.9%
window contains '('                     65.6%    25.7%    63.3%
window contains 'the'                   66.0%    63.7%    61.2%
window contains '}'                     67.2%    12.7%    48.6%
window contains ')'                     67.3%    26.3%    53.3%
window has adjacent duplicate tokens    68.1%    26.1%    46.0%
window has 3+ punctuation tokens        75.3%    92.5%    98.4%
window contains '='                    77.2%    24.2%    68.4%
window contains '\\'                   77.8%    12.3%    58.8%
window contains '{'                    82.2%    13.2%    69.1%
last token is punctuation               85.1%    25.7%    76.5%
window contains '$'                    90.4%    18.4%    89.5%
window contains <unk>                   91.3%    67.5%    94.5%
window contains '"'                    96.0%     9.9%    92.2%
last token is a digit                   96.7%     2.8%   100.0%
window contains <|im_start|>            97.4%    14.7%    95.4%
last token is '.'                       99.5%     3.5%   100.0%
```

Mean **67.1%**, min 48.0%, max 99.5%.

## My rarity hypothesis was wrong, and the table says so
The explanation I gave on 2026-11-11 was "the honest concepts are the rare ones, and rare ones starve". The correlation
between prevalence and honesty across all 36 concepts is **pearson -0.021, spearman +0.114**: essentially nothing. A
concept true in 0.1% of windows ('?') and one true in 92.5% ('3+ punctuation') sit at opposite ends of the honesty range,
and so do a 67.5% token (dead at 62.4%) and a 3.5% one (99.5%). **Prevalence does not predict honesty.**

What does separate them is the token itself: rare, unambiguous, structurally distinctive tokens and positions score
85-99.5%; common function words score 48-67%.

## The cross-firing test: 15 of 36 neurons secretly detect something else
For each neuron, the label it predicts better than its own (by more than 2 points):

```
neuron                            own      better at                        that label
window contains 'a'              67.2%    last token is '?'                  84.7%   (+17.4)
window contains 'is'             63.6%    last token is '?'                  76.8%   (+13.2)
window contains 'for'            73.2%    last token is '?'                  85.8%   (+12.5)
window contains 'and'            64.0%    last token is '?'                  75.0%   (+11.0)
window contains ':'              73.5%    window contains '"'                84.5%   (+11.0)
window contains '*'              87.4%    last token is '?'                  96.1%   (+ 8.7)
window contains '-'              66.3%    window contains 'x'                74.9%   (+ 8.6)
window contains ')'              73.9%    window contains '`'                82.0%   (+ 8.1)
window contains 'x'              82.1%    last token is '?'                  91.8%   (+ 9.7)
window contains 'to'             61.5%    window contains <|im_end|>         70.7%   (+ 9.1)
window contains '`'              63.5%    last token is a digit              70.6%   (+ 7.1)
window contains '2'              68.9%    last token is a digit              74.1%   (+ 5.2)
window has adjacent duplicate    78.7%    window contains '{'                84.1%   (+ 5.4)
window contains 'in'             74.8%    last token is '?'                  78.1%   (+ 3.3)
window contains '''             62.2%    last token is '?'                  64.3%   (+ 2.2)
```

**15 of 36.** This is exactly the failure the brief asked to be able to catch: a neuron whose name says one thing and
whose behaviour says another. Note the direction - many of them leak toward "last token is '?'", the rarest concept, which
is the one a neuron cannot be honest about (its own recall is 0.0%).

## What this changed
The concept set is now chosen HONESTLY, before training, using a train-split probe: a logistic regression on the same
input the concept neurons see (mean embedding + last-token embedding), fitted on TRAIN windows only, so the held-out
split cannot leak into the choice. `tools/select_concepts.py` keeps the 12 concepts whose train probe says the input can
support them:

```
last token is '?'            last token is '.'          last token is punctuation
last token is a digit        contains <|im_start|>      contains '$'
contains '`'                 contains '\'              contains '{'
contains '<unk>'             contains <unk>             contains '}'
```

The 24 dropped include every common function word ('the' probe 84.1%, 'is' 72.2%, 'for' 62.6%) - they were predictable in
a probe but never honest in the trained model, and now they are not named in the model at all.

**Prediction for the 12-concept arm, written before the run:** balanced honesty on held-out at or above **90%**, token
accuracy between **8% and 14%** - below the 19.68% of the 36-concept model because 12 rare concepts carry less predictive
signal. Above 15% means the honest set is free; under 8% means it is too thin to predict with.
