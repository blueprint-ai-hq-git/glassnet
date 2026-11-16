
## The limit, measured: the row-content term does nothing either, and the transparency guarantee is exact attribution, not concept content (2026-11-16)

Predictions in `lens/runs/CONTENT-ARM-PREDICTIONS.txt`, written before the run - including the admission that this term had been retired that
morning at **2000% of the cross-entropy gradient** and was being given a fair re-test at a measured weight. Numbers from
`lens/runs/CONTENT-ARM.txt`.

```
final VAL accuracy        23.73%     (shipped 4M: 24.58%, so -0.85)
final VAL perplexity     203.50
concepts passing          51/53
median own-word mass      0.0008     (Z1 predicted above 0.002)   -> FAILED
explain() max |residual| 2.980e-06   | concept share 1.000000     -> PASSED
both gates before training: gradient check 3.749e-10 GRADIENT CORRECT, wiring TERM IS WIRED AND LOCAL
```

**Z1 failed and the falsifier fired in its written form.** The term ran for twelve epochs at a weight that put it at ~13% of the measured
cross-entropy gradient, and own-word mass is **0.0008** - identical to the free answer matrix it was supposed to improve, to the fourth
decimal, for a cost of 0.85 accuracy points.

## The five-mechanism record, all measured today

```
mechanism                                        own-word mass   VAL      verdict
free answer matrix (shipped 4M)                      0.0008       24.58%   baseline
structural rows + correction (unpriced)              0.0008       24.33%   no change
structural rows + priced correction                  0.0013       23.66%   +0.0005 for 0.67 pts
structural rows + structural scale 10                0.0012       23.58%   no change
pure structural rows, no correction                  0.0019       13.74%   +0.0011 for 10.6 pts
row-content supervision at 13% of the CE gradient    0.0008       23.73%   nothing
plus: post-hoc scaling drove mass to 0.0000; support restriction reached 0.63 mass at 0.0022 accuracy
```

**Five mechanisms, six measurements, none of them moved own-word mass above 0.002 without destroying accuracy.** The content arm is the
cleanest of the lot: the term was verified correct, ran at a sane weight, and changed the quantity it directly optimises by **zero**.

## The honest statement of the limit

**This project's transparency guarantee is EXACT ATTRIBUTION, and it delivers that to machine precision:**

- `explain()` residual **2.4e-06** to **2.98e-06** across every checkpoint measured today,
- concept share **1.000000**, i.e. the named concepts account for the prediction to within rounding,
- no hidden path, verified on the shipped model rather than assumed.

**It does NOT deliver concept CONTENT**, and that is now measured five different ways:

- a concept's answer row puts **0.08%** of its mass on the words that name it, at the median,
- the row is a diffuse, mildly code-flavoured direction (20.4% of its mass on code/maths vocabulary),
- the input side detects its name at AUROC **0.9997-0.9999** while the output side barely contains it - the two sides are **decoupled**
  (rank correlation **-0.307**),
- training under any pressure aimed at the row (price, cap, scale, direct supervision) leaves the mass at 0.0008-0.0019,
- and next-token prediction itself pushes the rows away from meaning, even with no free correction present.

**So the accurate sentence is**: *the model's answer is provably a weighted sum of 53 named handles, and each handle is a diffuse
vocabulary-wide direction rather than a concentrated detector of its name.* Not "the concepts mean what they say". The README and STATUS now
say exactly that, because the alternative is a transparency claim that is arithmetic-true and semantically empty.
