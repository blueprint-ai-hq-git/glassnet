
## The share WAS fixed and meaning still did not appear: the hard cap flips 0.9999 -> 0.2087 and own-word mass stays at 0.0016 (2026-11-16)

Predictions in `lens/runs/HARD-CAP-PREDICTIONS.txt`, written before the run. Numbers from `lens/runs/HARD-CAP.txt`. This is the decisive
version of the experiment, because for the first time the intervention **worked exactly as designed on the quantity it targeted**.

```
final VAL accuracy        14.48%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     373.25
concepts passing          49/53
median correction share   0.2087     (unpriced: 0.9999)   -> H1 PASSED, the share flipped
median ||structural||     4.4949     (unpriced: 1.5466)
median ||A_c B||          2.6950     (unpriced: 87.42; capped)
rows with correction >50% : 10 of 53  (unpriced: 53 of 53)
median own-word mass      0.0016     (H2 predicted above 0.05)   -> FAILED
explain() max |residual| 1.048e-06   | concept share 1.000000     -> PASSED
```

## The falsifier fired, and this time nothing is confounded

**H1 passed for the first time in the whole family of row-level experiments**: the correction's energy share fell from **0.9999 to 0.2087**, the
structural term became the dominant carrier of the row (norm 4.49 against a capped 2.70), and the number of rows where the correction carries
more than half the energy fell from **53 of 53 to 10 of 53**.

**And meaning still did not appear: own-word mass 0.0016**, three orders of magnitude below the 0.10 bar and below even the arm's own 0.05
prediction. The cost was **10 points of accuracy** (24.33 -> 14.48).

**So the previous conclusion needed refining, and this measurement is what refines it.** I had been saying "the share is a ratio and cannot be
won by acting on the correction". That was too weak. The share CAN be won - this arm won it, decisively - and **winning it does not produce
meaning.** Structural dominance of a row is not the same thing as the row meaning its name, and this is the cleanest possible demonstration
because the intervention succeeded on its own terms.

## The family, complete

```
arm                                   corr share   own-word mass   VAL
free answer matrix (shipped 4M)          --           0.0008       24.58%
structural + correction (unpriced)      0.9999        0.0008       24.33%
structural + priced correction          0.9968        0.0013       23.66%
structural + structural scale 10        0.9877        0.0012       23.58%
structural + HARD CAP (share flipped)   0.2087        0.0016       14.48%
pure structural, no correction          0.0000        0.0019       13.74%
row-content supervision at 13% of CE      --           0.0008       23.73%
next-token concept supervision           (running)     0.0008 at ep8 25.7%
```

Six row-level mechanisms, spanning the entire range of correction share from **0.0000 to 0.9999**, and own-word mass stays inside
**0.0008-0.0019** in every one of them. The mass does not track the share, the magnitude, the scale, or the support of the row.

**What this establishes, stated as narrowly as the evidence allows**: within this architecture at 16M parameters, a concept's answer row can be
made structurally dominant without becoming meaningful, and every knob that changes what the row *does* leaves what the row *contains*
untouched. Meaning is not a control on this quantity.
