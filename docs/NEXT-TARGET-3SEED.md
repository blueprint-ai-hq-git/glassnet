
## Next-token concept supervision lifts validation accuracy by +1.54 points on three seeds, and leaves concept-row content at baseline (2026-11-16)

Decision rule `lens/runs/NEXT-TARGET-DECISION-RULE.txt` was written BEFORE any seed 1 or 2 endpoint existed. Predictions
`lens/runs/NEXT-TARGET-PREDICTIONS.txt` were written before the code changed. Numbers from `lens/runs/NEXT-TARGET-3SEED.txt`,
`lens/runs/NEXT-TARGET-3SEED-MASS.txt` and `logs/exposure/nodirect1024_nexttarget_data4m_e12_s*.log`.

## The accuracy half - R1 was satisfied on every clause

```
seed   next-target VAL   shipped VAL   paired delta
 0        25.97%            24.57%        +1.40
 1        26.24%            24.53%        +1.71
 2        26.16%            24.64%        +1.52
mean     26.12% (sd 0.139)  24.58%       +1.54 (sd 0.156)
```

**All three seeds land above the shipped best seed of 24.64%**, the mean is 26.12%, and the comparison is **paired on seed** (the rule this
project set after unpaired arm means misled it once). Three of three paired deltas are positive. Perplexity improves on every seed as well
(169.67 / 169.13 / 168.10 against the shipped 181.04).

## The meaning half - unchanged, and it is published with the same weight

```
seed 0 own-word mass 0.0008
seed 1 own-word mass 0.0008
seed 2 own-word mass 0.0010
            baseline 0.0008      concepts at or above 0.05: 0 of 53 on every seed
```

## What this means, both ways

**A real gain, on a properly seeded comparison.** Labelling a concept 1 only when one of its own words is the *token being predicted* is a better
training signal than labelling it when its words merely appear in the window: **+1.54 points of validation accuracy, 3 of 3 seeds, paired, with
better perplexity on all three.**

**And it buys nothing for concept-row content.** Median own-word mass sits at 0.0008, 0.0008 and 0.0010 - the free-matrix baseline, with zero
concepts above 0.05 on any seed. So the supervision change improves prediction while the answer rows remain prediction devices that do not favour
their own words. This is the tenth mechanism measured on that question, and it is the one that improves accuracy while confirming the wall.

The two facts belong in one sentence, which is why the timing of the mass measurement was fixed before the third seed landed: a publication that
quoted +1.54 without the mass numbers next to it would have been a half-truth, and the half it dropped is the one this project exists to answer.

## What remains open

`explain()` stays exact (residual 2.384e-06 at the seed-0 endpoint, concept share 1.000000), so the system still truthfully reports which
concepts drove each decision. What is not true is the stronger claim that a concept's NAME predicts the words it promotes. Ten mechanisms,
correction shares from 0.0000 to 0.9999, two supervision targets, and own-word mass never leaves 0.0008-0.0019.
