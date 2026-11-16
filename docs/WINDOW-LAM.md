
## Breadth does not survive weaker window supervision: coverage 0.292 -> 0.133 at half strength (2026-11-16)

Predictions written before the run: `lens/runs/WINDOW-LAM-PREDICTIONS.txt`. Numbers from `lens/runs/WINDOW-LAM.txt`.

```
arm                        VAL acc   VAL ppl   coverage T=0.7   repetition   KL
window labels, lam 1.0      24.58%*   181.04       0.292          0.027     17.73   (shipped)
window labels, lam 0.5      25.38%    168.98       0.133          0.127     21.48
                                        * three-seed mean
```

## The falsifier fired, and harder than the bar

I predicted coverage would stay at or above 0.25 and fixed the falsifier at below 0.20. It came in at **0.133** - not merely under the bar, but the **lowest
coverage of any arm measured in this project**, and **halving window supervision more than halved breadth**.

Scored: **V1 FAILED** (0.133 vs 0.25), **V2 PASSED** (25.38 is inside 24.9-25.7), **V3 FAILED** (repetition 0.127 vs a 0.08 limit). The arm is not promotable:
coverage 0.133 is nowhere near the 0.27 bar fixed before the run.

## What this establishes

**Breadth in the window-label family is sharply sensitive to supervision strength**: 0.292 at full strength, 0.133 at half. Compare the next-token family,
where coverage barely moved with strength (0.173 / 0.167 / 0.150). So the two label definitions respond to strength completely differently, and neither
responds the way I predicted at first.

**The shipped configuration is the only arm in the project that reaches 0.292 coverage**, and every weaker setting of either label definition lands between
0.133 and 0.190. Whether 0.292 is a ceiling or a point on a rising curve is now the open question, and the arm running answers it by testing strength **2.0**:
if coverage exceeds 0.292 the shipped setting was never tuned for breadth, and if it does not, the shipped configuration is a local optimum in both
directions - a stronger statement than "we picked a setting".

## The honest summary of the whole day on this axis

Six configurations, one per setting, and **not one of them beats the shipped model on both accuracy and breadth**. Accuracy can be bought with weaker
supervision (up to 27.01% +/- 0.03 with none at all); breadth cannot be bought back at any setting tested. The shipped model remains the shipped model
because it is the only one that keeps the vocabulary, and the concept layer's price is now measured from both ends.
