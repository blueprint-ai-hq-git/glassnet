# Breadth follows concept supervision, not activation spread - and my second explanation is dead too (2026-11-15)

Two explanations for the readable model's narrow vocabulary have now been measured and both are wrong. This document records the
second one, the result that killed it, and what the data actually shows.

## What ran

One change from the 53-concept readable arm: the concept supervision weight, `--lam-concept`, from 0.3 to 0.6. 18 epochs, d1024,
seed 0, everything else identical. Predictions were written before the run in `lens/runs/ACTIVATION-COLLAPSE-PREDICTIONS.txt`.

## Result: breadth up, accuracy slightly down, activations MORE concentrated

All four checkpoints measured with the same instrument (`lens/scripts/measure_activation_spread.py`, 4,000 held-out windows):

```
checkpoint                     concept dims   effective dims        top-5 var   distinct tokens
direct-path control (36c)           36        2.97 of 36 (8%)          0.80           539
readable 53c, lam 0.3 (baseline)    53       10.37 of 53 (20%)         0.54           135
readable 53c, lam 0.6               53        9.69 of 53 (18%)         0.53           227
```

```
lam 0.3 baseline: final VAL 22.04%, ppl 240.55, held-out concept honesty 0.5884, trained-label 0.8058
lam 0.6        : final VAL 21.68%, ppl 252.31, held-out concept honesty 0.6298, trained-label 0.9257
```

- **Breadth rose 68%: 135 -> 227 distinct tokens.**
- **Concept names got more honest: 0.5884 -> 0.6298**, the first time the held-out median has crossed the 0.60 target.
- **Concepts became far more faithful on their own labels: 0.8058 -> 0.9257** trained-label AUROC.
- **Accuracy fell 0.36 points: 22.04% -> 21.68%.**
- **The activations did NOT spread: effective dimensionality fell 10.37 -> 9.69.**

## The explanation that died

`docs/history/ACTIVATION-COLLAPSE.md` argued that breadth was capped because the 53 concept numbers moved through only ~10 effective
directions, and that spreading them would widen vocabulary. **This run is that theory's own test, and it fails it: vocabulary nearly
doubled while the effective dimensionality went DOWN.** Whatever makes the readable model talk narrowly, it is not how concentrated the
activation cloud is. That is the second hypothesis to die today - the first was the answer layer's rank, refuted in the same document
by sampling the same 53-dimension family at random and reaching 341 tokens where the model used 135.

## What the data actually says

The one quantity that moved WITH breadth is **label fidelity**: trained-label AUROC 0.8058 -> 0.9257 while held-out honesty rose to
0.6298 and vocabulary rose to 227. Stronger supervision makes each concept respond more selectively to its own trigger, and a more
selective set of 53 numbers takes more distinct combinations across inputs - which is more distinct argmax tokens - even though the
cloud as a whole becomes slightly more concentrated. **Breadth tracks how decisively concepts fire on their names, not how spread out
their joint distribution is.**

## Prediction scorecard (honest)

- **L1 FAILED**: predicted final VAL at or above 21.80%; got **21.68%**.
- **L2 PASSED**: predicted held-out honesty above 0.5884; got **0.6298**.
- **L3 FAILED**: predicted effective dimensionality above 10.37; got **9.69** - the prediction was in the wrong direction, and it is
  the prediction that carried the theory.
- **L4 PASSED**: predicted more than 86 distinct tokens; got **227**.
- **FALSIFIER NOT TRIGGERED**: it required effective dimensionality at or below 11.0 **and** diversity at or below 86 in both arms;
  diversity came in at 227, so the arm does not license the "architectural, needs decorrelation" conclusion. The falsifier is what
  stopped me from writing that conclusion anyway.

## A measurement caveat worth stating plainly

The same control checkpoint reports **226** distinct tokens under `tools/diversity_diagnosis.py` (2,000 positions) and **539** under
`lens/scripts/measure_activation_spread.py` (4,000 windows, different sampling). Both are real measurements of a proxy for breadth; they
are not interchangeable. **Every breadth claim in this project must name its instrument**, or two documents will quietly disagree.

## What is next

`--lam-concept 1.0` is running as the second point on this curve, and the useful question it answers is the **trade-off rate**: how
much accuracy per unit of breadth. If 1.0 buys more breadth at a similar cost, the next arm should find the knee between 0.3 and 0.6
rather than push further right.


## Second point: lam 1.0, and the trade-off is now a curve

```
arm                                   distinct tokens   effective dims   final VAL   final ppl   concepts
direct-path control (36c)                   539           2.97 of 36        17.73%     54,751       --
readable 53c, lam 0.3 (baseline)            135          10.37 of 53        22.04%     240.55     52/53
readable 53c, lam 0.6                       227           9.69 of 53        21.68%     252.31     53/53
readable 53c, lam 1.0                       364          10.09 of 53        21.08%     269.30     52/53
```

**Breadth is monotonically controlled by the supervision weight, and it is now a dial rather than a mystery**: 135 -> 227 -> 364 as
`--lam-concept` goes 0.3 -> 0.6 -> 1.0, at a cost of roughly half an accuracy point per step (22.04 -> 21.68 -> 21.08).

**The readable model is still ahead of its matched control on accuracy at every point on the curve** - including at lam 1.0, where it
has 68% of the control's breadth (364 of 539) and is **+3.35 points more accurate** (21.08% vs 17.73%), with 52 of 53 concepts still
answering to their names.

That reframes the remaining work: it is no longer "can the readable model talk widely" but "**where on the curve is the best**
**operating point, and what happens at equal breadth**". At roughly lam 1.5-2.0 the readable curve should cross the control's 539, and
that is the fair fight: match breadth, then compare accuracy.


## Three paired seeds: breadth is solid, the honesty claim is NOT (correction against my own single-seed claim)

Predictions in `lens/runs/LAM06-3SEED-PREDICTIONS.txt`. lam 0.6 at seeds 1 and 2, against the lam 0.3 arms already trained at every
seed.

```
seed   lam0.3 VAL   lam0.6 VAL   delta     lam0.3 tok   lam0.6 tok   delta
  0      22.04%       21.68%     -0.36        135          227        +92
  1      21.98%       21.67%     -0.31        131          238       +107
  2      21.99%       21.61%     -0.38        131          207        +76

accuracy delta : mean -0.350, sd 0.036   (loses on 3 of 3 seeds)
breadth delta  : mean +91.7, sd 15.5     (gains on 3 of 3 seeds)
```

**Both of those are solid.** Breadth nearly doubles on every seed and the accuracy cost is nearly constant at 0.35 points.

Now the part that has to be corrected. I published "held-out concept honesty 0.5884 -> 0.6298, first time above the 0.60 target" from
one seed. Paired across three seeds:

```
seed   lam0.3 held-out   lam0.6 held-out   delta     lam0.3 trained   lam0.6 trained
  0        0.5884            0.6298       +0.0414        0.8058           0.9257
  1        0.6021            0.5977       -0.0044        0.7895           0.9290
  2        0.5962            0.6148       +0.0186        0.7669           0.9192

held-out  : mean +0.0185, sd 0.0229, positive on 2 of 3 seeds   <- WEAK, not established
trained   : mean +0.137,  positive on 3 of 3 seeds              <- SOLID
```

**The held-out gain is not established**: +0.0185 with a standard deviation of 0.0229, and seed 1 is negative. The "first time above
0.60" was my own single-seed reading and is withdrawn as a finding. What IS established is that stronger supervision makes concepts
much better detectors of **their own trained keyword lists** (+0.137, all three seeds) while their transfer to **unseen synonyms**
barely moves.

That sharpens the project's central worry rather than resolving it: supervision buys fidelity to the words a concept was trained on,
not fidelity to the concept's meaning. It also means the bulk of the breadth gain is being delivered by a mechanism that makes the
neurons better keyword matchers, which is exactly the kind of thing a reader of `explain()` would be misled by if it were reported as
"the concepts got more honest".

## Scorecard

- **P1 PASSED**: breadth above 180 on every seed (227, 238, 207).
- **P2 PASSED**: the accuracy cost appears on 3 of 3 seeds, not 2.
- **P3 PASSED only against the fixed threshold I wrote, and FAILS as a paired claim**: above 0.5884 on 3 of 3 seeds, but paired
  deltas are +0.0414 / -0.0044 / +0.0186. I wrote the prediction against a single baseline value instead of against paired deltas,
  which was a weaker test than the data allowed - noted so the next prediction is written as a paired delta.
- **FALSIFIER NOT TRIGGERED**: breadth exceeded 180 on all three seeds.
