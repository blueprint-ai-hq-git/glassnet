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
