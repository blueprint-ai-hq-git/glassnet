

---

# DEPTH AT 1200 STEPS: THE FALSIFIER FIRED, BUT THE TRANSPARENCY DELIVERABLE WORKS (2026-11-19 12:49)

The audit said the core was 1.3% of the model, one tanh step cannot compose, and effective context is 2 tokens. `lens/model2.py` adds a second
recurrent layer and gives layer 1 its own readout on the same next-token objective (deep supervision plus transparency). Gate: 4.026e-10.

Predictions: P1 depth2 beats the 6.07% single-layer baseline by >= 0.5; P2 layer 1's own readout scores above zero and within ~2 points of layer 2;
P3 the layers do not make identical predictions. **Falsifier: depth2 at or below 6.07% means depth is not the lever at this budget.**

```
arm                                  HELD-OUT            train ema   held-out loss
single layer (winoff, 1200 steps)     6.07%                7.0492       6.5013
two layers + per-layer readout        5.68% (layer 2)      9.1805       8.4180
                                       4.90% (layer 1)
difference                            -0.39 points
```

- **P1 FAILED**: -0.39 points. **The falsifier fired.**
- **P2 MET**: layer 1's own readout reaches **4.90%**, above zero and 0.78 points from layer 2. **The transparency deliverable works** - each layer
  produces its own next-token prediction, and both are scored every run.
- **P3 MET**: 5.68% vs 4.90% means the layers are not producing identical predictions.

## Why I am not yet accepting "depth does not help"

The training statistics say depth2 was **harder to optimise, not merely worse**: its train EMA is 9.18 against the baseline's 7.05, and its held-out
loss is 8.42 against 6.50. A second layer doubles the depth of the recurrence with the SAME learning rate, the same initialisation scale and the same
gradient clip of 1.0. A model that trains worse is not evidence that the architecture is wrong; it is evidence that the optimiser settings were chosen
for a different model.

So the honest status is: **"depth with a one-layer learning rate does not help"**, which is a weaker claim than "depth does not help".

## The one retry that is fair to the hypothesis

Same two-layer model, ONE change: learning rate 0.03 -> 0.01 (a deeper recurrence needs a smaller step). Predictions written before running:
- if it beats 6.07% by >= 0.5 points, the falsifier result was an optimisation artefact and depth is back in play;
- if it still fails, the claim becomes "depth does not help at 1.8M parameters and 1200 steps", and the next repair is FLAW 4 (`<unk>` is 5.34% of
  the corpus) rather than more architecture.

Tuning the learning rate once, with a pre-registered outcome, is not the same as fishing - but doing it a second time after seeing the result would be,
so this is a single retry and not a sweep.
