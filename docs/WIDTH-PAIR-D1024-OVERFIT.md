# At double width the direct-path model overfits and the readable model catches it

Date: 2026-11-14 19:33. Matched d1024 pair: `nodirect1024_d512_s0` (readable, no direct pathway) and `direct1024_d512_s0`
(direct ON), both 9 epochs, d512->d1024 fixed width for both, seed 0, 600k windows of the full 294,225,090-token corpus, trained
fresh in one run. Predictions `lens/runs/NODIRECT-D1024-PREDICTIONS.txt` written before launch. Logs
`logs/exposure/{nodirect1024,direct1024}_d512_s0.log`. This completes the width question begun in
`WIDTH-DOES-NOT-HELP-READABLE.md`.

## All three endpoints, because this project's own rule forbids quoting one

```
endpoint                     readable    control    difference
final epoch VAL accuracy       19.95%     18.57%     +1.38  readable
last-3-epoch mean VAL acc      19.59%     19.51%     +0.08  readable
held-out best (by position)    21.57%     23.46%     -1.89  control
final VAL perplexity           291.04     987.76     readable 3.4x better
concept fidelity               34 of 36   36 of 36      -2
```

## The trajectories explain the gap, and the gap flips with the endpoint

```
readable  VAL acc   10.02  13.84  15.82  17.38  18.12  18.64  19.21  19.60  19.95   <- monotonic, still improving
readable  VAL ppl     663    488    412    369    342    322    309    299    291   <- monotonic, still improving
control   VAL acc   17.12  19.41  20.07  20.32  20.54  21.05  20.19  19.76  18.57   <- peaks at epoch 6, then falls
control   VAL ppl     318    263    262    276    318    363    512    679    988   <- a 3.8x explosion after epoch 3
```

**The direct-path model overfits catastrophically at this width**: its training accuracy climbs to 55.46% while validation
accuracy falls from 21.05% (epoch 6) to 18.57% (epoch 9) and validation perplexity goes from 263 to 988. The readable model, forced
through its 36-concept bottleneck, improves monotonically on both accuracy and perplexity for all nine epochs and is still improving
when it stops.

So the readable model is **ahead on the final epoch (+1.38) and on the last-3 mean (+0.08), and behind on best-of-N (-1.89)**,
because best-of-N rewards the control's single lucky epoch 6 before it collapsed.

## Predictions, scored

- **P1 FAILED**: I predicted the readable arm would fall **more than 2.0 points** behind its control at d1024. It does not - it is
  **ahead** on the final epoch and on the last-3 mean, and behind only on best-of-N (-1.89). My reasoning was that a wider model
  routes around the bottleneck more; on a readable model it cannot route around anything, and on a direct-path model the extra width
  became overfitting instead of accuracy.
- **P2 HOLDS** (readable diversity improved in absolute terms: 44 -> 49 distinct tokens).
- **P3 HOLDS** (explain exact at d1024: mean residual 6.217e-07, `concept_share` 99.999996%, 100 of 100 windows).

## What this does and does not establish

- The readable model at d1024 is **not** worse than its direct-path control on the endpoint that matters most for a shipped model
  (final epoch and last-3 mean), and is far better calibrated in perplexity terms.
- It is however **behind on best-of-N (1.89 points)**, so a claim like "readable is better at width" is not supported either. The
  honest statement is: **at double width the direct-path model overfits and the readable model catches it, and which one wins depends
  on the endpoint.**
- **The readable arm had not converged** - accuracy and perplexity were still improving at epoch 9. A longer run is needed before any
  claim about its ceiling. That is the next experiment, not a conclusion.
- One seed per arm. The d512 pair agreed within 0.17 points on the final epoch; this pair reverses sign on best-of-N, so the ordering
  is endpoint-dependent, not stable.

## Where this leaves the concept-capacity question

Unchanged and still the top item: 15 of 36 concepts fire on a median window at both widths, diversity is 49 against the control's
191, and hidden width bought 0.55 points. **The concept layer is still the binding constraint**; width is now measured out twice over.
