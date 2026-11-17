
## Falsifier fired: the input-in-readout change did not rescue the cell harness (2026-11-17)

The pre-registered change was made exactly as written: `logits = concat(c_t, x_t) @ Wc + bc`, so the readout sees the current token embedding as well as the state.
Gate fixed beforehand: **above 5% next-token accuracy AND consecutive-state cosine below 0.90.**

```
HELD-OUT at start: loss 9.4722  acc 0.00%  cosine 0.9575
step  50  train 9.3952  acc 0.00% | HELD-OUT 9.3872  acc 0.00%  cosine 0.9632
step 100  train 9.3781  acc 0.00% | HELD-OUT 9.3784  acc 0.00%  cosine 0.9651
step 150  train 9.3725  acc 0.00% | HELD-OUT 9.3676  acc 0.00%  cosine 0.9686
step 200  train 9.3726  acc 0.00% | HELD-OUT 9.3625  acc 0.00%  cosine 0.9701

FINAL: held-out acc 0.00% | cosine 0.9701
GATE (>5% acc AND cosine <0.90): FAIL
```

**The falsifier fired**: accuracy is still zero with the input present. The pre-registration says: abandon this harness rather than re-scale it again. That is what happens here - no further learning rates, no
further readout inits, no third attempt.

## The honest detail inside that verdict

The loss **is** falling monotonically (9.4722 -> 9.3625, and it tracks down on held-out too), so this is not a broken gradient or a dead pipeline - it is learning, at a rate that would need far more than 200
steps to matter. `ln(12002) = 9.3936`: the model has moved from slightly-worse-than-uniform to slightly-better-than-uniform and no further.

So the accurate statement is: **the input-in-readout change did not produce measurable prediction at this scale, and the pre-registered rule ends the attempt here.** It does not establish that a
gated cell cannot learn this corpus, and it does not show the change was harmful - it shows the harness as configured does not get there, and the rule exists so that "just try a bigger learning rate" cannot
become the answer every time.

## Why the rule is being obeyed rather than argued with

Three attempts at this harness have now failed to produce non-zero accuracy: readout 0.02, readout 0.2, and readout-with-input. Each time the honest next step looked like a scale change. **A pre-registered
falsifier that is only obeyed when it agrees with the plan is not a falsifier.** The cell is certified, the composition is certified, and the harness is abandoned.

## What is left standing, and what comes next

Certified and reusable: `lens/cell.py` with gradients checked at T = 1, 2, 3, 8, 16, 32, and the end-to-end composition check (11/11 parameters). If a future arm wants a linear gated state, the module is
ready and its correctness is not in question - what failed is the training setup around it.

**Next work goes back to the thread with a confirmed cause.** The one mechanism this week that was measured rather than inferred is the spectral clamp: `renorm_recurrent(p, TARGET_RADIUS)` runs every step and
holds `Wh` at radius 0.9, which contracts perturbations by construction and took the k=16 memory response from **17.81% at initialisation to 0.885% after training**. Post-hoc rescaling to 1.0 gave a
resolvable but small free-running gain (5.00% -> 6.75%, CI [+0.44, +3.38], accuracy change within noise).

The untested version is the one that matters: **train the real scan model with `TARGET_RADIUS=1.0` from the start**, so the transition is never clamped below 1, and judge it on the same instruments - k=16
memory shift, held-out accuracy, free-running accuracy and repetition. That is a config change to an existing trainer, needs no new module, and tests the confirmed cause rather than a new hypothesis.
