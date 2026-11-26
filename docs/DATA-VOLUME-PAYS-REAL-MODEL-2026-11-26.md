# 2026-11-26 - ten times the distinct data is worth +0.63 nats on the REAL model, at equal epochs

Source: lens/runs/OLD-MODEL-REPLAY_poolA.json and _poolB.json, journal of unit glassnet-datalever3. Model:
direct1024_d512 (d=1024, 36 named concepts, direct route, tied embeddings, 47.5M parameters). Both arms start from identical
weights and are scored on the CANONICAL held-out set (lens/runs/OLD-HELDOUT-CANONICAL.npz, 4,000 windows, seed 20261126) - the
same windows for both arms, which is the fault I had to fix twice before this run was valid.

Equal EPOCHS, not equal steps: the big-pool arm gets steps in proportion to its pool.

    arm                                   after          gain vs before (6.9218 nats / 13.33%)
    A: 200,000 windows, 3,000 steps      6.7375 nats    +0.1843 nats   +0.05 accuracy points
       13.38%  (seeds 13.35, 13.40)
    B: 2,000,000 windows, 30,000 steps   6.1032 nats     +0.8187 nats   +0.29 accuracy points
       13.62%  (seeds 13.60, 13.63)

    B versus A: -0.6343 nats and +0.24 accuracy points for ten times the distinct data and ten times the steps.

## Prediction, on record before the run

"the 2M-window arm beats the 200k-window arm by at least 0.3 nats." It came out at **0.6343 nats**. MET.

## Why this matters more than anything else measured today

Three brain mechanisms failed to move accuracy on this model: concept loss neutral, coverage harmful, replay +0.04 then -0.03.
The data lever moved it **+0.29 accuracy points and 0.82 nats** in one afternoon, and it was the only lever that needed no new
idea - just ten times more distinct windows and the steps to read them.

Note what that also says about the mechanism tests: they were all run at 3,000 steps on 200,000 windows, which is roughly 0.24
epochs. The mechanism questions deserve a second look at a budget where the model is actually learning something, and nothing
measured at 13% and 6.9 nats should be treated as final.

## What is not yet known

Whether it keeps paying. 10x data bought 0.63 nats here; the earlier small-architecture work found the lever had NOT saturated
between a 200k and a 1M pool, but every lever in this project eventually flattened. That is the next measurement, not an
assumption.
