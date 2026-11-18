# Re-warming the learning rate buys 1.31 points - and the schedule was part of the plateau (2026-11-18)

The plateau report left one explanation untested: by step 6000 the learning rate had decayed **0.03 -> 0.0030**, so the flat band was measured while the model was taking
increasingly tiny steps. `lens/scripts/run_lr_rewarm.sh` resumed `scan12_r090_6k_ep6000` and trained **1500 more steps with the rate re-warmed to 0.03**, one variable changed.

```
step 6000 (plateau)   25.89%   loss 6.0640     <- the flat band 4500-6000 was 25.24-26.01
step 6250             26.53%   loss 5.9681
step 6500             27.27%   loss 5.9166
step 6750             27.18%   loss 5.8918
step 7000             27.32%   loss 5.8783
step 7250             27.69%   loss 5.8653
step 7500             27.20%   loss 5.8903     <- ENDPOINT
```

## Predictions, scored

- **P1** "held-out rises above 26.01%, the top of the measured band" - **MET**: 27.20%.
- **P2** "loss falls below 6.0640" - **MET**: 5.8903, a fall of **0.174 nats**.
- **P3** "if P1 fails, the plateau is capacity" - not reached; P1 passed.
- **FALSIFIER** "held-out stays inside the plateau band 25.24-26.01%" - **did not fire**. **Six of six** curve points after the re-warm sit above the old band's top.

## What it means, with its size attached

At the endpoint the effect is **+1.31 points = 1.85 standard errors** (SE of a difference ~0.71 at this protocol). That is **just under the 2-SE bar on a single comparison**, and the honest statement is: the schedule explains part of the
plateau, worth roughly **1.3-1.8 points**, not all of it.

The shape matters more than the size. Accuracy climbed for 500 steps and then **flattened at ~27.3% while the loss kept falling** - the same confidence-without-accuracy pattern the original plateau showed, now at a higher level. So a
re-warm is a **one-time lift, not a new regime**, and further re-warms should not be expected to keep paying.

## The n=1 problem this exposed, and the fix

Every arm in this line has been **one draw**, and it was structural: the scan trainer hardcoded **both** the weight initialisation (`SD.init(..., seed=0)`) and the batch shuffle
(`np.random.default_rng(0)`). A second independent draw was impossible to run.

Both now follow `SEED` (default `0`, so every existing arm is bit-identical and reproducible, and `SEED` is recorded in every checkpoint's metadata). `run_rewarm_seed1.sh` re-runs the
identical recipe at `SEED=1`:

- **P1** seed 1 also lands above the old band top (>26.01%)
- **P2** the two draws agree on the size of the effect within their error: |seed1 - seed0| <= 2 points, since the gap sd measured at this scale was 2.02
- **P3** loss falls below 6.0640 again - the objective improves for the same reason, not by chance
- **FALSIFIER** seed 1 stays inside 25.24-26.01%: then +1.31 points was a shuffle artifact, **the claim on this page is withdrawn**, and the plateau stands as capacity-limited.

That falsifier is the reason this entry is written as "1.31 points, one draw" rather than as a result.
