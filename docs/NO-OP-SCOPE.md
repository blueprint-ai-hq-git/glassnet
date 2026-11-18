# A zero-initialised component is a no-op in the FORWARD pass, not in the TRAINING TRAJECTORY (2026-11-18)

The memory-channel arm was launched with one safety claim: because `mem_Wo` is initialised to zero, step 0 "reproduces the resumed model exactly". The live run let that claim be tested
properly for the first time, and it is **true of the forward pass and false of the training run**. The refinement matters, because the difference is a mechanism, not a rounding error.

## What is exactly true

```
step 0, lr = 0 (tools/scan_identity_check.py, real checkpoint):
  loss with the channel    9.396832162837e+00
  loss without the channel 9.396832162837e+00      <- identical to twelve significant digits
  pre-existing gradient sha256              identical
```

The forward pass and every pre-existing gradient are unchanged. That claim stands.

## What is NOT true, and why

The optimiser clips by the **global** gradient norm, and the new parameters are inside that norm:

```
step 0 gradient norm, same batch:
  channel off   6.042480944661e-01
  channel on    6.289183837069e-01     (+4.08%)

clip scale s = min(1, clip/norm), clip = 1.0:
  off  1.654949      on  1.590031       -> the first update is 3.92% SHORTER for every shared parameter
```

So the first update already differs, and after that the trajectories separate. This is visible in the live runs at matched schedule (`STEPS=1500` in both, so the learning rate is
identical at 0.0296):

```
                       loss      acc      gnorm
channel off  (step 3025)  6.1269   17.43%   2.25
channel on   (step 3025)  6.1375   17.27%   3.44
```

## The second, larger mechanism

`mem_Wo` starts at zero but it **receives a gradient of 4.870e-03 on the very first step**, which is the whole point of the component - it has to be able to grow. So after one update the
channel is no longer zero, and from step 2 onward the forward pass genuinely differs. **A component designed to start as a no-op cannot stay one for more than a single step, by
construction.**

## The refined rule

"Verified no-op" needs a scope. The accurate statement is:

- **forward pass**: exact no-op at initialisation (loss bit-identical, pre-existing gradients bit-identical).
- **first update**: differs whenever the optimiser normalises by a global norm, because the new parameters join it (here: +4.08% norm, 3.92% shorter step).
- **second step onward**: genuinely different, because the output projection has left zero.

If a future component must be a no-op for the *whole run* until switched on, the gradient into its output projection has to be masked to zero, not merely its weights initialised to zero.

## Status of the arm

`memchan` is training normally: `step 3125/4500`, `lr 0.0278`, params 4,932,586 (channel present), resuming correctly from step 3000 - the resume fix is working in a live long run.
Its endpoint is k=16 memory against the resumed model's own 1.322%, with P2 asking whether accuracy survives, because every previous memory mechanism has paid for memory in accuracy.
