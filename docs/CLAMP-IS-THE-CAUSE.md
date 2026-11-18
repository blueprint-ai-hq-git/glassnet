# The clamp was the cause of the forgetting: unclamping gives 5.4x the memory and costs 3 points (2026-11-18)

`renorm_recurrent(p, TARGET_RADIUS)` runs every step and holds the recurrent matrix's spectral radius at a fixed value. Two arms, identical recipe, same seed, same data,
1500 steps each, **one number changed**:

```
                       radius 1.0        radius 0.9
Wh spectral radius      1.0000            0.9000        <- the clamp provably does what it claims
held-out (teacher-forced)  13.23%          16.27%
free-running             5.81%            5.00%         paired CI [-1.38, +3.00] NOT RESOLVABLE
repetition               0.908            0.915

memory: state movement from a token k steps back
   k     radius 1.0    radius 0.9    ratio
   1       39.144%       50.383%      0.8x
   2       34.776%       32.044%      1.1x
   4       25.860%       12.735%      2.0x
   8       13.968%        3.060%      4.6x
  16        4.730%        0.878%      5.4x
  32        0.587%        0.079%      7.4x
```

## What this settles

**The clamp is causally responsible for the long-range forgetting.** This is no longer a correlation between "trained models forget" and "the matrix is clamped": changing only
that one number, from 0.9 to 1.0, multiplies the k=16 memory by **5.4x** and the k=32 memory by **7.4x**. A contraction by construction does exactly what it says - it
contracts perturbations away.

Note also the shape of the curve: at k=1 the unclamped model retains *less* (39.1% against 50.4%). The clamped model is very reactive to the immediate past and forgets fast; the
unclamped one carries further but is less sensitive to single recent tokens. That is what a longer memory constant looks like.

## What it costs, and what it does not fix

- **Teacher-forced accuracy: 13.23% against 16.27%, a 3.04-point loss.** Real and not small.
- **Free-running accuracy: 5.81% against 5.00%** - the paired difference is +0.325 tokens per 40, bootstrapped CI [-1.38, +3.00], so **not resolvable at this sample**. The model
  still does not generate, unclamped or not.
- **The memory gate still fails**: k=16 is 4.730% against a 10% bar. Five times better is not enough.

## The next question, pre-registered

Is the 3-point accuracy cost a property of the radius, or an artefact of training length? The clamped model has been trained for 1500 steps at a radius where its recurrent weights
are comfortable; the unclamped one may simply need longer to find equally good weights while staying expansive.

Two matched arms at **3000 steps**, radius 1.0 against radius 0.9 (`lens/scripts/run_scan_r100_3k.sh`, `run_scan_r090_3k.sh`):

- **P1** radius 1.0 keeps at least **3x** the k=16 memory of radius 0.9
- **P2** the teacher-forced gap narrows **below 1.5 points** - most of the 3.04-point cost was under-training, not the radius
- **P3** neither arm reaches the 10% memory gate
- **FALSIFIER** the accuracy gap stays at or above **2.5 points** at 3000 steps: then radius 1.0 costs accuracy permanently, and the honest conclusion is that the
  memory/accuracy trade is real at this scale rather than a training-length artefact.
