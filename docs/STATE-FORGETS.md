
## The state forgets exponentially: effective memory is single-digit tokens (2026-11-17)

Three facts needed reconciling: the state is load-bearing (**-16.68 points** if zeroed), order-sensitive (**-11.04 points** if shuffled), yet accuracy is **flat from 16 to 128 tokens**
(15.62% at both). `tools/state_memory.py` measures the boundary directly rather than inferring it: perturb the token at position `t-k`, run the recurrence to `t`, and measure how far
the state moves.

```
   k    state shift        (||h_t^orig - h_t^perturbed|| / ||h_t||, 12 windows)
   1       47.97%
   2       29.98%
   4       12.05%
   8        3.485%
  12        2.36%
  16        0.885%
  24        0.309%
  32        0.081%
  48        0.010%

state norm by position: t=1 1.680 | t=8 2.959 | t=32 3.060 | t=64 2.946 | t=128 2.962   (stable, no collapse)
mean cosine(h_t, h_t+1): 0.5711
```

## The number that explains the thread

**A token eight steps back moves the state by 3.5%. Sixteen steps back: 0.9%. Thirty-two: 0.08%.** The influence decays roughly geometrically with a half-life of about **1.5 tokens**,
so the model's **effective memory is single-digit tokens** despite a 256-dimensional state and 128-token training windows.

The state does not collapse - its norm settles at ~2.96 and consecutive states turn by a cosine of 0.57 - it simply **stops integrating anything older than a handful of tokens.**

## What this settles, item by item

- **Why accuracy is flat past 16 tokens**: information older than ~8-16 tokens has essentially zero influence on the state. More context cannot help if the state discards it.
- **Why training on its own rollouts could not help**: with single-digit memory there is nothing for the model to recover from its own history; feeding it its own tokens gives it the same
  short window of information as feeding it corpus tokens. That is exactly the "both metrics fell, repetition did not move" result the rollout arm produced.
- **Why generation loops**: a predictor whose effective context is a handful of tokens behaves close to an n-gram model, and n-gram-ish models repeat. Repetition 0.89-0.93 is that.
- **Why the readout and the concept work measured fine**: they measure *instantaneous* prediction quality from a short window, which this model does reasonably. Nothing in those
  instruments could see a memory-length defect.

## The lever this names

The decay is the **vanishing-influence** problem along the time axis: each step's perturbation is attenuated by the state Jacobian, and after ~8 steps only 3.5% survives. The standard
families that address it are all now targetable with the instrument that found it:

1. **State transitions with larger gain / orthogonal initialisation for `Wh`** - so the Jacobian does not shrink the perturbation.
2. **Gating** - a learned per-dimension write/erase, which is precisely a mechanism for *keeping* information against that decay.
3. **Longer-range training signal** - but note the rollout arm already showed that training alone, without changing the transition, moved nothing.

**The gate for whichever is tried, written now:** the perturbation shift at **k=16 must rise above 10%** (today **0.885%**, an order of magnitude short), while held-out accuracy holds at or
above 16%. **Falsifier:** if the k=16 shift rises above 10% and free-running accuracy is still under 6%, then memory length was not the binding constraint on generation either, and the
loop must be sought elsewhere.

## The honest caveat

This is measured on one checkpoint (`scan9_ep1500`, 4.7M parameters, 1,500 steps). Whether the decay is what training *produced* or what the initialisation *imposed* is not yet separated -
`Wh` was initialised small, and a small `Wh` attenuates each step by construction. That separation is the next measurement, and it decides whether the fix is an initialisation change or a
training change.
