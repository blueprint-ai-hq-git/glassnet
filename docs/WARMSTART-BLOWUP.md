
## The blow-up is not a learning rate, and the cause is a norm I specified but did not build (2026-11-17)

Two pilots with gradient clipping, differing only in learning rate:

```
lr 1e-3, clip 1.0 :  6.5691 -> 1.7836 by step 10 -> 10.1121 by step 13
lr 2e-4, clip 0.5 :  6.5691 -> 1.8424 by step 10 -> 10.0918 by step 13
```

**A tenfold learning-rate change produced nearly the same trajectory.** Same descent, same step-10 minimum, same step-13 blow-up. That rules out the learning rate as the
cause and rules out my previous explanation as sufficient: clipping fixed the *first* explosion (the zero-init projections), but a second, deterministic instability
sits behind it.

Same val loss in both runs: 6.5581 and 6.5673 nats, against **6.06 nats** for the shipped model on the same unmasked held-out text (8.7393 bits/token). So the pilot
ends *worse* than the model it started from, which is what a diverged run looks like.

## The cause, and it is my spec contradicting my artifact

The spec says the path is `... -> RMSNorm -> project to K named concepts -> answer`. **The warm-start artifact has no final RMSNorm** - the block's output goes
straight from a 32768-vector into `concept.W (32768, 53)`.

That is exactly where the overflow appears: the earlier warning was `overflow encountered in exp` on `-(flat @ cW + cb)`. Embeddings have a controlled scale
because they were trained that way; a block's residual output has no such guarantee, and a handful of windows with large activations produce huge concept logits,
saturating the sigmoid and destroying that step's loss.

So the instability is structural: **the concept layer is being asked to consume an unnormalised 32768-vector whose scale nothing controls.**

## The design tension this exposes, which has to be resolved deliberately

Step-0 equivalence requires that the block contribute **exactly zero**, which is why the output projections are zero-initialised. A final RMSNorm **changes the
scale** of the concatenated vector even when the block contributes nothing, so adding it **breaks exact equivalence** - the warm-start would no longer reproduce the
shipped logits.

Three ways out, and they are a real choice rather than a patch:

1. **Keep equivalence, normalise later.** Start from the exact shipped model, and treat the final norm as one of the things training introduces. Accept that the
   first steps are not equivalent to the shipped forward.
2. **Keep the norm, give up exact equivalence.** Initialise the block as a no-op *and* the final norm as identity-by-scale, then measure the equivalence gap and
   report it instead of claiming zero.
3. **Normalise inside the block only.** Add a third RMSNorm at the block output but with its gain initialised to zero, so at step 0 it multiplies the normalised
   value by zero - contribution zero, equivalence preserved, and the norm grows into place during training.

**Option 3 is the one I will build**, because it keeps both properties this project has insisted on: the new path starts as an exact no-op, *and* the quantity feeding
the concept layer is scale-controlled. Its falsifier is the obvious one - if all three gains stay near zero after training, the norm is not being used and the
instability will return, which the loss curve will show.

## What this week keeps repeating

Every step forward has been preceded by an instrument failing: a gradient that was wrong, a gate that could not fail, a metric that failed correct code, a runner that
reported COMPLETE over nothing, and now a spec that specified a component the artifact did not contain. The rule that keeps working is the same one: **when a number
misbehaves, check whether the thing you are looking at is the thing you think you are looking at.**
