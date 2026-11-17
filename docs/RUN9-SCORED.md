
## Run 9 scored against its own pre-registration: F4 fails, and the readout is provably not what separates inputs (2026-11-17)

`tools/eval_run9.py lens/runs/chat/scan9_ep1500.npz` evaluates the endpoint against the criteria written before the run.

```
F1  KL across six prefix-free words       1.1702    bar >0.05    PASS
F2  state cosine across those inputs      0.1736    bar <0.50    PASS
F3  held-out accuracy                    16.93%     bar 5.34%    PASS
F4  trained readout vs noise readout      trained 1.1702  noise 1.1660  ratio 1.0036x   needs >2x and >0.030   FAIL
G1  gain_ro bounds                       0.978 .. 1.119                    bounded, not saturating
G2  mean predictive entropy              8.835 = 94.1% of uniform          no collapse
```

## F4 is the gate that matters, and it did its job

The whole point of F4 - added as "the sharpest" criterion before this run - was to catch the case where the readout separates inputs for a reason that has nothing to do with training.
It caught exactly that.

**The trained readout scores 1.1702 KL. A random-noise readout at the same scale scores 1.1660.** The ratio is **1.0036x**. In other words the trained answer layer is
*indistinguishable from noise* at separating these six inputs.

**So F1's headline pass is not evidence of learning.** F1 was written as "KL above 0.05", and the number cleared it by 23x - but the noise control clears it too, which means the 1.17
is produced by the *random projection downstream of the state*, not by anything training achieved. Without F4 the project would have recorded a 23x-over-bar win today. That is the
single most useful thing this run produced.

## The guards rule out the two ways this metric can lie

G1: the readout gain stayed within **0.978-1.119** - it did not run away, so the separation is not a gain explosion.
G2: predictive entropy is **8.835 nats, 94.1% of uniform over 12,002 words** - the distribution has not collapsed toward a point, so the separation is not softmax saturation.

Both guards pass, which makes F4's failure a clean statement rather than an artefact.

## The pre-registered falsifier, read exactly

It said: *"if F4 fails **while F1 fails**, the readout normalisation is not the lever either, and the next suspect is the target itself."*

F1 did **not** fail, so the literal falsifier did **not** fire. But the substance is stronger than the falsifier anticipated: **F1 passed precisely because the noise control
passes too.** The conclusion the falsifier was aiming at - that readout normalisation is not the lever - holds regardless, with better evidence than the falsifier required.

**Recorded honestly as: falsifier not fired in its literal form; its intended conclusion is nonetheless supported, by a measurement the falsifier did not anticipate.**

## What this run settles

1. Our own readout normalisation does **not** make the answer layer input-sensitive. Trained and noise readouts are the same to 0.4%.
2. The states are decorrelated (cosine 0.1736) and held-out accuracy is fine (16.93%), so the earlier stages are working.
3. **The failure is between the state and the answer layer** - and it is not a matter of scale, gain or entropy. It is that the readout's response to different inputs carries
   no more information than a random projection of the same state.

Per the falsifier's own redirect, the next suspect is **the target**: a 12,002-way softmax over a 128-token window may be asking this readout for more separation than the data
supports.
