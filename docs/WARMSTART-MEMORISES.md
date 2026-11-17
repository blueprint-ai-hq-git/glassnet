
## The warm-start arm memorises: train down 2.13 nats, held-out up 0.008 (2026-11-17)

120 steps, batch 8, learning rate 1e-4, per-group gradient clipped. Baseline re-measured in the same run, on the same windows and the same positions.

```
step   0   train 6.2029
step  70   train 5.1754
step  80   train 4.7526   VAL 6.5686
step 100   train 4.2849   VAL 6.5770
step 119   train 4.0702   VAL 6.5775

RESULT train 6.2029 -> 4.0702   (-2.13 nats)
RESULT val   6.5694 -> 6.5775   (+0.0081 nats, WORSE)
SHIPPED MODEL on the identical 40 windows: 6.5694 nats
speed 3.22 s/step at batch 8 = 149 windows/minute on four cores
```

**The training loss falls by 2.13 nats while the held-out loss rises slightly.** That is not "training is slow" - the model is fitting the windows it sees and
getting very slightly worse on the ones it does not. The gap widens with every checkpoint, which is the signature of memorisation rather than learning.

## What this settles and what it does not

**Settled:** at these settings the warm-start arm does **not** learn anything that generalises. Adding one certified block to the shipped model, then training the whole
stack with plain clipped SGD at 1e-4 on 4,000 available windows, produces a fit to the training windows and no held-out gain.

**Not settled:** whether depth helps. The falsifier for this arm - breadth under 0.20 with depth in place means the concept bottleneck K is binding, not depth - remains
**untested**, because it is a statement about a model that has learned something, and this model has not.

## Why this is the right result to have

The project's own rule is that a model must learn rules, not memorise. Here the memorisation is visible in a single pair of numbers, in 120 steps, on identical windows:
train **-2.13**, held-out **+0.008**. That is a cheap, unambiguous detector, and every future arm gets it for free because the baseline is now re-measured in the
same run rather than quoted from a different protocol.

## The next question, stated before it is run

Train loss falling while held-out rises, with only 4,000 windows and no epoch structure, is a **data and batching** signature, not an architecture one. The next arm
must therefore fix the amount of distinct data per step before touching anything else: shuffle over all 4,000 windows per pass instead of sampling with
replacement, evaluate once per pass on held-out, and report the gap at the end of every pass. If the gap closes and held-out improves, the block is doing work and the
depth question can be asked. If the gap still widens while held-out falls, the block is adding capacity that only memorises, which is itself an answer.

## Cost, so the decision is arithmetic

149 windows/minute on four CPU cores. A 4M-window pass would be **448 hours**. Any real arm needs the GPU; nothing about this result changes that.
