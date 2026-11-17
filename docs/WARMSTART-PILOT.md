
## The warm-start loop works, and the divergence was a learning-rate scaling problem in two groups (2026-11-17)

First pilot: loss **6.5691 -> 10.2597** in 12 steps with `exp` overflow. That is a failure, so it was instrumented rather than re-tuned by guesswork.

`tools/dbg_warmstart.py` asked three questions in order:

```
1. TRANSFER CHECK
   loss with the block as saved            10.722558
   loss bypassing the block entirely       10.722558
   difference 4.679e-08                    transfer is exact

2. GRADIENT MAGNITUDES over 8 windows (L2 norm, summed)
   emb 6.97e+01   concept.W 7.15e+01   concept.b 4.48e+00
   answer.W 2.04e+01   answer.b 6.34e+00
   block.n1 0.00e+00  block.Wq 0.00e+00  block.Wk 0.00e+00  block.Wv 0.00e+00
   block.Wo 2.39e+03   block.n2 0.00e+00  block.Wg 0.00e+00  block.Wu 0.00e+00
   block.Wd 2.79e+03

3. DOES A SMALL lr DESCEND?
   lr 1e-4 : 5.950523 -> 5.925244 after 5 steps   DESCENDS
   lr 1e-5 : 5.925244 -> 5.922725 after 5 steps   DESCENDS
```

**The diagnosis is unambiguous.** The two zero-initialised output projections (`Wo`, `Wd`) carry gradients **34-40x larger** than every other parameter group, because
the gradient of a projection that starts at exactly zero is a large outer product of its inputs. The pilot used a single learning rate of 0.01 for every group, so
those two groups were driven far too hard in the first step and the activations ran away.

The loop itself is correct: at 1e-4 and 1e-5 the loss descends immediately, and the transfer from the shipped model is exact to **4.7e-08**.

## What changes

**Per-group learning rates.** The zero-init projections get their own, smaller rate; everything transferred keeps the rate the shipped model trained with. This is
standard practice for adapters bolted onto a trained model, and here it is not a matter of taste - it is a factor of 34 that the gradient norms measured.

## One flaw in my own diagnostic, recorded

The second lr run started from the state the first one left behind (5.95 rather than 5.925 as its "before"), so the two runs are not independent replicates. Both
descend, so the conclusion holds, but the numbers should not be read as a controlled comparison of 1e-4 against 1e-5. A cleaner version resets the weights between
runs, and the arm runner will do that.

## Where this leaves the arm

The warm-start arm is now blocked on one piece of engineering, not one of science: split the learning rate by parameter group, re-run the pilot, and confirm descent
over enough steps to see a trend rather than a wobble. Speed is measured at **1.09 s/step for a batch of 2** on two CPU cores, i.e. about 93 windows per minute,
which means a real arm needs either many cores or CuPy. That decision is now arithmetic instead of instinct: 4M windows at 93/minute is 716 hours on CPU.
