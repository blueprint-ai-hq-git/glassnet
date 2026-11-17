
## Training destroyed the memory the initialisation already had (2026-11-17)

The decay curve on `scan9_ep1500` (token 16 steps back moves the state by 0.885%) had one obvious alternative explanation: `Wh` is initialised small, so a small `Wh` attenuates every step by
construction and the defect might be built in before training starts. `tools/memory_init_vs_trained.py` runs the identical measurement against both.

```
   k        init      trained
   1     48.662%      48.530%
   2     46.023%      31.116%
   4     40.622%      13.093%
   8     28.917%       3.231%
  16     17.812%       0.858%      <- the gate I set was 10%
  32      5.711%       0.091%

Wh largest singular value : init 1.9753 -> trained 1.3188
Wh Frobenius norm         : init  15.996 -> trained  10.497
|Wx| init 15.986 trained 15.981   |emb| unchanged at 0.0798
```

## The initialisation was adequate; training removed it

**At initialisation the k=16 shift is 17.81% - comfortably above the 10% gate I wrote before this measurement.** After 1,500 training steps it is **0.858%**, a **20.8x loss of
memory**.

The spectral picture says the same thing: `Wh`'s largest singular value falls from **1.98 to 1.32**. A perturbation multiplied by a matrix with spectral norm near 2 grows; one multiplied by 1.32
shrinks steadily. **Training walked the state transition from expansive toward contractive.**

`Wx` and the embeddings are essentially untouched (norms 15.986 -> 15.981, |emb| 0.0798 either way), so this is specifically the **recurrent transition** that changed.

## What this means for the fix - and it changes the target

The fix is **not** a better initialisation. The initialisation already had the property the gate asks for. The lever is that **gradient descent had no reason to keep it**, and a strong reason to
spend it: next-token cross-entropy on this corpus is largely satisfied by short-horizon statistics, so a transition that attenuates old information is a *cheaper* solution than one that carries
it. Memory is a capacity the objective charges for and does not reward.

That reframes the three candidate levers from the previous entry:

1. **Initialisation / orthogonal `Wh`** - now known to be insufficient on its own: the initialisation already had 17.8% at k=16 and training took it away. An init change only helps if something
   stops training from undoing it.
2. **Gating** - still the most promising, because a gate gives the model a way to *hold* information that a plain tanh recurrence cannot express, and the gradient can learn to use it when
   the loss rewards it.
3. **A loss term that pays for memory** - the direct counter to "the objective charges for capacity and does not reward it". Without this, both of the above are fighting the gradient.

## The honest limits of this result

- One checkpoint, one seed, one arm of 1,500 steps. The direction (memory falls during training) is measured; the *rate* is not characterised across runs.
- The measurement is on the state's *sensitivity*, which is a necessary condition for memory, not sufficient. A state could be sensitive to old tokens and still not use them usefully.
- It does not show that 1,500 steps is enough for the model to have explored a memory-using solution and rejected it. A longer run might recover memory later; that is a separate,
  pre-registerable question.

**The gate for any fix stands as written:** k=16 shift above **10%** while held-out accuracy holds at or above **16%**. The new information is that this target is **already achievable with these
weights** - it was achieved before training started.
