
## The definitive calibration numbers, with all four guards in place (2026-11-17)

Every earlier calibration number in this thread failed for a different reason. `tools/calibration_final.py` fixes all four in one place: the entropy expression is verified by
printing **`sum(probs)`**, the **forward path is named and includes the direct `Wd` term**, **`<unk>` is masked** exactly as the sampler masks it, and **accuracy, entropy and top-1
confidence are reported together** so no proxy can stand in for the real thing.

Path: `readout(state) @ Wa + ba + relu(emb[last_input] @ Wd) @ emb.T`. Held-out = last 3% of the corpus, 5,120 tokens, never trained on.

```
TRAINED (full path)    entropy 6.6505 nats (70.80% unif) | sum(pr) 1.000000000000 | top1 0.1256 | acc 16.70%
NOISE READOUT control  entropy 8.8683 nats (94.42% unif) | sum(pr) 1.000000000000 | top1 0.0027 | acc  0.00%

vocab 12002 | chance 0.0083%
```

## What this says, correctly this time

**The model is slightly under-confident, not saturated.** It is right **16.70%** of the time and assigns its top choice a mean probability of **0.1256**. A perfectly calibrated model at
16.7% accuracy would sit near 0.167 on average; this one is a little more conservative than that. That is a normal, mildly cautious distribution.

**The noise control is the contrast that matters:** entropy 8.87 nats (94% of uniform), top-1 **0.0027**, accuracy **0.00%**. So the trained readout is **47x sharper** in top-1
probability and it actually predicts, where the noise readout never does.

This is what F4 should have measured. Its KL-across-six-prompts proxy failed; this replaces it with next-token behaviour on real text, and the answer is unambiguous: **the readout works.**

## The last hypothesis in this thread, also dead

The generation output is gibberish that loops (`"the and of the"`). I had suggested the cause was a one-hot distribution. **It is not** - the distribution is broad and slightly
under-confident, with a top-1 probability of 0.126. A model with this calibration, decoded greedily, produces a *plausible but wrong* word 83% of the time rather than a stuck loop.

So the looping is **not** a calibration defect and **not** a saturation defect. It has to be about the decode path (repeated context driving repeated predictions) or about the
window/context construction at generation time differing from training. That is now the next honest question, and it is testable on the same checkpoint without retraining.

## Verified standing numbers for run 9

```
held-out accuracy        16.70% (this measurement) / 16.93% (trainer log) / 17.97% (a 12-chunk slice)
held-out entropy         6.6505 nats = 70.80% of uniform   sum(pr) = 1.0
held-out top-1 conf      0.1256   (slightly under-confident at 16.7% accuracy)
noise-readout control    top1 0.0027, accuracy 0.00%
state cosine (F2)        0.1736   - states decorrelate
probe entropy (G2)       8.835 nats = 94.1% of uniform (context-free, expected)
```

The four earlier errors that produced different numbers - a broken entropy expression, a missing UNK mask, a forward path without the direct term, and a proxy metric - are each
recorded in their own document. **Four wrong numbers, four different causes, one correct method.**
