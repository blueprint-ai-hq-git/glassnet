# The model is 0.70 percentage points better than always guessing the most frequent token

Measured with `tools/marginal_baseline.py` on `scan8_ep1500.npz`, over 20,000 held-out positions drawn from
the validation tail, with the frequency baseline scored on exactly the same positions:

```
accuracy of always predicting the most frequent token (',') : 0.0399
model held-out accuracy on the same positions               : 0.0469
model improvement over the frequency baseline               : +0.0070  (+17.5%)

correlation of log(model average prediction) vs log(unigram) : +0.1588
KL(unigram || model average prediction)                      : 1.8330 nats
```

## What this says
The trained model scores **4.69%** where a model that ignores its input entirely and always answers `,` scores
**3.99%**. The entire advantage from 4.7M parameters, a recurrent state, four verified hand-written gradient
paths, a direct token path, decorrelation and a readout with its own normalisation is **0.70 percentage
points**.

It is also **not** simply reproducing the unigram: the correlation between its average prediction and the
token frequencies is only **+0.1588**, with **1.8330 nats** of KL between them. So it is neither learning
context structure nor faithfully copying the base rates - it has collapsed onto a handful of answers, as
`ARGMAX-DECIDED-BY-STATE.md` measured independently (three distinct argmax tokens across 200 contexts).

## Why this reframes everything before it
Every arm from run 5 to run 10 has been hunting an expressiveness problem in the state or the readout: state
collapse, direct-path dominance, readout ignorance, map compression, resolution. All four were tested and
falsified (`RESOLUTION-NOT-COMPRESSION.md` and the documents it links). This measurement says why those hunts
kept failing: **there is almost no learned signal to express.** A readout cannot be blamed for failing to
carry information the model never learned.

## The honest reading of the published 37.37%
The earlier best figure of 37.37% was measured "by position" on a different task and split. It is not
comparable to this 4.69%, and nothing in this document revises it. What this document establishes is the
current state of the 32-token talk model: **barely above its own frequency baseline**. That is the number to
beat, and it is much lower than the headline figure suggests.

## Where the next work goes
Capacity and training signal, not more state surgery:
1. a matched capacity test at larger width on the same data and steps, to see whether the baseline gap grows;
2. a curriculum or objective change that forces different contexts onto different answers, since frequency
   matching is currently the cheapest way to lower loss;
3. a hard check in every future arm: report the frequency-baseline gap beside every accuracy number, so a
   model that only learned the marginal can never look like progress again.
