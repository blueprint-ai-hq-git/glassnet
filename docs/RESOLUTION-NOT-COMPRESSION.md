# The state-to-answer map responds, but at far too low a resolution (2026-11-10)

Follow-up to the collapse diagnosis, which found the state decides the argmax but only ever produces three
distinct answers across 200 contexts. The obvious explanation was that the map compresses: bigger state
differences get squashed into the same prediction. This measures that directly.

`tools/state_distance_response.py`, 200 held-out contexts, 4000 pairs, `scan8_ep1500.npz`:

```
state distance range: 0.0000 .. 0.7582 (mean 0.1888)

Spearman(state distance, logit distance) : rho +0.610
Spearman(state distance, |KL|)           : rho +0.248

state-distance decile    mean logit dist      mean |KL|
decile 1                        17.5409         0.0677
decile 5                        21.0896         0.0611
decile 10                       27.9183         0.1378
```

## The map is not compressing
Rank correlation between state distance and logit distance is **+0.610**, and the top state-distance decile
produces **27.92** of logit distance against **17.54** in the bottom decile - a **59% increase**. Prediction
differences really do grow with state differences, monotonically. So "the map squashes everything to one
answer" is wrong.

## What is actually wrong: resolution
Even at a state cosine distance of **0.7582** - states pointing almost completely apart - the two predictions
differ by only **0.138 nats** of KL, and the model's top-1/top-2 probability gap is about **0.008**. With
12,002 candidates the argmax is decided by margins that small, so a handful of frequent tokens win nearly
always. The map has enough sensitivity and nowhere near enough dynamic range: it can only whisper, and the
argmax needs a shout.

That also explains why simply scaling the readout (run 9's untrained gain = 1.0) raised KL 63x while the
trained-vs-noise control stayed at 1.00x: scaling amplifies the whisper without making it correct.

## What this rules out, and what it points to
Ruled out by measurement: state collapse, direct-path swamping, readout ignorance of the state, map
compression. All four were plausible and all four are false.
Remaining: the model is fitting the marginal. A 4.7M-parameter model trained on ~23M tokens against a
12,002-way target is in an underfit regime where matching the frequency distribution minimises loss, and the
conditional structure that separates contexts is exactly what it lacks the capacity or the signal to learn.
The next arms should therefore test capacity and training signal, not more state surgery.
