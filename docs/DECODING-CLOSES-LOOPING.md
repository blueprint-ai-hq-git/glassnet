
## The loop was the decoder: sampling eliminates it, and the cost is fidelity (2026-11-15)

Predictions in `lens/runs/DECODING-REAL-PREDICTIONS.txt`. Three seeds of the best configuration (4M/12ep), three decoders, all measured
through `lens.load()` + `net.forward()`. The earlier decoding grid is withdrawn because it used the broken hand-built path.

```
decoder                          repetition        realistic coverage   KL(gen||targets)
                                 (mean of 3 seeds)  (mean of 3 seeds)   (mean of 3 seeds)
A greedy, no controls               0.275                0.343                17.21
B sample T=0.8                      0.0013               0.217                19.51
C sample + top-k 50 + rep 1.3       0.000                0.222                19.07

per-seed repetition: A 0.318 / 0.252 / 0.256 | B 0.002 / 0.000 / 0.002 | C 0.000 / 0.000 / 0.000
```

- **D1 PASSED**: repetition goes to **0.000-0.002** at B and C, far below the 0.10 bar.
- **D2 FAILED**: coverage at B and C is **0.217-0.222**, below the 0.30 bar I set.
- **D3 FAILED narrowly**: KL at C is **19.07**, just above the 19.0 bar (B is 19.51).
- **FALSIFIER NOT TRIGGERED**: repetition is nowhere near the 0.20 that would have made the training-protocol arm mandatory.

## The finding that closes the looping question

**Looping is a property of GREEDY decoding, not of the model.** The same weights that repeat themselves 27.5% of the time under argmax
produce **0.13%** repetition under plain sampling at T=0.8. Every attempt this project made to fix looping by changing the model - data
volume, context length, epochs, supervision strength, concept count, decorrelation, interface width - was aimed at a defect that lives in
the decoder.

That also means the **training-protocol arm (scheduled sampling / unlikelihood) is NOT required** for looping, and its falsifier says so
explicitly. It stays in the queue only as a way to close the remaining fidelity gap, not as a fix for repetition.

## The trade that actually remains

```
greedy   : repetition 0.275, coverage 0.343, KL 17.21      faithful, loops
sampling : repetition 0.001, coverage 0.217, KL 19.51      varied, drifts to the tail
```

Neither end is right. Greedy hugs the true distribution and then degenerates into a loop; sampling escapes the loop by wandering into the
long tail, which is why coverage falls and KL rises. The useful question is now whether there is an intermediate that keeps looping near
zero without giving up 0.12 of coverage - i.e. a temperature or truncation setting that sits on the knee rather than at either end. That
is the next arm and it is config-only:
`lens/runs/TEMPERATURE-KNEE-PREDICTIONS.txt`.

## Correction to my own scorecard

Yesterday's and this morning's repetition findings - the "looping survived eight changes" line - were all measured under greedy decoding
through a broken forward. They were wrong twice over. What is true, measured properly: **under greedy the best configuration loops 0.275
+/- 0.037; under sampling it does not loop at all.**
