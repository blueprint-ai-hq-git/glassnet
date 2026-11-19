

---

# THE MEMORY BANK FAILED, AND THE REBUILD THAT REPLACES IT (2026-11-19 10:11)

## The failure, measured

The token-space bank ran against its no-bank control, same seed, data, 600 steps:

```
bank K=16   HELD-OUT 5.33% (loss 6.3339)
no bank     HELD-OUT 5.27% (loss 6.3339)     -> +0.06 points. Nothing.
all 16 slots decoded to the same word: 'management'
```

Then the diagnosis, on the trained weights:

```
slot-to-slot cosine: mean 0.864   max 1.000   min 0.553
effective distinct slots (1/mean(cos^2)): 1.3 of 16
each slot vs the mean slot direction: cos 0.934
```

**Sixteen slots holding 1.3 slots' worth of information.** My previous "fix" (content-addressed writes plus a per-slot bias) did not
address the real cause: the write was still `w = softmax(scores)` over slots, so **every token was written a little into every
slot**. Each slot therefore held a running average of the corpus, all slots converged to the same average, and the decoder mapped
that average to whichever word sits nearest the centre of the embedding table. Predictions scored: P1 (both learn, above the 14.70%
trivial constant) **FAILED** - both were far below it; P2 (bank within 1 point of control) **MET** at +0.06; P3 (slots hold
different words) **FAILED**.

## The rebuild: a readable context window, not a learned store

`lens/winmem.py`. The bank IS the last K tokens. Slot j holds the embedding of exactly one real token and is replaced **wholly** -
never blended. The addressing is a ring buffer, so it is discrete, needs no write policy to learn, and cannot collapse. Reading is
learned: `a = softmax(K q / sqrt(H))`, `r = a C`, `h <- h + tanh(r Wo)`, so the model chooses which past tokens matter while being
unable to smear them together in storage.

```
tokens fed:       the captain sailed west storm harbour again sailed captain
window contents:  storm | harbour | again | sailed
slot-to-slot cosine: -0.081      (was 0.864)
finite-difference: worst abs 4.334e-10   GATE: PASS
```

**Three gradient bugs were found by that one check, in order:**
1. the embedding gradient for slot contents was never wired: numeric 1.423356 against analytic 0.0;
2. attribution was gated on the slot being filled *before* the write, skipping the first K steps: off by 1.40;
3. the pre-write slot was zeroed when this step's read had touched its old content: off by 0.63.

Each was invisible in the code and obvious in the difference. That is the argument for the gate, not a formality about it.

## Honest status

This is a working, gated, readable window bank - **not yet a result**. It has not been trained at scale or compared against a control,
because the trainer wiring is the next step. What is established: the design is implementable, its gradient is exact, and it does not
suffer the collapse that killed its predecessor.
