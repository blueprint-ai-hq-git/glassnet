# Raising the concept count did not close the matched gap - what a concept COSTS does (2026-11-18)

The width hypothesis was testable and it is now tested. The previous matched comparison had 13 concept units against the plain control's 21 hidden units at the same parameter
count, so "readability costs" and "there were fewer units" could not be separated. This arm raised the concept layer to 21 named concepts.

```
READABLE: 24,945 params   patches 16 x 64  ->  21 concepts  ->  10
PLAIN   : 24,655 params   784 -> 31 -> 10   (1.16% difference in size)
delta -1.47 points        concepts passing 20/21
previous run (13 concepts): -1.53 points
```

**The gap did not move: -1.53 -> -1.47, which is 0.06 points, noise.** Predictions P1 (-1.0 to +0.5) and P2 (within +-0.5) **failed**. P3 held: 20 of 21 concepts honest = 95.2%.
The falsifier as written ("at or below -1.5") did **not** strictly fire at -1.47, and it is not being claimed either way - the honest statement is that raising the concept count changed nothing.

## Why, as arithmetic rather than opinion

The plain control is sized to match the readable model's **parameter count**, not its unit count. So adding concepts also grew the control:

```
one CONCEPT unit = 16 patches x 64 dims + 10 answer weights = 1034 params
one PLAIN   unit = 784 pixels          + 10 answer weights =  794 params
ratio 1.30  ->  at matched parameters the readable model gets 68-77% of the units
measured: 21 concepts vs 31 hidden units = 68%
```

A named concept sits on a **wider input** (16 patches x 64 dims = 1024 features) than a plain hidden unit (784 pixels). Every concept therefore costs 30% more, and at a fixed
parameter budget you can afford fewer of them. **The matched gap is consistent with being a unit-count gap that is itself caused by the concept layer's input width** - not with
a mysterious cost of readability.

## The next arm, and its prediction, written before it runs

`d = 49` makes 16 x 49 = **784** concept features, exactly the per-unit cost of a plain hidden unit. At matched parameters the readable model should then afford the **same
number of units** as the control.

- **P1** concept count and hidden count land within 2 of each other (parity achieved)
- **P2** the gap moves from -1.47 to between -0.5 and +0.5 points
- **P3** honesty stays at or above 85%
- **FALSIFIER** with unit cost equalised the gap is still at or below -1.0 points. Then per-unit cost was not the explanation either, and what remains is the price of the
  concept **constraint** itself - every middle neuron forced to be a named, human-checkable formula.

The layer-by-layer evidence already says the architecture is real, not decorative: zeroing the concept layer drops the model from 84.15% to 10.17%, and shuffling it drops it to
9.55%, so the answer layer reads per-sample concept content rather than the layer's average.
