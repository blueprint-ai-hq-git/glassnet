

---

# DECORRELATION IS A NO-OP ON THE WINNING CONFIGURATION - AND THE PLAIN CONTROL IS BLOCKED BY DESIGN (2026-11-20 13:15)

## windec: neither prediction met, falsifier not fired, no reason to keep the knob

```
                              reference (decorr 0)     windec (decorr 0.5)    delta
held-out accuracy             12.18%                   12.10%                 -0.08
perplexity                    712.92                   714.51                 +1.59
concepts passing audit        25 of 36                 25 of 36               0
POSITIVE SUPPLY               8.476x                   10.107x                +19%
concepts supplying positive   20 of 36                 18 of 36               -2
```

P1 required accuracy >= 12.68%, **or** within 0.5 points of 12.18 while supply fell **below** 8.476x. Accuracy does sit within 0.5
(12.10), but supply went **19% the wrong way** (8.476x -> 10.107x), so **P1 NOT MET**. The falsifier (accuracy more than 1.0 below
12.18) did **not** fire - the drop is only 0.08.

**The reading is therefore not "decorrelation hurts" but "decorrelation does nothing here"** - it is a lever only on the bottlenecked
shape (where it cut supply 236x -> 38x) and a no-op on the configuration that actually performs. Best-known setting stays
`lam_decorr 0`, by parsimony: a knob that buys nothing should not be in the shipped configuration.

## The concept line is now closed, and here is the full accounting

```
configuration                          acc      ppl      audit      supply
36 concepts + direct (decorr 0)     12.18%   712.92  25/36 (69%)    8.476x   <- best accuracy
36 concepts + direct (decorr 0.5)   12.10%   714.51  25/36 (69%)   10.107x   <- no-op
36 concepts, no direct              7.31%   953.87  25/36 (69%)  236.039x
9 concepts, no direct (decorr 0.5)   7.37%  1275.33   9/9 (100%)   19.839x   <- best honesty
9 concepts + direct (decorr 0.5)     9.35%  1269.90    5/9 (56%)   80.896x
```

No candidate hypothesis remains: count is not the lever (36 -> 9 costs accuracy when the path is present and only buys honesty when it
is absent), decorrelation is not the lever on the winner, concentration never moves (top-half 74-82%, entropy 0.906-0.950), and no
configuration is both honest and accurate.

## The plain control: blocked by an explicit design decision, not by my oversight

The comparison your question actually needs - custom architecture vs **a regular model at matched parameters** - cannot be run today,
and I checked rather than assumed:

```
$ lens.Net(vocab=100, concepts=[], d=32, window=8, seed=0)
ValueError: LensNet needs at least one named concept; the concept layer IS the architecture.
```

The model **refuses** an empty concept layer on purpose: the concept layer is not a bolt-on, it is the architecture. So a plain
control is a **new model class**, not a flag - it must share the embedding, the window handling and the answer shape while replacing
the concept layer with a plain hidden layer, and its parameter count must be matched to the readable arm and printed before any
comparison.

That is queued as the next build, with the spec and this evidence, rather than rushed at the end of a wake. Until it exists, **no
statement that our architecture beats or loses to a regular model on the chat task has been measured** - the image model's -0.08
parity is the only matched figure this project has, and it is an image model.
