

---

# ANATOMY SCAN: WHAT IS ACTUALLY INSIDE, NAMED TENSOR BY TENSOR (2026-11-20 14:45)

The directive: "take a normal model, scan it, what it has - layers, features, tokens, KV cache, everything - but all of it is
hidden, so find it." `tools/anatomy.py` does that from a checkpoint alone: no training, no data. Every number below is summed from
the checkpoint's own arrays, and the full inventory is written to `lens/runs/ANATOMY.json`.

## The three trained architectures we own, scanned

```
architecture                          params      table%  compute%  concept%   state carried per token
LensNet 36 concepts + direct        2,225,284      89.0     11.0       1.7     none (window is the whole context)
LensNet  9 concepts + direct        1,873,609      88.4     11.6       0.5     none
tokmem + readable window bank       1,848,064      83.1     16.9       -       256 floats + the bank
```

## The finding that matters, in one line

**The readable part is 1.7% of the model** (and 0.5% at 9 concepts). Sixty-nine percent of `arch_direct_on` is a **token lookup
table** (1,536,000 parameters of `emb`), another 20% is the answer matrix, 9.4% is the direct shortcut - and the entire named concept
layer, the thing the whole project is about, is **36,900 parameters**.

That is the honest anatomy of "our readable model": it is a word table with a small concept hat. Whatever the concept line
measured earlier, it was measuring a 1.7% component and hoping it explained a model that is 89% lookup.

## Per-tensor inventory, `arch_direct_on_ep1.npz` (2,225,284 params, sha256 7eedfcfddae85cec)

```
tensor        shape        role            params     share
emb           12000x128    token table    1,536,000   69.03%
answer.W      36x12000     answer layer     432,000   19.41%
dout          16x12000     direct path      192,000    8.63%
concept.W     1024x36      concept layer     36,864    1.66%
dproj         1024x16      direct path       16,384    0.74%
answer.b      12000        answer layer      12,000    0.54%
concept.b     36           concept layer         36    0.00%
```

## Two things the scan caught in itself, which is why it prints them

1. **My role map was too thin.** The first run reported **208,384 parameters as OPAQUE** (`dout`, `dproj`) when those tensors ARE a
   named component - the direct path - that my map only knew under the name `dmix`. An inventory tool that cannot name a component is
   not evidence the component is opaque; it is evidence the map is thin. Fixed, and the checkpoint now reports **100.0% of tensors
   named**. This matters because "transparent" must mean "we found everything", not "we found the things we thought of".
2. **State per token is `none` for LensNet and 256 floats for tokmem.** That is a real architectural difference, not a gap in the
   scan: LensNet has no carried state at all - its entire context is the 8-token window, so there is no KV cache to build for it.
   Any "own KV cache" work therefore belongs to the recurrent family (tokmem) or to a new attention block, not to LensNet.

## What this means for the next part

The scan says the architecture is 89% lookup and 1.7% meaning. So "10x smarter each layer" cannot be won by tuning that shape - the
shape has almost nothing to tune. The next part of the programme therefore starts where the parameters actually are: a **transparent
attention block written by us**, with every tensor named, **its own KV cache**, and a reference implementation computed by hand so
"transparent" is proven rather than asserted.
