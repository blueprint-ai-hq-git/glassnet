
## One keyword, 8.55x faster, outputs identical to ten decimals (2026-11-17)

The previous entry predicted that rewriting three vocab-sized einsums as folded matmuls would move a step by at least 4x, and set an explicit falsifier. **The prediction was
wrong and the falsifier fired - but the investigation found something better.**

## The falsification, first

`tools/einsum_rewrite_test.py`, same shapes, same inputs, both forms:

```
float64:  contraction      einsum     matmul    speedup
          bkv,bkd->vd      0.1108s    0.1097s    1.01x
          bkv,vd->bkd      0.1220s    0.1344s    0.91x
          bkv,hv->bkh      0.1878s    0.1795s    1.05x
          TOTAL            0.4206s    0.4236s    0.99x     equivalence MATCH (rel diff 4.0e-16)
```

**The matmul rewrite buys nothing.** My stated explanation - that einsum leaves BLAS unused - was false *for optimize=True*. But those timings did not match the live step, which spends
**6.86 s per einsum** on the same shapes. A 40x gap had to be explained rather than papered over.

## The gap, and the real cause

`grep -n einsum lens/scan_direct_batched.py` shows all four calls pass **no `optimize` argument**. NumPy's default is `optimize=False`, which uses a generic looping contraction
instead of planning a BLAS path. Measured on the real shapes:

```
contraction      default     optimize=True    matmul
bkv,bkd->vd      4.1435s       0.1049s       0.1177s
bkv,vd->bkd      2.1501s       0.1250s       0.0959s
bkv,hv->bkh      5.0815s       0.1522s       0.1359s
TOTAL           11.3752s       0.3820s       0.3495s

default -> optimize=True : 29.8x
default -> matmul        : 32.5x
optimize -> matmul       :  1.09x   => the same thing
```

**`optimize=True` *is* the matmul rewrite.** NumPy does the folding itself, given the instruction. The fix is one keyword, not 247 lines and not a restructure.

## The A/B on the real training step

Same inputs, same seed, current code vs the added keyword, `tools/ab_step.py`:

```
BEFORE  loss 9.9041865856  acc 0.0000000000  gnorm 1.1704698742  time 32.47s
AFTER   loss 9.9041865856  acc 0.0000000000  gnorm 1.1704698742  time  3.80s
```

**Loss and gradient norm identical to all ten printed decimals, and the step is 8.55x faster.** No arithmetic changed; no gradient check is needed because nothing in the mathematics moved -
the equivalence is demonstrated on the actual function, which is stronger than a gradient check for this kind of change.

## What this says about the whole GPU thread

The arm was never GPU-bound, never einsum-bound in any deep sense, and never needed a port. **It was calling a library function without the flag that makes it fast.** The measured
chain that got here: loop is 0.024 s (not the cost) → profile says 88% einsum → matmul rewrite gives 0.99x (prediction falsified) → the 40x benchmark-vs-live gap points at a missing
flag → one keyword, 8.55x, identical outputs.

Four wrong or partial explanations died on the way, each to a number rather than to an argument.

## Caveat on the ratio

32.47 s is a 2-core pin; the live arm runs at ~10.5 s on more cores. Both forms scale with cores, so the **8.55x ratio is the claim** and the absolute seconds are not. The running arm
is unaffected - it imported the old code - and the next launch gets the faster path.
