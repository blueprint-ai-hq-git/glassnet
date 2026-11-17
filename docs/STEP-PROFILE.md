
## 88% of a run-9 step is four einsums over the vocabulary (2026-11-17)

Last wake established the recurrent loop is 0.024 s of the step. `tools/profile_step.py` profiles one real `train_step` with the arm's own config
(V=12002, D=128, H=256, CHUNK=128, B=32, K=64, readout norm on):

```
one step, wall clock (warm): 31.12 s        [pinned to 2 cores; the live run gets ~10.5 s]

   ncalls  tottime  cumtime  function
        4   27.431   27.431  numpy c_einsum          <- 88% of the step, 6.86 s per call
        1    3.226   31.209  train_step:20           <- the Python body itself
        1    0.335    0.335  ndarray.copy
      146    0.124    0.124  ufunc.reduce
        1    0.043    0.043  ndarray.argmax
```

**Four einsum calls hold 88% of the cost, at 6.86 s each.** The recurrence, the embedding gather, the readout `argmax`, the renorm, the gradient dict allocation - all of it
together is under 4 s. (Absolute seconds here are inflated by the 2-core pin; the *proportions* are the finding and they are not sensitive to core count.)

## The four contractions, and why two of them are miswritten

```
line  94   "bkv,bkd->vd"   dlgB(B,K,V) x dsel(B,K,D)  -> (V,D)      B*K*V*D  ~ 3.1e9 ops
line  96   "bkv,vd->bkd"   dlgB x emb(V,D)            -> (B,K,D)    B*K*V*D  ~ 3.1e9 ops
line 100   "bte,de->btd"   dpre x Wd(D,D)             -> (B,T,D)    small
line 102   "bkv,hv->bkh"   dlgB x Wa(H,V)             -> (B,K,H)    B*K*V*H  ~ 6.3e9 ops
```

Lines 94 and 96 are **batch-of-outer-products and a plain matrix product in disguise**. `einsum("bkv,bkd->vd", A, Bmat)` is `A.transpose(1,0,2).reshape(K*B? ...)`
- more usefully, it is a matrix product once the batch is folded: `einsum("bkv,bkd->vd")` = `(A.reshape(-1,V).T @ Bmat.reshape(-1,D))` with V as the leading axis.
Written that way, **NumPy dispatches it to BLAS/OpenBLAS instead of its own einsum kernel.** The same is true of line 96 with `emb`, and line 102 with `Wa`.

Two further multipliers sit on top of that:

1. **Everything is float64.** The hot file casts to float64 in ten places, including `xs`, all four einsums, and the gradient dict. On the same hardware float32 is
   roughly twice the arithmetic throughput, and this kernel is pure arithmetic.
2. **`p["emb"].astype(np.float64)` and `dsel.astype(np.float64)` allocate a copy of the whole embedding table inside the step** (V*D = 1.5M doubles), which buys nothing
   because the source array is float32 either way.

## What this predicts, written before the change

Rewriting the three vocab-sized einsums as folded matmuls, plus float32, should move a step by **at least 4x** and plausibly far more, because 88% of the time is in code
that is currently leaving BLAS unused. **The falsifier is explicit: if the rewritten step is not at least 3x faster with gradients matching the old form to ~1e-6, then the
einsum form was not the cost and this explanation is wrong.** The rewrite is an arithmetic restructuring, so equivalence is checkable directly - old and new outputs on the
same inputs, elementwise.

That check is mandatory before any arm uses it. This project's rule is that a change to the arithmetic gets a numerical verification against the form it replaces, and the
rewrite is exactly such a change.
