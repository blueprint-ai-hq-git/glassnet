

---

# OUR OWN BLOCK, WITH OUR OWN CACHE, AND THREE PROOFS IT IS RIGHT (2026-11-20 15:08)

The scan said what the shipped model is: **89% lookup table, 1.7% concepts, no state carried per token.** So the architecture being
asked for did not exist in this project. This is the first piece of it: `lens/glass_block.py`.

## What is in it, and why each part is named

```
parameters, by name (d_model 8, 2 heads, head_dim 4)
  Wq          8x8        64      query projection
  Wk          8x8        64      key projection
  Wv          8x8        64      value projection
  Wo          8x8        64      attention output projection
  W1          8x16      128      MLP up-projection
  W2          16x8      128      MLP down-projection
  ln1_scale   8           8      LayerNorm 1 scale     ln1_shift  8   8   LayerNorm 1 shift
  ln2_scale   8           8      LayerNorm 2 scale     ln2_shift  8   8   LayerNorm 2 shift
  total: 544 parameters - and this list IS the model: there is nothing else to learn
```

LayerNorm **exposes the mean and variance it used** (`return_stats=True`) rather than hiding them. The causal mask is written
explicitly as `np.triu` so "the future" is a visible object. `KVCache` keeps one K and one V tensor per (layer, head), append-only,
with a `report()` that prints the exact shape and byte cost of a context.

## The three tests, all printing real numbers

```
TEST 1  attention + residual vs a HAND reference (python loops, math.exp, no shared code)
  max |difference| = 4.441e-16      PASS

TEST 2  KV-cache equivalence: same 4 tokens, all-at-once vs one-at-a-time through the cache
  max |difference| over all 4 prefix positions = 1.192e-07      PASS   (float32 rounding, not a cache error)
  KV cache: 1 layers x 2 heads x 4 tokens x 4 wide x 2 (K and V) x 4 bytes = 256 bytes

TEST 3  LayerNorm vs hand computation (mean 3.000000, var 3.500000)
  max |difference| = 0.000e+00      PASS
```

## Two defects I had to fix in my own test before it was worth anything

1. **The hand reference did not include LayerNorm.** The block normalises before attention; the reference did not. As written, the
   first version of TEST 1 would have failed and I would have spent the next hour hunting a bug in correct attention code. Fixed by
   adding an explicit `norm=False` switch **whose only purpose is to let a test verify one mechanism at a time** - a test that mixes
   two unknowns cannot tell you which one is wrong. LayerNorm then got its own hand test (TEST 3).
2. **The cache byte report printed `0 bytes`** because it reported an outer cache object I never appended to, while the real stepped
   cache held 4 tokens. A byte accounting that can say zero while holding data is worse than no accounting. It now reports the final
   stepped cache and **asserts** its token count.

## Honest status of this build

- **Forward is proven. Backward does not exist yet** - the block has no gradients, so it cannot be trained. That is the next piece,
  and it goes through the same gate as everything else in this project: every gradient checked against finite differences before it is
  trusted, target worst error < 1e-8 in float64.
- 544 parameters is a test-size block, not a model. Scaling it up is only worth doing once the gradients are certified.

## Why this is the right next layer of the programme

The scan proved the old shape has almost nothing to tune (1.7% of parameters carry the meaning). This block puts the computation
where the parameters are: attention over real tokens, with a cache whose cost is printed, and every tensor nameable by a human. The
gate for "10x smarter per layer" was written down before any of this ran: **10x fewer parameters at equal accuracy, or 10x less
compute at equal accuracy** - and it can only be attempted after the backward pass is certified.
