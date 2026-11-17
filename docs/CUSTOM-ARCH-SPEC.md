
# Spec for our own model: all the features of a normal transformer, still exactly explainable

Written 2026-11-17 after the reference comparison. Every gap below is a **measured** gap, not a hunch. Every parameter count is computed by
`tools/param_budget.py`, not estimated.

## What we measured about ourselves first

| Fact | Number | Source |
|---|---|---|
| Shipped model layers | **0** - embed, flatten, sigmoid, 53 concepts, linear | checkpoint tensors + `lens/model.py:434` |
| Bits per character | **2.1731** | `tools/bpc_ours.py` |
| Effect of destroying word order | **+32.0%** loss | `tools/position_ablation.py` |
| Concepts carrying > 0.05 bits when removed | **16 of 53** | `tools/concept_capacity.py` |
| Concepts that make the model worse | **8 of 53** | same |
| Information carried by concepts | **47%** | same |
| Out-of-domain words we cannot represent | **32%** (12 of 37) | tokeniser probe |

The last two together are the diagnosis: **the entire capacity between input and output is about sixteen numbers**, and one word in three cannot even be encoded.
No amount of supervision tuning fixes either.

## The design

One standard pre-norm transformer block, repeated, with **the concept bottleneck kept as the only path to the answer**:

```
tokens -> BPE embeddings (+ RoPE at every layer)
       -> [ RMSNorm -> attention (GQA) -> +residual
            RMSNorm -> SwiGLU MLP     -> +residual ]  x L
       -> RMSNorm -> project to K named concepts (sigmoid)
       -> answer from the K concepts only
```

The last two steps are ours and are unchanged in spirit from today's model: the answer is a function of named concepts and nothing else. That keeps `explain()` exact
- no residual stream, no skip connection and no attention output reaches the vocabulary directly. This is the one place we deliberately differ from a normal
transformer, and it is the whole point of the project.

**What each missing feature buys, stated so it can be falsified later:**

- **Depth + attention + RMSNorm + residual (one unit, not four)**: composition. Today concepts are computed once from a bag of positions; with blocks they can be
  computed from *relations between* positions. If depth does not move breadth, the bottleneck itself is the cause and the next suspect is K.
- **RoPE (relative position)**: anything beyond the window. Today position is absolute and welded to 32 slots.
- **SwiGLU MLP**: per-token nonlinear capacity, roughly 2/3 of the parameters in a normal transformer.
- **BPE with byte fallback**: removes the 32% unrepresentable-word class entirely; nothing becomes unspellable.

## Parameter budget, computed

`tools/param_budget.py`:

```
candidate                          total      embed     blocks    concept    answer
A  d256  L6  ff1024  h4 kv1 V16k   10,860,992   4,096,000   5,708,544   16,448   1,040,000
B  d384  L8  ff1536  h6 kv2 V16k   24,524,480   6,144,000  17,315,840   24,640   1,040,000
C  d512  L8  ff2048  h8 kv2 V16k   39,691,968   8,192,000  30,427,136   32,832   1,040,000
D  d384 L12  ff1536  h6 kv2 V16k   33,182,400   6,144,000  25,973,760   24,640   1,040,000
E  d512 L12  ff2048  h8 kv2 V16k   54,905,536   REJECTED: over the 50,000,000 limit
```

Weight footprint in float32 is 43-159 MB, and activations at batch 512 / context 512 are roughly 1.6-3.2 GB by the crude `6 * B * T * d` bound, which fits the 6 GB
card with room for the optimiser state.

**Recommendation: B (d384, 8 layers, 24.5M parameters).** Twice the shipped model, still well under the limit, leaves headroom to grow rather than starting at the
ceiling. C and D are the next rungs if B is cheap to train.

## Build order, one measurable step at a time

The comparison produced one insight worth acting on before any new architecture: **the tokeniser is separable and cheap to test**. It changes what the model sees
without changing what the model is, so it isolates its own effect.

1. **BPE tokeniser first, on the existing architecture.** Train a byte-level BPE on our own corpus (our own training, no borrowed vocabulary), vocab 16k, byte
   fallback. Same model, same data, same epochs, only the tokeniser changes.
   *Prediction: bits per character improves by 0.15-0.35 and the unrepresentable-word class goes to zero. Falsifier: improvement under 0.05 means the tokeniser was
   not what was costing us and the effort belongs entirely in architecture.*
2. **The block stack (depth + attention + RMSNorm + SwiGLU + RoPE), tokeniser fixed at step 1's result.** Behind the full gate: analytic gradients for every new op,
   a wiring proof, step-0 equivalence to the current model when the stack is zeroed, a smoke run, then training.
   *Prediction: breadth above 0.292 for the first time, because there is finally somewhere for variety to live. Falsifier: breadth still under 0.20 with depth in
   place means the concept bottleneck K is the binding constraint, and the next experiment is K, not L.*
3. **Then**, and only then, revisit supervision strength on a model that can compute - yesterday's entire strength sweep was measuring a bottleneck with no room in it.

## What does not change, and cannot

No pretrained weights. No borrowed training code. Our own engine, our own optimiser, our own KV cache. float32. MIT. Under 50M parameters. The answer continues to be
a function of named concepts only, so `explain()` stays exact and checkable - the residual stays reported next to every explanation.

**The licence line stays honest**: the reference model contributes its *published configuration* and its measured behaviour to this comparison. Every adopted feature
is re-implemented in our engine from the description, not copied from its weights or its training code.
