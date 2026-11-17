# Reference comparison: our shipped model against a normal 0.5B transformer

Started 2026-11-17 at Yahya's instruction: take an original model that exists in the world, compare it with ours **feature by feature, each one tested**, and
work out exactly what we are missing so our own custom model can be transparent *and* have everything a normal model has.

**Reference model**: Qwen2.5-0.5B-Instruct. Apache 2.0 licence (verified: the model card licence field; 0.5B class is Apache 2.0). Two forms are on this
machine already or fetched for measurement: the original safetensors directory `qwen2.5-0.5b-instruct-original` (290 bf16 tensors, 988 MB) and
`qwen2.5-0.5b-instruct-q4_k_m.gguf` (491,400,032 bytes, sha256 74a4da8c...) downloaded for inference with llama.cpp.

**Licence discipline**: the comparison uses the reference model's *published configuration, tokenizer and behaviour*. No weights or training code from it enter
our model. Every feature we adopt is re-implemented in our own engine, which keeps the from-scratch claim honest.

**Our model measured here**: `lens/runs/chat/nodirect1024_wide60_lam1.0_data4m_e12_s0_best.npz`, the shipped configuration. Its five tensors and its forward
path were read out of the checkpoint and `lens/model.py`, not taken from the docs.

---

## The single largest finding

**Our shipped model has no layers at all.** The forward pass, verified from `lens/model.py:434`:

```
x      = embedding(emb, ids)              # window of 32 tokens, each 1024-d, concatenated -> (n, 32768)
h      = sigmoid(concept.W @ x + concept.b)   # 32768 -> 53 named concepts
logits = answer.W @ h + answer.b          # 53 -> 12002 vocabulary
```

Five tensors, total 14,674,913 parameters: `emb (12002,1024)`, `concept.W (32768,53)`, `concept.b (53,)`, `answer.W (53,12002)`, `answer.b (12002,)`.

That is not "a small transformer". It is **one linear projection into 53 concepts, a sigmoid, and one linear projection to the vocabulary**. There is no
attention, no MLP, no normalisation, no residual stream, no positional encoding and no depth. The 32-token window is encoded only by *which columns of
`concept.W`* each token lands in.

Qwen2.5-0.5B is a 24-layer transformer. So the gap between the two is not a matter of scale - it is a matter of **five missing mechanism classes**: depth,
attention, normalisation, a gated MLP, and explicit positional encoding.

---

## Feature by feature

Legend: **PRESENT** = we have it, with the file where it lives. **DIFFERENT** = we have something in that role but it is not the same mechanism. **ABSENT** =
we do not have it at all. Measured differences are marked with a number and its source file.

### Structure

| Feature | Reference (Qwen2.5-0.5B) | Ours (shipped) | Status |
|---|---|---|---|
| Layers | 24 transformer blocks | **0** - single projection path | **ABSENT** |
| Hidden width | 896 | 1024 | PRESENT (differs by 14%) |
| Parameters | ~494M (0.5B class) | 14,674,913 | PRESENT (34x smaller) |
| Context window | 32,768 tokens | **32 tokens** | **ABSENT** (1024x smaller) |
| Positional encoding | RoPE, theta 1e6 | **none** - order only enters via which `concept.W` columns a token hits | **ABSENT** |
| Attention | 14 query heads, 2 KV heads (GQA 7:1), head_dim 64 | **none** - concepts are the only mixing operation | **ABSENT** |
| MLP / feed-forward | SwiGLU, intermediate 4864 (gate/up/down) | **none** | **ABSENT** |
| Normalisation | RMSNorm, eps 1e-6, pre-norm | **none** in the shipped checkpoint | **ABSENT** |
| Residual stream | yes, every block | none | **ABSENT** |
| Activation on the bottleneck | SiLU inside the MLP | **sigmoid on the 53 concepts** | DIFFERENT |
| Output embedding tying | `tie_word_embeddings: true` | **untied** - `answer.W (53,12002)` is a free matrix | DIFFERENT |
| Attention bias | Qwen2 has q/k/v biases | n/a | ABSENT (no attention) |

### Tokenisation and vocabulary

| Feature | Reference | Ours | Status |
|---|---|---|---|
| Tokeniser | BPE, 151,936 merges-based vocab | word-level, 12,002 vocab | **DIFFERENT** (structural gap) |
| Vocab size | 151,936 | 12,002 | 12.7x smaller |
| Unknown-word handling | byte fallback, nothing is unrepresentable | `<unk>` token exists | DIFFERENT |
| Merge table | `merges.txt` 1.67 MB | none | ABSENT |

The tokeniser difference is the one that quietly caps us: a word-level vocabulary of 12,002 cannot represent a word it has not seen, while BPE can spell any
string from bytes. This is measured below on our own held-out text rather than asserted.

### Numerics and serving

| Feature | Reference | Ours | Status |
|---|---|---|---|
| Weight precision | bf16 | float32 | DIFFERENT (deliberate: our rules say float32) |
| Quantised path | GGUF q4_k_m available from upstream | none | ABSENT |
| KV cache | standard, `use_cache: true` | our own cache exists in code, unused by this checkpoint (it is stateless per window) | DIFFERENT |
| Sampling controls | temperature, top-k, top-p, repetition penalty | temperature, top-k, repetition penalty (measured earlier today) | PRESENT |
| Chat template | ChatML-style control tokens | `<|im_start|>` / `<|im_end|>` | PRESENT (same family) |

### What we have that it does not

| Feature | Ours | Reference |
|---|---|---|
| Named middle neurons | 53 labelled concepts, each with a definition | none - unnamed dimensions |
| Exact explanation | `explain()` residual 2.4e-06, concept share 1.000000 (`docs/CAUSAL-EXPLAIN.md`) | none available |
| Concept-to-answer sparsity | answer built only from 53 numbers | dense 896-wide residual stream |
| Readable weight budget | 53 concepts a human can list | 494M unexplained parameters |

---

## Measurement slots, filled only when a real number exists

- **bits per character, ours**: pending - requires our held-out perplexity converted with our tokeniser's chars-per-token
- **bits per character, reference**: pending - `llama-perplexity` on the same held-out text
- **next-token accuracy, reference on our text**: pending
- **generation behaviour on identical prompts**: pending (repetition, coverage, distinct-n)

These are the numbers that make the comparison honest instead of rhetorical. A perplexity in word-tokens cannot be compared with a perplexity in BPE tokens,
which is exactly why the measurement is in bits per character: it is the only unit both models can be scored in on the same text.

## Ranked gap list, with the transparent-compatible choice for each

Filled in once the measurements land. The ranking criterion is: what plausibly costs us accuracy and breadth, given that our own strength sweep showed breadth
peaking at exactly one supervision setting - which is the signature of a model that cannot compute, only interpolate.

---

## First measurements (2026-11-17, morning)

**Our model, bits per character: 2.1731** (`tools/bpc_ours.py`, 59,879 word tokens, 240,804 characters of held-out text, every token scored with no
masking, arithmetic taken from the checkpoint itself). That equals 8.7393 bits per word-token, i.e. word-level perplexity 427.37 *in this unmasked setting*.
The 181.04 in the shipped docs is a different quantity - assistant-turn-masked scoring - and the two must not be quoted as if they were the same number.

**Tokenizer gap, measured two ways, because one of them is a trap**:
- On our own held-out text the out-of-vocabulary rate is **0.00%** - but that is by construction: the corpus was tokenised with this vocabulary, so it
  *cannot* contain a word the vocabulary lacks. This measurement says nothing about the real world and is recorded here only so nobody quotes it as if it did.
- On 37 ordinary out-of-domain words the corpus never saw, **12 (32%) are absent from our vocabulary**, and there is no mechanism to spell them:
  `zurich`, `kolkata`, `mitochondria`, `tokenizer`, `backpropagation`, `llamacpp`, `pls`, `thx`, `asap`, `btw`, `imo`, `tbh`.
  A BPE vocabulary cannot fail this way - it spells any string from bytes. This is not a small gap, it is a structural one: names, new technical terms and
  abbreviations are unrepresentable rather than merely rare.

**Reference model, bits per character: PENDING** - the scoring run needs approval (the launch gate timed out). The command is one line:
`llama-perplexity -m data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf -f data/models/val_text.txt -c 512 --chunks 12 -t 4`.

---

## Position: tested, and my first verdict on it was wrong

I predicted from the architecture that positional information is absent. **Measured, it is not absent - it is implicit and fixed.**

`tools/position_ablation.py`, 3000 held-out windows, identical tokens and targets, only the order changed:

```
condition                  bits/token
true order                   8.7502
random shuffle              11.5540   (+2.8038, +32.0%)
reversed                    11.5879   (+2.8377, +32.4%)
```

Destroying order costs **32% of the loss**, so the model genuinely uses order - it gets it from *which columns of `concept.W` each position writes into*,
because the window is concatenated position by position. What it lacks is not position, it is **relative** position: there is no mechanism that knows two
tokens are three apart regardless of where they sit. A transformer's RoPE gives relative position for free at any distance; ours is welded to absolute slots
in a 32-token window.

**Correction recorded**: the first version of that script printed "the model is effectively order-blind" next to a 32% effect - the verdict-check bug this
project keeps catching, a correct number under a wrong conclusion. The verdict is now computed from the number against a stated threshold, so the wording
cannot drift from the measurement again.

## Ranked gap list (updated with the measurements)

1. **Depth (0 layers vs 24)** - the largest single difference. Our model cannot compose features: concepts are computed once, linearly, from a flattened window.
   No amount of supervision tuning changed that, which is exactly what yesterday's strength sweep looked like from the outside.
2. **Relative position (none vs RoPE)** - absolute slots only work inside 32 tokens. Every sequence task beyond that window is unreachable, not merely harder.
3. **Tokenizer (word-level, 12,002 vs BPE, 151,936)** - 32% of ordinary out-of-domain words are unrepresentable, with no spelling fallback. Names, new terms and
   typos are structurally impossible rather than rare.
4. **Attention (none vs GQA 14/2 heads)** - the only mechanism that lets distant tokens interact. Ours mixes tokens exactly once, through a 53-wide bottleneck.
5. **Normalisation (none vs RMSNorm)** - no RMSNorm, no residual stream: there is nothing to keep activation scale stable in a deep stack, so depth cannot be
   added safely until this is.
6. **Gated MLP (none vs SwiGLU 4864)** - no per-token nonlinear capacity. The 53-concept sigmoid is the only nonlinearity in the whole model.
7. **Output tying (untied vs tied)** - a small parameter saving and a mild regulariser; genuinely low priority.
8. **KV cache** - our own exists in code but this checkpoint is stateless per window; needed once context exceeds the window, which is item 2's consequence.
9. **Quantised export** - a deployment concern, not a capability gap.

## What that implies for the build, in order

Items 1, 2, 4 and 5 are one piece of work, not four: a residual stack with RMSNorm, relative position and one attention block per layer cannot be added
separately. The honest sequence is a single block (normalise, attend, residual, normalise, gated MLP, residual) repeated to whatever depth 6 GB allows, with
the concept layer kept as the *only* path to the answer so `explain()` stays exact.

That is a real architecture change and it goes in behind the same gate as every loss change: gradient check, wiring proof, step-0 equivalence, smoke run - one
transition at a time, with the concept bottleneck preserved so readability is never traded away silently.

**Still unmeasured**: the reference model's bits per character. The scoring run needs approval and has not been retried. Without that number the gap list above
is ranked by mechanism, not by measured cost.
