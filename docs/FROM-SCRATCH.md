# What "from scratch" means here, precisely

This word gets abused, so here is exactly what is ours and what is not.

## Written by us - no framework, no borrowed training code

| Piece | Where |
|---|---|
| Tensor type, forward pass, reverse-mode differentiation | `lens/tensor.py` |
| Every operation's derivative (add, multiply, matmul, ReLU, sigmoid, cross-entropy, BCE, L1, reshape, stack) | `lens/tensor.py`, `lens/ours_ops.py` |
| Layer definitions (Dense, compute core, tied readout) | `lens/model.py` |
| Optimiser (agreement-scaled steps) | `lens/opt.py` |
| KV cache | `lens/kvcache.py` |
| Training loop, gates, checkpointing, resume | `lens/scripts/train_chat.py`, `lens/unified.py` |
| Concept layer, naming and audit | `lens/concepts.py` |
| Live inspector, exports, local server | `lens/liveview.py`, `lens/serve.py` |

**Verified:** a grep across every `.py` file in the repository finds **zero** imports of `torch`, `sklearn`,
`keras`, `tensorflow` or `jax`. No pretrained weights are used anywhere.

## Not written by us - libraries we stand on

| Library | Used for | Why it is allowed |
|---|---|---|
| **NumPy** | the array arithmetic under our own tensors | it is an array library, not a learning framework: it supplies no gradients, no layers, no optimisers |
| **matplotlib** | documentation charts | plotting only |
| **CuPy** | optional GPU arrays (same code path as NumPy) | array library, not a framework |
| **OpenCV** | reading image files for the vision experiments | file decoding |
| **pyarrow / datasets** | reading prepared data shards | file formats |
| **safetensors** | writing the standard weight container | file format |

## The honest sentence for a reviewer

> GlassNet implements its own tensors, gradients, layers, optimiser, cache and training loop, and uses NumPy
> purely as the array arithmetic backend. No machine-learning framework and no pretrained weights are
> involved. Datasets are used only where the licence is recorded in DATA_LICENSES.md.

If anyone finds a framework import or a borrowed training routine we missed, that is a documentation bug -
tell us and we will correct this file, not argue about it.
