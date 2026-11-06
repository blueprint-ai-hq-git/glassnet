---
license: other
license_name: blueprint-ai-license-1.0
license_link: https://github.com/blueprint-ai-hq-git/glassnet/blob/main/LICENSE
language:
  - en
library_name: glassnet
tags:
  - from-scratch
  - research
  - not-for-production
  - interpretable
pretty_name: GlassNet
---

# GlassNet

**A language model you can look inside.** Written from scratch: own tensors, own backward pass, own
optimiser, own KV cache. **No PyTorch. No transformers. No autograd library. No pretrained weights.**

This is **not a normal model**. You cannot load it with `AutoModel.from_pretrained` - there is no
`transformers` implementation, because there is no framework underneath it at all. The whole architecture is
about 7,000 lines of plain NumPy, and you can read every line of it.

> **Status: early research.** The machinery works and is measured. The model is not good yet, and the numbers
> below say exactly how good it is.

## How it works - all four stages, no hidden path

```
tokens + position
      |
      v
[ compute core ]    recurrent: state carried across all 32 positions
      |
      v
[ concepts ]        256 named neurons: "is_a_question", "mentions_self", ...
      |
      v
[ answer ]          tied readout over the 12,002-token vocabulary
```

| Stage | What it does | Cost |
|---|---|---|
| embedding | turns token ids into vectors | 0.0053 ms/token |
| compute core | carries a state vector forward instead of using attention | 0.0027 ms/token |
| concepts | 256 neurons, each audited against a measurable definition | 0.0008 ms/token |
| answer | tied readout sharing the embedding matrix | 0.0273 ms/token |

## How to look inside it

This is the entire point of the project. Three ways, all of them real:

1. **`explain()`** - ask why an answer happened. It prints which named concepts fired and how much each one
   pushed the output. If the stated reason does not match the input, that is a reportable bug.
2. **The live inspector** - `lens/liveview.py` serves real weights and live neuron activations in your
   browser on port 8770. Not a mock: it reads the checkpoint's tensors and shows what fires for a given input.
3. **`docs/ARCHITECTURE.md`** - every derivative written out by hand, and the per-operation gradient check
   that proves them (9/9 PASS, worst absolute error 7.5e-10).

## Measured results - quote these, not better ones

| Checkpoint | Held-out accuracy (by position) | Params |
|---|---:|---:|
| `glassnet-d512-ext6-37pct` | **37.37%** (28.57% under the legacy by-id metric) | 15,567,566 |
| `glassnet-d256-2m-24pct` | **24.20%** | 5,908,686 |

**Context window: 32 tokens.** About a short sentence. Not a paragraph, not a conversation, and not a setting
you can raise. Anyone claiming this model "remembers" a chat is wrong.

## What does not work yet

- **Coherent conversation.** Grammar is real; content is weak.
- **Arithmetic on unseen operands:** 0.0% exact across three implementations (a per-digit probe shows a
  partial rule forming: digit 0 reaches 100% on unseen operands).
- **Attention:** our bare single-head block *lost* 5.41 points (18.69% vs 24.10%). It lacked positions,
  residual and normalisation - so that block was worse than no attention, not attention itself.
- **Our own architecture:** lost to the plain shape on both seeds (3.87/4.07 vs 5.53/5.87). Cause found and
  documented: an unnormalised residual saturating the state.

## This is not a normal model - practical differences

| Normal model | GlassNet |
|---|---|
| `pip install transformers` | `pip install numpy safetensors` |
| billions of parameters | 5.9M or 15.6M parameters |
| thousands of tokens of context | **32 tokens** |
| weights only, no explanation | named concept neurons + `explain()` |
| confidence in chat | 37% next-token accuracy, stated plainly |
| closed training recipe | every run log published, including failures |

## Files in this repository

| File | What it is |
|---|---|
| `model.safetensors` | the weights, standard container |
| `config.json` | architecture + **the measured metrics, including the bad ones** |
| `labels.json` | the concept-name tables (text cannot live inside safetensors) |
| `examples/load_weights.py` | load them with numpy - no framework |
| `examples/run_locally.md` | exact commands to serve this checkpoint |
| `LICENSE` | Blueprint AI License 1.0 |

## Licence: free for most people, paid at scale

Free for individuals, students, educators, researchers and non-commercial use. Free for organisations under
**1,000,000 USD** annual revenue. Above that a commercial licence is required (standard terms: 1% revenue
share). Harmful use is not licensed. **Not OSI-approved open source** - that is deliberate.


## Our organization

All Blueprint AI HQ models, the org card and how to work with us: **https://huggingface.co/blueprint-ai-hq-main**

## Everything else

- Code, tests, all results and all failures: **https://github.com/blueprint-ai-hq-git/glassnet**
- Status and honest percentages: `STATUS.md` in that repository
- How to test us and report a number: `docs/TESTING.md`
- Contact: **blueprintai.help1@gmail.com**

*If a claim here cannot be traced to a run, treat it as a bug and tell us.*
