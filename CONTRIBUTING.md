# Contributing to GlassNet

Two rules matter more than style:

1. **One change, one measurement.** Change a single thing, report what it did to the number.
2. **Failed experiments get published.** A negative result with a clean measurement is a real contribution.

## Getting started

```bash
git clone https://github.com/blueprint-ai-hq-git/glassnet && cd glassnet
python -m venv .venv && . .venv/bin/activate
pip install numpy==1.26.4 matplotlib   # CPU only: the tested pin, gradcheck runs on it
# GPU (cupy-cuda12x 14.2.0): it declares numpy<2.6,>=2.0, so the GPU venv needs
#   pip install "numpy>=2.0,<2.6" cupy-cuda12x==14.2.0
# with numpy 1.26.4 installed, `import cupy` fails: numpy.core.multiarray failed to import
python tests/test_gradcheck.py     # must be all PASS before anything else
```

## What we need most

| Need | Detail |
|---|---|
| **Independent reproductions** | run the baseline on your hardware, report the number even if it is lower |
| **Readability tests** | run `explain()`, tell us when the stated reason is wrong - that is a bug we care about |
| **Negative results** | tried an idea that did not work? That is publishable here |
| **Docs fixes** | anything confusing is a bug in the docs, not in you |
| **Small-model efficiency** | runs must fit in 5 GB VRAM and not freeze the user's desktop |

## Ground rules

- No pretrained weights, no PyTorch `nn`/`autograd`, no borrowed training code. The whole point is scratch.
- Every gradient you add gets a numerical check before it is trusted.
- Do not commit datasets, checkpoints (`.npz`/`.npy`/`.safetensors`), or secrets.
- Do not soften a number. If it is bad, say it is bad - that is the house style.

## Licence and CLA

Contributions are under the [Blueprint AI License 1.0](LICENSE) and require the [CLA](CLA.md), which keeps the
project able to offer commercial licensing. You keep copyright of your own work.

## Sign-off

Add a line to your commit:

```
Signed-off-by: Your Name <you@example.com>
```
