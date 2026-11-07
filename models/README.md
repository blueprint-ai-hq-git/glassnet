# Published models

These are the checkpoints we have actually published, with the number each one measured. Every
figure traces to a log line in `docs/history/`. **All of them are experimental: none can hold a
conversation, and none should be used in a product.** The context window is 32 tokens.

| model | architecture | held-out next-token accuracy | safetensors sha256 |
|---|---|---|---|
| [`glassnet-d512-ext6-37pct`](https://huggingface.co/blueprint-ai-hq-main/glassnet-d512-ext6-37pct) | d=512, 15,567,566 params | 37.37% by position / 28.57% legacy by-id | `55b80baebd7c57584e5aed6cdb240f913a76332de3848c479177688b202b853d` |
| [`glassnet-d256-2m-24pct`](https://huggingface.co/blueprint-ai-hq-main/glassnet-d256-2m-24pct) | d=256, 5,908,686 params | 24.20% | `794636c8831092dc967f83ee5c16c683f3502f37191649405801e02e16d8233d` |

## How to load one without PyTorch

```bash
pip install numpy safetensors
python examples/inference_numpy.py --model blueprint-ai-hq-main/glassnet-d512-ext6-37pct
```

The loader refuses a checkpoint whose vocabulary does not match its `labels.json`, because a silent
mismatch decodes every id to the wrong word. That happened once and is fixed.

## What the accuracy number means

It is next-token accuracy on held-out text: given 32 tokens, how often the correct next token is the
model's top pick. Random guessing over a 12,002-word vocabulary is about 0.008%. It is not
conversational ability, and we do not report it as such.

## Not published here

Training-ladder checkpoints from the data-scale experiments (500k / 2M / 4M / 8M windows) stay local.
Their numbers are in `docs/history/NOGATE-*-RESULT.md`, and the best d256 rung reached 24.51% held-out
at 3,811,534 parameters. They will be uploaded only if a person can do something useful with them.
