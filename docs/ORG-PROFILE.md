# Blueprint AI HQ

**We build models you can look inside.**

GlassNet is our first project: a language model written entirely from scratch - own tensors, own backward
pass, own optimiser, own KV cache. No PyTorch, no transformers, no pretrained weights, nothing borrowed.

## Why we exist

Most small models are black boxes: you give them text, you get text back, and nothing between those two
points is inspectable. We think that is the wrong direction. Our models carry **named concept neurons** so a
person can read *why* an answer came out, publish **every failed experiment** with its number, and state the
real accuracy instead of implying more.

## Our rules, applied to every release

1. **No number without a run behind it.** Every claim names the file it came from.
2. **Failed experiments are published.** Our own architecture lost to the simple one and that is on the record.
3. **If a claim cannot be audited, it is dropped** rather than softened.
4. **No ML framework.** NumPy is the array backend; everything above it is ours.
5. **We say how bad it is.** GlassNet scores 37.37% held-out. It is not an assistant yet.

## Models

| Model | Held-out accuracy | Parameters |
|---|---:|---:|
| [glassnet-d512-ext6-37pct](https://huggingface.co/blueprint-ai-hq-main/glassnet-d512-ext6-37pct) | 37.37% | 15.6M |
| [glassnet-d256-2m-24pct](https://huggingface.co/blueprint-ai-hq-main/glassnet-d256-2m-24pct) | 24.20% | 5.9M |

## Code and results

**https://github.com/blueprint-ai-hq-git/glassnet** - engine, model, tests, charts, every run log including
the failures.

## Want to help?

The most useful thing you can send is **a number we can reproduce** - including when it is worse than ours,
because we publish it next to our own. Ideas, disagreements and failed experiments are all welcome.

**Contact:** blueprintai.help1@gmail.com
