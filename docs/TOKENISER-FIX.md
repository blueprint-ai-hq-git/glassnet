# The tokeniser was deleting every space, and the model could not say so

*A defect that made readable English impossible, found by decoding the training corpus instead of trusting the pipeline.*

## What was wrong

The word tokeniser's pattern had no alternative that matched whitespace:

```python
TOKEN_RE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")
```

Text went in, tokens came out, and every space between words was silently discarded. A held-out conversation decoded as:

```
human:howcanistorefoodifidon'thaveapantry?assistant:youcouldstorethefoodinarefrigerator
```

Measured on the prepared corpus: **0 of 12,000 vocabulary entries contained a space**, and a decoded sample contained 0 spaces.

This is not a training problem. The model had **no space character in its alphabet**, so it could never emit a word boundary, no matter how long it trained, how wide it was, or how much data it saw. Every earlier judgement about "the model writes word salad" was downstream of a tokeniser that could not write a sentence.

## First fix: keep the space

A tokeniser that attaches a leading space to each word and punctuation token (GPT-2 style) and keeps newlines:

```python
TOKEN_RE_SP = re.compile(r"\n| ?[A-Za-z']+| ?[0-9]+| ?[^\sA-Za-z0-9]")
```

Same corpus now decodes as text a person can read:

```
human: how can i store food if i don't have a pantry?
assistant: you could store the food in a refrigerator, the top cupboards in your kitchen, ...
```

`11,158 of 12,000` vocabulary entries now carry a space. The switch is off by default, so every earlier corpus and checkpoint reproduces exactly.

## Second defect: the model used `<unk>` as an "I do not know" sink

A fixed 12,000-word vocabulary cannot cover English. On this corpus **2.28% of training tokens were `<unk>`, and 33.3% of the tokens the model GENERATED were `<unk>`** — roughly fourteen times its own training rate. It had learned the placeholder as a safe fallback for uncertainty.

A byte-level BPE has no unknown token in principle: every byte is representable. Ours is trained from scratch on the project's own text, 6,000 merges, **6,256 symbols**, and it round-trips exactly on both training and held-out text (`decode(encode(x)) == x`, byte for byte).

| | 12k word tokeniser | 6,256-symbol BPE |
|---|---|---|
| `<unk>` in training text | 2.28% | **0.000%** |
| `<unk>` in generated text | 33.3% | **0.0%** |
| repeat-rate, 5-token shingles | 0.0% | 0.0% |
| held-out bits per character | 1.793 | **1.581 ± 0.005** |
| held-out assistant accuracy | 21.49-24.21% | 41.37-41.74% |

Bits per character is quoted on purpose: per-token accuracy flatters a smaller vocabulary, so a like-for-like measure is the only honest comparison. **1.581 ± 0.005 over three seeds (1.587 / 1.581 / 1.577) against 1.793** — an **11.8% improvement** at the same width, window and epoch count.

This does not overturn the project's earlier finding that BPE costs about 3.1% more tokens on our own text, measured on token efficiency. That finding stands. It was never a measurement of coverage, and coverage was the defect.

## An instrument fault found on the way, and a gate that now gates

The assistant-mask walk matched whole tokens (`"assistant"`, `":"`). A byte-level tokeniser splits that word, so the mask found **0 of 129,112 training targets** and the run trained on nothing while reporting `loss 0.0000, acc 0.00%` — a void arm wearing the shape of a result.

Two fixes: a text-span fallback that locates the speaker markers in the **decoded** text and maps them back to token spans through cumulative token lengths (works for any tokeniser that round-trips), and an empty mask now **raises and refuses to train** instead of printing a warning and continuing. A gate that cannot fail is worse than no gate.

## Longer training than the old recipe allowed

The six-epoch setting had been tuned on the word corpus, where validation perplexity bottomed at epoch 6 and then rose. On the BPE corpus validation was still falling at the last epoch, so six was truncating it. Twelve epochs: assistant perplexity **52.17**, accuracy **50.18%**, bits per character **1.467** (one seed, replication running).

## What is still wrong

The model writes short English fragments with real words and correct spacing, e.g. *"often to do your people in find to or such as your own how like my fo. For example, to do"*. Grammar is rough and sentences do not hold together. It is not fluent and this document does not claim it is.
