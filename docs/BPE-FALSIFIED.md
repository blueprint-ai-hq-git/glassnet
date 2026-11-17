
## Built our own BPE from scratch, and it falsified my own prediction: the tokeniser is not what costs us (2026-11-17)

Yesterday's spec predicted that a tokeniser change would buy **0.15-0.35 bits per character** and fixed the falsifier at *"improvement under 0.05 means the tokeniser
was not what was costing us and the effort belongs entirely in architecture."* This is the measurement, and the falsifier fired.

**Built**: `tools/train_bpe.py`, a byte-level BPE written from scratch with incremental pair statistics, trained on 3.0 MB decoded from the **train** split only
(595,697 tokens, 9,747 unique words): **8,000 merges, vocabulary 8,256** (256 byte ids + merges). Encoder and evaluation in `tools/eval_bpe.py`.

```
held-out text: 59,911 word-tokens, 240,804 characters

word-level vocabulary (ours now):  59,911 positions   4.019 chars/token
our own BPE (8k merges):           61,773 positions   3.898 chars/token
                                   -> 3.1% MORE positions for the same text
```

**On our own text the word-level tokeniser is more efficient, not less.** That is the opposite direction to the prediction. The reason is structural: our corpus was
tokenised with this vocabulary by construction, so every word in it is already exactly one token. A BPE has to split them.

## What the tokeniser change is actually for

Not compression - **generalisation**. The two measurements together:

```
in-domain      word-level 4.019 chars/token   vs  BPE 3.898   -> word-level wins by 3.1%
out-of-domain  32% of ordinary words UNREPRESENTABLE (word-level)  vs  0% (byte fallback BPE)
```

So the honest statement is: **on our own distribution the tokeniser is already near-optimal, and on anything outside it the tokeniser fails completely.** The change
buys the ability to encode `zurich`, `kolkata`, `pls`, `tbh` - not fewer bits on text we already cover. Effort for a bits-per-character gain belongs in architecture,
exactly as the falsifier said.

## Two more facts measured while checking

- **Every one of the 12,002 vocabulary ids appears in the corpus** - the vocabulary is fully used, not partly dead.
- **The distribution is extremely skewed**: in a 60M-token sample the top 10 tokens are **26.3%** of all tokens and the top 100 are **60.9%**. That is a
  high-frequency, formulaic corpus, which is consistent with the model reaching 24.58% next-token accuracy while its concept bottleneck holds only ~16 useful
  numbers: much of the signal is a small set of very common patterns.

## Correction recorded

I wrote the spec's step 1 as "tokeniser first, prediction 0.15-0.35 bits/char". The measurement says the prediction was wrong and the falsifier fired. The spec is
corrected here rather than left standing: **the tokeniser moves to a generalisation feature to be adopted when we train on text outside this corpus, not a
performance lever for the current one.** The architecture step (depth, attention, norm, gated MLP, relative position) is the only remaining item on the measured
list, which is where the build order now points.
