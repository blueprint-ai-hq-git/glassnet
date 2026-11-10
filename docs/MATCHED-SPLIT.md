# Matched validation split built, and the size of the original mismatch corrected (2026-11-10)

## What was built
`tools/make_matched_split.py` carves the validation set out of the corpus by hashing conversation ids, so the
holdout is drawn from the same mixture as the training data instead of being the conversations at the end of the
file. It writes `split_matched_v1.npy` (one byte per conversation, 740,665 bytes) and a JSON report, and it never
touches `ids_v12000.npy`, so a running trainer is unaffected.

```
corpus: 294,225,090 tokens, 740,665 conversations

shipped split : 4,501 validation conversations (0.608%), 1,507,128 tokens (0.512% of corpus)
matched split : 7,404 validation conversations (1.000%), 2,882,786 tokens (0.980% of corpus)
                held-out conversations by quarter of the file: 24%, 26%, 23%, 25%
```

`train_chat.py` gained `--split-file`, which replaces the shipped split and **validates its length against the
corpus** so a split cannot be silently applied to the wrong data.

## Measured composition, at window level, on the real corpus
Sampled through the trainer's own `chat_data.make_windows`, 4,000 windows per side, symbol share of target
tokens:

```
SHIPPED tail split : val 31.42%  train 28.78%   gap +2.65 pp
MATCHED split v1   : val 28.52%  train 28.90%   gap -0.38 pp
```

## Correction: +2.65 pp, not +16 pp
`DATA-SPLIT-SHIFT.md` reported symbol share 13.29% in the first 200,000-token block against 44.04% in the last
200,000-token block of the corpus, and framed the split as "the model learns prose and is graded on code".

Those two figures are real, but they describe the **extremes of the file**, not the validation split. The shipped
validation split is 4,501 conversations spread across the corpus, and measured through the trainer's real window
sampler its composition gap is **+2.65 pp**, not +16 pp. The mismatch is real, small enough to be a nuisance
rather than a scandal, and the matched split removes it (**-0.38 pp**).

Earlier I described it in terms the numbers do not support. The direction was right (the holdout is
somewhat more code-like, and it contains genuine Python), the magnitude was overstated by roughly six times.
Both of today's corrections have the same cause: I measured the most extreme blocks and then generalised to the
whole split without measuring the split itself.

## A bug this work caught
The first version of my verification script loaded `chat_data.build()`'s **default** corpus - 13,318,447 tokens,
86,360 conversations - not the 294,225,090-token corpus the models actually train on, because the real corpus is
selected by `CHAT_DATA_DIR`. My 740,665-entry split was silently indexed into an 86,360-conversation corpus and
"matched" beautifully. The `--split-file` length check now makes that failure impossible in the trainer, and the
negative test confirms it: 86,360 against 740,665 is refused.

## Next
Retrain with `--split-file data/chat/prepared_mix_ctrl_honest/split_matched_v1.npy`, then every accuracy in this
repository gets a within-distribution validation number for the first time. Until that arm finishes, the existing
held-out numbers stay labelled as measured on the code-heavy tail.
