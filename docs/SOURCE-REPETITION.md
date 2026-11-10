# Which part of the corpus teaches the model to repeat itself (2026-11-10)

`COPY-PENALTY-VERDICT.md` established that the true next token is already visible in the window 40.33% of the time,
which makes copying correct two times in five. This measures which source produces that, per source corpus, on each
source's own held-out conversations (`tools/source_repetition.py`, 3,000 sampled conversations per source):

```
source                    tokens   convs   in-win   modal   symbols   mean len
prepared_ext_honest        38.8M    156K   38.32%   4.50%    17.28%       266
  conversational: hh + oasst1 + ultrachat + orca_math
prepared_kn2               255.2M   584K   51.75%   7.94%    30.25%       444
  knowledge/code: metamath, cosmo_math, hendrycks, openr1, textbooks,
  gutenberg, magicoder, codefeedback, opencode

in-win = share of targets already present in the window
modal  = accuracy of always predicting the most frequent token in the window
```

## Two findings, and a correction to my own account

1. **The code/math/books source repeats itself far more**: 51.75% of its targets are already visible, against
   38.32% for conversation, and it is also more symbol-dense (30.25% against 17.28%).
2. **That source is 87% of the training corpus.** 255.2M of 294.2M tokens are math, code and books; only 38.8M
   are conversation.

**Correction:** earlier today I wrote that "the model learns prose and is graded on code". The corpus is not
mostly prose - it is mostly math, code and books, by a factor of nearly seven. I based that claim on decoding an
early window of the corpus and let it stand for the whole thing, which is the same mistake as the split magnitude
error: measure a sample, describe the whole.

So the looping is not a mysterious pathology and not a fixable loss term. The model spends most of its life on text
where the answer is frequently visible in the window, and it behaves accordingly.

## The arm this launched
`lens/scripts/run_talk_ext_d512.sh` trains the same architecture, same recipe, same seed on the conversational
portion only (38.8M tokens, 156,520 conversations), on its own matched holdout (1,590 validation conversations,
1.016%, composition gap -0.24 pp of symbol share). One change: the data. Prediction, recorded before the result:
if the repetition is mostly a corpus property, this arm should show lower in-window copying and less looping than
the merged-corpus arms at similar accuracy; if it does not, my account of the cause is wrong again and the next
place to look is the architecture's own feedback path.

## Incidental finding
The conversational corpus ships a split array with `len(bounds)` entries while the merged corpus ships
`len(bounds) - 1`. Both work because the trainer indexes only the first `len(bounds) - 1` positions, but a length
check written against `len(bounds) - 1` fails on the first corpus and one written against the shipped length fails
on the second. Recorded so the next tool does not have to rediscover it.
