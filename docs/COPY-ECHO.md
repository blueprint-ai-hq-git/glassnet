# How much of what the model says is something it can already see (2026-11-10)

The symbol mask cured the punctuation and the loop moved to words (`SYMBOL-MASK-VERDICT.md`), so the question is
the echo itself. This measures every emitted token across 12 fixed prompts x 24 greedy steps
(`tools/copy_echo.py`):

```
checkpoint                        in window   == prev   == mode   window mass   unigram expects
matched_d512_s0_best (3 ep)          74.3%      0.0%     25.0%        30.8%            8.6%
matched_mask_d512_s0_best (3 ep)     74.7%      7.3%     28.8%        32.5%            8.5%
gpu_d512_s0_best (6 ep, 37.37%)      80.6%     61.5%     58.3%        48.4%            7.1%
```

PREDICTION, recorded before measuring: emitted tokens would sit in the window far above chance, and the winning
role would be "most frequent token in the window" rather than "the token just before it".

## Partly falsified, and the falsification is the interesting part
The first half holds emphatically: emitted tokens are already in the window **74-81%** of the time, against **7-9%**
by unigram expectation - an 8.6x to 11x over-selection of visible tokens. The model's probability mass on window
tokens is 30.8-48.4% where frequency alone would give 7-9%.

The second half is wrong for the fresh models and right for the well-trained one:
- the 3-epoch matched models echo the immediately-preceding token **0.0%** and **7.3%** of the time, so they are
  not bigram copiers;
- the 6-epoch model that scored the published 37.37% echoes the previous token **61.5%** of the time and the modal
  window token **58.3%** of the time.

**So the echo is not a fixed property of the architecture. It grows with training, and the model that scores best
is the one that copies hardest.** That is consistent with everything measured today: the 6-epoch checkpoint is also
the one that emitted 67.7% symbols, and a copy path that keeps improving at "repeat what you can see" is exactly
how a next-token model converts extra epochs into extra repetition instead of extra understanding.

## The masked arm's final numbers
```
              epoch 1   epoch 2   epoch 3
unmasked       19.69%    24.26%    25.52%
masked         19.19%    22.53%    23.31%
```
The symbol mask costs **2.21 pp** by epoch 3 and removes the punctuation entirely. Both facts are reported together:
it does what it was built for, and it does not address the disease.

## What this points at
1. **A copy-pressure penalty is the next honest lever, not another token-class ban.** The measurable target is the
   `window mass` column: it should fall from 30.8-48.4% toward the 7-9% that frequency justifies, while accuracy is
   watched for its price.
2. **Epoch count is now a variable with a known bad direction.** More epochs raise accuracy AND raise copying;
   the ratio between them is what a "learns rules, not memorises" claim actually depends on.
3. Every future generation result must report `in window` and `window mass` beside accuracy, because accuracy alone
   rewarded the behaviour that produced the loops.
