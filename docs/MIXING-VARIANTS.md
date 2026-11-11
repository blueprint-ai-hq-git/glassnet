# Mixing a copy pathway into a language model: six variants, measured (2026-11-11)

All arms: d512, matched split, 90% or 50% corpus batches, the readable pointer as the copy route, evaluated on identical
held-out items. The question throughout: the pointer copies words it has never seen at **58.0%** on its own, so what
happens when a language objective shares the model?

```
variant                                              language   copy taught   copy held-out
pointer only (no corpus at all)                          -         95.0%         58.0%
pointer + 50/50 corpus mixture                         29.8%       85.7%          4.2%
pointer + 50/50, 24,000 steps                          28.9%      100.0%          8.5%   (plateau)
pointer + corpus, pointer matrices FREEZEN              29.2%       84.5%          0.2%
pointer + corpus + bounded confidence gate             29.3%       95.3%          3.8%
pointer + corpus + UNBOUNDED gate (1 + K*sigmoid)      29.8%       96.0%         14.8%   (peak 16.5%)
pointer + corpus + unbounded gate + SEPARATE emb table  29.3%       93.3%          3.3%   (at 4k steps)
```

## What is established
1. **The pointer genuinely generalises on its own**: 58.0% on words it never saw, 97.0% with an oracle pointer, and it
   puts its weight on the right position 85.7% of the time.
2. **A language objective costs most of that generality.** Every mixture lands between 0.2% and 16.5%.
3. **A bounded gate cannot help; an unbounded one can.** 0.2% frozen -> 3.8% bounded -> 14.8% unbounded. The reason is
   arithmetic, not tuning: the pointer's logits are smaller in magnitude than the competing pathway's, so a factor in
   [0,1] can only attenuate a voice that is already too quiet.
4. **Freezing the pointer's own matrices does not protect it** (0.2%), because it reads the SHARED embedding table and
   corpus batches keep moving that table underneath it. Freezing the pointer's matrices was never the same as freezing
   the pointer.
5. **Giving the pointer its own embedding table did NOT help either** (3.3% at 4,000 steps against 14.8% with a shared
   table). This is the one result that contradicts the prediction I recorded before running it, and it suggests the copy
   path benefits from sharing a space with the language path rather than being isolated from it.

## Best configuration so far
Language **29.77%** (against a 24.27% starting point on the same plain next-token ruler) together with **96.0%** taught
copying and **14.8%** held-out copying. That is a model that talks and copies, but it copies unseen words one time in
seven rather than the six in ten the pathway manages alone.

## What the next arm should be, and why it is not another gate
Gates, freezes and separate tables have all been tried. The remaining untried lever is the ORDER of learning: train the
copy operation to convergence first, then introduce language at a LOW learning rate so the copy operation is bent rather
than overwritten. That is different from freezing (which does not protect anything, as measured) and different from a
mixture from step one.

Prediction for that arm, recorded before it exists: held-out copying 30% or better with language at or above 25%.
