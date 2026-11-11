# Copying from the window: 17.5% on taught words, 0.0% on held-out words (2026-11-11)

The first benchmark task forced to a score. 3,000 steps, lr 0.3, batch 256, from the 25.52% checkpoint, on synthetic
in-window copy windows in the chat template. Half the vocabulary is never shown in these windows, so the run can show
whether the model learns the ACT of copying or just the LIST of answers.

```
step     loss     copy accuracy, taught words    copy accuracy, held-out words
 500    6.864              0.0%                          0.0%
1000    6.328              0.3%                          0.0%
1500    5.770              0.8%                          0.0%
2000    4.915              4.2%                          0.0%
2500    3.270             12.5%                          0.0%
3000    1.735             17.5%                          0.0%
```

## Both predictions failed, and the failure is informative
- **C1 (taught words >= 50%) FAILED** at 17.5%.
- **C2 (held-out words >= 25%) FAILED** at 0.0%, and it was 0.0% at every single checkpoint.

The loss fell from 6.86 to 1.73 while held-out accuracy never once left zero. That combination is the signature of
**fitting the answer set, not learning the operation**. If the model had learned to copy, a word it had never been
trained on would still be sitting in its window and would still be copyable. It is not copyable, because nothing in the
network can reach back and take one position out of the input.

## Why this is structural rather than a tuning problem
The architecture flattens the 32-token window and passes it through a gated mix (`relu(x_flat @ dmix) @ emb.T`). There
is no attention, no per-position addressing, no pointer. Copying requires exactly that: choose a position, take what is
there. Without it the readout must approximate a lookup by superposition, which is what the 17.5% looks like.

This single defect explains the whole benchmark table: copying, arithmetic, sequencing, yes/no and retrieval all need
the model to hold and address specific things in its input, and none of them can work by bag-of-window mixing.

## What changes next
The next arm adds an EXPLICIT POINTER: a learned scoring of every position in the window followed by a copy of the
chosen position's embedding into the readout, with the pointer weights readable (so `explain()` can say "I copied
position 6"). That is a small, honest architectural addition rather than more training, and it is exactly what the
readable-architecture goal needs anyway.

Prediction for that arm, recorded before it exists: taught-word copying rises above 60% and held-out-word copying
above 30% at the same 3,000 steps, because the operation becomes representable instead of approximated.
