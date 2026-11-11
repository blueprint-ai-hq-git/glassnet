# The pointer keeps its aim and loses the vote (2026-11-11)

Six mixing variants dropped the copy pathway from 58.0% held-out copying to between 0.2% and 16.5%, and none of them
said WHERE the loss happened. This tool separates the two possibilities on the trainer's own items:

- **AIM**: does the pointer still put its probability mass on the window position that holds the word to copy?
- **VOTE**: does the assembled model actually output that word?

Prediction recorded before the first measurement: aim would survive mixing and the loss would be in the vote. Measured
with `tools/pointer_alpha_mixed.py` on 600 taught and 600 held-out items dumped by the trainer itself (`--dump-items`):

```
checkpoint            items    AIM mass on answer slot   pointer alone   assembled   chance
copy_ponly (pointer)  taught        0.7979                   83.8%          93.8%      0.0312
copy_ponly (pointer)  held-out      0.7706                   58.0%           0.5%      0.0312
ampgate               taught        0.6880                   83.8%          96.0%      0.0312
ampgate               held-out      0.6575                   63.0%          14.8%      0.0312
```

**Confirmed.** On held-out words the pointer's attention still lands on the right slot with 0.77 of its mass against
0.031 for chance, and its own answer is right 58-63% of the time. The assembled model outputs it 0.5-14.8% of the time.
The copy operation is intact; the mixture outvotes it.

## What this rules out
- **It is not erosion.** The corpus does not teach the pointer to stop looking at the right word. Aim only falls from
  0.798 to 0.658, and its standalone accuracy actually improves from 58.0% to 63.0% during mixing.
- **It is not a missing operation.** Standalone accuracy and oracle tests already showed the mechanism works.
- **It is not curable by freezing the wrong thing.** Aim was never the thing being lost.

## What it points at
The answer is a VOTING problem. Held-out words give the pointer a correct aim and, inside the assembled model, a
minority opinion. Whatever decides the final token is choosing the concept path's answer. The fix has to make the
pointer's vote count when its aim is sharp, which is a routing decision, not a training-length or learning-rate
question. Every arm that tried to fix this by training harder (24,000 steps), freezing, isolating the embedding table or
lowering the learning rate was addressing the wrong half.

## Tool note (a mistake worth recording)
The first version of this tool built its own copy items instead of using the trainer's. It reported 0.0% on a model that
scores 58.0% in its own harness, because my hand-built items had no position cue - it was measuring the tool, not the
model. The trainer now dumps its exact items (`--dump-items`) and every diagnostic consumes those. Diagnose the model
with the model's own data, never with a reimplementation of it.
