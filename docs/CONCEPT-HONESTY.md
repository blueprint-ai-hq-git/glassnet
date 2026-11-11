# Concept honesty, measured honestly: 64.9%, not 79.2% (2026-11-11)

The Phase 2 gate reported concept honesty at 79.2%. That number was inflated by class imbalance and this file corrects it.

## The trap, stated plainly
The median named concept is true in only **16.3%** of held-out windows (min 7.0%, max 76.8%). So a neuron that always
answers "no" scores **83.7%** plain accuracy across this concept set - higher than the 79.2% the model actually achieved.
Plain accuracy on a rare concept measures the class balance, not the neuron.

## The honest metric, and what it says
Balanced accuracy = mean(recall on positives, recall on negatives). A dead neuron scores 50 and cannot be gamed.

```
CONCEPT BAR : always answering NO would score 75.6% plain accuracy; a dead neuron scores 50.0% balanced
WITH rules  : balanced concept honesty 64.9% mean over 36 concepts
worst five  : 'as' 50.3%, '3' 50.7%, '*' 51.0%, 'that' 51.2%, 'is' 51.9%
```

**64.9% balanced honesty, and five concepts are effectively dead.** The prediction recorded before this run was
"balanced honesty lands between 50% and 75%" - it landed at 64.9%, so the prediction passed while the concepts failed.

## What did NOT change
Token accuracy is identical at **19.68%**, the ablation at **20.43%**, and the readability price still **0.75 points**.
Only the honesty verdict changed, because only the honesty metric was wrong. That is the useful part: the earlier
conclusion "the concepts are 79% honest" was an artefact of the metric, and the mechanism was never that good.

## Why the concepts are weak
Every concept is "this window contains token X", a bag-of-words property, while the layer that must predict the next token
is fed the last position. Concept supervision is also one term beside a token cross-entropy that is several times larger.
Two fixes are queued in order: raise the concept loss weight so the concepts are not drowned (running now, prediction
balanced honesty >= 75% with token accuracy within 1 point), and if that is not enough, replace the concept set with
positional and structural properties ("last token is a digit", "window ends with a question mark") that a next-token rule
can actually use.
