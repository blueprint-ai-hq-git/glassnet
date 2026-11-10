# The held-out split is a different dataset, not a held-out sample (2026-11-10)

Found while chasing the symbol attractor. The training corpus is a concatenation of sources in order, and the
validation tail is the end of that concatenation - which is a completely different kind of text.

Symbol-token share, measured in 200,000-token blocks:

```
train first 200k   : 13.29% symbols
train middle 200k  : 28.08% symbols
held-out last 200k : 44.04% symbols
```

Decoded samples from the same positions:

```
TRAIN (offset 1000):
  . <|im_start|> user how should i dress for a job interview ? <|im_start|> assistant <unk> , it ' s really hard
  to answer this question ! but i ' ll tell you what , i ' ll tell you my answer , and then i ' ll ask you ...

HELD-OUT (offset n-50000):
  " <unk> the specified amount from the balance if possible . : param amount : a float representing the amount
  to withdraw . " " " if amount > 0 : if self . balance > = amount : self . balance - = amount else : print (
```

## What this means
The model trains mostly on conversational text and is evaluated on **Python code**. The validation split is not
a held-out sample of the training distribution; it is a different dataset that happens to sit at the end of the
file.

That single fact explains the day:

1. **The low held-out accuracy is partly a distribution shift**, not only a learning failure. Training loss
   falls while held-out accuracy stays near the frequency baseline because the held-out tokens come from text
   the model was never shown anything like.
2. **The symbol attractor is the code tail**: 44% of the held-out targets are symbols, and the model responds
   by predicting symbols 68.3% of the time. Measured over-prediction against the corpus-wide share is **1.65x**
   (`tools/symbol_attractor.py`), but a large part of that is the model being asked code-shaped questions after
   being trained on prose.
3. **Every accuracy number measured on this tail is a cross-domain number.** That includes the 4.69% for the
   talk core and the 11.00-31.50% context curve for the readable model. None of them is wrong, but none of them
   is a measure of within-distribution generalisation, which is what they have been read as.

## The fix, in order
1. **Randomised holdout.** Split the corpus by randomly sampled chunks (about 1%), not by taking the last N
   tokens, so train and validation have the same composition. Until that exists, no generalisation claim is
   valid.
2. **Per-source reporting.** Report accuracy separately for the conversational proportion and the code
   proportion, because the two demand different behaviour and an average hides both.
3. **Then re-measure everything**: the talk core, the readable model, and the context-length curve, on the
   matched split. Some numbers will move up, some down; the ones that move tell us which failures were real.

## What I am not claiming
This does not excuse the talk core's 4.69%. That number was measured against a frequency baseline on the same
positions, so the comparison stands as a like-for-like measurement; what changes is the interpretation of the
absolute figure, not the comparison. It also does not change the generation finding: the model still loops on
symbols and still cannot answer a question.
