# Where the punctuation loop actually comes from (2026-11-10)

The model is open, so this is not a guess. It has two logit paths and they can be separated exactly:

```
concept path : logits_c = sigmoid(concept(x)) @ answer.W + answer.b     (36 readable numbers)
copy path    : logits_d = relu(x_flat @ dmix) @ emb^T                   (position-aware echo of the window)
total        : logits   = logits_c + logits_d
```

**Prediction, recorded before measuring:** the loop comes mostly from the copy path, not the concept layer,
because the window already contains the repeated symbol and the tied readout lets the model echo it cheaply.

## Result: prediction correct, and it is not close

Trace of `the capital of france is`, 14 generated tokens, symbol probability attributed per path:

```
st  token   p(sym) total   concept path   copy path   top1 total   top1 copy
0   'the'       0.071        0.008         0.041      'the'        'the'
1   '`'         0.028        0.008         0.021      '`'          '`'
2   '`'         0.104        0.008         0.065      '`'          '`'
3   '`'         0.972        0.008         0.965      '`'          '`'     <-- locks in
4   '`'         0.339        0.007         0.314      '`'          '`'
5   '`'         0.428        0.007         0.398      '`'          '`'
...
13  '`'         0.307        0.007         0.320      '`'          '`'
```

The concept path's symbol probability never moves: **0.007-0.008 for all 14 steps.** The copy path goes from
0.021 to **0.965** and carries essentially the entire loop.

**Why it never stops:** each emitted backtick is appended to the window, and the copy path's job is to echo what
is in the window. So every step feeds the next one. It is a self-reinforcing echo, not a preference for
punctuation, and it explains why the decode guard could not fix it: banning one repeated symbol leaves the copy
path free to echo any other symbol in the window, which is exactly the 46.5% symbol share the guard still showed.

## The second finding: the readable concepts have no clean meaning
Every one of the 36 concepts has a symbol weight of essentially **0%** of its total weight, so none of them is a
"punctuation concept". But their top-voted tokens are not clean either:

```
c0  'might' 'pound' 'german' 'bills' 'flood' 'luck'
c5  'bb' 'maze' 'donation' 'reside' 'pulse' 'cattle'
c14 'vibe' 'cosmic' 'account' 'gum' 'beach' 'donate'
c30 'stove' '94' 'labels' 'fog' 'tasks' '2004'
```

That is a scatter of unrelated words, not a meaning a person could name. So the concept layer is not the cause of
the loop, and it is also not yet the readable middle layer it was supposed to be. Both facts belong on the
record together.

## What this changes
1. **The fix has a target now.** The copy path is where symbols come from, so damping it is a specific edit with a
   specific measurement, not a search: penalise copy-path logits on pure-symbol tokens during training while
   leaving the concept path free to choose punctuation when it genuinely wants it.
2. **The measurement is defined before the change:** the copy path's symbol mass should collapse and generated
   text should stop looping without a decode guard. If symbol mass only moves a little, the edit failed.
3. **The concept layer needs its own work**, separately and later, because a layer whose neurons vote for
   incoherent word lists cannot explain anything to a person even if accuracy is fine.
