# Bench v0: the first real benchmark, and the honest baseline is zero (2026-11-11)

Six tasks, 200 generated items each, every answer a single token that exists in the vocabulary, scored with the chat
template the model was trained on, with `<unk>` allowed and banned, plus a frequency baseline.

```
task      chat   chat-<unk>-banned   raw    freq    what the model actually answers
add       0.0%          0.0%        0.0%   0.0%    "what is 8 + 6 ?"            -> 'the'
sub       0.0%          0.0%        0.0%   0.0%    "what is 6 - 0 ?"            -> 'the'
seq       0.0%          0.0%        0.0%   0.0%    "2 , 4 , 6 , 8 ,"            -> 'the'
copy      0.0%          0.0%        0.0%   0.0%    "the secret word is ... is"  -> 'the'
yesno     0.0%          0.0%        0.0%   0.0%    "is 0 greater than 1 ?"      -> 'the'
recall    0.0%          0.0%        0.0%   0.0%    "the capital of russia is"   -> 'the'
OVERALL   0.0%  over 1,155 scorable items
```

**The model scores zero on every task.** It cannot add 8 + 6, cannot continue 2, 4, 6, 8, cannot repeat a word that is
sitting in its own window, cannot answer yes or no, and knows no capitals. Its answer to everything is `the`, and with
`<unk>` unbanned it is `<unk>`.

## Two flaws in this harness that had to be fixed first
1. **World knowledge scored 22.5%** on the first run. It was fake: the prompt was built from the wrong variable
   ("the capital of moscowrussia is") and, worse, any expected answer missing from the vocabulary was scored as `<unk>`
   so answering "I do not know" counted as correct. Items whose answer is not in the vocabulary are now dropped and
   counted (45 of 200 for world knowledge).
2. **`<unk>` was never banned in earlier generation tests**, which hid the fact that on unfamiliar input the model's
   true first choice is "I do not know". The benchmark now reports both.

## Why this baseline matters more than any accuracy number published so far
Every figure this project has reported until today was next-token accuracy on its own held-out split: 25.52%. That
number was never comparable to anything, and it concealed the fact that the model cannot perform a single checkable
task. A next-word predictor at 25% is not a weak assistant; on this evidence it is not an assistant at all.

This also puts Yahya's target in measurable terms: being better than an average model at 512 on every benchmark starts
from **zero on all six**, and the distance is now visible instead of assumed.

## What the suite is not
It is small and hand-built: 1,155 items, single-token answers, six abilities. It says nothing about conversation,
reasoning depth, code generation or long context. It is a floor, not a ceiling, and it is deliberately honest about
being one.
