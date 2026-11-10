# The punctuation is cured; the loop is not (2026-11-10)

The copy-path symbol mask was built on measured evidence: the punctuation loop comes from the copy path, whose
symbol probability climbed 0.021 -> 0.965 while the 36 concepts stayed at 0.007-0.008 (`SYMBOL-FORENSICS.md`). The
mask forbids the copy path from emitting any of the 62 pure-symbol tokens and leaves that right with the concepts.

12 fixed prompts, 24 greedy tokens each, same harness both sides:

```
                          symbol share   longest symbol run   distinct 4-grams
unmasked (25.52% arm)          8.7%              3                54.0%
masked   (19.19% ep1)          0.0%              0                47.6%
```

```
unmasked  hello , how are you ?   -> by the current of the world of the world of the world of the world of
unmasked  what is your name ?     -> the ` function of the ` function ` function . the ` function ` function is
masked    hello , how are you ?   -> the current of the function of the function of the function of the function is
masked    what is your name ?     -> the current of the function of the function of the function is the number of the
```

## Two results, and the second one matters more
1. **The punctuation is gone.** Symbol share 8.7% -> **0.0%**, longest symbol run 3 -> **0**, on every prompt. The
   mask does precisely what it was designed to do, and it cost 0.50 pp of validation accuracy at epoch 1
   (19.19% against 19.69%).
2. **The loop is not gone.** It moved. Suppress the symbols and the same self-reinforcing echo runs on ordinary
   words: "of the function of the function of the function". Distinct 4-grams actually fell, 54.0% -> 47.6%, so the
   text is *more* repetitive without the punctuation than with it.

So the punctuation was a symptom, not the disease. The disease is that the copy path is self-reinforcing for
whatever is frequent in the window, and removing one class of tokens relocates the loop instead of ending it.

## Also worth recording
The older checkpoint `gpu_d512_s0_best` emitted **67.7%** symbols on the same prompts (`DECODE-GUARD.md`), while this
matched-split model at 8.7% emits far fewer. Same architecture, same mask off, different training length and data.
The appetite for punctuation grows with training, which fits a copy path that gets steadily better at echoing.

## What this rules out and what it points at
- Ruled out: punctuation-specific fixes, and decode-time suppression of any single token class.
- Still standing: the copy path itself is the problem, and the honest test of any future fix is whether
  repetition falls for ALL token classes at once, not whether one symptom disappears.
- The next measurement is therefore on the copy path's self-similarity, not on symbols: how much of each emitted
  token is explained by "it was already in the window", measured before and after any change.
