# The checking model passes its own gate for the first time (2026-11-13)

The data judge is the filter that is supposed to stop looping, scrambled or junk text from ever reaching training. The previous
version (`docs/DATA-JUDGE.md`) **failed two of its three gates**: false alarm **7.30%** against a 5% bar, and one-token loops
caught only **69.5%** against an 80% bar. This is the rebuild, and it passes all three, on three seeds.

```
                                   old judge            new judge (3 seeds)
AUROC (real vs corrupted)           0.9597               0.9923 +/- 0.0001     (bar >= 0.90)  PASS
false alarm on clean held-out        7.30%              2.59% +/- 0.02%       (bar <= 5%)    PASS
soup caught                         100.0%             100.0% +/- 0.0                       PASS
shuffled order caught                86.9%              80.9% +/- 0.3%                       PASS
one-token loops caught              69.5%               90.0% +/- 0.2%                       PASS
chunk repeats caught                80.0%               92.6% +/- 0.1%                       PASS
unknown flood caught                92.6%               99.8% +/- 0.0                       PASS
```

## What was wrong, and what fixed each failure
1. **The false-alarm promise was never tested.** The old judge chose its threshold on the same split it reported from, so its "5%"
   was not a held-out number. The rebuild uses **three disjoint splits**: fit on train, choose the threshold on a calibration split
   at 3% (inside the bar, leaving margin), and report on a test split the threshold never saw. Result: **2.59%** measured on text
   the threshold never touched.
2. **One-token loops were invisible to a bag-of-token feature set.** New features measure repetition directly: adjacent repeats,
   longest identical run, periodic structure at lags 2-8, longest repeated substring, bigram/trigram saturation, token entropy.
   Loops went 69.5% -> **90.0%**.
3. **Random-token soup was caught 0% in the first rebuild**, because soup sits at the *opposite* extreme from a loop and a single
   linear boundary cannot sit on both sides of clean text. Fixed with **two-sided deviation features**: every feature is also
   expressed as |robust z| against clean text, so damage in either direction scores high. Soup went 0% -> **100%**.
4. **Shuffled text was caught 8.8%** at first: it contains exactly the same tokens as clean text, so every bag-of-token feature is
   unchanged. Fixed with **order-plausibility and coverage features** - mean bigram/unigram log-probability from the training corpus,
   plus the fraction of adjacent pairs and triples that actually occur in real text. Shuffled went 8.8% -> **80.9%**.

## Honest limits of this result
- **Shuffled is only 0.9 points above its bar** (80.9% against 80%). It is stable across seeds (+/- 0.3), so it is not luck, but a
  different corpus or a longer window could push it back under. It is the weakest of the five kinds and should be treated as such.
- **It detects the five damage kinds it was trained on.** A kind of damage not in that list (topically wrong but well-formed text,
  for instance) is not what this measures - the earlier attempt at that failed for a different reason and is documented separately.
- **It is not wired into training yet.** Passing its gate is the precondition; the next step is filtering data with it and measuring
  what that does to the model, which is a separate experiment with its own numbers.

## Method
`tools/loop_judge.py`: 12 repetition/entropy features plus 3 order-coverage features, each also as a two-sided robust deviation
(30 columns); a logistic head fitted with our own gradient descent (no library); threshold fixed on a disjoint calibration split.
Windows come from the real corpus (`data/chat/prepared_mix_ctrl_honest`, 20,000 train / 10,000 calibration / 10,000 test windows,
vocab 12,002). Three seeds change only the corruption sampling.

## Reproduction
`.venv-gpu/bin/python3 tools/loop_judge.py --windows 20000 --seed {0,1,2}`, results in `runs/judge/loop_judge_s*.json`, logs in
`logs/judge/`.
