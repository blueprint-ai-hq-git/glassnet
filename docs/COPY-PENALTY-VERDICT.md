# The copy penalty failed, and the reason reframes the whole problem (2026-11-10)

## The arm: three epochs, full result
```
                 baseline          copy penalty      change
epoch 1 acc       19.69%             21.75%          +2.06 pp
epoch 2 acc       24.26%             24.24%          -0.02 pp
epoch 3 acc       25.52%             25.60%          +0.08 pp
```
Final checkpoints, measured with `tools/copy_echo.py`:
```
                                   baseline     penalised
emitted token already in window      74.3%        79.5%
emitted token == previous token       0.0%         0.0%
copy path mass on visible tokens     30.8%        29.1%
distinct 4-grams in generation       54.0%        49.2%
```

**Verdict: the penalty did not work.** The epoch-1 advantage (accuracy +2.06 pp, copy mass -7.2 pp) washed out by
epoch 3 to +0.08 pp and -1.7 pp, the previous-token echo is 0.0% for both, and generated text is slightly MORE
repetitive than the baseline's. This is the third lever today to fail (symbol mask: paid 2.21 pp and relocated the
loop; decode guard: removed repetition and left the appetite; copy penalty: transient gain, no lasting change).

## Why - measured, not guessed
PREDICTION before measuring: the copy path would be capacity-limited, mostly dead ReLU units, so a mass penalty
would have little leverage. **Falsified:** 62.9% of the 512 copy units are alive, and the top 10 units hold only
8.7% of the copy-path logit energy. The units are alive and evenly spread, so the penalty is not fighting a
capacity limit - **it is fighting the cross-entropy objective.**

That led to the measurement that matters:
```
matched validation windows, 4000
true next token is already present in the window   40.33%
always predict the modal token in the window        6.62%
always predict the previous token                   2.75%
always predict the globally most frequent token     4.11%
the readable model                                 25.60%
```

Two consequences, and both change the plan:

1. **Copying is often the correct answer.** In 40.33% of windows the right token is already visible, so a penalty on
   the copy path is a penalty on being right about two targets in five. That is why it was transient and why it
   bought no accuracy.
2. **The model is not a copier.** Every trivial copy baseline is far below it (6.62% and 2.75% against 25.60%), so
   the 25.60% is genuine modelling, not echo. What looked like a copy pathology in generation is mostly the data:
   this corpus repeats itself 40% of the time, and an autoregressive model trained on that will loop, because
   looping is what the next token distribution actually says.

## What this rules out and what it points to
- **Ruled out:** penalising copy mass; banning token classes; decode-time suppression. All three treat a symptom.
- **Pointing at the corpus:** the repetition rate of the training text is the variable that sets how repetitive
  generation can be. The next honest measurement is per-source self-repetition - which sources push the 40% - and
  the arm after that trains or generates on the low-repetition portion.
- **A correction to my own earlier framing:** I described the model as a copy machine that got worse with training.
  The measured truth is that it outperforms every copy baseline by a wide margin and lives in a corpus that repeats
  itself 40% of the time. The echo numbers were real; the interpretation I put on them was too simple.
