

---

# FOCUSING ON UNCERTAINTY IS NOT JUST USELESS, IT DESTROYS THE MODEL (2026-11-21 19:57)

The "learn 10x faster" lever was tested as a number, and the number is catastrophic. Recorded in full because a plausible idea that fails
loudly is worth more than a plausible idea left untested.

## Measured, 2 seeds, identical data and task, only the loss weighting differs

```
seed   arm          steps to 90% token acc   final exact held-out
 0     A_uniform            500                    100.00%
 0     B_focus              500                      4.50%
 1     A_uniform            350                    100.00%
 1     B_focus              400                      0.00%
```

- **P1 NOT MET:** no speed-up at all (0.0% and -14.3%).
- **P2 NOT MET, catastrophically:** accuracy fell by **95.50** and **100.00** points.
- **Falsifier fired** on both seeds.

## Why it destroyed the model, which is the useful part

The weight was `(1 - p_max)`: upweight the tokens the model is unsure about, downweight the ones it is confident about. On a carry chain that
is a trap - **the tokens the model has already learned are exactly the ones it is confident about**, so the mechanism systematically
starves the material it had consolidated and chases only the hardest token. The model ends up able to grind the hard column and unable to
reproduce the easy ones, which is why training token accuracy could reach 90% while exact whole-answer accuracy collapsed to 0-4.5%.

That is a real failure mode worth naming: **"focus on what is hard" and "forget what is easy" are the same instruction**, and this
architecture obeys both at once.

## What this establishes about the 10x goal

- **The loss is not where the speed is.** Uncertainty weighting, the most obvious "smart" weighting, is harmful.
- **The baseline is already fast:** uniform training reached 90% token accuracy in **350-500 steps** and then 100.0% exact on held-out
  pairs. A 10x claim against that would mean learning this task in ~35-50 steps, which is a demanding target and has not been approached.
- **What is eliminated:** the last remaining "cheap" ideas for speed are (a) loss weighting - now measured and harmful, (b) more compute
  passes per step - measured earlier today at **0.0 points** (loop R=1/4/8), (c) more data - measured at **+6.11 points** and missing its
  bar, (d) more capacity - measured at **+0.27 +- 0.31**, unstable.

## The next lever, and it is the last obvious one

If not the loss, the compute, the data volume or the capacity, the remaining candidate is **the order and mixture of the data** - a
curriculum. That is the next pre-registration, and its falsifier is written before it runs.
