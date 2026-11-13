# The private copy path failed: worse language than both alternatives, no better copy (2026-11-13)

The successor to freezing, and the only version that would not trade. Freezing the copy path on language batches protects held-out
copy by about 2 points (+1.98 over five seeds, +2.00 over three from a second starting checkpoint) but costs **-0.73 points of
language**, because language is forbidden from updating the pointer matrices. So the copy path was given its **own** pointer
matrices (`ptrW`, `Wq`, `Wk`), used only on copy batches; language keeps and updates the shared ones and is forbidden nothing.

Its mechanism was verified before the run: after 80 language-only batches the private tensors show max abs drift **0.000e+00** while
the shared ones moved (5.52e-03, 3.88e-03, 4.06e-03). Language provably cannot touch them.

## Result - prediction failed on language, falsifier triggered
```
arm            held-out copy (3 seeds)      language (3 seeds)     taught
none           73.9, 75.9, 75.7 -> 74.02    16.43,16.57,16.48 -> 16.30   84.1
private        71.9, 76.1, 75.0 -> 74.33    12.93,12.65,13.00 -> 12.86   84.3
frozen         75.97 (5 seeds)              15.56 (5 seeds)              84.3
```

Prediction was "held-out copy >= 74 **and** language >= 16.1".
- **Copy half passed, barely:** 74.33 against the 74.0 bar - and that is statistically the same as doing nothing at all (74.02).
- **Language half failed by 3.24 points**, and the falsifier written before the run said "if language falls below 15.8 the private
  path still costs language and the trade has not been removed". **It triggered: the private path is 3.44 points WORSE than doing
  nothing, and 2.70 points worse than freezing.**

So the private copy path is not a smaller trade - it is a **worse outcome on both counts** than the freeze it was meant to replace.
The freeze remains the best configuration in this line.

## My mechanism hypothesis was wrong, and the weights say so
The obvious story was "the copy gradient has to go somewhere, so it is redirected into the shared components language needs". The
saved weights refute it. Relative drift from the shared start, `keep_none` (nothing frozen) against `priv_copy` (private pointer):

```
tensor       drift none   drift priv   difference
ptrW            0.1890      0.1053       -0.0837
Wk              0.1017      0.0511       -0.0506
Wq              0.0775      0.0511       -0.0264
emb             0.0315      0.0302       -0.0012
concept.W       0.0319      0.0313       -0.0006
answer.W        0.0019      0.0018       -0.0001
dmix            0.0000      0.0000       +0.0000
```

The private arm moves the shared pointer **less** (as designed) and moves the shared components by the **same** amount to four
decimals. There is no redirected interference to point at. **The cause of the 3.4-point language loss is therefore unknown**, and this
document does not invent one - what is established is the outcome, not the mechanism.

## What this line has now measured
```
configuration                held-out copy   language   verdict
nothing frozen                 74.02          16.30     baseline
frozen on language batches     75.97          15.56     +2 copy for -0.73 language  <- best
own pointer parameters         74.33          12.86     worse on both counts
```

## Infrastructure notes from this run (both real, both fixed)
1. **My arms sat idle ~25 minutes** because the script waited on `systemctl --user is-active glassnet-keep3`. A unit stays
   `active` while its chain waits for the agent to exit, *not* while training runs, so the guard could never clear. Scripts now wait
   on `pgrep -f train_copy_task.py`. The finished keep3 unit was stopped by hand; its logs and checkpoints are all kept and measured.
2. **`unexpected EOF while looking for matching quote` at line 38** was caused by me **editing the script while bash was executing
   it** - bash reads a script incrementally, so a mid-run edit corrupts the rest of the read. `bash -n` was clean and all three arms
   completed; the error was pure self-inflicted. Never edit a running script; write a new one.

## Reproduction
`lens/scripts/run_priv_arms.sh` (prediction and falsifiers in the header), logs `logs/priv/priv_s{0,1,2}.log`, checkpoints
`priv_copy_s*` (with `preserve` tensors `ptrW_priv`, `Wq_priv`, `Wk_priv`), drift table from `tools/`-style numpy comparison against
`early_s0_ep1000.npz`.
