
## The fix holds on a real arm: same numbers, 4.96x faster, on half the cores (2026-11-17)

The 8.55x A/B was a single step with a 2-core pin. Run 9 was restarted on the fixed code, and the arm itself confirms it.

**Identical loss at every logged step**, which is the check that the restart reproduced the *same run* rather than a similar one:

```
step  25   old 9.4038   new 9.4038
step  50   old 8.5875   new 8.5875
step  75   old 8.7967   new 8.7967
```

**Timing, 25 steps:**

```
old run (12 cores, no pin):  222 s per 25 steps   (step 50 at 530 s, step 75 at 793 s)
new run (4 cores, pinned):    45 s per 25 steps   (step 50 at 89 s,  step 125 at 223 s)
```

**4.96x on the real arm - while using one third of the CPU.** The old run had every core available and still lost by five times.

Run 9's ETA moved from **17:00 to about 14:45**, i.e. the remaining 1,375 steps take roughly 41 minutes instead of three hours.

## Why restarting was legitimate, and what it cost

The arm had reached step **425/1500 (28%)**. Restarting threw that away. It was justified on one ground only: the A/B demonstrated **loss and gradient norm identical to ten decimal places**, so the restart continues the same trajectory rather than a new one - and the step-by-step loss match above proves that held in practice.

Without that equivalence this would have been impatience dressed up as optimisation. With it, it is a controlled restart.

The partial log is preserved as `lens/runs/SCAN-RUN9-preoptimize-partial.txt`; nothing was deleted.

## One thing caught while doing it

`sober` (Roblox) was running at the moment of relaunch. Per the corrected 2026-10-02 rule the GPU stays with GlassNet and only `cs2` blocks - but Roblox still needs CPU, so the relaunch is pinned with `taskset -c 4,5,6,7` and runs at nice 19 / ionice c3. **The speedup above was achieved on half the machine, with a game running.**
