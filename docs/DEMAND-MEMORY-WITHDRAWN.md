# WITHDRAWN: "demand training creates 16-step memory" is refuted by a frequency split (2026-11-18)

Three entries today claimed that delayed-target training gives the model measurable 16-step recall, with the memory bank worth +3.45 points. **That interpretation does not survive a
control that splits accuracy by the frequency of the target token, and it is withdrawn.**

## The test

If a model recalls *which* token occurred 16 steps back, it must get rare tokens right sometimes. If instead it has merely learned a better prior over frequent tokens, all of its gain will
sit in the most common tokens and it will score **nothing** on the tail. Frequency ranks are computed on the **training region only**, so the split cannot borrow held-out information.

```
delay 4    (250 steps)        top-20    ranks 21-500    tail (888)     tail n
  td4 arm                     35.16%       4.83%         0.00%          888
  base                        19.61%       4.75%         0.00%          888
  same-window constant        24.87%       5.56%         0.56%

delay 16   (1000 steps)       top-20    ranks 21-500    tail (892)     tail n
  td16 arm                    16.15%       1.60%         0.00%          892
  base                        10.09%       2.14%         0.00%          892
  same-window constant        23.65%       6.85%         0.56%
```

## What the numbers say

1. **All of the gain is in the frequent band.** At delay 4: +15.55 points on the top-20 tokens, **+0.08** at ranks 21-500, **0.00** on the tail. At delay 16: +6.06 on top-20,
   **-0.54** at ranks 21-500, **0.00** on the tail.
2. **Rare-token recall is exactly zero at both delays** - 0 correct out of 888 and 892 positions, against a constant predictor that gets 0.56%. One correct answer would have shown as
   0.11%, so this is not a rounding artefact; the model never once recalled a rare token from a delay.
3. **The arm is below a per-window constant predictor in every band** at delay 16 (16.15 vs 23.65, 1.60 vs 6.85, 0.00 vs 0.56), and below it in two of three bands at delay 4.

So what the delayed-target objective changed is **how the model distributes probability mass across frequent tokens** - a readout/prior shift. It did **not** create token-specific memory,
and the earlier "above a fair constant predictor" framing was measuring the prior shift, not recall.

## Why the earlier claims looked positive

They compared **pooled** accuracy against a **pooled** constant baseline. Pooled numbers are dominated by the top-20 tokens, which are 3325 of the 6552 scored positions. In that band the
arm does beat the pooled constant, so the pooled comparison turned positive - while the band where memory would have to show, the tail, sat at zero the whole time. **A pooled comparison
hides exactly the failure this project exists to catch.**

## What still stands from today

- The **memory module** itself holds a fact for 16 steps in isolation: 81.25% recall at 16 steps, control at chance, probe gradient certified at 6.789e-08, one bank per sample. That result
  is untouched by this correction - it is a different measurement with a different control.
- The **bank's contribution** to the delayed-task numbers (+3.45 points at delay 16) is real *as an accuracy difference*, but it is a difference in the same frequency band, so it is not
  evidence of memory either.
- The **infrastructure** stands: `tools/eval_delayed.py`, `eval_delayed_ci.py` (window-level bootstrap) and now `eval_delayed_split.py`.

## The lesson, stated as a rule

**A pooled accuracy gain is not evidence of the capability the task is named after.** Any claim about recall must be split by target frequency first, because the frequent band is large
enough to carry a pooled number on its own. This control should have been run before the first publication, not after three.


---

# DOSE-RESPONSE ENDPOINT: 47.30% pooled, and still ZERO rare-token recall (2026-11-18)

The delay-4 arm finished its 1000 steps. Frequency-split, ranks from the training region only:

```
delay 4 endpoint        top-20    ranks 21-500    tail (888)    pooled
  td4_r090_ep7000       81.74%       16.29%         0.00%       47.30%
  base (no demand)      19.61%        4.75%         0.00%       11.65%
  same-window constant  24.87%        5.56%         0.56%        7.13%
```

**The pooled number quadrupled - 11.65% to 47.30% - and rare-token recall is still exactly zero.** Not "low": zero, out of 888 positions, at every checkpoint this arm produced (0.00% at 250, 500 and 1000 steps). One correct rare token would have read 0.11%.

## What the bands say

- **Top-20: 81.74% against a 24.87% constant.** The model became very good at the frequent tokens.
- **Ranks 21-500: 16.29% against 5.56%** - a genuine 2.9x improvement, so this is not pure frequency: the model is using local context for mid-frequency tokens too.
- **Tail (rank > 500): 0.00% against 0.56%** - the model is *worse than a constant predictor* on rare tokens, because a constant at least guesses one of them occasionally.

So the arm learned **the local statistics of tokens it can afford to guess**, and has no ability whatsoever to retrieve *which* rare token occurred four steps back. A four-step-back target was sitting in the model's input four tokens earlier - a copy mechanism would get some of them. It gets none.

## The line's conclusion, now with a mechanism

The synthetic delayed-recall probe proved the **memory bank itself** holds a fact for 16 steps when a task demands it (81.25%, control at chance, gradient certified 6.789e-08, one bank per sample). The real model, trained with genuine delayed demand, still cannot retrieve a rare token across 4 steps.

**The gap is not the bank and not the objective. It is the base model's ability to represent token identity in its recurrent state at 4.7M parameters.** Both levers this line could pull - the memory mechanism and the training signal - have now been pulled and measured, and neither moved rare-token recall off zero.

## The rule this validates

Published one wake earlier, before this endpoint existed: *a pooled accuracy gain is not evidence of the capability the task is named after; split by target frequency first.* The endpoint is its strongest test yet - a **47.30% pooled score with 0/888 rare-token recall** - and the pooled number alone would have read as a success.


---

# Why the memory-module result is immune to the frequency artefact that killed the other claims (2026-11-18)

The delayed-recall numbers on **real text** were withdrawn because all their gain sat in the 20 most frequent tokens, with **0 rare-token recalls out of 888-892 positions**. The obvious next question is
whether the one surviving memory result - the **8-slot bank holding a fact for 16 steps in the synthetic probe (81.25%)** - is the same kind of artefact.

It cannot be, and that is now measured rather than asserted.

**The synthetic task draws every token uniformly.** Verified on 51,000 draws from the probe's own generator:

```
per-token share     min 0.0298   max 0.0328   (perfectly uniform = 1/32 = 0.03125)
chi-square vs uniform   30.5    df 31    5% critical ~43.8   -> consistent with uniform
```

Consequences:

1. **There is no frequent band to exploit.** The top-20 tokens are 62.5% of a 32-token vocabulary simply because there are only 32 tokens; every token is equally likely, so a "prior over
   frequent tokens" is just a uniform prior over all of them.
2. **The best constant predictor scores exactly 1/32 = 3.125%.** The control arm - same readout, memory channel removed - measures **3.03-3.22%** across delays 1, 4, 8 and 16. It sits exactly
   where a constant predictor must sit, which is the check that the wiring leaks nothing.
3. So the bank's **81.25% at a 16-step delay** is 26x the chance rate with **no frequency structure available to inflate it**. Whatever it is doing, it is carrying token identity.

**What this does and does not do.** It does not make the real-text result better - that stays withdrawn. It establishes that the module-level capability is real, which is what makes the
real-text failure informative: the bank can hold a token for 16 steps, the real model cannot, so the limiter sits in the base model rather than in the memory mechanism.


---

# Extended training at a 16-step delay ERODES the middle band - measured, not inferred (2026-11-18)

Chasing the "it just needs more steps" hypothesis produced a result in the opposite direction, and it survives a significance test:

```
delay 16, ranks 21-500 band, n = 2379 positions per checkpoint
  td16  ep7000    1.60%   = 38 hits
  td16x ep7500    1.47%   = 35 hits
  td16x ep8250    0.59%   = 14 hits

ep7000 vs ep8250:  1.60% vs 0.59%   z = 3.35   95% CI on the difference [+0.42, +1.60] points
VERDICT: RESOLVABLE decay
```

Meanwhile the frequent band stayed flat (16.15% -> 15.60% -> 15.79%) and the tail stayed at **0 of 892** throughout.

So 1,250 further steps of the same delayed-target objective **actively removed** about a point of delayed ability from the middle-frequency band, while neither the frequent band nor the tail improved. That is not "needs more training": the objective is
converging onto the frequency prior and away from whatever partial delayed information it had briefly encoded.

**A bound I am NOT overstating.** The tail result is `0 of 892`, which only implies a true rate below **0.34%** - one in 300 rare targets could exist and this sample would not see it. The high-N run (1,040 windows, ~8,900 tail positions) is in flight to
tighten that to **0.034%**. Until it lands, "zero" means "below 0.34%", not "exactly none".


## Endpoint of the delay-16 extension: monotone erosion, zero tail (2026-11-18 20:30)

```
delay 16       POOLED   next-tok   top-20   ranks 21-500   TAIL(892)
ep7000          8.67%     16.54%   16.15%      1.60%        0.00%
ep7500                              15.60%      1.47%        0.00%
ep8250                              15.79%      0.59%        0.00%
ep8500          8.26%     11.86%   16.09%      0.55%        0.00%
```

The middle band declines **monotonically** across all four checkpoints (1.60 -> 1.47 -> 0.59 -> 0.55) while the frequent band is flat (16.15 -> 15.60 -> 15.79 -> 16.09) and the
tail never moves off zero. The pooled number, which had been the only thing improving, also falls at the endpoint (8.67 -> 8.26).

So the delay-16 line closes on four consecutive checkpoints: **no training length produces rare-token recall, and the middle band is actively eroded by continuing.**


---

# HIGH-N TAIL TEST: the zero is now solid, and one published magnitude was overstated (2026-11-18 20:45)

The rare-token band had only ~890 positions per checkpoint, which bounds a zero result at merely **below 0.34%**. Re-measured on **1,040 windows (10x)**:

```
delay 4, 8728 tail positions
  td4_r090_ep7000       top-20 82.37%    ranks 21-500 13.11%    TAIL 0.00%
  same-window constant  top-20 25.17%    ranks 21-500  3.89%    tail 0.38%

95% upper bound, 0 hits in 8728:  true rate < 0.034%
```

## Predictions, scored

- **P1** "zero tail hits, bound below 0.05%" - **MET**: 0 of 8728, bound **0.034%**.
- **P2** "the bands reproduce the low-N values within ~2 points" - **PARTIALLY MET, and this is the correction.** Top-20 reproduced well (82.37% vs 81.74%, 0.63 points apart). The **middle band did not**: **13.11% at high N against the 16.29% published from low N, a 3.18-point move** - outside the stability window I set in advance.
- **Falsifier** ("any tail hit") - **did not fire**.

## The correction

The published tn4 endpoint entry says *"ranks 21-500 16.29% against a 5.56% constant"*. The high-N measurement is **13.11% against 3.89%**. The qualitative claim survives - the middle band is still **3.4x its constant**, so real mid-frequency delayed learning exists - but the **magnitude was overstated by about 3 points**, because 104 windows was too few to pin that band down.

That is the same lesson as the tail, applied to a band I had assumed was well-sampled: **only the frequent band was stable at 104 windows.** Anything derived from the middle or tail bands at that sample size should be treated as an estimate with a ~3-point uncertainty, and published with that caveat or re-measured at 10x.

## What now stands, at the resolution actually available

- **Frequent band**: 82.37% vs 25.17% constant at delay 4 - large, stable, reproduced.
- **Middle band**: 13.11% vs 3.89% constant - real but 3 points smaller than first published.
- **Tail**: **0 of 8728**, true rate below **0.034%** - this is now a defensible zero rather than a rhetorical one.


---

# WITHDRAWN: the "monotone erosion" claim and its z=3.35 (2026-11-18 20:55)

I published, two wakes ago, that extending delay-16 training *erodes* the middle band, with a two-proportion test giving **z = 3.35, CI [+0.42, +1.60] points**. **That claim is withdrawn.** Two independent defects, both mine:

## 1. The z-test assumed independent positions, which they are not

The test treated **2379 scored positions as 2379 independent trials**. They are not: 64 positions sit inside each window and share recurrent state and a memory bank, so hits cluster within windows. This is *precisely* the error I corrected for pooled numbers earlier the same evening - I wrote the cluster-robust rule and then broke it for the band numbers.

## 2. The low-N midpoint was biased, and the high-N measurement proves it

The evaluator draws windows from a seeded RNG, so **104 windows is a strict prefix of the 1040-window draw** - not an independent smaller sample. Measured on the *same* checkpoint:

```
delay 16, ranks 21-500, td16_r090_ep7000
  first  104 windows (2379 positions):  38 hits = 1.60%
  all   1040 windows (23364 positions): 100 hits = 0.43%
  -> the remaining 976 windows contribute only 62 hits in 20985 positions = 0.30%
```

So the first 104 windows are roughly **5x easier** than the rest of the corpus, and the low-N number was an artefact of which windows happened to be drawn first. Every mid-band and tail figure I quoted at 104 windows inherits that bias.

## Corrected delay-16 numbers, at the resolution that is available

```
delay 16, td16_r090_ep7000, 1040 windows
  band                 model    same-window constant
  top-20               16.53%          24.94%
  ranks 21-500          0.43%           4.26%
  tail (>500)           0.00%           0.19%   (0 of 8727, true rate < 0.034%)
```

**The correction makes the delay-16 result worse, not better.** At high N the model is *below* the trivial constant predictor on **every** band: the middle band by about 10x (0.43% vs 4.26%). The earlier "8.67% vs a 6.59% constant, so delay 16 slightly beats trivial" reading was propped up by prefix-selected windows.

## What survives

- **The tail zero survives and is now solid**: 0 of 8727, true rate **< 0.034%**.
- **The delay-4 frequent band survives**: 82.37% vs a 25.17% constant, reproduced at high N within 0.63 points.
- **The erosion question is now OPEN, not answered** - the four checkpoints must be re-measured at 1040 windows before any trend claim can be made.
