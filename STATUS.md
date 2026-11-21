## Scoping my own claim: "about six epochs" is true of this recipe, not of the project

I published that both exposure arms peak at epoch 6 and that ~6 epochs is the useful training length here, treating it as general.
Testing it against all 52 curves on disk with at least six epochs:

```
peak epoch distribution across 52 arms: 1 x4, 2 x15, 3 x19, 4 x2, 5 x1, 6 x9, 8 x1, 11 x1
median peak epoch 3.0 | inside 5-7: only 10 of 52
```

Across the project models peak **early** (median epoch 3); the 100-epoch arms peak at 1-3. **Inside the matched d512/d1024 direct-tied
recipe** it is different: 3 of 4 long runs peak at epoch 6 (29.81 / 28.51 / 30.06) and one at epoch 8 (29.78).

So the claim is now stated in its proper scope: **peak at epoch 6 holds for this recipe at 2M windows, three of four long runs; the
project-wide median peak is epoch 3, so six is not a rule for other configurations.** Runs of six or fewer epochs are excluded from
the distribution because they cannot peak later than their last epoch - several apparent "epoch 6 peaks" are that artefact.

`docs/D1024-BEATS-D512.md`.
## Three-seed paired number: +3.95 (was +3.88 on two pairs), computed by a published tool

Both arms now have three seeds, so `tools/compare_arms.py --paired` recomputes the equal-3-epoch result from the history files:

```
paired by seed (3 pairs):
  seed 0: best +3.86   seed 1: best +3.92   seed 2: best +4.08
across seeds:
  best   mean +3.95  sd 0.11  positive 3/3
  last3  mean +4.45  sd 0.51  positive 3/3
```

**+3.95 replaces +3.88** - three paired seeds instead of two, all positive, spread only 0.11 points, and the arithmetic comes from a
published tool rather than from a number I typed. It does not change the matched 9-epoch result (**+1.30 to +1.71**): +3.95 answers
"width at equal 3-epoch training", not "width in general". No t-statistic is quoted - three pairs is not that kind of evidence.

`docs/D1024-BEATS-D512.md`.
## FINAL VERDICT: width is worth +1.30 to +2.85 at matched 9-epoch training - every endpoint rule agrees

Both arms 9 epochs, matched schedules, same seed, printed by `tools/compare_arms.py`:

```
              best-of-N   final    last-3   all-epoch mean
d1024 @9ep      29.81      29.17    29.18       28.49
d512  @9ep      28.51      27.67    27.47       25.64
width           +1.30      +1.50    +1.71       +2.85
```

**The first comparison here where every endpoint rule agrees in sign.** At matched training, width is worth **+1.30 to +2.85 points** and
no endpoint choice changes the direction (the earlier cross-epoch comparison flipped to -0.17 under one rule).

**Predictions:** B1 **FAILED** - d1024 reached 29.81 and plateaued, 0.57 short of the 30.38 floor, so extra epochs do not keep helping
the wide model. B2 **PASSED** - the gap shrank from +3.86 to +1.30. Arm A's falsifier was not triggered (d512 best 28.51, below 29.38).

**The number for width now:** +3.86 at equal-3-epoch training remains a correct answer to that question, but the better-controlled
matched-9-epoch comparison gives **+1.30 to +1.71**. The honest summary is that **most of the +3.86 was training length, not
parameters** - width is real and consistently positive, worth roughly one to two points on this task, not four. Both curves peak at
**epoch 6**, so the useful training length here is about six epochs and everything after is noise around a plateau.

Next: seeds 1-2 at 9 epochs on both sides (the matched line is one seed each), then 6-epoch arms since that is where both peak.
`docs/D1024-BEATS-D512.md`.
## New tool: `tools/compare_arms.py` prints every endpoint rule, so no single number can be cherry-picked

Takes any number of `*_history.jsonl` files and prints each arm's full curve, **best-of-N**, **final epoch**, **last-3-mean** and
**all-epoch mean**, plus its **selection inflation**, and warns when arms have different epoch counts. On the two arms that disagreed:

```
best-of-N       +0.87   (29.38 vs 28.51)
final epoch     +1.71   (29.38 vs 27.67)
last-3 mean     -0.17   (27.29 vs 27.47)
all-epoch mean  +1.65   (27.29 vs 25.64)
```

Its final line: "no single number is printed as the answer - that is the point of this tool." Usage in `docs/D1024-BEATS-D512.md`.
## The width effect depends on how the endpoint is picked: -0.17, +0.87 or +1.71 points

Both curves bounce two to three points per epoch, so the endpoint rule decides the sign of the result. The same two runs:

```
how the endpoint is chosen              d1024 @3ep   d512 @9ep   width advantage
best-of-N (what this project quotes)      29.38        28.51        +0.87
final epoch                               29.38        27.67        +1.71
last-3-epoch mean (robust end state)      27.29        27.47        -0.17
```

**By end-state averaging the reference is marginally ahead.** By best-of-N the wide model is ahead by 0.87. Same two runs, different
questions. Selection inflation is measurable: d1024's best epoch is **+2.09** above its own mean, d512's is **+2.87** above its own mean
with **six more chances** to be lucky - so best-of-9 against best-of-3 favours the arm with more epochs by construction.

**Not established:** that width is worth +3.86 in general. That figure is a correct equal-3-epoch, equal-data, seed-matched result, but
it is fragile to the endpoint rule once the reference trains longer. **Not yet settled:** 9 epochs against 3 is unequal; d1024 at 9
epochs is running now. **Established:** single-epoch values cannot carry a conclusion on this recipe, and any width claim must state
which endpoint rule it used. From here every such claim quotes best-of-N, final epoch and last-3-mean together.

`docs/D1024-BEATS-D512.md`.
## Exposure test, arm A: the reference gains +2.99 points from more training and closes to within 0.87 of the wide model

d512 at **9 epochs** (same windows, stretched lr schedule):

```
epoch   1      2      3      4      5      6      7      8      9
acc %  19.58  24.18  25.13  25.01  25.99  28.51  27.07  27.65  27.67
```

**Best 28.51% (epoch 6), final 27.67%.** Its 3-epoch value was 25.52, so extra training is worth **+2.99 points** to the reference.

A1 PASSED (improves, stays below 29.38) and the falsifier did **not** trigger - but the reference came within **0.87 points** of d1024's
3-epoch 29.38, not the 3.86 the 3-epoch comparison implied. **Most of the +3.86 gap can be closed on the reference side by training it
longer.** The 3-epoch number is still valid as an equal-training result, but the take-away is now "+3.86 at equal 3-epoch training, and
at most +0.87 versus a 9-epoch reference unless d1024 also improves with 9 epochs" - which is arm B, running now.

The reference curve bounces hard (28.51 -> 27.07 -> 27.65 -> 27.67), so single epochs are unreliable and the checkpoint-selected best is
quoted, as everywhere else here. `docs/D1024-BEATS-D512.md`.
## The 7-epoch arm failed its prediction, and its data had to be rescued from a log I destroyed

```
epoch   by-position %   val ppl    lr
1          24.41        131.03    0.400
2          28.13        100.80    0.367
3          28.86         95.48    0.333
4          29.18         89.02    0.300
5          28.78        101.60    0.267
6          30.06         96.90    0.233
7          29.74        111.45    0.200
```

**Prediction FAILED and the falsifier triggered.** I predicted >= 30.88% at epoch 7; the best epoch is **30.06** (epoch 6) and it ends
at 29.74, only **+0.68** over the 3-epoch 29.38, below the 1.0-point threshold I set for "more data helps". The curve bounces by about
the seed spread (sd 0.52), which is a plateau, not learning.

Confounds, written before the number arrived: `--epochs 7` stretches the lr decay over 7 epochs instead of 3, so epochs are not
comparable across the two runs; and the token diagnostic is one-pass, so this is "more epochs at a slower-decayed rate", not "more data".

**The log was destroyed by my own edit** - I modified the running shell script, bash re-read it at a shifted offset, ran a mangled command
(`in_chat.py: command not found`, rc=127) and its redirect truncated the log. The numbers survived only because the trainer writes
`history.jsonl` and per-epoch checkpoints; the entire table above is re-derived from that file. The 84-byte truncated log is archived,
not deleted. **The 3-seed result is untouched** (29.11 mean, paired +3.88, equal data, shared schedule); what this shows is that d1024
plateaus near 29-30 at this data, and whether d512 catches it is the exposure arm now running.

`docs/D1024-BEATS-D512.md`.
## CORRECTION: a caveat I published cited a number that is in no log

I wrote that d1024's 1.35 tokens per parameter was measured against **8.91** for the d512 reference. **8.91 is in no log in this
project** - a figure written from memory instead of read. What the runs actually print:

```
run                                    params        printed tokens/parameter
d512 reference (and both r2 seeds)    15,567,566        4.11   (adequate)
d1024 seed 0                          47,468,238        1.35   (UNDER-TRAINED)
```

The true ratio is **3.05x**, exactly the parameter ratio - same tokens, spread thinner. The nearest real figure to my invented 8.91 is
8.22, and it belongs to a different full-corpus rung, not the matched reference.

Two consequences that invalidate parts of my own plan: the diagnostic counts **one pass and ignores epochs**, so my P3 prediction on
the 7-epoch arm is impossible by construction; and **7 epochs does not match the reference's exposure** - cumulative tokens give
12.3 per parameter for the reference against 9.4 for the 7-epoch arm (76%), with **9.1 epochs** needed for a true match.

**The three-seed result is not affected**: mean 29.11, paired +3.88 on two pairs, falsifier untriggered. The error was in my
explanation of the comparison's imperfection, not in the comparison. `docs/D1024-BEATS-D512.md`.

## Seed-matched comparison: +3.88 paired, on two pairs

d512 seed 1 came in at **19.88 / 23.44 / 25.54** against reference seed 0's **19.69 / 24.26 / 25.52** - the reference reproduces to
**0.02 points** at epoch 3, so the arms can now be compared paired on seed as this project requires:

```
seed    d1024    d512     paired difference
0       29.38    25.52         +3.86
1       29.45    25.54         +3.91
paired mean                      +3.88   (sd 0.04, 2 of 2 seeds positive)

unpaired arm means: d1024 29.11 (n=3) - d512 25.53 (n=2) = +3.58
```

**Quote +3.88.** It rests on two pairs, so no t-statistic is worth reporting (n=2); the honest statement is "two matched seeds, both
positive, +3.86 and +3.91". d512 seed 2 is still running and started low (18.59 / 21.03), which is precisely why the paired number is
the one quoted rather than the arm means. Falsifier not triggered: no d512 seed has reached 29.0% (best 25.54).

`docs/D1024-BEATS-D512.md`.

## Correction: the epoch-2 claim in the d1024 result was a single-seed margin

I published that "d1024's epoch 2 already beats the reference's final epoch". With three seeds that is still true in direction and
**overstated in size**, so it is corrected:

```
d1024, three seeds    epoch 1   epoch 2   epoch 3
mean                    24.49     27.22     29.11
sd                       0.11      1.04      0.52
worst seed              24.36     26.09     28.51
```

All three seeds do exceed the reference's 25.52% at epoch 2, but the worst seed's margin is **+0.57**, not the **+2.62** the
single-seed sentence implied. Epoch 1 is very stable (24.49 +/- 0.11) while epoch 2 varies by two points across seeds, so mid-run
epochs are not interchangeable measurements on this recipe. The epoch-3 result stands unchanged: mean **29.11**, gap **+3.59**.

Also recorded: the d512 reference's epoch 1 is now a triple - 19.69 / 19.88 / 19.88, **sd 0.11** - so the reference's own wobble is
small against the effect under discussion. `docs/D1024-BEATS-D512.md`.

## d1024 three-seed result: 29.11% mean (sd 0.52), gap +3.59 over the single-seed reference

The wider side now has three seeds, each re-derived from its own log:

```
seed    epoch 1   epoch 2   epoch 3
0        24.36     28.14     29.38
1        24.53     27.43     29.45
2        24.57     26.09     28.51
mean                          29.11   (sd 0.52, spread 0.94)
```

The prediction written before those runs passed on both clauses (worst seed 28.51 against a 27.88 floor; mean 29.11 against a 28.5
floor) and the falsifier (any seed below 26.0%) was not triggered. Against the reference's 25.52% the honest three-seed gap is
**+3.59** - averaging the winner correctly costs 0.27 points off the single-seed +3.86.

**One side of the comparison is still open:** the d512 reference has one finished seed. Its seeds 1 and 2 are running with the
reference recipe verified field by field (`--lr 0.4`; I had written 0.6 and caught it before the arm ran). Until they land, +3.59 is a
three-seed mean against a single run and must be quoted that way. Details: `docs/D1024-BEATS-D512.md`.

## The first 50M-class model beats the 15.5M reference by +3.86 points

Shape chosen from measurement: the direct pathway only contributes with `route` off, and `route`/`direct` are mutually exclusive. So
the 50M model is the plain shape, wider: d=1024, direct/tied, no route, **47,468,238 parameters**.

```
                          parameters   ep1     ep2     ep3     VAL ppl (1/2/3)
d512 reference            15,567,566   19.69   24.26   25.52   187.93 / 141.99 / 126.79
d1024                     47,468,238   24.36   28.14   29.38   131.49 / 101.54 /  93.13
difference                               +4.67   +3.88   +3.86
```

Assistant-only accuracy by position on the matched split, from each run's own log. **d1024 is ahead at every epoch, and its epoch 2
already beats the reference's final epoch.** Perplexity 126.79 -> 93.13. The prediction (>= 28.0% at epoch 3) passed at **29.38%**,
and was met a full epoch early. VRAM peaked at **2,145 MiB** of the 5 GB budget.

**Caveats, written before the result and still binding:** one seed per side; and d1024 is data-starved by the trainer's own
diagnostic - **1.35 tokens per parameter (UNDER-TRAINED)** against the reference's **8.91** - so the margin is not a clean estimate
of width at matched tokens. Next: d1024 seeds 1-2, then a matched-token run, and only then treat 29.38% as the number to beat for the
50M line. Details: `docs/D1024-BEATS-D512.md`.

## The route/direct repair is implemented: identical behaviour, half the parameters

Rather than wire the discarded direct term INTO the route (measured: copy -82.8 points), the model builder now simply does not
create it when `route` is on - `GLASSNET_ROUTE_SKIP_DIRECT=1`, off by default so existing checkpoints load byte-identical. Verified
on a real route-on checkpoint, same 200 held-out windows:

```
guard        dmix built   parameters   language
OFF (today)      yes      16,354,001     13.00
ON               no        7,965,393     13.00
```

**Half the parameters, identical output** - 8,388,608 parameters and the model's largest matmul gone from every route-on arm, with no
behaviour change. Getting the guard in took three edits, each caught by running it: the first fell through an `elif` chain to
low-rank tensors that were never built (`AttributeError: dproj`), the second hit `UnboundLocalError: copy_out` in the return dict,
and the guard-off path was re-verified unchanged after every edit.

Final state of the question: including the term in the route is **rejected** (-82.8 copy), not building it is **adopted** (free),
and `direct` without `route` remains the only configuration where the direct pathway contributes (+18.33 ablation). The switches
stay mutually exclusive, but the exclusivity now costs nothing. Details: `docs/ROUTE-VS-DIRECT-VERDICT.md`.

## Route and direct must stay mutually exclusive - both half-fixes measured, both rejected

With `route` on, the direct term is computed and discarded (drift 0.0000, 8.4M params). Wiring the term INTO the route, trained with
it from the start, one flag apart in one script, paired seeds:

```
arm           held-out copy   taught   language
direct_out      83.1 / 85.1     83.4     12.62     (current: term computed and discarded)
direct_in        1.4 /  1.2     30.5      8.91     (term included in the route base)
difference       -82.82         -52.9     -3.71
```

**Including the direct term destroys the copy pathway: -82.8 points of held-out copy.** The falsifier (more than 3 points lost)
triggered by a factor of twenty-seven, and the other prediction failed in the informative direction: on the checkpoint actually
trained with the term, zeroing `dmix` **improves** language by **1.25 points**, so the term reaches the output and is unhelpful there.

Mechanism: the direct term is ~**28x** the magnitude of the concept/answer logits (2.63 against 0.093), so putting it inside the
route base makes the base drown the pointer readout - the switch stops being a choice between comparable readouts.

**The correct repair is not to wire the term in, but to stop paying for it:** when `route` is on, do not build `dmix` at all. That
changes no behaviour and saves 8.4M parameters plus the model's largest matmul on every step. `route` and `direct` should be
mutually exclusive in the model builder, with a clear error if both are requested.

Two of my own mistakes en route, both caught by reading artifacts: an arm that passed `--direct` to a trainer with no route flag (it
became a reference replication), and an arm that passed `--route-gate` without `--pointer --pointer-qk` (copy read 0.0% on both
sides, impossible for a copy task). Neither became a claim. Details: `docs/ROUTE-VS-DIRECT-VERDICT.md`.

## Removing the direct pathway costs ~5 points at every epoch - it earns its budget

The retraction said `dmix` matters in a trained model; this is the training answer. The shipped pretraining recipe with `--direct`
omitted, three seeds, by position on the matched split:

```
epoch            1        2        3      params
WITH dmix     19.69    24.26    25.52    15,567,566   (1 seed, its own log)
WITHOUT mean  15.30    18.73    20.29     7,178,958   (3 seeds, sd 0.36/0.09/0.03)
deficit       -4.39    -5.53    -5.23
```

**4.4-5.5 points lost at every epoch and roughly double the perplexity** for 8.4M fewer parameters. My prediction ("removable if
within 1.0 point") is **falsified** and its falsifier ("losing >=3 points means it earns its budget") is triggered in all three
seeds. Caveat stated: the reference is **one** seed, while the no-dmix arm is three with a spread of sd 0.03 at epoch 3 - the gap is
far outside plausible seed variation, but that side has no measured spread.

## And the ordering fix cannot be retrofitted

`lens/scripts/run_routefix.sh` applied the fix, then tested itself: on a route-on checkpoint the direct term now reaches the output
and language accuracy collapses **14.75% -> 0.25%**, with zeroing `dmix` restoring exactly 14.75%. Cause: every route-on checkpoint
has `dmix` frozen at initialisation (drift 0.0000), so including its term injects 8.4M untrained parameters. The script reverted
itself and `model.py` is byte-identical to the backup. **The fix needs a retraining arm, not a patch.** The script's verdict line was
also wrong (it tested a signed difference, so it called a catastrophic change "no change"); that is corrected. Details:
`docs/PARAM-DIRECT-MATTERS.md` and `docs/ROUTE-FIX-NEEDS-RETRAIN.md`.

## RETRACTION: dmix is one of the most important components, not dead weight

My parameter-budget audit measured `dmix` on a COPY-finetuned checkpoint and found it worth 0.00, then advised cutting it from the
50M design. **That was the wrong model for a language-budget question.** On the pretrained language model
(`matched_d512_s0_best`, route off, baseline 23.93%), zeroing each tensor gives:

```
emb            +18.33 points   (39.5% of params)
dmix           +18.33 points   (53.9% of params)   <- joint largest
answer.W        +3.60 points    (2.8%)
concept.W       +1.93 points    (3.8%)
answer.b        +0.47 points
```

**`dmix` is tied with `emb` as the single most important component of the language model.** My advice would have deleted the
joint-largest contributor.

The two measurements are not contradictory, they are different states: during pretraining (route OFF) `dmix` learns normally and
every tensor shows non-zero drift; in the copy arms (route ON) `forward()` captures `_route_base` **before** the direct path adds
its term, so the term is discarded - no contribution, no gradient, drift **exactly 0.0000**, ablation 0.00 on every task.
**Turning on `route` silently disables the direct pathway**, so every route-on arm has been computing and discarding 8.4M
parameters.

Corrected advice for 50M: **keep the direct pathway** and fix the ordering (either include the direct term in the route combination
or stop enabling `route` where `dmix` matters). The real question is which components earn budget *in the same model state*. What I
got wrong: I measured a component in a state where a bug had already disabled it and published that as a property of the component.
Details: `docs/PARAM-BUDGET-RETRACTION.md`.

## dmix is not dead weight - it is computed and then DISCARDED by the route switch

A correction to the entry below: the 0.00 damage figure is right, the explanation was wrong. Measured on `early_s0_ep1000`
(`route=True`):

```
language accuracy   dmix intact 14.75  |  dmix zeroed 14.75  |  restored 14.75
mean-ablation +0.00   zero-ablation +0.00     relative drift 0.0000
ReLU units firing: 67.2%   hand-computed dmix term magnitude: 2.63 vs main logits 0.093
```

The block is alive, its term is ~28x the main logits, and the forward pass throws it away: when `route` is on, `forward()` captures
`_route_base = logits` **before** the direct path contributes and mixes that base with the pointer readout at the end, so `copylog`
never reaches the output. No output contribution -> no gradient -> 0.0000 drift -> 0.00 in every ablation. It is the inverse of a
bug already fixed in this file (pointer-only applied mid-forward with the copy path added back on top).

**So 8,388,608 parameters - 51.3% of the model - and its largest matmul are computed every forward pass and discarded.** Either the
direct term belongs in the route combination, or `dmix` should not be built when `route` is on; the second is free and immediate.
The budget advice is unchanged (do not scale a component that contributes nothing) but the reason changes from 'dead weight' to
'discarded by a routing branch, therefore unlearned'. Details: `docs/PARAM-BUDGET-AUDIT.md`.

**Pointer-only ship number confirmed on five seeds:** paired copy gain **+1.86 +/- 0.70 (t = 5.9, **5 of 5 seeds positive**), language cost **-0.69 +/- 0.29** - the prediction (gain >= 1.5, >= 4 of 5 seeds) passed and the falsifier did not trigger. Seed 3 fell to 71.2 against 76-79 on the first three, and its unfrozen pair fell with it (69.7), which is why the five-seed gain (+1.86) is smaller than the three-seed figure (+2.07). Three-tensor and four-tensor sets are now statistically indistinguishable; the simpler set ships because concept.W was measured as contributing nothing.

## Half the model's parameters are measured dead weight on both objectives

The 50M target is a budget question, so this measures which parameters earn their place on the **language** objective, the same way
the copy line did for copy. Checkpoint `early_s0_ep1000`, baseline 16.53%, 16,354,001 parameters:

```
tensor        params      language damage   share
emb           6,145,024       +16.53        37.6%
ptrW            262,144       +14.40         1.6%
Wq              262,144       +10.87         1.6%
Wk              262,144       +10.47         1.6%
concept.W       589,824        +0.40         3.6%
answer.W        432,072        -0.07         2.6%
dmix          8,388,608        +0.00        51.3%
```

**`dmix` is 51.3% of the model and worth 0.00 points of language accuracy** - and it was already 0.0 points of copy damage with
0.0000 drift. Dead on both objectives. **The pointer matrices are 4.8% of the parameters and carry 10.5-14.4 points**, roughly
**100x** `dmix`'s value per parameter.

So a 50M model built the current way would spend about **26M parameters on a component measured at zero**. The budget should go to
width and the pointer path instead: at `d = 1024` without `dmix` the model is **about 17.3M parameters** (emb 12.29M, concept
1.34M, answer 0.49M, ptrW 1.05M, Wq/Wk 2.10M), inside the 50M limit with room to widen. Details: `docs/PARAM-BUDGET-AUDIT.md`.

## CORRECTED: concept.W alone does nothing; the pointer is the whole mechanism

An earlier version of this entry compared arm averages and credited `concept.W` alone with +1.28 points of copy for no language
cost. Those arms are **seed-matched** and that was an unpaired comparison: paired on seed the differences are +0.2 / 0.0 / +0.1.

```
arm              copy gain (paired)   t     seeds up   language change (paired)   t
concept.W only        +0.10         1.7      2 of 3            -0.03            -1.0
pointer only          +2.07         3.9      3 of 3            -0.79            -6.4
full freeze           +1.95         6.2      5 of 5            -0.74             -
```

**`concept.W` alone does nothing measurable.** The **pointer matrices are the whole mechanism**: freezing `ptrW`/`Wq`/`Wk` gives
**+2.07** points of held-out copy (3.9 SE, 3 of 3 seeds) against the full freeze's +1.95, at a language cost of -0.79. Ship
configuration is therefore the **simplest set that works: freeze the three pointer matrices**.

Also withdrawn: the claim that freezing `concept.W` on top of the pointer makes copy worse - both numbers come from separate arms
with different seed counts, and the paired gain is the same within error. This is the second pairing error in this line, which is
why paired-on-seed numbers are the only ones quoted here. Details: `docs/COPY-FREEZE-SUBSET.md`.

## The private copy path failed: worse on both counts than freezing

Freezing the copy path buys ~2 points of held-out copy for -0.73 language, because language is then forbidden from updating the
pointer. So the copy path was given its **own** `ptrW`/`Wq`/`Wk`, used only on copy batches - mechanism verified first at
**0.000e+00** drift under 80 language-only batches.

```
arm        held-out copy        language
none       74.02                16.30
private    74.33                12.86
frozen     75.97                15.56
```

**Prediction (copy >=74 and language >=16.1) failed on language by 3.24 points**, and the written falsifier triggered: the private
path is **3.44 points worse than doing nothing** and 2.70 worse than freezing, while its copy is statistically the same as doing
nothing. It is not a smaller trade, it is a worse outcome on both counts. **The freeze remains the best configuration.**

My mechanism story (gradient redirected into shared components) is **refuted by the saved weights**: the private arm moves the
shared pointer *less* (ptrW drift 0.105 vs 0.189) and moves `emb`/`concept.W`/`answer.W` identically to four decimals. The cause of
the language loss is unknown and is recorded as unknown rather than dressed up. Details: `docs/COPY-PRIVATE-PATH-FAILED.md`.

Two infrastructure faults, both fixed: arms idled 25 minutes waiting on a systemd unit that stays 'active' while the chain waits
for the agent (now waits on the trainer process), and an EOF error caused by editing a bash script while bash was reading it.

## The copy-freeze win replicates from a different starting checkpoint

Five seeds earlier gave +1.98 points of held-out copy from `early_s0_ep1000`, so every arm had started from that one checkpoint.
Repeating the pair from `early_s0_ep2000` (a start already decayed to 64.3 percent):

```
seed   keep_scoped   keep_none   difference
0         72.0         69.2         +2.9
1         72.4         70.5         +1.9
2         72.7         71.4         +1.3
mean   72.37          70.37        +2.00
```

**+2.00 points, positive on 3 of 3 seeds** - same effect, same size, from a start about 7 points lower. The written falsifier (gap
below 1.0 from this start means the effect is specific to ep1000) did not trigger. Combined across two starting checkpoints:
**+1.98 on 5 of 5 seeds** and **+2.00 on 3 of 3**.

It is still not free: the language cost is **-0.73 points** (15.56 vs 16.30 over five seeds). The successor meant to remove that
trade - a copy path with its **own** pointer matrices, invisible to language gradients - is built, its mechanism verified at
**0.000e+00** drift under 80 language-only batches, and its arms are running. Details: `docs/COPY-FREEZE-REPLICATES.md`.

## The copy win is real: 5 seeds, paired, +1.98 points (t = 6.2)

Seeds 3 and 4 landed for the per-batch freeze, which I had recorded as "suggestive, not decisive" with a written falsifier (gap
below 1.0 point on five seeds = noise). It did not trigger - and the test I used first was the wrong one:

```
seed   keep_scoped   keep_none   difference
0         76.2         73.9         +2.3
1         78.9         75.9         +3.0
2         76.8         75.7         +1.1
3         71.5         69.7         +1.8
4         76.5         74.8         +1.7
mean   75.97 +/- 2.72   74.02 +/- 2.52   +1.98
```

**Unpaired, this looks weak** (+1.98 with SE 1.66, about 1.2 SE) - that is what I quoted before. **But these arms are seed-matched**:
seed 3 is low in both arms, seed 1 high in both, so the unpaired test counts shared variation against the effect. Paired on seed
the differences are **+2.3, +3.0, +1.1, +1.8, +1.7** - mean **+1.98**, sd **0.71**, paired SE **0.32**, **t = 6.2**, and **5 of 5
seeds favour the freeze** (sign test p = 0.031).

**The effect was real; my statistical test was wrong.** Language cost on the same seeds: **15.56%** scoped against **16.30%** unfrozen (paired -0.73), so the trade is **+1.98 points of copy for -0.73 of language**; taught-word copy is unchanged (84.26 vs 84.14). The recorded successor - a copy path with its own
parameters, invisible to language gradients - is still unbuilt; this result is the target it has to beat. Details:
`docs/COPY-FREEZE-SEEDS5.md`.

## Retraction: the data judge already passed, and my |z| clip is not the fix

I took on the judge as "still fails its gate" and produced a mechanism, a fix and a claim. The control killed the claim:

```
arm                          AUROC     false alarm   one-token loop   verdict
clipped |z|<=50 (3 seeds)    0.9820-22    0.0422        89.7-90.1%      PASS
UNCLIPPED control, seed 0    0.9923      0.0257        90.1%           PASS
```

**The unclipped control passed and passed better** - false alarm **0.0257** against clipped **0.0422**, AUROC 0.9923 against
0.9822. My hypothesis (that zero-MAD features reaching |z| = 1e9 explained the historical 69.5% miss) is **retracted as a cause**
by the rule written before the run. The 1e9 values are real; they were not what broke the gate. Clipping now defaults **off** so
the tool behaves as verified.

**And the bigger correction: the judge was never failing.** `docs/JUDGE-V2.md` already records the rebuild passing (AUROC 0.9923,
false alarm 2.59%), and my uncapped run reproduces it to the digit - 0.9923 and 0.0257. The "still fails its gate" line in my own
status summary described the OLD judge in `docs/DATA-JUDGE.md`. That is a reporting error on my part, and the second time a stale
status line nearly cost a wake of work.

Verified now: AUROC **0.9923** (bar 0.90), false alarm **2.57%** (bar 5%), all five corruption kinds >= 80% - with one standing
caveat, `shuffled order` sits on its bar at **80.7%**, the corruption that keeps every token and only destroys order. Details:
`docs/JUDGE-ZCLIP-VOID.md`.

## The KL anchor stops the damage: RL is safe now, but still not competitive

`docs/RL-DEGRADES.md` measured unanchored RL costing a working capability **-5.94 points** (3 of 3 seeds worse). This adds a KL
penalty against a frozen copy of the warm-start policy - gradient **numerically checked before use** (max abs 6.096e-11, rel
1.557e-09, PASS):

```
arm        warm starts        deltas                 mean delta
beta 5     32.98/22.96/26.60   -2.61, +6.12, -1.60    +0.64 +/- 4.78
beta 20    32.98/22.96/26.60   +0.00, +6.12, -0.00    +2.04 +/- 3.53
unanchored                     -13.61, -1.02, -3.20   -5.94
supervised, same data                                 34.66 +/- 6.16
```

**The anchor removes the damage** - the first non-negative RL result in this line. But the beta-20 zeros are the honest caveat: a
strong anchor buys 'not degrading' with 'not learning'. **Safe but not useful.** Verified before believing it: every log shows the
anchor active with the right beta, the two seed-1 traces differ (reward 0.070 vs 0.057), and the identical +6.12 is a
coarse-metric coincidence (13 of 214 items).

**RL has now failed to beat supervised in eight configurations.** Two of those failures are not reward hacks: 'cannot start' and
'does not improve'. Per the falsifier committed before the run, the line is not broken - the anchor works - but it is recorded as
**not competitive with supervised learning here**, and stops being tuned. Next: the data judge, which fails its own gate (false
alarm 0.0730 vs a 5% bar, loops 69.5% vs 80%) - an engineering problem with a measurable pass condition. Details:
`docs/RL-KL-ANCHOR.md`.

## RL does not hold a working capability either: it degrades it in all three seeds

The reachability result removed the last excuse - the task is solvable, so RL can be measured against a real target. So the arm
no previous run tested: **warm start on sums >= 10 so the model CAN answer, then binary-verifier RL**. The reward is satisfiable
and is actually collected. Prediction: stays >= 25% and improves >= 3 points.

```
seed   acc_gen after warm start   after 400 RL steps   delta
0              32.98                   19.37          -13.61
1              22.96                   21.94           -1.02
2              26.60                   23.40           -3.20
mean           27.51                   21.57           -5.94
```

**Zero of three seeds improved; all three got worse**, seed 0 losing more than a third of the capability. So the RL line stands
like this, every row measured: from a policy that cannot emit two digits, binary reward gives zero advantage (no learning),
per-digit reward answers '1' to everything (acc_in 22 -> 2.34), shape reward answers '98' (acc_in -> 0.00); from a policy that
CAN (33%), binary reward **costs 5.94 points**. Supervised on the same data at equal compute: **34.66 +/- 6.16**.

**RL has not beaten supervised in any configuration, and the last one is the standard recipe.** The failure is no longer
'cannot start' and no longer a reward hack - with a satisfiable reward and a capable starting policy, 400 policy-gradient steps
still cost accuracy, which points at the update itself (group-8 sampling noise, nothing keeping the policy near one that works,
entropy added on top).

Next arm: a **KL anchor** to the warm-start policy - the standard remedy for this exact failure. If a KL-anchored update still
cannot improve on its own starting point, this RL implementation is not competitive here and the line stops rather than being
tuned further. Details: `docs/RL-DEGRADES.md`.

## The carry task IS reachable: 34.66% +/- 6.16 with demonstrations, 0.00 everywhere without

Every RL arm sat at `acc_gen` 0.00 on sums >= 10, so before spending another one the task had to be shown reachable by
*something*. Supervised, 800 steps, same 7,342-parameter model, only the training distribution changed:

```
arm        acc_gen (3 seeds)         mean +/- sd     acc_in (3 seeds)      mean
sup_in     0.00,  0.00,  0.00          0.00          28.04, 25.85, 30.28    28.06
sup_both  40.84, 28.57, 34.57         34.66 +/- 6.16   1.40,  0.98,  2.29     1.56
sup_gen   41.36, 27.04, 34.04         34.15 +/- 7.02   0.00,  0.00,  0.00     0.00
```

**The task is reachable - ~35% on sums it never saw demonstrations for, when it is given two-digit answers to learn from.** So the
earlier conclusion is not a task artefact: same model, same compute, supervised **34.66%** against RL from a verifier at **0.00**
for all three reward shapes.

**Second finding, about capacity rather than reward:** learning carry costs in-distribution accuracy. `sup_both` is given BOTH
distributions and still cannot hold both - 34.66% on carries, **1.56%** on singles. That is the same competition already measured
between copying and language, now reproduced on arithmetic, and it explains why every carry-trained RL arm lost the singles.

Next: warm-start from a checkpoint that can already produce two digits, then RL with a binary verifier - with the reward
satisfiable the advantage is no longer zero, so the question becomes whether RL holds or improves a working capability. Details:
`docs/RL-REACHABILITY.md`.

## Reward the shape and the shape gets hacked: a constant two-token nonce

The last arm of the carry investigation stopped rewarding digits and rewarded the **structure** - +0.2 for emitting any
two-token answer - because every training answer is one token and the second token was unreachable:

```
seed   acc_in   acc_gen   answers after training        confidence
0       0.00     0.00     '98' on all 7 probes           0.31-0.67
1       0.00     0.00     '83' on 6, '89' on 9+0          0.28-0.54
2       0.00     0.00     '81' / '84'                     0.35-0.68
mean reward at step 400: 0.168 against a 0.2 bonus
```

**Two-token answers appeared (prediction's first half passed) but `acc_gen` stayed 0.00 and `acc_in` fell to 0.00** (the falsifier
triggered). Mean reward **0.168 of a 0.2 bonus** says plainly what happened: the model collected the shape bonus and nothing else,
finding the cheapest constant two-token answer - a different one per seed. Third reward hack in the investigation, each shaped by
the reward it was given: binary pays nothing (cannot start), per-digit pays for a right first digit (answers '1'), shape pays for
two tokens (answers '98').

**Conclusion: on this task and this model, RL from verification cannot produce carry** - implementation, reward resolution,
sample count, learning rate, entropy, data mix, frequency bias and output-structure reachability have all been measured and
excluded. What is NOT shown is that the task is reachable at all at 7,342 parameters: `acc_gen` is 0.00 for every method ever
run, including 800 supervised steps. So the next arm is a **reachability check** - supervised training with two-digit answers in
the data - before another RL arm is spent on a task nothing can solve. Details: `docs/RL-SHAPE-HACK.md`.

## Frequency correction cannot fix the 'answer 1' shortcut - and why

The obvious repair after the reward-hacking result is a frequency-corrected reward. Implemented, it changed **nothing**:
identical probe answers, `fit_metric -3.9e-05`, `acc_in 2.3364%`. The reason is structural, not a bug: **a correction that is
constant within a group is invisible to a group-relative advantage** - it cancels in `(r - mean)/std`. On sums >= 10 the
correct first digit is always `'1'`, so the correction is constant for every group member and the advantage is arithmetically
unchanged. It would work against an *absolute* baseline; it cannot work inside this estimator.

The deeper blocker: every answer the model was ever trained on is a single token, so a two-digit answer needs a token pattern
it has never produced. Position-1 credit is unreachable, position-0 credit is free - exactly the shape of a reward that pays
for a constant guess. Next arm rewards the *structure* (a small bonus for emitting any second token). Details:
`docs/RL-REWARD-HACK.md`.

## The collapse was reward hacking: partial credit taught the model to always answer '1'

`docs/RL-GEN-VERIFY.md` reported per-digit credit destroying in-distribution accuracy (22% -> 3.13%) without naming why. Three
variables are now ruled out, each by bit-identical output when the variable changed:

```
configuration                                          acc_in seed 0   seed 1
per-digit, lr 0.02                                        2.34          2.93
per-digit, lr 0.005                                       2.34          2.93
per-digit, lr 0.02, RL batch mixed in-dist + >=10          2.34          2.93
per-digit, lr 0.02, entropy_beta 0.0                       2.34          2.93
```

So it is **not** the learning rate, **not** single-distribution training, **not** the entropy floor. The run now prints the
model's greedy answers after training, and they name it:

```
3+4 = '1' (true 7)  conf 0.83      8+7 = '1' (true 15) conf 0.86
9+0 = '1' (true 9)  conf 0.88      6+6 = '1' (true 12) conf 0.89
```

**It answers '1' to everything at 83-89% confidence.** On sums >= 10 the correct first digit is always 1 (10-18), so the cheapest
way to collect the position-0 part of the reward is to emit '1' unconditionally - and the position-1 part is unreachable because
the model never emits a second token. That is **reward hacking of a partial-credit reward**, and in-distribution accuracy falls
to the rate of pairs summing to 1, which is exactly what 2.34% is.

The requirement for the next reward is therefore concrete: it must not pay for a constant guess. Frequency-corrected per-digit
credit is the next arm, prediction written before it runs. Details: `docs/RL-REWARD-HACK.md`.

## RL from a verifier on the carry problem: nothing learns it, and partial credit destroys what was known

The honest test RL exists for - a checkable objective with **no demonstrations**. The supervised warm start only sees sums <= 9,
so it cannot imitate an answer to 8+7; RL is handed sums >= 10 plus a checker. 3 seeds, equal compute:

```
arm          acc_gen (3 seeds)   acc_in (3 seeds)
gen_sparse   0.00, 0.00, 0.00     18.22, 22.93, 21.10   -> 20.75
gen_dense    0.00, 0.00, 0.00      2.34,  2.93,  4.13   ->  3.13
sup_gen      0.00, 0.00, 0.00     28.04, 25.85, 30.28   -> 28.06
```

**`acc_gen` is exactly 0.00 in all nine runs.** And the prediction that partial credit would help failed catastrophically in the
other direction: per-digit credit **collapsed in-distribution accuracy from ~22% to 3.13%** - it did not trade one ability for
another, it destroyed what the warm start knew.

The mechanism for why binary verification cannot start is measured, not asserted: mean reward **-0.006** with all group members
receiving the same reward, so the group-relative advantage is exactly zero and every token gets zero weight. It is not learning
slowly - it is receiving no signal at all.

**A correction to my own script framing:** `sup_gen` was labelled the ceiling "when labels exist", but the tool trains it on the
in-distribution split, so it never sees a sum >= 10 either. It shows the un-generalising model cannot do carry; it does **not**
price the label advantage, and that prediction was not testable as written.

Before blaming the RL method: every method sits at 0.00 on sums >= 10 at 7,342 parameters, so the task itself has to be shown
reachable first. Next running: is the collapse a learning-rate artefact, and does mixing in-distribution pairs prevent it.
Details: `docs/RL-GEN-VERIFY.md`.

## RL: more samples per prompt did not help either - the comparison is what is wrong

Third cheap lever on the RL line, after the implementation and reward density were both tested and rejected:
group-relative advantage used only 8 samples per prompt, so its baseline is noisy. 32 samples per prompt, same seeds, equal
compute, same reward:

```
seed        group 8    group 32
0            21.03      21.50
1            20.00      18.54
2            18.81      18.81
mean    19.95 +/- 1.11   19.62 +/- 1.64     diff -0.33
supervised, equal compute   28.06 +/- 2.21
```

**Prediction (+2 points) failed and the written falsifier triggered**: -0.33 points, every seed-matched difference inside 1
point. Quadrupling the samples bought nothing.

So the method is not being starved of signal in any of the three ways the project guessed (implementation, reward density,
sample count) - **the comparison itself is the problem.** Every arm asks RL to beat supervised on a task where supervised has
every token labelled, which is the one setting RL is least likely to win; it has never been tested where RL is actually for, an
objective that is checkable but not demonstrable. That test is now written with its prediction: consistency between `a+b` and
`b+a`, where the reward never sees a label, plus a falsifier for the real risk that the model learns to be consistently wrong.
Details: `docs/RL-GROUP-SIZE.md`.

## Reward density is not what was holding our RL back either

`docs/RL-V1-METHOD.md` says our RL loses to supervised because a supervised step labels every answer token while the RL reward
was one bit per sequence, and names reward density as the fix. Built it: per-digit verifiable reward (`glassnet/rl.py:
token_rewards`) plus a per-token advantage against the same digit position of other samples of the same prompt
(`tools/rl_compare.py --dense-reward`). Equal compute, three seeds:

```
arm      acc_in mean +/- sd      ece_in
sup      28.06 +/- 2.21          0.087
sparse   18.84 +/- 1.00          0.278
dense    19.95 +/- 1.11          0.267
```

**Both predictions failed.** Dense beats sparse by **+1.11 points** against a 2.0 bar, and closes **12%** of the
sparse-to-supervised gap against a 50% bar. With spreads of 1.11 and 1.00 that is about **1.3 standard errors** - not
distinguishable from noise, and nowhere near enough to close the **8.1-point** deficit.

So both diagnoses for why RL loses - a wrong implementation, then coarse reward - have now been tested and rejected by
measurement. What is left, in the order this result makes it attractive: (1) group size, since 8 samples per prompt is a noisy
advantage estimate - queued next with its prediction; (2) a task with **no demonstrations**, because every comparison so far
asks RL to beat supervised on a task where supervised has full labels, which is the case RL is least likely to win. Details:
`docs/RL-DENSITY.md`.

## Rebuilding the copy path structurally does not make it immune - refuted

The successor to freezing was to stop trading: build the copy path so language cannot erode it - an **identity readout** (nothing
learned to overwrite) plus the copy path's **own embedding table** - and freeze nothing. Warning up that construction on copy
batches first (so it is not judged untrained), then running 4000 language-only steps:

```
PHASE A warm-up (copy-only, 3000 steps)        PHASE B (language-only, 4000 steps, nothing frozen)
before: taught 18.0 / held 0.0                  step 3000: taught 0.7 / held 0.2 / language 25.20
step  500: taught 78.3 / held 0.0               step 3500: taught 0.8 / held 0.2 / language 25.52
step 3000: taught 50.5 / held 0.0               step 4000: taught 0.8 / held 0.2 / language 26.15
```

**The prediction (held-out >= 60% after warm-up) is falsified, and not marginally: held-out copy is 0.0% for the entire 3000-step
warm-up.** The construction memorises trained words (peaking at 78.3%, then eroding to 50.5% during warm-up) and cannot copy a
word it has never seen. Language training then takes even the memorised copy from 50.5% to 0.8%.

So a copy path built this way is not immune - it is **more** fragile than the shared one. The best copy protection measured
remains the per-batch freeze (77.3 +/- 1.4 held-out, 0.85 language points, `docs/COPY-KEEP-LEARNING.md`). The `lens/model.py`
note claiming the identity readout lifts held-out copy 71.5 -> 85.3 is **not a drop-in recipe** and is not an available win until
the difference is identified. Ground truth, first void attempt included: `docs/COPY-STRUCT-REFUTED.md`.

## Keeping the copy path learning beats freezing it

Freezing the copy path halved the decay but stops it learning, so this runs the successor: **freeze it only on LANGUAGE
batches, let it keep learning from COPY batches** - the separated-store idea applied to a mechanism. Mixed stream (50/50),
4,000 steps from `early_s0_ep1000` (71.0% held-out).

```
arm            held-out copy (late, 3 seeds)   taught   language
keep_scoped    77.3 +/- 1.4                    84.3%    15.64 +/- 0.19
keep_none      75.2 +/- 1.1                    83.9%    16.49 +/- 0.07
keep_always    71.3  (n=1)                     82.5%    15.97  (n=1)
```

**The scoped arm is the best copy result in the study**: 77.3% held-out, above the 71.0% it started from, above no-freeze
(+2.1) and far above always-frozen (+6.0). The prediction's copy half passed; its language half failed by **0.85 points**
(15.64 vs 16.49, separated by more than the spreads, so the cost is real and small).

**Honest statistics:** +2.1 points with a combined standard error of ~1.03, i.e. about two standard errors on three seeds -
**suggestive, not decisive**. The always-frozen comparison is a single seed and cannot carry a claim alone. What stands without
hedging is the direction: every arm that shielded the copy path kept more copy, and the one that also let it keep learning kept
the most.

Still untested and recorded as next: a copy path with its **own** parameters that trains on copy batches and is invisible to
language gradients - the full version of the idea, which would remove the language cost instead of trading for it. Details:
`docs/COPY-KEEP-LEARNING.md`.

## Freezing the copy path halves the copy decay

All arms continue from `early_s0_ep1000` (held-out copy **71.0%**) and train on **language-only** batches for 4,000 steps:

```
arm             late held-out copy (mean of last 3 evals)   drop from 71.0   language
freeze_copy     68.7%  (69.5, 68.5, 68.2)                    -2.3 points      16.15%
freeze_control  60.5%  (61.3, 63.5, 56.8)                   -10.5 points      17.70%
freeze_none     60.5%  (61.5, 63.3, 56.7)                   -10.5 points      17.70%
```

**Freezing `ptrW`, `Wq`, `Wk`, `concept.W` is worth +8.2 points of held-out copy**, taught-word copy flat at 83% in every arm.
Prediction passed on all three counts.

**The control is what makes it mean something:** freezing `answer.W` and `dmix` gave numbers identical to no freezing at all
(60.5% both, per-eval within 0.1), so it is the choice of what to freeze doing the work, not the act of freezing. The cost is
**1.55 language points** (16.15% vs 17.70%) for 8.2 copy points - a real trade, stated rather than hidden.

Limits: the decay here is 10.5 points, not the original curve's 25.2, because this run continues from ep1000 with
language-only batches (different recipe), so it is comparable internally but not to that curve. And freezing protects a
capability by refusing to update weights - the copy path stops learning. The untested successor is giving the copy path its own
parameters that keep learning from copy batches only, i.e. the separated-store idea applied to a mechanism instead of a table.
Details: `docs/COPY-FREEZE.md`.

## Which weights carry copying, and what language training rewrites

The decay is in **held-out** words (71.0% -> 45.8%) while taught words stay flat, so the thing that breaks is the copy
**mechanism**, not the stored facts - a memorisation store would have fixed the wrong half. Two measurements on the real
checkpoints find where it lives.

**Mean-ablation (ep1000, taught 83.8% / held-out 69.5%):** `emb`, `ptrW`, `Wq`, `Wk` each destroy copying entirely. Then the
interesting one:

```
concept.W      +15.0 points of HELD-OUT copy, and +1.0 (i.e. nothing) on taught words
dmix (8.4M)     +0.0 - half the model's parameters have no role in this task at all
```

**Drift between ep1000 and ep4000 (relative L2):** the copy-carrying tensors move the most - `ptrW` 0.566, `concept.W` 0.269,
`Wk` 0.242, `Wq` 0.230 - while the memorised-answer path barely moves: `emb` 0.096 and `answer.W` **0.0166**. `dmix` moved
**0.0000**.

The two agree: memorised answers sit in weights language barely touches (so taught copy holds at ~83%), the copy mechanism sits
in weights language rewrites (so held-out copy loses 25 points). That also explains why the separated-store win (96.1 -> 0.0
points of forgetting) is real but does not apply here: it fixed forgetting of FACTS, this task decays the MECHANISM.

Next: freeze `ptrW`, `Wq`, `Wk`, `concept.W` during language-only training, with a control that freezes `answer.W`/`dmix`
instead - if the control also holds copy, the freeze is not what is doing the work. Details: `docs/COPY-WEIGHTS.md`.

## Copy decay measured off the checkpoints, not the log

Rebuilt the copy task's evaluation from its own definitions (same vocab, same taught/held-out word split from rng(0), same three
prompt forms, same item seeds, 600 items per side) and scored the saved checkpoints directly:

```
checkpoint               taught words   held-out words
early_s0_ep1000.npz         83.0%           71.0%
early_s0_ep2000.npz         83.8%           64.3%
early_s0_ep3000.npz         82.5%           57.2%
early_s0_ep4000.npz         84.3%           45.8%
```

**The decay is real and it is in the weights**: held-out copy falls **71.0 -> 45.8** (a 25.2-point loss) while taught-word copy
stays flat at 82.5-84.3%. The recorded figures reproduce to within a point (I measure 71.0% vs the record's 70.2% at step 1000,
and 45.8% vs 46.5% at step 4000) - a ~0.8 point offset because my evaluation items are rebuilt from the same seeds rather than
reusing the trainer's arrays, stated rather than rounded away.

This is now the reference the memorisation-store fix has to beat: a separated store must hold held-out copy near 71% across the
same 4,000 steps with taught accuracy near 83%, on a 15.5M-parameter model with 4,852 taught and 4,853 held-out words. Details:
`docs/COPY-DECAY-CURVE.md`.

## The separated memorisation store works: 96.1 points of forgetting -> 0.0

The project owner's proposal tested directly. 128 key->value facts trained to 100%, then **language-only** training (no rehearsal), which is
the situation behind the project's measured 70.2% -> 46.5% copy decay. Three seeds each:

```
arm      recall after warm-up   recall after 1500 language-only steps   language
joint            100.0%                  3.9%  (drop 96.1 +/- 0.0)       100.0%
split            100.0%                100.0%  (drop  0.0 +/- 0.0)       100.0%
```

**The joint model forgets the facts completely; the separated store forgets nothing, at identical language accuracy.** The
mechanism is structural, not lucky: the store reads through its own key embedding and writes through its own read-out, so the
language gradient updates a disjoint set of parameters and interference is impossible by construction.

Two of my own design errors had to be fixed before the test meant anything: the first version rehearsed the facts on every
language step (which prevents forgetting by construction and measured nothing - both arms hit ~100%), and the first split arm
shared the key embedding with the language network, a half-measure that left recall frozen at 17.6%.

Limits: this proves the mechanism in a controlled setting, not on the real corpus copy task, and it does not help language at
all (language is identical between arms). Porting it to the real copy task with its 70.2% peak is the next experiment.
Details: `docs/MEM-SPLIT.md`.

## The checking model passes its own gate for the first time

The data filter that is supposed to stop looping or scrambled text reaching training failed two of three gates before
(false alarm 7.30% against a 5% bar; one-token loops caught 69.5% against 80%). Rebuilt, and it now passes all three on
three seeds:

```
                                   old judge     new judge (3 seeds)
AUROC                               0.9597      0.9923 +/- 0.0001     (bar >= 0.90)  PASS
false alarm on clean held-out        7.30%      2.59% +/- 0.02%       (bar <= 5%)    PASS
soup caught                         100.0%      100.0% +/- 0.0                       PASS
shuffled order caught                86.9%      80.9% +/- 0.3%                       PASS
one-token loops caught               69.5%      90.0% +/- 0.2%                       PASS
chunk repeats caught                 80.0%      92.6% +/- 0.1%                       PASS
unknown flood caught                 92.6%      99.8% +/- 0.0                       PASS
```

Each failure had a named cause and a fix: the old threshold was chosen on the split it reported from (now three disjoint
splits, threshold picked at 3% on calibration and reported on unseen test); bag-of-token features cannot see a one-token loop
(now explicit repetition features); random soup sits at the *opposite* extreme from a loop (now two-sided |robust z|
deviation features - soup went 0% -> 100%); and shuffled text keeps its tokens, only its order changes (now bigram/unigram
plausibility plus pair/triple coverage from the training corpus - 8.8% -> 80.9%).

Honest limits: shuffled clears its bar by only 0.9 points (stable across seeds, but a different corpus could push it under), it
detects the five damage kinds it was trained on, and it is **not wired into training yet** - that is the next experiment, with
its own numbers. Details: `docs/JUDGE-V2.md`.

## Our own RL trainer, built properly: still does not beat supervised

First honest measurement of a from-scratch RL method with a **verifiable** reward (addition: right or wrong, no reward model).
Group-relative advantage over 8 samples per prompt, repetition penalty in the reward, hinge entropy floor, two new ops with
hand-written backward passes, both **gradient checked (4.9e-11)** before use.

```
arm                          in-distribution accuracy   2-digit generalisation   ECE
supervised, 800 steps         51.12 +/- 3.30 %                 0.00%              0.131
supervised, 400 steps (warm)  20.93 %                         0.00%                -
RL, entropy bonus 0.05        20.56 +/- 3.69 %                 0.00%              0.219
RL, entropy bonus 0.20        22.31 +/- 0.69 %                 0.00%              0.109
RL, entropy hinge 0.5/2.0     18.84 +/- 1.00 %                 0.00%              0.278
```

**Against the fair baseline (its own 400-step warm start at 20.93%), RL is roughly neutral, slightly negative.** I have to
correct myself: earlier today I reported RL "destroyed 29 points" - that compared 400+400 steps against 800 supervised steps,
two differences at once. It never cost 29 points and it never gained any.

Three of my own bugs were caught by measurement rather than by reading: a sign error in the new weighted log-prob op, float32
quantisation of the loss scalar (which made the finite-difference check read a staircase), and an entropy term added as a
constant bonus that swamped the policy loss.

**Verdict: RL does not beat supervised, second time, now with a correct implementation.** The mechanism is measurable - the
supervised signal is dense, the RL reward is one bit per sequence, so 400 RL steps buy less than 400 extra supervised steps. On
a task where every label is known, RL has no structural advantage; the RL argument belongs to tasks with a checker but no
labels. Details: `docs/RL-V1-METHOD.md`.

## Phase 2 gate, current state on the image task: 4 of 5 items measured

Shipped configuration: 12 named concepts -> 32 linear rules (3 connections each) -> answer, alpha 3, per-rule variance floor
0.10, three seeds.

```
item                                   result
1. accuracy vs baseline                81.73% +/- 0.54 after rule pruning (76.26 +/- 4.62 before)
                                       PHASE 1 BASELINE 86.69% +/- 0.59 -> readable is 4.96 points BELOW it
2. connections per rule                3.00 exactly (min 3, max 3) on every kept rule, all seeds
3. rules a human must read             27 / 26 / 21 kept of 32 (mean 24.7), each one a 3-concept sentence
4. concept honesty                     0.625 +/- 0.034 mean |rho|; 1-4 of 12 below 0.50
   sanity test (another name fits)     2 / 2 / 3 of 12 (mean 2.3) better explained by a DIFFERENT name
5. layer ablation                      NOT YET MEASURED on this config (needs training runs; a game was running)
```

Two honest notes belong beside those numbers. **The readable model does not beat the Phase 1 baseline** - it is 4.96 points
below it - though within the readable family linear rules plus pruning moved it from 76.10 to 81.73 (+5.63). And item 4 can
pass at 0.968 honesty for a concept that changes **0.00%** of predictions (minimum influence is 0.00% in two of three seeds),
so concept honesty is necessary for readability and demonstrably not sufficient.

Item 5 is the one open gate item; the runner support exists and it was deliberately not launched while the machine was busy.
Details: `docs/PHASE2-IMAGE-GATE.md`.

## Output-variance floor: a better ship (81.73% +/- 0.54) and dead concepts still survive

Seventh lever, aimed at the quantity the mechanism named - a rule that emits a constant makes no decisions.

**A hypothesis killed before it cost a run:** the obvious version (`--rfloor`, floor on each rule's answer row) died on a
pre-check - in the shipped arm the per-rule answer-row magnitude has **min 1.747, median 5.648, and 0 of 32 rules below 0.5**,
yet some rules are worth exactly 0.000. Weight is not the quantity, one level up either. Not trained.

`--varfloor F` penalises rule output variance below F. **Gradient check exact (0.00e+00)** away from the relu kink.

```
setting                    pruned accuracy (3 seeds)   worst min influence   mean influence   concepts <1%
ship config, no floor      81.19 +/- 0.98                  0.00%               23.56%          3 of 12
varfloor 0.02              80.64 +/- 1.63                  0.00%               25.27%          1-2 of 12
varfloor 0.10              81.73 +/- 0.54                  0.00%               24.49%          0-2 of 12
```

**New ship configuration: `alpha 3 / 32 rules / linear / varfloor 0.10 / pruned` = 81.73% +/- 0.54**, per seed
82.35 / 81.33 / 81.51, 21-27 rules to read. That is +0.54 points and, more importantly, **half the seed spread** - half the
spread is what makes a claim reproducible.

**But the minimum concept influence is still 0.00% in two of three seeds.** Seventh lever, seventh failure on that specific
point. The pattern across all seven: the two levers that changed the right thing (linear rules, variance floor) also improved
accuracy; every lever aimed at *weights* did nothing. On this architecture weight magnitude is not a proxy for function at
either level.

So the defect stays **open**, and the honest description of the ship is: 12 named concepts, 22-27 load-bearing rules, and one
concept that means what its name says and decides nothing - a caveat that belongs beside the accuracy, not in a footnote.
Details: `docs/VARIANCE-FLOOR.md`.

## Coverage-aware pruning tested: the last sub-cause, and why it cannot work

The mechanism named three sub-causes for dead concepts; linear rules killed saturation, and this tests the last one - never
remove a rule that is the only reader of some concept (`--keep_coverage`).

**Prediction:** minimum concept influence rises above 1%, accuracy cost under 1.5 points. **Both failed.** The guard fired on
1 of 3 seeds (`protecting rule #23: last reader of ring_ink`), cost **0.79 points** (82.26% -> 81.47%), and:

```
rule #23 (the protected last reader): removing it changes accuracy by +0.000 points
switching off ring_ink, ordinary mask:  changes accuracy by +0.000 points
switching off ring_ink, coverage mask:  changes accuracy by +0.000 points
minimum concept influence: 0.00% in both masks
```

**The protected rule is itself worth exactly 0.000 points** - the guard keeps a rule that decides nothing so that it can read a
concept that then still decides nothing.

So the three sub-causes collapse into one: a concept matters only if a rule reading it is load-bearing, and that is a property
of the **rule layer's value distribution**, not of concepts or edges. The one lever that helped confirms it - the median rule is
worth **+6.65 points at 8 rules and +0.07 at 48**, so dead concepts are a symptom of rule-layer redundancy.

**Six levers, full ledger:** delete dead concepts **-8.62**; weight-coverage penalty **none**; dispersion selection **-5.97**
(count-confounded); concept dropout **-10.52**; **linear rules +5.09 and halved dead concepts**; coverage-aware pruning
**-0.27, no effect**.

Two of my own tool bugs were caught and fixed here: the coverage guard aborted the whole prune and restored every removed rule
(seed 0 reported 4 removals and an accuracy identical to the unpruned model), and both prune modes wrote the same filename, so
the coverage run replaced seed 0's ordinary mask - the mode is now in the filename and both masks were regenerated. Details:
`docs/COVERAGE-PRUNING.md`.

## Linear-rule ladder: the price of honest concept names falls to 1.18 points

Every earlier calibration here was done with sigmoid rules, and linear rules changed the landscape, so the grid was retrained:
R in {16,32,48} x alpha in {1,3} x 3 seeds, each pruned, honesty and influence measured in one pass.

```
alpha  rules  pruned accuracy   mean |rho|   misnamed   rules to read   min influence
1.0     16    81.20 +/- 1.26       0.451      6.7 of 12       13.7            0.00%
1.0     32    82.37 +/- 0.76       0.422      7.3 of 12       20.3            0.00%
1.0     48    81.62 +/- 2.31       0.463      6.0 of 12       27.3            0.00%
3.0     16    72.10 +/- 6.85       0.603      2.3 of 12       14.0            0.17%
3.0     32    81.19 +/- 0.98       0.640      2.3 of 12       20.0            0.00%
3.0     48    78.79 +/- 5.32       0.632      1.7 of 12       25.3            0.00%
```

**Prediction:** alpha 3 stays within 2 points of alpha 1 at the same rule count. **Held at R=32 (cost 1.18), failed at R=16
(9.10) and R=48 (2.83)** - so the saturation explanation is right about the mechanism and wrong about universality: honest
names are cheap only where the rule layer has capacity to absorb the constraint.

**Price of honest names, three measurements on this task:** sigmoid concept-weight dial **-9.88 points**, sigmoid best-honest
vs best **-3.68**, LINEAR at R=32 **-1.18** for the same halving of misnaming.

**Ship candidate: `alpha 3 / 32 rules / linear / pruned` = 81.19% +/- 0.98, 2.3 of 12 concepts misnamed, 20 rules to read** -
1.18 points off the grid's best and **+2.15 over the original Phase 2 gate with ten fewer rules**. R=32 is also the only
*stable* honest setting (R=16 gives +/- 6.85, R=48 gives +/- 5.32).

Still broken: minimum concept influence is 0.00% in four of six configurations. Coverage-aware pruning - the one sub-cause of
the three not yet tested - is next. Details: `docs/LINEAR-LADDER.md`.

## Linear rules are the best readable configuration measured on this task: 81.19% +/- 0.98 with 20 rules

The dead-concept mechanism pointed at sigmoid saturation, and the model had supported linear rules since it was written - but
the **trainer never exposed the flag**, so that branch had never been trained once. Now `--rule_linear`.

```
arm                                accuracy (3 seeds)   rules to read
sigmoid rules, a3 R32              72.01 +/- 4.35            32
sigmoid rules, a3 R32, pruned      76.10 +/- 2.40            29
alpha-1 R16, pruned                79.78 +/- 0.48            15.7
LINEAR rules, a3 R32               76.71 +/- 2.39            32
LINEAR rules, a3 R32, PRUNED       81.19 +/- 0.98            20 (19-22)
original Phase 2 gate (sigmoid)    79.04 +/- 2.38            30
```

Per seed with pruning: **82.26 / 80.95 / 80.35**; pruning itself worth **+4.48 +/- 1.52** (10-13 of 32 rules removed). That is
**+5.09 over the best sigmoid honest arm**, **+1.41 over the accuracy champion** and **+2.15 over the original Phase 2 gate**,
with **20 rules to read instead of 30** and the tightest spread measured on this task (+/- 0.98).

The audit detects the mode itself: `rule mode: linear reproduces 79.47% (sigmoid 20.03%) -> MATCH` - a sigmoid read of a
linear-rule model collapses to 20.03%, so the forward pass must match the training pass or every later number is fiction.

Concept honesty on this arm: mean |rho| **0.660**, 1 of 12 below 0.50, 2 of 12 tracking another name. Dead concepts halve -
**2 / 0 / 0 of 12** under 1% influence - but one seed still has two, exactly as the mechanism predicts: saturation was only one
of the three sub-causes. Details: `docs/LINEAR-RULES.md`.

## Concept dropout tested - and the real mechanism of dead concepts found

**First run of the dropout test was invalid, by my own mistake:** I used non-inverted dropout, leaving training and evaluation
with different activation magnitudes. It scored 54.80% +/- 5.04. Corrected to inverted dropout (1/(1-p) rescaling):

```
setting                        accuracy (3 seeds)   min influence   mean influence
no dropout, pruned (reference) 76.10 +/- 2.40         0.00%            20.90%
non-inverted dropout 0.3       54.80 +/- 5.04         0.00%            18.80%
INVERTED dropout 0.3           65.58 +/- 1.83         0.00%            17.72%
dropout 0.6                    16.67 +/- 5.70         0.00%            11.10%
```

The correction recovered **10.8 points** - my implementation flaw was worth more than the lever. Even corrected, dropout costs
**10.52 points**, minimum influence stays **0.00%** and mean influence falls. Prediction failed on both halves. Dropout assumes
redundancy; a concept layer has none, so dropping a concept destroys information rather than forcing robustness.

**The mechanism, measured** (per concept: causal influence vs the rules that read it):

```
concept        causal inf   best rule reading it    sub-cause
ring_ink           0.00%     0.00, 0 readers        every connection pruned away
top_heavy          0.01%     0.00, 2 readers        wired only into inert rules
edge_balance       0.00%     0.00, 3 readers        wired only into inert rules
corner_ink         1.49%    10.04, 2 readers        rule saturated
grad_x_mean        3.14%    15.60, 3 readers        rule saturated
row_ink_std       38.15%    15.60, 9 readers        healthy
```

Three sub-causes: **(1)** the edge does not exist any more, **(2)** the rule it feeds is worth nothing, **(3)** the rule's
output is saturated so the concept cannot move it. All four failed levers attacked *concept usage*; influence is a property of
the **concept-to-rule edge**. Next levers follow from that: coverage-aware pruning that never cuts a concept's last edge,
linear rules (cannot saturate), and training-time rule necessity. Details:
`docs/CONCEPT-DROPOUT-AND-MECHANISM.md`.

## Selecting concepts by dispersion does not remove dead concepts either

Third attempt at the same defect, and the measured reason influence tracks variance (dead concepts std 0.101 vs 0.160 live)
predicted this one should work. `tools/select_concepts.py` now rejects candidates whose activation std is below a bar before
training. **Prediction: accuracy within 1.5 points of 76.10% +/- 2.40 and minimum influence above 2%. Both failed.**

```
arm                             accuracy (3 seeds)   min influence   mean influence   concepts <1%
12 concepts, unpruned a3 R32    72.01 +/- 4.35         0.00%            20.90%          3 of 12
10 dispersion-selected a3 R32   66.04 +/- 2.38         0.00%            31.90%          1-2 of 10
change                          -5.97 points           unchanged        +11.00          -
```

**Honest confound:** this arm changed the concept set AND the count (12 -> 10) at once, and the earlier K=9 arm gave 67.48%, so
most of the drop is probably the count, not the filter. It should have been run at matched K.

The finding that survives: **even when every selected concept varies by construction (std >= 0.12), training still produces
concepts at 0.00% influence.** Deadness is created by training, not inherited from the feature set. Three levers now agree -
deletion costs accuracy, the weight-coverage penalty changes nothing, dispersion filtering changes nothing - while mean
influence rises every time the layer narrows (20.9 -> 31.9) and the minimum stays at zero.

Next lever, and it is a different kind: **concept dropout** - randomly zero a concept during training so the model cannot lean
on the others, making each concept necessary rather than merely readable. Details: `docs/CONCEPT-DISPERSION.md`.

## Coverage penalty built and gradient-checked - and it does not work

Built the coverage lever (`--cov`): a training penalty requiring every concept to attract at least 0.30 of rule weight, from
the engine's own ops. **Gradient check PASSES exactly** (max abs diff **0.00e+00**), after two of my own harness errors: an
identical 4.633e-02 "failure" across six unrelated ops (float32 perturbation - a harness tell), and a 1.36e-03 "failure" that
was just the kink of `|x|` at x=0 under central differences.

Result, three seeds per setting:

```
setting       accuracy (3 seeds)   min concept influence   mean influence
no coverage   72.01 +/- 4.35          0.00%                  20.90%
cov 0.5       72.41 +/- 4.99          0.00%                  19.03%
cov 2.0       72.44 +/- 4.58          0.00%                  19.09%
```

No accuracy cost, no accuracy gain, and **the minimum influence stays 0.00% while mean influence falls slightly**. The penalty
did exactly what it was built to do - per-concept weight usage rose as designed - and the model still ignored those concepts.

**Why, measured:** influence tracks how much a concept's activation VARIES across images, not how much weight points at it.
Concepts with influence < 1.5% have mean activation std **0.101**; those above 10% have **0.160**. A concept that barely moves
cannot change an answer no matter how much rule weight it attracts.

So the next levers are selection by dispersion (`top_heavy` std 0.093 and `edge_balance` 0.075 would have been excluded before
training) and, if a training term is wanted at all, penalising variance shortfall rather than weight. Details:
`docs/CONCEPT-COVERAGE.md`.

## Deleting the causally dead concepts makes the model worse

`explain()` says three of twelve concepts carry no decision (top_heavy 0.01%, ring_ink 0.00%, edge_balance 0.00%). Dropping
them and retraining on nine concepts was the obvious cleanup. Prediction: accuracy within 1.0 point of 76.10% +/- 2.40, mean
influence up from 20.90%. **Half failed.**

```
arm                       accuracy (3 seeds)      mean causal influence per concept
12 concepts, pruned       76.10 +/- 2.40              20.90%   (3 concepts below 1.5%)
 9 concepts (3 dropped)   67.48 +/- 10.01             31.24%   (lowest single concept 0.92-13.83%)
change                    -8.62 points                +10.34 points
```

Seeds of the nine-concept arm: **55.59 / 74.30 / 72.55** - seed 0 collapsed and its own connection pruning cost 10.74 points.

- The **influence half held**: with the three gone, every remaining concept changes 1-80% of predictions.
- The **accuracy half failed by 8.62 points** against a 1.0-point tolerance, and the spread grew from +/- 2.40 to +/- 10.01.
- **The decorative concepts were not decorative in combination.** The concept layer is the only path from pixels to answers,
  so a concept that never wins alone is still a channel the rules combine with others. Nine channels are much worse than
  twelve - not "cleaner at the same accuracy".
- The uncomfortable trade: you can have a layer where a third of the concepts never matter (**76.10%**) or one where every
  concept matters (**67.48%**), and the first is 8.62 points better.

The right lever is not deletion but **coverage** - require every concept to be used by at least one load-bearing rule. That
keeps all twelve channels and makes "every named concept matters" true by construction. Details:
`docs/DEAD-CONCEPT-REMOVAL.md`.

## Retraction: rules sharing a concept triple are thresholds, not duplicates

Last wake I published that the 29 kept rules contain only 20 distinct wirings, so "a human must read 20, not 29". **That was
wrong.** Grouped by wiring, the rules have different biases (`+0.2925 / -0.0201 / -1.1766 / -0.6214` on one group) - they
are a **piecewise response** in the same three concepts, not one rule written four times.

Tested the way it should have been tested the first time - collapse each group to its strongest member and re-measure:

```
seed 0: 76.60% -> 29.84%   (-46.76 points)   9 of 29 removed
seed 1: 73.48% -> 47.18%   (-26.30 points)   5 of 23 removed
seed 2: 78.21% -> 41.26%   (-36.95 points)   6 of 23 removed
```

One two-rule group on the same concepts (biases +0.90 / -0.89) alone carries **12.22 points**; a four-rule group on another
seed carries **29.99**. My prediction was "under 1.0 point" - wrong by 26 to 47 points.

**The corrected claim: 29 rules is the honest reading count.** Nothing is overstated. The real product fix is in `explain()`:
present a rule group as one shape ("when these three concepts are high / medium / low") instead of four separate reasons.
`docs/EXPLAIN-IMAGE.md` is corrected in place with the retraction left visible; full measurement in
`docs/RULE-CONSOLIDATION.md`.

## explain() on the readable image model: real output, and two defects it exposes

`tools/explain_image.py` prints, for a real image: the class, the rules that fired for it, the 3 concepts each rule is wired
to, and what switching the strongest concept off does. Verbatim:

```
image 1: predicted Pullover (score +3.201)   truth: Pullover   CORRECT
  I said Pullover partly because of RULE #18: fires 0.85 -> pushes Pullover +1.410
     when: ink_amount=0.82  AND  left_heavy=0.60  AND  symmetry_ud=0.55
  I said Pullover partly because of RULE #21: fires 0.72 -> pushes Pullover +1.363
     when: ink_amount=0.82  AND  left_heavy=0.60  AND  symmetry_ud=0.55
  if 'ink_amount' were switched off: Pullover -> Sandal
```

**Defect 1 - duplicate rules.** 29 kept rules contain only **20 distinct 3-concept wirings**; 5 wirings are shared by 14
rules (two pairs with answer-row cosine 0.91 and 0.95, two with -0.96 and -0.83, i.e. a rule and its negation). A human
must read **20 rules, not 29** - the reported rule count overstates the reading work by 31 percent.

**Defect 2 - honesty is not influence.** Honesty passes the gate (mean rank agreement **0.735**, all 12 concepts >= 0.556,
none below 0.50) yet the two most label-honest concepts are causally dead: `top_heavy` **0.968 honest / 0.01 percent
influence**, `ring_ink` **0.910 / 0.00 percent**. The highest influence belongs to `grad_ratio` (0.556 honest, **47.87
percent**) and `ink_amount` (0.605, **46.79 percent**). Correlation between honesty and influence: **-0.483**.
Mean influence 20.90 percent.

A neuron can mean exactly what its label says and still decide nothing. Details: `docs/EXPLAIN-IMAGE.md`.

## The pruned ladder: pruning a third of a wide rule layer is worth +4.09 points

Class-prior census per arm (constant-output rules that still push a class): **12% of rules at 8-16 rules, ~31-34% at
32-48** - the degeneracy appears once the layer is wide enough to afford it. That is exactly where pruning pays.

| arm | rules | unpruned | pruned | change | removed | left to read | class-prior share |
|---|---|---|---|---|---|---|---|
| a1 R16 | 16 | 79.88 +/- 0.31 | 79.78 +/- 0.48 | -0.10 | 0.3 | 16 | 12% |
| a3 R16 | 16 | 70.00 +/- 3.41 | 70.39 +/- 3.40 | +0.39 | 1.0 | 15 | 12% |
| a3 R32 | 32 | 72.01 +/- 4.36 | 76.10 +/- 2.40 | +4.09 | 3.3 | 29 | 34% |
| a3 R48 | 48 | 74.73 +/- 1.73 | 76.20 +/- 1.71 | +1.47 | 5.0 | 43 | 31% |
| a3 R8 | 8 | 69.48 +/- 5.35 | 69.48 +/- 5.35 | +0.00 | 0.0 | 8 | 12% |

- **Best honest arm is now `a3 / 32 rules / pruned`: 76.10% +/- 2.40 with 28.7 rules to read**, 1.3 of 12 concepts
  misnamed. Pruning is worth **+4.09** there, the largest single gain measured on this layer.
- It **matches the 48-rule arm (76.20% +/- 1.71) with 14 fewer rules to read** - same accuracy, 29 sentences not 43.
- **Pruning is not universally positive**: on the 16-rule alpha-1 arm it cost 0.30 points on seed 0 and removed nothing on
  the other two; on the 8-rule arm nothing was removable. The gain tracks the class-prior share.
- The accuracy champion is unchanged at `a1 / 16 / pruned` = 79.78% +/- 0.48 with 7 of 12 names wrong. Pruning cannot fix
  concept honesty; that is a separate defect with its own measured price.

Gap between accuracy champion and best readable honest arm: **3.68 points**, with 5.4x fewer misnamed concepts.
Details: `docs/PRUNED-LADDER.md`.

## Rules that read nothing are class priors - and deleting them RAISES accuracy

**15 of 48** rules in the honest arm never read a concept (input strength ~1e-4, output constant across all 10,000 test
images) yet still push a class through their answer row (0.34-0.79). They are hidden class priors wearing a rule's name -
unreadable by construction, and invisible to per-rule ablation, which ablates inputs these rules do not have.

Structural fixes measured: rule bias removed -> max logit shift from a constant rule **0.364 -> 0.297**; centred rule
outputs -> **0.038**; both -> **0.0013** (channel closed). Either alone is insufficient, because `sigmoid(bias) != 0.5`.

Pruning rules that cost <=0.05 train points to delete, complete removal of weights and answer row, three seeds:

```
seed  before   after    change   removed    rules to read
 0   76.56%   77.36%   +0.80     4 of 48        44
 1   74.51%   77.00%   +2.49     7 of 48        41
 2   73.12%   74.23%   +1.11     4 of 48        44
MEAN 74.73%   76.20%   +1.47 +/- 0.90          43
```

**Pruning raises accuracy on all three seeds** - the removed rules were net-harmful constants on borderline images. The
gap to the 79.88% accuracy champion narrows from 5.15 to **3.68 points**, and the honest arm now carries 43 rules with 1.0
of 12 concepts misnamed. Also fixed a bug in my own pruning tool: it sorted candidates the wrong way and printed "removed 0
of 48" for models where 29 of 48 were removable. Details: `docs/CLASS-PRIOR-RULES.md`.

## The honesty tax is not fixed: rules recover half of it at the honest setting

Prediction written first - "at alpha 3, R=32 reaches >= 72% keeping misnamed <= 1.5 of 12". **PASSED** (72.01%, 1.3 of 12,
0.736), and R=48 went further.

| rules | accuracy (3 seeds) | mean \|rho\| | misnamed | median per-rule | dead rules | interaction |
|---|---|---|---|---|---|---|
| 8 | 69.48 +/- 5.35 | 0.740 | 1.3 of 12 | +6.65 | 0.7 of 8 | 9.9% |
| 16 | 70.00 +/- 3.41 | 0.698 | 0.7 of 12 | +1.11 | 1.7 of 16 | 7.8% |
| 32 | 72.01 +/- 4.36 | 0.736 | 1.3 of 12 | +0.31 | 6.3 of 32 | 5.4% |
| 48 | 74.73 +/- 1.73 | 0.743 | 1.0 of 12 | +0.07 | 13.0 of 48 | 6.1% |

- **The tax shrinks from 9.88 points to 5.15**: accuracy at alpha 3 rises 69.48 -> 70.00 -> 72.01 -> **74.73%** as rules go
  8 -> 16 -> 32 -> 48, against the 79.88% alpha-1 ceiling. The 48-rule arm is also the most stable (**+/- 1.73**).
- **Honesty is now insensitive to rule count**: mean \|rho\| 0.698-0.743 and misnamed 0.7-1.3 of 12 across 8-48 rules. Once
  the concept loss is weighted, extra rules extract accuracy from honest concepts instead of diluting them.
- **The price moves to the rules**: dead rules 0.7 -> 1.7 -> 6.3 -> **13.0 of 48**, median per-rule +6.65 -> **+0.07**. At
  48 rules a quarter are inert - the same defect the original 30-rule gate had.

So the trade is no longer accuracy against honest names; it is accuracy against how many rules a human must read.
Details: `docs/HONESTY-TAX-SHRINKS.md`.

## Re-timing the concept layer instead of weighting it: tested, rejected

Hypothesis: the concept head is under-*trained*, not under-*weighted*, so give it its own learning rate (`--lmul`).
Prediction written first - "concept lr x3 reaches honesty >= 0.65 within 1 point of 79.88% accuracy". **Failed.**

| concept lr x | accuracy (3 seeds) | mean \|rho\| | misnamed | median per-rule | dead | interaction |
|---|---|---|---|---|---|---|
| 1 | 79.88 +/- 0.31 | 0.478 | 7.0 of 12 | +0.35 | 1.0 | 9.5% |
| 3 | 68.85 +/- 7.79 | 0.499 | 6.0 of 12 | +0.74 | 1.7 | 4.9% |
| 10 | 65.47 +/- 6.53 | 0.591 | 3.7 of 12 | +0.95 | 0.7 | 1.2% |

Honesty barely moves (0.478 -> 0.591 at x10) while accuracy collapses (79.88% -> 65.47%) and the x3 seed spread blows out
to +/- 7.85. Measured lever efficiency on the same concept set:

```
lever                  accuracy cost   rho gain   rho per accuracy point lost
alpha 1 -> 3               -9.88       0.478 -> 0.698          0.0220
concept lr x10            -14.41       0.478 -> 0.591          0.0078
concept lr x3             -11.03       0.478 -> 0.499          0.0019
```

**Concept-loss weighting is ~3x more efficient than re-timing.** Closed as a lever. One lead kept: the lr lever drives
concept interaction to **1.2%** with every rule load-bearing - useful only where additive explanations matter more than
accuracy. Two bugs in my own audit tool were also caught and fixed: a hard-coded metadata filename that made nine valid
models report parse-fail, and a rule-mode choice that picked the flattering forward pass instead of the one matching the
recorded accuracy. Details: `docs/CONCEPT-LR-LEVER.md`.

## Concept-set size is not the lever; concept pressure is

Tested the obvious fix for the honest configuration (12 concepts at alpha 3 = 70.00% +/- 3.54, 0.7 of 12 misnamed): give it
**18 concepts**. Prediction written first - "+3 accuracy points, <=2 of 18 misnamed". **Failed on both counts.**

| concepts | alpha | accuracy | mean \|rho\| | misnamed | median per-rule | dead |
|---|---|---|---|---|---|---|
| 12 | 1.0 | 79.88 +/- 0.31 | 0.478 | 7.0 of 12 (58%) | +0.35 | 1.0 |
| 12 | 3.0 | 70.00 +/- 3.41 | 0.698 | 0.7 of 12 (6%) | +1.11 | 1.7 |
| 18 | 1.0 | 79.68 +/- 0.59 | 0.485 | 11.0 of 18 (61%) | +0.65 | 1.0 |
| 18 | 3.0 | 72.17 +/- 6.69 | 0.684 | 5.7 of 18 (31%) | +0.81 | 2.0 |

- Accuracy at alpha 1 is flat (79.68 +/- 0.58 vs 79.88 +/- 0.32); at alpha 3 it reaches 72.17% but the seed spread
  explodes to **+/- 6.59** (75.35 / 76.67 / 64.48), so a stable 70.00 was traded for a volatile 72.17.
- Misnaming rose to **5.7 of 18**, not <=2.
- **The rate is the finding**: at low concept pressure **58% of concepts are misnamed at K=12 and 61% at K=18** - the same
  fraction at either size. The concept set decides *what could* be named; the training objective decides *whether it is*.
- The **12-concept set holds both records** (79.88% +/- 0.32 accuracy at alpha 1; 0.7 of 12 misnamed at alpha 3) and is
  the one to ship. Details: `docs/CONCEPT-SET-SIZE.md`.

## A de-duplicated concept set: best readable accuracy so far, honesty still needs pressure

Built a 26-feature candidate pool, then selected concepts that are both learnable (ridge probe on pixels) and mutually
distinguishable (|Spearman| < 0.75). The diversity ceiling binds hard: only **6 / 11 / 12 / 15 / 18** of the 26 survive at
limits 0.60 / 0.70 / 0.75 / 0.80 / 0.85, and `bright_core` failed learnability outright (probe **-0.004**).

| alpha | accuracy (3 seeds) | mean \|rho\| | misnamed | median per-rule | dead | interaction |
|---|---|---|---|---|---|---|
| 1.0 | 79.88 +/- 0.31 | 0.478 | 7.0 of 12 | +0.35 | 1.0 | 9.5% |
| 3.0 | 70.00 +/- 3.41 | 0.698 | 0.7 of 12 | +1.11 | 1.7 | 7.8% |

- **79.88% +/- 0.32** with 12 de-duplicated concepts and **16 rules** - the best three-seed readable accuracy measured on
  this task, beating the original 30-rule 79.04% +/- 2.38 by 0.84 points with **half the rules** and a **7x tighter spread**.
- **At alpha 3 the de-duplicated set reaches 0.7 of 12 misnamed** - lowest of any configuration here - but accuracy falls
  to 70.00% +/- 3.54.
- **My prediction failed**: at alpha 1 de-duplication made misnaming *worse* (7.0 of 12 vs 4.7 of 14) and honesty fell
  0.587 -> 0.478, confirming again that misnaming is not label confusion.

Six checkpoints under `runs/phase2_sweep/SEL12_A*_s*/`. Details: `docs/CONCEPT-SET-DEDUP.md`.

## The concept-honesty dial on Fashion-MNIST: honest names are expensive here

Three concept-loss weights, three seeds each, 16 rules:

| alpha | accuracy | mean \|rho\| | misnamed | median per-rule | dead | interaction |
|---|---|---|---|---|---|---|
| 1.0 | 77.93 +/- 1.40 | 0.587 | 4.7 of 14 | +0.19 | 0.7 | 8.7% |
| 3.0 | 73.65 +/- 3.06 | 0.759 | 2.3 of 14 | +1.38 | 2.0 | 16.0% |
| 10.0 | 63.32 +/- 1.54 | 0.885 | 1.7 of 14 | +1.84 | 3.3 | 10.8% |

Honesty climbs exactly as it did on the chat corpus (**mean rank agreement 0.587 -> 0.885**, misnamed **4.7 -> 1.7 of
14**) but the price is far steeper: accuracy **77.93% -> 63.32%**. Halving the misnaming costs **4.3 points**; nearly
eliminating it costs **14.6**. On the chat model the same dial cost 1.86 points.

Recommended operating point: **alpha = 3, 16 rules** - 73.65% +/- 3.04, honesty 0.759, 2.3 of 14 misnamed. Also
falsified my own collinearity hypothesis: only 1 of the 4 misnamed pairs is collinear, and the worst offender
(`quad_LR_ink`, own rho -0.084) pairs with its apparent target at just 0.160 - the neuron learned something unrelated.
Details: `docs/CONCEPT-WEIGHT-DIAL.md`.

## Rule-count sweep on the original Phase 2 task: 16 rules is the balance

Twelve models (4 rule counts x 3 seeds) on the original Fashion-MNIST architecture. The sweep **reproduces the recorded
Phase 2 result exactly at R=30** (79.04% +/- 2.38 vs the gate's 79.04 +/- 1.97), which validates it:

- **accuracy rises with rules**: 76.75% (8) -> 77.93% (16) -> 78.93% (24) -> 79.04% (30)
- **necessity collapses**: median per-rule ablation loss **+3.01** points at 8 rules, **+0.01** at 30
- **dead rules**: 0.3 -> 0.7 -> 3.7 -> **6.7 of 30**
- **concept interaction**: 16.4% -> 8.7% -> 3.9% -> 4.9%
- **concept honesty is independent of rule count**: mean |Spearman| 0.55-0.62 at every size, with 4.7-6.3 of 14 concepts
  better explained by another name everywhere

**Recommendation: 16 rules** - 1.11 points off the ceiling, 0.7 dead rules instead of 6.7, median rule worth +0.19 instead
of +0.01. The original 30-rule setting is the least faithful of the four. Details: `docs/PHASE2-RULE-SWEEP.md`.

## Re-audit of the original Phase 2 model with today's instruments

`tools/phase2_reaudit.py` reproduces the recorded **80.75%** accuracy exactly (validating the reconstruction), then
disagrees with three of the original gate's claims:

- **Concept honesty:** mean Spearman **0.591**, but **4 of 14 neurons track a different named feature better than their
  own** - `quad_LR_ink` correlates **-0.084** with its own label. AUROC >= 0.55 did not test this.
- **"0 dead rules of 30" is not true by necessity:** median per-rule ablation loss is **+0.01** points and **6 of 30**
  rules are worth nothing (one is negative at -0.19).
- **Rule plausibility:** median pair lift **1.21**, with **10 of 30** rules joining features that do not co-occur.
- **Concept interaction 5.3%**, so per-concept explanations account for ~95% of the effect (chat model: 0.4-3.7%).

The layer is legible but loosely attached. Details: `docs/PHASE2-REAUDIT.md`.

## explain() in the library, by measurement

`gn.load_rules(path)` + `gn.explain_rules(m, tokens)` now exist and are exported from the package root. Real output on a
held-out window - *"at the heart of efforts to increase" -> said 'to' because `shape: last token is 6+ chars` is worth
+1.92 (switching it off favours 'department' by +6.63)* - and per-window additivity is **exact** (interaction 0.0%),
because when no rule crosses its ReLU boundary the model is exactly linear in its concepts.

**Caught a fabricated explanation in my own library:** the first version padded short windows with the vocabulary's id 0,
which is `<unk>`, making `window contains <unk>` spuriously true - so the explainer answered `<unk>` because of a
concept the explainer had created. Training never pads, so the library now raises on short input instead of inventing
context. Details: `docs/EXPLAIN-LIBRARY.md`.

## Additivity replicated on 9 models: per-concept explanation is complete to within 3.7%

Three seeds for each of three configurations, comparing the sum of individual concept effects against switching every
concept off: **UNTIED-11 2.7% +/- 0.6**, **TIED-23 2.1% +/- 1.6**, **TIED-23 distinct pairs 1.3% +/- 1.1** interaction.
Worst single measurement **3.7%**, so a per-concept explanation accounts for **at least 96.3%** of a decision's effect.

My prediction said 'under 3% on every seed' - it passed on all three means and failed on one seed (3.7%), so the claim
that survives is the worst case, not the mean. This does not depend on top-level design choices: tied models add a
rank guarantee, untied models add rule necessity, and **both give an additive measured concept explanation**.
Details: `docs/ADDITIVITY-REPLICATED.md`.

## Correction: concept-level explanation is additive in EVERY configuration

Last wake concluded the architecture 'trades one kind of faithfulness for another'. Measuring concept interaction across
three models falsifies my own prediction that additivity would hold only under a tie: interaction is **1.5%** (tied,
distinct pairs), **0.4%** (tied, repeated pairs) and **2.1%** (untied). Concept-level explanations are quantitatively
complete everywhere.

So the **untied 24-rule model has both** every rule load-bearing (median 0.36) **and** an additive concept explanation
(97.9% of the effect accounted for). What only the tied model provides is the rank proof that the answer layer holds no
hidden degrees of freedom - bought by making rules redundant. Details: `docs/ADDITIVITY-CHECK.md`.

## Rule redundancy is structural; concept-level explanation is faithful and additive

De-duplicating rule pairs under the tie did **not** restore rule necessity: median per-rule ablation loss stayed **0.000**
with distinct pairs (16.05% / 14.30% accuracy over two seeds). Redundancy is intrinsic to the tie - every rule is a
linear combination of the same concept rows, so any rule's contribution can be absorbed by the others.

So `tools/explain_measured.py` explains a decision **causally at concept level**: switch each true concept off and report
how much the predicted word's score moves. Example: *"I said 'the' partly because the window contains 'the' (+1.55) and
partly because it contains a comma (+1.23)."* And the decomposition is **additive**: over 2,000 held-out windows the
individual concept effects sum to **-4.661** against **-4.592** for switching all concepts off, i.e. concept interaction
accounts for **1.5%** of the effect. A reader who follows the concept list has the whole causal story.
Details: `docs/CONCEPT-LEVEL-EXPLANATION.md`.

## Tying gives complete EXPLANATIONS but not faithful CAUSES

Two ablations on the tied configuration, three seeds each:

| config | median per-rule ablation loss | rule necessity | explanation structure |
|---|---|---|---|
| 24 rules UNTIED | **0.358 +/- 0.154** | every rule needed | arbitrary push direction |
| 24 rules TIED-23 | **0.000 +/- 0.000** | every rule dispensable | push = sum of concept rows (rank 22 of 23) |

Under the tie, deleting any single rule changes nothing on **all three seeds** - every rule's row is built from the same
shared concept rows, so rules become alternative routes to one output. Concept-level influence survives (forcing a
concept flips the answer on **29.7%** of windows) but honesty still does not predict influence (**pearson -0.142**), and
`content: window has <unk>` has **zero** influence at 63.6% honesty.

So each defensible configuration holds exactly one property: UNTIED-24 makes every rule necessary but leaves the push
direction arbitrary; TIED-23 cannot be wrong about *why* but its named rule is not necessary. Neither is fully faithful,
and that is the honest state. Details: `docs/TIED-FAITHFULNESS.md`.

## Widening the tied channel: complete explanations at NO accuracy cost

Tying the answer layer to the concepts cost -2.83 points with 11 concepts - and the reason was structural: under a tie
the answer matrix has rank at most **C**, so 11 concepts give only 11 independent directions for 12,002 outputs. Widening
the concept set to 23 recovers all of it:

| config | accuracy (3 seeds) | honesty | layer price | explanation proven |
|---|---|---|---|---|
| untied, 11 concepts | 15.08% +/- 0.46 | 78.0% | 1.57 pp | no |
| tied, 11 concepts | 12.25% +/- 0.99 | 79.2% | 4.97 pp | yes (rank 11) |
| **tied, 23 concepts** | **15.44% +/- 0.22** | 75.2% +/- 2.9 | 3.46 +/- 0.23 pp | **yes (rank 22 of 23)** |

TIED-23 is **+0.36 points** from untied-11, i.e. inside noise: the complete, rank-verified explanation is free in
accuracy. It still costs more in the layer itself (3.46 vs 1.57 points) and in concept honesty (75.2% vs 78.0%, because
the added concepts are weaker). Details: `docs/TIED-WIDE.md`.

## Tied answers: a complete, provable explanation for 2.83 accuracy points

The explanation had a hole: a rule's firing was checkable (two named concepts) but its *effect* was a free 12,002-dim
vector unrelated to those concepts. Tying the answer layer to the concepts - each rule's answer row becomes the **sum of
its concepts' rows** - closes it, and the closure is provable: the saved answer matrix has **rank 11 against 11
concepts**, where the untied model is full rank 24.

| config | accuracy (3 seeds) | honesty | layer price |
|---|---|---|---|
| 24 rules untied | **15.08% +/- 0.46** | 78.0% +/- 1.7 | 1.57 +/- 0.51 pp |
| 24 rules TIED | **12.25% +/- 0.99** | 79.2% +/- 1.1 | 4.97 +/- 1.11 pp |

The tie costs **-2.83 points (5.2x the spread)** and makes the readable layer three times as expensive: a concept's output
row must serve every rule that wires to it, so rules lose the ability to say rule-specific things. Concept honesty is
unaffected. This is the first configuration where 'nothing is hidden' is literally true.
Details: `docs/TIED-ANSWERS.md`.

## RETRACTION: 'plausibility predicts necessity' does not survive seeds

Three seeds per arm, produced by ablating six saved checkpoints:

| arm | median per-rule ablation loss | pearson(lift vs necessity) |
|---|---|---|
| 24 rules | **0.358 +/- 0.154** (positive on every seed) | +0.062 +/- 0.107 |
| 96 rules | **0.000 +/- 0.000** (zero on every seed) | +0.076 +/- 0.036 |

**HOLDS:** at 24 rules every rule is load-bearing; at 96 rules the median rule is exactly inert on all three seeds.
**RETRACTED:** the plausibility-vs-necessity correlation reported earlier as +0.263 and +0.184. Its other seeds are
-0.013 and +0.015, so that figure was a lucky seed; the 96-rule arm is the more consistent of the two at +0.076.
Neither arm shows a real effect: whether a rule makes sense still does not predict whether the model needs it.
Six levers were tried today on the concept vocabulary and the wiring rule and none moved this - so the cause is in the
objective, not the vocabulary. Details: `docs/FAITHFULNESS-SEEDS.md`.

## The rule-count trade, measured with 3 seeds on both sides

Identical recipe, only the rule count differs:

| rules | accuracy (3 seeds) | honesty | readability price |
|---|---|---|---|
| 24 | **15.08% +/- 0.46** | 78.0% +/- 1.7 | 1.57 +/- 0.51 pp |
| 96 | **16.49% +/- 0.17** | 77.8% +/- 1.4 | 0.29 +/- 0.31 pp |

The gap is **+1.41 points, 2.9x the combined seed spread - real**. And it inverts the usual framing: at 96 rules the
readable layer is nearly free (0.29 points) *because 86 of its 96 rules are inert*, while at 24 rules the layer costs
1.57 points **and every rule is load-bearing** (median per-rule ablation loss 0.35-0.50). Concept honesty is identical
either way (78.0 vs 77.8), so rule count trades accuracy against faithfulness, not honesty.
Details: `docs/RULE-COUNT-SEEDS.md`.

## Seed robustness: half of today's accuracy claims were noise

Every result earlier today was **one seed** (the RNGs were hardcoded). With `SEED` added, the best configuration run
three times gives **15.08% +/- 0.46 accuracy** (range 0.88 points), **78.0% +/- 1.7** concept honesty, and a readability
price of **1.57 +/- 0.51** points.

Re-graded against that 0.88-point noise: **word order +6.18, both inputs +4.30, 96->12 rules -2.55** are real;
12->24 rules +1.40 is likely real; pair->facet +0.98 is borderline; and **AR_L1 +0.48, dropping the dead concept
-0.56, 24->48 rules +0.17, cross-facet wiring +0.07, semantic wiring +0.02 are not distinguishable from noise.**

The honest headline is therefore **15.08% +/- 0.46 against a 5.42% baseline**, not the 15.78% single run.
Details: `docs/SEED-ROBUSTNESS.md`.

## Concept selection by data: the 'dead neuron' was an empty label

`shape: last token is capitalised` occurred in **0 of 20,000** training windows (lowercased corpus), so its neuron could
only ever sit at exactly **50.0%** honesty - a concept-set error, not a broken neuron, and nothing in the loss curve
showed it. `FACET_SEL` now drops any label with no positive examples before training, and dropped exactly one:

| config | accuracy | honesty mean/min | median per-rule | worst rule | lift-corr |
|---|---|---|---|---|---|
| 12 concepts (prev best) | 15.78% | 74.8 / 50.0% | 0.500 | -0.550 | +0.263 |
| 11 concepts, dead label dropped | 15.22% | 79.5 / 61.8% | 0.487 | -0.125 | +0.184 (s+0.433) |

Removing one impossible label raised mean honesty **+4.7 points**, healed the net-harmful rule (-0.550 -> -0.125),
left no dead concept, and cost 0.56 points of accuracy (inside single-seed noise, recorded as a move).
Details: `docs/CONCEPT-SELECTION.md`.

## explain() output on the best model, with causal checks

On the 15.78% model (24 rules, 12 facet concepts) `explain()` now prints cross-facet reasons a person can evaluate, e.g.
`rule #17 fires 0.375 -> pushes 'you' +1.85 (reads: earlier: token at -2 is punct AND shape: last token is a digit)`.
Forcing a concept ON changes the answer on **36.2%** of held-out windows, so the readable path is load-bearing. Two
defects recorded: one concept is **dead** (`shape: last token is capitalised`, flip 0.0%, honesty exactly 50.0%), and
concept honesty does **not** predict causal influence on this model (**-0.018**), while rule-level plausibility does
(+0.263). Details: `docs/EXPLAIN-OUTPUT.md`.

## Answer-row sparsity: best middle layer so far (15.78%)

The existing L1 only shrank concept->rule wires, never the rule->answer rows, so a rule could keep full answer weights
while contributing nothing. Adding answer-row pressure (`AR_L1=0.02`) to the 24-rule arm improved four of five metrics
at once: accuracy **15.30% -> 15.78%**, readability price **2.13 -> 1.65 points**, median per-rule ablation loss
**0.350 -> 0.500 points**, plausibility-vs-necessity **+0.166 -> +0.263** (spearman +0.402), and implausible rules among
the heaviest fell 5 of 10 to 4 of 10. One defect survives: a single rule is still net-harmful (-0.550 points when
removed). Details: `docs/ANSWER-ROW-SPARSITY.md`.

## Rule-count sweep: 24 rules is the knee

Four arms, only the rule count changed, each ablated rule by rule:

| rules | accuracy | median per-rule loss | plausibility-vs-necessity |
|---|---|---|---|
| 12 | 13.90% | 1.613 pts | +0.236 |
| 24 | 15.30% | 0.350 pts | +0.166 |
| 48 | 15.47% | 0.000 pts | +0.102 |
| 96 | 16.45% | 0.000 pts | -0.012 |

**The layer stops being fully load-bearing between 24 and 48 rules.** At 24 rules the model gives up only **1.15
points** against 96 rules while every rule participates and plausibility still predicts necessity. Two anomalies are
recorded rather than hidden: one 24-rule rule actively hurts (-0.650 when removed), and half the load-bearing rules
are still implausible at every count. Details: `docs/RULE-COUNT-SWEEP.md`.

## Sparse rule layer: 12 rules where EVERY rule matters

Cutting the layer to 12 rules (cross-facet wiring kept) gives **13.90%** accuracy, and the ablation changes completely:
median per-rule loss **0.000 -> 1.613 points**, i.e. every rule now changes the answer, and the correlation
between rule plausibility and rule necessity turns **positive for the first time (+0.236**, was -0.012). Cost:
**-2.55 points of accuracy** (16.45% with 96 rules) and the readability price rises to **3.65 points**.
So the honest framing is a trade between two true claims: 96 rules is legible but 86 of them are inert; 12 rules is
legible AND faithful - every rule matters - at a measured price. Details: `docs/SPARSE-RULES.md`.

## Rule ablation: the load-bearing rules are NOT the plausible ones

Deleting rules from the answer layer and measuring what accuracy is lost: **correlation between a rule's concept-pair
lift and its usefulness is pearson -0.012, spearman -0.044** - zero. **7 of the 10 most load-bearing rules have
lift below 1** (their concepts do not co-occur), and the one plausible rule in that list costs only 0.100 points
while an implausible one costs 1.475. Deleting the 44 plausible rules costs 7.15 points; deleting 44 random rules
costs 8.10 - indistinguishable. **Median per-rule loss is 0.000 points**: roughly ten rules carry the model and 86
are individually inert. Readable is achieved; faithful is not. Details: `docs/RULE-ABLATION.md`.

## Cross-facet wiring: no impossible rules, best accuracy, cheapest readability price

Restricting every rule to join concepts from DIFFERENT facets: accuracy **16.45%** (highest of any concept set),
ablation 18.00%, so the readability price is the lowest measured at **1.55 points**. **Zero** rules join
contradictory pairs, and the heaviest rules are now plausible - `shape: last token is a digit AND content: window
has '{'` (lift 2.06), `shape: last token is 6+ chars AND position: chat tag in last 8` (lift 1.79). The
influence-vs-lift correlation turned POSITIVE for the first time (**+0.030**, against -0.104/-0.100/-0.119 on the
earlier concept sets). Still open: median pair lift 0.89, and 15 of 96 rules join independent concepts.
Details: `docs/CROSS-FACET-RULES.md`.

## Facet concepts: best concept set so far (16.38%), but the model still prefers impossible rules

12 concepts split into four near-independent facets - shape of the last token, content anywhere, position of the
chat tag, punctuation earlier - score **16.38%** against 15.28% for single tokens and 15.40% for positional pairs,
with the cheapest readability price of the three (**1.92 points**). Cross-facet rules now appear among the
heaviest, e.g. `position: chat tag in last 4 AND earlier: token at -3 is punct`. But the second-heaviest rule is
still `shape: last token is punctuation AND shape: last token is 6+ chars`, lift **0.00** (impossible), influence
74.3: a random wiring still offers contradictions and the objective still likes them. Cross-facet wiring is
running. Details: `docs/FACET-CONCEPTS.md`.

## Token-pair concepts: rules are nonsense because the concepts are mutually exclusive

Replacing single-token concepts with positional/pair concepts ("previous token is punctuation", "last two tokens
both digits") held accuracy (**15.40%** vs 15.28%) but median rule pair lift FELL to **0.72** while median |PMI|
rose to **2.18 bits** - the concepts are strongly related, negatively. The model's heaviest rule is
`previous token is punctuation AND last two tokens both digits`, lift **0.00**: not rare, **impossible**, because a
token cannot be both punctuation and a digit. Conjunctions of positional concepts are therefore either
contradictions or tautologies, which is the structural reason 2-3-concept rules read as nonsense - a design error
in the concept set, not in training. Concept honesty also fell to **74.8%**. Details: `docs/PAIR-CONCEPTS.md`.

## Rule semantics: the biggest rules join concepts that avoid each other

`tools/rule_semantics.py` measures every rule against the data. In the unconstrained model **23 of 96 rules join
statistically independent concepts**, median pair lift **0.98** (the wiring was, on average, exactly independence),
and three rules join **logically impossible** pairs (lift 0.00) - `last token is punctuation AND last token is a
digit` - while still carrying influence 50.7. Constraining the mask to pairs with lift >= 1.5 raised median lift to
**1.38** and PMI to **1.279 bits** and removed the impossible pairs at no accuracy cost (**15.30%** vs 15.28%), but
the influence-vs-lift correlation stayed **negative** (-0.127): the model changed which rules exist, not which ones
it trusts. Concept honesty fell 84.0% -> 79.0%. Details: `docs/RULE-SEMANTICS.md`.

## Causal test: concepts drive the answer, but the rules read as nonsense

`tools/intervention.py` forces one concept neuron on or off and recomputes the answer. Forcing a concept changes the
answer on **43.2%** of held-out windows on average (up to **84.8%** for "last token is '?'"), so the readable path
is genuinely used. But honesty and usefulness are only weakly linked (pearson **+0.253**): `contains '}'` is 93.1%
honest and flips only **20.3%**. And the printed reasons expose the real defect - high-weight rules are arbitrary
conjunctions like "window contains <unk> AND window contains '}'", mechanically true and humanly meaningless.
Accuracy cannot detect that. Rule semantics is the next gate. Details: `docs/CAUSAL-EXPLAIN.md`.

## Honesty is a tax, not a state: the concept warmup interpolates the trade-off

Holding the concept loss at 5x for 800 steps then releasing it to 1x gives **15.28% accuracy / 84.0% honesty**,
which sits on the line between 1x (**16.65% / 71.8%**) and 5x (**13.47% / 92.0%**) instead of above it. That is
about **1.2-1.4 points of accuracy per 6 points of honesty** at every setting, so honesty costs accuracy in each
step it applies rather than being a state the model keeps. Readability price at the balanced point: **2.47 points**.
Details: `docs/HONESTY-IS-A-TAX.md`.

## PHASE 2 GATE: all five items PASS on the deliverable configuration

12 train-probe-selected concepts -> 96 rules (2 connections each) -> answer, 2,692,780 params: accuracy **13.47%**
vs **5.42%** trivial (**+8.05**); **96 rules** to read; concept honesty **92.0%** balanced against an 85% bar
(worst concept 63.8%); ablation **17.18%**, so readability costs **3.71 points**. The 92.0% cost 3.18 points of
accuracy against the same concepts at 1x (16.65% / 71.8%), and the best accuracy ever reached with this layer is
**19.68%** at 64.9% honesty - honesty and accuracy trade through one dial, measured across six settings.
Details: `docs/PHASE2-GATE-FINAL.md`.

## The full 36-concept table: rarity does NOT explain honesty, and 15 neurons are dishonest

The whole table exists now (`tools/concept_table.py`, no retraining): mean **67.1%** balanced, min 48.0%, max 99.5%
('last token is .'). Prevalence vs honesty across 36 concepts: **pearson -0.021, spearman +0.114**, so the earlier
'rare concepts starve' explanation is falsified. The cross-firing test finds **15 of 36 neurons predicting another
concept's label better than their own** - e.g. `contains 'a'` at 67.2% own but 84.7% for "last token is '?'".
The concept set is now selected honestly by a train-split probe (`tools/select_concepts.py`, 12 of 36 kept).
Details: `docs/CONCEPT-TABLE.md`.

## Concept honesty vs accuracy: one dial, three settings

Same recipe, only the concept-loss weight changed: **1x -> 19.68% accuracy / 64.9% balanced honesty**, **2.5x -> 19.20% /
74.6%**, **5x -> 17.82% / 84.0%**. The middle dose costs **0.05 points of accuracy per point of honesty**, half the price of
5x (0.10). So the concepts were under-weighted, not dead, and label-honest concepts cost **1.86 points** of
prediction. The readability price also rises with honesty: the rule layer costs 0.75 points against its ablation at
1x and 2.13 points at 5x. Caveat in the doc: one seed per point, and 84.0% still misses the 85% bar with 'as' dead
at 52.8%. Details: `docs/CONCEPT-TRADE.md`.

## CORRECTION: concept honesty is 64.9%, not 79.2%

The Phase 2 gate's 79.2% was plain accuracy on concepts that are true in a median of only **16.3%** of windows, so
always answering 'no' scores **83.7%** - higher than the reported figure. Balanced accuracy, where a dead neuron
scores 50, puts the concepts at **64.9%** with five of 36 effectively dead (worst 'as' 50.3%). Token accuracy is
unchanged at 19.68% and the readability price unchanged at 0.75 points; only the honesty verdict was wrong.
Details: `docs/CONCEPT-HONESTY.md`.

## PHASE 2 GATE, all five items measured

Three-layer model (36 concepts -> 96 rules -> answer), 2,696,644 parameters: accuracy **19.68%** against a **5.42%**
trivial baseline (**+14.26 points**); **2 connections per rule** after pruning from 3; **96 rules** to read; concept
honesty **79.2%** mean (weakest 57.6%, FAILS the 85% bar); ablation without the rule layer **20.43%**, so
readability costs **0.75 points**. Input choice matters most: window mean 9.20%, last token 15.38%, both 19.68%.
Details: `docs/PHASE2-GATE.md`.

## Phase 2: word order is worth +6.2 points, and readability costs 1.27

The three-layer model (36 concepts -> 96 rules -> answer) scores **9.20%** when the concept layer sees the window
mean and **15.38%** when it sees the last token, against a **5.42%** trivial baseline - so order is worth
**+6.18 points**, the biggest effect measured on this layer. Concept honesty moves the other way: **86.5%** with
the window mean, **70.7%** with the last token, because the concepts are named for window contents. The ablation
with no rule layer scores **16.65%**, so readability costs **1.27 points**, measured. Details:
`docs/RULES-ORDERED.md`.

## Phase 2 rule layer: built and honest, and it fails against the right bar

36 concepts -> 96 sparse rules (3 wires each, pruned to 2) -> answer now exists. On held-out windows it scores
**5.42%**, and 'always output the most common token' also scores **5.42%**: it learned only the frequency prior.
My prediction ('within 2 points of the ablation') passed, which is exactly how a bad gate hides a null result -
the ablation scores 3.55%, below a constant. Cause: mean pooling destroys word order, so there is nothing to
condition on. Readability numbers are real: 2 connections per rule, 96 rules, concept honesty 82.9% mean.
Details: `docs/RULES-GATE.md`.

## Identity pointer: 83.3% held-out copying, a new best

Deleting the pointer's learned readout matrix and tying it straight to the embedding table raises held-out copying
from a previous best of **70.2%** to **83.3%** at step 1,000 (1,500 held-out items), taught copying 84.7%, with
language 13.45% at that point. The reasoning is exact rather than empirical: `(alpha-weighted window embedding) @
E^T` is the attention-weighted sum of one-hot token scores, so attending to a position outputs the token there,
with no parameter in between for language training to overwrite. Prediction on record: copy PASSED (75% bar),
language FAILED at that checkpoint (13.45% against 16%). Details: `docs/IDENTITY-POINTER.md`.

## The library works in five lines, and two-stage training fails in all three forms

`import glassnet as gn` now exposes the real engine: `gn.load_chat()`, `gn.model()`, `gn.fit()`, `gn.why()`.
Real run: auto-pilot chose 15,567,566 params, d512, lr 0.30, loss 7.86 -> 6.08 in 3.09 s, **held-out 4.70%**
after three epochs, and `explain()` returned the concept decomposition with residual 11.72 and `exact: False`
(the readable path does not explain the whole logit - the same 7.3% share already published). Three epochs is a
working pipeline, not a good model, and it is labelled that way. `docs/LIBRARY.md`.

Separately: language-first two-stage training was measured three ways and **failed every time** - held-out copy
6.8% unfrozen, 3.2% frozen, 2.5% frozen with a separate frozen embedding, against predictions of 40-50%. Freezing
also cost 8-10 points of language. `docs/TWO-STAGE-FAILS.md`.

## The copy peak, measured properly: 69.8% held-out copying at 1,000 steps

On 3,000 evaluation items per checkpoint (up from 600): held-out copying **70.2 / 63.1 / 56.8 / 46.5%** for seed 0
and **69.3 / 65.5 / 55.1 / 47.2%** for seed 2 at steps 1,000/2,000/3,000/4,000, while language climbs 14.6 ->
20.4%. The prediction (50%+ at step 2,000 on both seeds) **passed**. Copying decays monotonically as language
improves: a measured trade-off, not a tuning problem. Checkpoints are now written at every evaluation, so the
70.2% model is a file (`early_s0_best.npz`), not a line in a log. Details: `docs/COPY-PEAK.md`.

## Copy generalisation peaks early: 63.7% at 2,000 steps across three seeds

Routing, three seeds. Held-out copying at step 2,000 is **64.3 / 66.7 / 60.2%** (**mean 63.7, spread 6.5**) while
language is still ~16%. By step 12,000 language reaches **25.4%** (mean of three seeds, spread 0.6) and held-out
copy drifts and swings (1.7-56.2%). Mean of last three: copy **26.9%**, language **25.4%**. So the two abilities
trade against each other along one run, and the copy operation is strongest at the START of a mixed run. Next
arm trains language first and adds a short copy stage at the end. Details: `docs/ROUTE-SEEDS.md`.

## The data judge exists and fails its own gate

Fifteen readable features, five corruption kinds, held-out conversations. Without order features: AUC **0.922**
but shuffled word order detected only **1.0%**. With them: AUC **0.98**, but **29.4%** of clean held-out windows
flagged. With a calibrated threshold: AUC **0.9597** PASS, false alarm **0.0730** FAIL (bar 5%), one-token-loop
detection **69.5%** FAIL (bar 80%). Not wired into training: a filter that throws away 7% of good data costs more
than the damage it removes. Next design recorded: score a window against its own neighbours instead of a global
threshold. Details: `docs/DATA-JUDGE.md`.

## Routing beats adding: the first model that talks AND copies

Switching between the pointer's readout and the other pathway's readout, instead of adding them, gives held-out
copying of **39.5 / 3.2 / 56.2%** over the last three checkpoints (**mean 33.0%**) with language at **25.8%**
(mean). Every earlier mixture sat between 0.2% and 16.5% held-out. The reason is documented in
`docs/POINTER-AIM-VS-VOTE.md`: the pointer never loses its aim, it loses the SUM. A switch does not have to
out-sum anything. Caveat stated in the doc: this metric swings hard between checkpoints, so 33.0% needs three
seeds before it is a stable number. Details: `docs/ROUTE-BEATS-ADD.md`.

## The pointer keeps its aim and loses the vote

Measured on the trainer's own items: on held-out words the pointer's attention still sits on the correct window
slot with **0.77 mass against 0.031 chance**, and it gets the word right **58.0% (63.0% after mixing)** alone -
but the assembled model outputs it only **0.5-14.8%** of the time. So the copy operation is intact and the
mixture outvotes it: the problem is voting, not erosion, and every arm that tried to fix it by training longer,
freezing, isolating the embedding table or lowering the learning rate addressed the wrong half. Details:
`docs/POINTER-AIM-VS-VOTE.md`.

## CORRECTION: separate embedding table 15.7% at the finish, and single checkpoints must not rank arms

I published that the separate embedding table was worse (3.3%) by comparing its step-4,000 reading against a
finished 12,000-step arm. Its finished figure is **15.7% held-out copying, the highest today**, with **30.15%**
language. But held-out copy swings 7.0 / 9.3 / 7.5 / 15.7 across checkpoints on identical settings, so single
readings rank coin flips: mean of the last three is **14.8%** for the gate arm and **10.8%** for the separate
table, i.e. **the two are indistinguishable**. What is established is the direction 0.2% -> 3.8% -> 14.8-15.7%.
Details: `docs/SEP-EMB-CORRECTION.md`.

## Six ways to mix a copy pathway into a language model, measured

The pointer copies unseen words at **58.0%** alone. Every mixture measured today lands between **0.2% and
16.5%**: 50/50 mixture 4.2%, 24k steps 8.5% (plateau), frozen pointer 0.2%, bounded gate 3.8%, **unbounded gate
14.8%**, separate embedding table 3.3%. Two findings worth the space: a gate bounded in [0,1] can only attenuate
an already-too-quiet voice, and freezing the pointer's own matrices protects nothing because it reads the shared
embedding table that corpus batches keep moving. Best configuration: language **29.77%** with **96.0%** taught
and **14.8%** held-out copying. Details: `docs/MIXING-VARIANTS.md`.

## A model that talks and copies at once: 29.82% language, 85.7% taught copy, 4.2% held-out

First arm to train both abilities together (50% chat corpus, 50% synthetic copy, pointer instead of the old
echo path, 8,000 steps): language **29.82%** on matched windows against **24.27%** for the same model before
the arm (**+5.55 pp**, same ruler - note this is NOT the project's 25.52% assistant-by-position scale),
copying taught words **85.7%**, held-out words **4.2%** against 58.0% for a copy-only model. Held-out copying
was still rising at every checkpoint, so the next arm runs three times longer. Details:
`docs/MIXED-COPY-FIRST.md`.

## The pointer copies unseen words at 58.0%; the old copy path was outvoting it

Three checkpoints on the same 600 mixed items: **pointer-only 83.8% taught / 58.0% held-out** (prediction was
50% or better: passed), pointer+Q/K **94.2% / 0.5%**, pointer+Q/K+gate **96.0% / 0.7%**. So the copy operation
generalises - a word the model never trained on is copied correctly most of the time - and the `dmix` echo
path is what destroys it. The 0.7% reading was also masked by a bug of ours: `pointer_only` was applied before
the `dmix` path was added, so it was never actually pointer-only. Details:
`docs/POINTER-GENERALISES.md`.

## The pointer works, it is being drowned out by the other pathways

The 6,000-step arm scored **94.2% taught / 0.5% held-out**, which looked like memorisation. An oracle test on
the same checkpoint says otherwise: the pointer pathway ALONE copies **84.0%** of taught words and **66.3%**
of held-out words, and lands on the answer position **83.5% / 85.7%** of the time. So the copy operation
generalises; the full model's other pathways simply outvote it and answer `the`. A correct mechanism that
cannot be heard is indistinguishable from a broken one until you isolate it. Next: a learned gate so the
pointer can be heard - prediction, held-out copying at 50% or better. Details:
`docs/POINTER-WORKS-DROWNED-OUT.md`.

## Copying from the window: 17.5% taught, 0.0% held-out, so it cannot copy at all

First benchmark task forced to a score, 3,000 steps from the 25.52% checkpoint: loss fell **6.86 -> 1.73**,
taught words reached **17.5%**, held-out words **0.0% at every checkpoint**. Learning the act of copying would
copy unseen words too; fitting the answer list would not. So it is memorising, and the cause is structural:
the architecture flattens the 32-token window through a gated mix with **no attention, no per-position
addressing, no pointer**, so it cannot reach back and take one token. This explains every 0.0% in the
benchmark table. Next arm adds an explicit readable pointer. Details: `docs/COPY-TASK-STRUCTURAL.md`.

## First benchmark: 0.0% on all six tasks, and that is the baseline

`tools/bench_v0.py` scores arithmetic, subtraction, sequence continuation, in-window copy, yes/no and world
knowledge, 200 generated items each, chat template, `<unk>` allowed and banned, with a frequency baseline.
Result: **0.0% everywhere, 1,155 scorable items**, and the model answers `the` (or `<unk>`) to everything. So
the 25.52% next-token figure was never comparable to anything and concealed that the model cannot do a single
checkable task. Two harness bugs were fixed first: a fake 22.5% from scoring missing answers as `<unk>`, and
`<unk>` never having been banned in earlier generation tests. Details: `docs/BENCH-V0.md`.

## Vocabulary decision: keep word tokens (measured, 2.1049 bits/char, best on both counts)

Three tokenisations, identical model shape and compute, bits per character: words **2.1049** with
**0.2561 tokens/char**, hybrid **2.1648** / 0.2670 (+2.8% bits, 0% unknown), pieces **2.2729** / 0.2751
(+8.0% bits, 0% unknown). So removing the `<unk>` cap costs 2.8-8.0% in-domain. The cap is real (13.73% of
positions want an unsayable word; 23.01% of out-of-domain tokens destroyed) but the insurance is only worth
its premium when we have an out-of-domain benchmark. Details: `docs/VOCAB-DECISION.md`.

## Words versus pieces: the word list wins in-domain by 10.9% in bits per character

Same text, same shape, output tied to embeddings, same compute. Words **2.1003 bits/char** at 12,002 entries with
2.11% unknown; pieces **2.3581 bits/char** at 16,000 entries with 0.00% unknown and 32% more parameters. So
switching to byte-level pieces costs efficiency on in-domain text even though the vocabulary cap is real. The
measured answer is a HYBRID: keep word tokens, add byte fallback only for unknown words. Details:
`docs/TOKENS-VS-PIECES.md`.

## Anchored RL: collapse cured, calibration still worse than supervised

Adding a supervised anchor inside the RL loss kept the distribution alive (**entropy 3.529** against the
reward-only collapse to **1.191**) and put **more probability on the right answer than any other arm**
(**0.189** against 0.150). Accuracy tied the control (**24.50% vs 24.17%**), but echo was worse
(**36.8% vs 31.0%**) and confidence error worse (**0.340 vs 0.287**). Plain conclusion: at this size our RL
method does not beat supervised training on accuracy, its unique ability is trading accuracy for repetition
control. Details: `docs/RL-ANCHORED.md`.

## RL dose response: penalty works, accuracy pays for it, confidence collapses

Echo falls monotonically with the repetition charge **57.4 -> 42.4 -> 13.3 -> 0.6%** and accuracy falls with it
**20.17 -> 20.08 -> 14.17 -> 6.25%**, against a **24.17%** supervised control. At charge 0.1 echo actually
rose above the start point (36.8 -> 42.4%), because reward-only training pushes echo up by itself. Confidence
error climbs **0.287 -> 0.534 -> 0.726 -> 0.920**: the model does not get honest, it gets loudly wrong. The
lever is the reward SHAPE, not the dose. Details: `docs/RL-DOSE-RESPONSE.md`.

## First RL ladder: repetition 36.8% -> 0.6%, and accuracy 24.17% -> 6.25%

Our own policy-gradient trainer, three arms from the same checkpoint, 2,000 steps each. Control (ordinary
training) reached **24.17%** with **31.0%** window-echoing. Reward-only reached only **20.17%** while echo
rose to **57.4%**. Adding a repetition charge crushed echo to **0.6%** and crushed accuracy to **6.25%**: at
`lam_rep=1.0` the repetition term was three times the size of the correctness term, so the model did what it
was paid to do. First lever in this project that ever moved repetition. Details: `docs/RL-LADDER-1.md`.

## 2026-11-10, the day in one page

Ten arms trained or measured, seven lost. The loop survived four levers (symbol mask, decode guard, copy
penalty, exposure bias - the last finished at **19.94% vs 25.52%**) and four explanations were falsified, the
last being confidence sharpening. Capacity is not the answer (**26.02% vs 25.52%** for +89% params, and the
readable path got *less* aligned, **8.2% -> 4.3%**). Readability costs **5.22 pp** of accuracy but also
**0.18 mean AUROC** of honesty (**30/36 vs 36/36** concepts passing). Full writeup:
`docs/NIGHT-2026-11-10.md`.

## The readability tax is bigger than accuracy: concepts stop tracking their own labels

Same 800 matched windows, project gate (`lens.api.audit`, AUROC bar 0.55): readable-only **30/36** concepts pass
against **36/36** for the copy-path model, mean AUROC **0.6886 against 0.8695** (drop 0.1810), mean fire rate
**47.4% against 28.3%**. Six concepts fail, all exclusively in the readable model - `mentions_body` 0.4275,
`mentions_a_name` 0.4662, `mentions_sport` 0.4774, `mentions_a_quantity` 0.5435, `mentions_family` 0.5437,
`is_a_list` 0.5471 - so the formulas are not wrong; they fail only when the concepts must carry the whole
prediction. Details: `docs/CONCEPT-LABEL-AUDIT.md`.

## CORRECTION: the loop is not the copy path's fault

The concept-only arm (no copy path at all) loops **worse**: symbol share **27.8%** against 8.7%,
longest symbol run **23** against 3, distinct 4-grams **19.8%** against 54.0%, copy rate at steps 17-24
**100.0%**. So the copy path carried the loop in one model; it is not its cause. The loop forms in a small
next-token model under greedy decoding whichever route the answer takes. Details:
`docs/READABLE-ARM-VERDICT.md`.

## Readable-only arm: readability costs 5.22 pp, not 19 pp

`--direct` dropped so the 36 concepts are the only route to an answer, same recipe and matched split:
**15.41% / 18.70% / 20.30%** against **19.69% / 24.26% / 25.52%**, so readability costs **5.22 pp** - the
earlier 18.50-19.17 pp was the price of retrofitting concepts onto a copy-trained model, not of a readable
model. Blemish: only **28/36** concepts pass the label-honesty gate here against 35/36. Details:
`docs/READABLE-ARM-VERDICT.md`.

## Capacity verdict: doubling width bought 0.50 pp, all of it unreadable-path

d=512 -> d=768, 15.57M -> 29.42M params, matched split, one change: accuracy **19.69/24.26/25.52% ->
20.30/24.16/26.02%**, so **+0.50 pp** against a measured single-seed spread of 0.26-3.07 pp. The trainer flags
the arm UNDER-TRAINED at **2.18 tokens per parameter**, so capacity and budget are confounded. Same
decomposition both widths: concept-only accuracy **3.00% -> 4.67%**, copy-only **20.17% -> 21.50%**, and
concept agreement with the model's actual answer **8.2% -> 4.3%** - so the extra capacity went to the copy
path and the model became **less** readable. Details: `docs/CAPACITY-VERDICT.md`.

## Self-context final verdict: prediction falsified, and the day's synthesis

Three epochs, matched split, one change: accuracy **19.94% against 25.52%** (-5.58 pp), copy rate over steps
9-24 **89.9% against 81.3%** (I predicted a fall below 90%), previous-token echo **30.2% against 0.0%**,
modal echo **41.7% against 25.0%**, symbols **16.7% against 8.7%**. Only distinct 4-grams improved (76.2%
against 54.0%). The epoch-1 improvement reversed - the second lever today to do that, which makes
single-epoch comparisons non-evidence in this setup. The confidence-sharpening explanation was also
falsified (baseline entropy **rises** 4.19 -> 5.02 while its in-window rate **falls** 75.0% -> 70.8%).
**Synthesis: the loop is a decoding artefact with a working fix (no-repeat 4-gram blocking, 39.3% -> 100%
distinct 4-grams); the content is a model-and-data problem no decoding trick touches.** Details:
`docs/SELFCTX-FINAL-VERDICT.md`.

## Self-context, epoch 1: text much less repetitive, copy curve barely moved

Against the baseline's epoch 1 (both equally trained): repeated 4-grams **63.9% -> 85.7%**, symbol share
**52.4% -> 24.3%**, longest symbol run **16 -> 7**, at a cost of **4.95 pp** of accuracy (19.69% -> 14.74%)
and 1.84x the training time. But the mean copy rate over steps 9-24 only moves **91.4% -> 87.5%**, and it
rises at steps 5-8 and 21-24, so the curve is nudged rather than reshaped. The output trades a backtick loop
for a comma loop and still answers nothing. Interim: one seed, epoch 1 of 3. Details:
`docs/SELFCTX-EP1.md`.

## Run 9: the state finally reacts, the readout still does not

`tools/eval_run9.py` on `scan9_ep1500.npz`: **F1 KL 1.1702 PASS** (bar >0.05), **F2 state cosine 0.1736 PASS**
(bar <0.5, against 0.9716 at run 6), guards clean (**gain 0.978-1.119**, entropy 94.1% of uniform), but
**F4 FAIL**: the trained readout scores **1.1702** against **1.1660** for a random-noise readout, a ratio of
**1.00x** where 2x is required. So the state now carries prompt-specific variation and the answer layer does
not use it - the same shape of finding as the readable model, on a different architecture. Run 9's own sample
answers are still symbol soup, so this is not a talking model. Details: `docs/RUN9-VERDICT.md`.

## The loop is autoregressive feedback, measured by generation step

Share of emitted tokens already in the window, by step: conversational model **6.2% at steps 1-4, 96.9% at
steps 9-12**; merged model **28.1% then 90.6%**. The model writes its own tokens into the window, and after
about eight steps almost anything it can say is something it can already see. Three rival explanations were
falsified today: corpus repetition rate, boilerplate density (math/code carries MORE boilerplate and loops
less), and memorisation - every 16-gram of the training corpus was hashed and **0.0%** of generated 16-grams
appear verbatim in it, so the loop is generated, not recalled. Details:
`docs/FEEDBACK-LOOP-MECHANISM.md`.

## Prediction falsified: the conversational corpus loops worse

Training on the conversational portion only, with the prediction recorded in advance: accuracy **17.35%**
against 25.52% on the merged corpus, in-window copying 69.1% against 74.3%, **copy mass 46.2% against
30.8%**, previous-token echo **39.6% against 0.0%**, symbol share in generation **58.3% against 8.7%** and
the longest symbol run **23 against 3**. So the conversational data, which repeats itself less by direct
measurement, produces a model that loops more - the repetition rate of the corpus is not what drives
looping. The visible culprit is boilerplate: `human`, `assistant`, quotation marks and backticks are cheap
and self-reinforcing under the copy path. Details: `docs/CONV-ARM-FALSIFIED.md`.

## Which part of the corpus teaches the repetition

Per source (`tools/source_repetition.py`): the target is already visible in the window in **38.32%** of
conversational windows against **51.75%** of math/code/book windows, and that math/code/book source is
**255.2M of the 294.2M training tokens, 87%**. Symbol share 17.28% against 30.25%. So the corpus is mostly
math and code, not prose - **correcting my earlier claim** that the model learns prose and is graded on code.
The looping follows from the data: the model spends most of its training on text where the answer is often
already visible. An arm training on the conversational portion only (38.8M tokens, matched holdout, one
change) is running. Details: `docs/SOURCE-REPETITION.md`.

## The copy penalty failed, and the reason reframes the problem

Three epochs, matched split: accuracy 19.69/24.26/25.52% baseline against 21.75/24.24/25.60% penalised, so the
epoch-1 gain washed out to **+0.08 pp** and the copy mass to **-1.7 pp**; generated text is slightly more
repetitive. The copy path is not capacity-limited either: **62.9%** of its 512 units are alive and the top 10
hold only **8.7%** of its energy, so the penalty fights the loss, not a bottleneck. Then the number that
reframes it: the true next token is already visible in the window **40.33%** of the time, while trivial copy
baselines score 6.62% and 2.75% and the model scores **25.60%**. So copying is often correct and the model is
not a copy machine - the repetition in generation is mostly the corpus repeating itself. Details:
`docs/COPY-PENALTY-VERDICT.md`.

## Copy-pressure penalty, first epoch: accuracy +2.06 pp, copying down

Interim, one seed, epoch 1 of 3, matched split, one change (`--lam-copy-penalty 2.0`). Against the baseline's
epoch 1: assistant accuracy **19.69% -> 21.75%**, previous-token echo **38.5% -> 24.3%**, modal-window echo
**45.8% -> 38.5%**, copy path's mass on visible tokens **36.9% -> 29.7%** (frequency justifies about 10%).
Source: `lens/runs/MATCHED-COPY-D512.txt` against `lens/runs/MATCHED-D512.txt`, measured by
`tools/copy_echo.py`. This is the first lever today that reduced the copying while also raising accuracy,
but it is one seed and one epoch - the final epoch and two more seeds decide it. Details:
`docs/COPY-PENALTY-EP1.md`.

## How much of what the model says is just visible in the window

12 prompts x 24 greedy steps (`tools/copy_echo.py`): emitted tokens are already in the window **74-81%** of
the time against **7-9%** by unigram expectation. The role splits by training length: the 3-epoch models
echo the previous token **0.0% / 7.3%**, while the 6-epoch model that scored 37.37% echoes the previous
token **61.5%** and the modal window token **58.3%**. So the echo grows with training, and the best-scoring
checkpoint is the one that copies hardest. The masked arm finished at **19.19% / 22.53% / 23.31%** against
**19.69% / 24.26% / 25.52%** unmasked, so the symbol mask costs **2.21 pp**. Details: `docs/COPY-ECHO.md`.

## The punctuation is cured; the loop is not

12 fixed prompts, 24 greedy tokens each, same harness: the copy-path symbol mask takes symbol share from
**8.7% to 0.0%** and the longest symbol run from **3 to 0** on every prompt, at a cost of **0.50 pp** of
validation accuracy (19.19% against 19.69% at epoch 1). But the loop did not stop, it moved: `hello , how are
you ?` now gives `the current of the function of the function of the function`, and distinct 4-grams fell from
**54.0% to 47.6%**. So punctuation was a symptom; the disease is that the copy path is self-reinforcing for
whatever is frequent in the window. Details: `docs/SYMBOL-MASK-VERDICT.md`.

## The price of readability: 19.17 pp

Same harness, 600 matched validation windows (`tools/concept_only_eval.py`): **full 22.17%**, **copy path
only 20.17%**, **concept path only 3.00%**, and the concept path agrees with the full model **8.2%** of the
time. So giving the 36 named concepts the decision costs **19.17 pp**, and concept-only accuracy lands
**below** the 4.23% frequency baseline - a person reading the readable numbers would score worse than always
answering `,`. The model's current ability is, numerically, its copy path. Details:
`docs/READABILITY-PRICE.md`.

## The readable half explains 7.3% of the model's decisions

Splitting the logits the model already computes into its 36-number concept path and its copy path, on 233
matched validation windows: logit variance **0.7 from the concepts** against **2.3 from the copy path**, and
the concept path alone picks the same token as the full model only **7.3%** of the time. The concept rows
are near-orthogonal (participation ratio **35.97** of 36), so the readable layer is not redundant, it is
simply not the deciding path. So `explain()` can name which concepts fired and still not be the reason for
the answer, and it must say so. Measurements: `docs/READABLE-PATH-EXPLAINS-7PCT.md`.

## Where the punctuation loop comes from, traced in the open model

The architecture separates into a 36-number concept path and a copy path that echoes the window. On
`the capital of france is`, tracing 14 generated tokens: the **concept path's** symbol probability stays
at **0.007-0.008 for all 14 steps**, while the **copy path** goes from 0.021 to **0.965** and carries the
entire loop. Each emitted backtick is appended to the window and echoed again, so the loop is
self-reinforcing, which is why a decode guard could not fix it. Separately, all 36 concepts have a symbol
weight of about **0%**, but their top-voted tokens are unrelated word scatters (`c0: might pound german
bills flood luck`), so the concept layer is neither the cause of the loop nor yet a readable layer.
Trace and prediction: `docs/SYMBOL-FORENSICS.md`.

## A decode guard fixes the looping, not the meaning

Measured on `gpu_d512_s0_best.npz`, 12 prompts, 24 tokens each (`tools/decode_guard.py`):

```
guard      symbol share   longest symbol run   distinct 4-grams
greedy          67.7%            15.0               39.3%
norepeat        57.6%             5.0              100.0%
symrun          50.7%             2.3               69.0%
both            46.5%             2.4              100.0%
```

The guards work - repetition is eliminated - but the text stays meaningless and the model still spends
**46.5%** of its emitted tokens on symbols against **27.7%** in the corpus, about **1.7x** the rate it was
shown. So the preference lives in the weights, not the decoder: decoding tricks are ruled out as a fix.
Details: `docs/DECODE-GUARD.md`.

## Matched validation split built; the original mismatch was +2.65 pp, not +16 pp

`tools/make_matched_split.py` now draws a validation holdout by hashing conversation ids, and
`train_chat.py --split-file` uses it with a length check against the corpus. On the real 294,225,090-token
corpus, sampled through the trainer's own window builder (4,000 windows per side, symbol share of target
tokens): shipped split **val 31.42% vs train 28.78%, +2.65 pp**; matched split **val 28.52% vs train
28.90%, -0.38 pp**. The 13.29% vs 44.04% figures in `docs/DATA-SPLIT-SHIFT.md` are real but describe the
extremes of the file, not the validation split - so the mismatch is real and small, and my earlier
framing overstated it roughly six-fold. Correction and details: `docs/MATCHED-SPLIT.md`.

## Correction: the code tail is the model's easiest region, not a handicap

I previously wrote that the code-versus-conversation split mismatch partly explains a low held-out
accuracy. Measured by region, the opposite is true: the held-out code tail is where the readable model
performs **best**, at **33.00%** accuracy and **perplexity 107.6**, against **17.33-25.00%** and
**270-635** in the conversational regions it trained on. So the split mismatch **flatters** the headline
rather than depressing it. What survives: the split really is compositionally different (13.29% symbols
in the first 200k against 44.04% in the tail), the symbol attractor really is partly the code tail
(47.7% of targets are symbols, 71.3% of predictions), and per-source reporting is needed because an
average hides a 17-33% spread. Details: `docs/CORRECTION-TAIL-IS-EASIEST.md`.

## The held-out split is a different dataset, not a held-out sample

The corpus is a concatenation of sources and the validation tail is the end of it. Symbol-token share:
train first 200k **13.29%**, train middle **28.08%**, held-out last 200k **44.04%**. Decoded, the training
text is conversational (`<|im_start|> user how should i dress for a job interview ?`) while the held-out
tail is **Python code** (`def withdraw ... if amount > 0 : ...`). So the model learns prose and is graded
on code. That partly explains the near-baseline held-out accuracy, and it explains the symbol attractor
(44% of held-out targets are symbols; the model predicts symbols 68.3% of the time). Every accuracy
number measured on this tail is a cross-domain number. Fix: randomised holdout, then re-measure.
Details: `docs/DATA-SPLIT-SHIFT.md`.

## What the architecture that learns actually says

Sampled output from the readable model, which scores 31.50% next-token accuracy at 24 tokens of context:
`the capital of france is` -> `8 * chapter 1 . * * * * * *`;  `i feel sad because` -> `it transaction by -
levels nicotine concepts and saving back to handle t brand government addressing . however`.

Local grammar is real (`we can set the following equation as follows`, `which chapter 2`, `however`) but
global meaning is not, there is no factual recall, and a symbol attractor takes over within a few steps
under both greedy and sampled decoding. So the model is **not** talk, and the symbol attractor is now the
top open item: a model can be at 31.50% next-token accuracy and still be unusable if decoding walks into
`*` and never leaves. Details: `docs/GENERATION-REPORT.md`.

## The readable architecture has real signal; the recurrent talk core does not

Same corpus, same vocabulary, same held-out tail, same measured frequency baseline of **3.99%**. The
readable concept/answer model scores **11.00%** with just one token of context (**2.8x** baseline) and
**31.50%** at 24 tokens (**7.9x**). The recurrent talk core scores **4.69%**, which is **1.18x** - within
noise of answering `,` every time. Context is demonstrably used by the readable model (accuracy rises
with length); the core's own measurements show no such rise. So talk should be built on the
concept/answer family. Details: `docs/ARCH-COMPARISON.md`.

## The model is 0.70 points better than always guessing the most frequent token

Over 20,000 held-out positions, scored against a frequency baseline on the same positions: always
predicting `,` gives **3.99%**, the trained model gives **4.69%** - a **+0.70 pp** advantage, **+17.5%**
relative. The model is also not just copying the unigram (log-correlation **+0.1588**, KL **1.8330**
nats), so it neither learns context structure nor reproduces the base rates faithfully; it has collapsed
onto a handful of answers. That is why every readout and state hunt failed: there is almost no learned
signal to express. Every future arm reports the frequency-baseline gap beside its accuracy.
Details: `docs/MARGINAL-BASELINE.md`.

## The map responds but at too low a resolution - it is not compression

Over 4000 pairs of real held-out contexts: Spearman(state distance, logit distance) **rho +0.610**, and the
top state-distance decile gives **27.92** logit distance against **17.54** in the bottom decile, a **59%**
increase. The state-to-answer map is therefore NOT squashing different states into one answer. What is
wrong is resolution: even at state cosine distance **0.7582**, predictions differ by only **0.138 nats**
of KL, against a top-1/top-2 gap of about **0.008** with 12,002 candidates. Ruled out by measurement now:
state collapse, direct-path swamping, readout ignorance, map compression. Details:
`docs/RESOLUTION-NOT-COMPRESSION.md`.

## Who decides the answer: the state does, and it says one of three things

On 200 real held-out contexts (`scan8_ep1500`): the full argmax equals the **state-only** argmax **87.0%**
of the time and the **direct-only** argmax only **8.5%**, so the recurrent core is where the decision is
made. But the top choice takes **just 3 distinct values across 200 different contexts** - functional
collapse in the answer, which is what the repetition loop looks like from inside. Mean top-1 probability
**0.0334** against a uniform **0.000083**, top-1/top-2 gap **7.975e-03**, mean entropy **8.987** of 9.393.
So the problem is not that the readout ignores the state, it is that the state-to-answer map is degenerate.
Details: `docs/ARGMAX-DECIDED-BY-STATE.md`.

## Run 8 final: decorrelation works, the readout does not, generation collapses

Held-out loss **8.8631**, accuracy **5.60%** (run 5: 5.34%), train/held-out gap **+0.2994**. Criteria as
written before the run: **E1 PASS** state cosine **0.4643** (predicted 0.47 twelve minutes before it landed),
**E2 FAIL** KL **0.0151**, **E3 PASS** 5.60% at +0.26 pp, **E4 FAIL** generation collapses to
`. . the , the the the the` for every prompt, and **F4 FAIL** trained readout 0.0151 against noise 0.0153,
ratio **0.99x**. The state separation move worked; the answer layer is unchanged. Details:
`docs/RUN8-VERDICT.md`.

## A trap in our own headline test, found before it fooled us

Taking run 8's weights and merely setting the new readout gain to **1.0** - RMS normalisation with no
training at all - multiplies the headline KL by **63x**, from **0.0151** to **0.9590**, which would pass
the bar that eight runs failed. The trained-vs-noise control exposes it: at gain 1.0 the trained readout
scores **0.9590** and a readout of random numbers scores **0.9604**, ratio **1.00x**. All of the apparent
improvement is scale, none of it is learning. So F1 alone is gameable by arithmetic, and run 9 is judged on
F4 (trained must beat noise by 2x at the same scale) plus G1/G2. Details: `docs/METRIC-TRAP.md`.

## Where the input dies: measured, not guessed

The logits split into a state term and a direct term. The obvious hypothesis was that the direct term
swamps the state. Measured on `scan8_ep1000`: the state term is **larger** (sd 0.111-0.124 against
0.065-0.085) and carries most of the variation, contributing **0.0108** of the **0.0151** KL when the
direct term is frozen. So the failure is one of **scale**, not attribution: logit differences of about 0.1
across six very different prompts are washed out by a 12,002-way softmax. That is exactly what the new
readout normalisation amplifies. Details: `docs/READOUT-DECOMPOSITION.md`.

## The trained readout separates words no better than random noise

Instrument check on `scan8_ep750`: the KL metric responds monotonically to deliberate readout damage
(`0.0056` at x0.25 scale through `0.5989` at x8) and reproduces its baseline exactly when unchanged, so
the metric is sound. Then the decisive result: a readout of **random numbers** at the trained one's own
scale gives KL **0.0152**, against the trained readout's **0.0151**. Training made the answer layer no
more sensitive to the input than noise is. That is why the states differ (E1 passes) and the output does
not (E2 fails).

## Decorrelation works, the readout is now the bottleneck

Run 8 pairs the direct path with the state decorrelation penalty. Measured on its own checkpoints: state
cosine **0.1717** at step 500 and **0.2261** at step 750, against **0.9716** with no penalty, so E1 passes
(bar < 0.5). KL across six prefix-free words stays at **0.0150** and **0.0151**, against a bar of 0.05, so
E2 fails. Two different inputs now produce genuinely different internal states and still produce nearly
identical next-token distributions: the bottleneck has moved from the state to the answer layer.
Full notes are in the private working log; this line is the measured result.

## Decorrelation penalty verified

The named alternative is implemented and gradient-checked: with lambda 0.5, finite differences match at
**1.018e-08** (seed 5) and **1.009e-09** (seed 23); with the term off the earlier **1.684e-10** is
unchanged, so nothing was disturbed. Run 8 pairs it with the direct path: E1 state cosine below 0.5,
E2 KL above 0.05. Sources: `docs/EVIDENCE.md`, `docs/EVIDENCE.md`.

## Run 6 mid-run: state collapse returns, and my plateau call was wrong

At step 3750 (`scan6_ep3750.npz`): KL across six prefix-free words **0.0091** nats (K1 needs 0.05, trending
FAIL) and state cosine **0.9716** (K3 bar is 0.9, so FAIL). Two conclusions: longer training does not raise
input sensitivity, so the readout limit is structural; and the drift I called a plateau at **0.8088** at
step 1000 had reached **0.9716** by step 3750 - that inference from four points was wrong and is recorded
as wrong. Source: `docs/EVIDENCE.md`.

## Incident: run 6 overwrote run 5's checkpoints

All 15 files matching `scan5_ep*.npz` carry mtimes after run 6 started, so **run 5's weights were
overwritten and are gone**; its log and published numbers survive. Root cause: the checkpoint prefix
was hardcoded in the driver, so a run launched with a different log still wrote the same filenames.
Fixed: the driver reads `CK_PREFIX` from the environment and each runner pins its own. The 15 files
were renamed, not deleted, with a sha256 receipt. Source: `docs/EVIDENCE.md`.

## Direct path at 25 steps: D1 FAILS, ranking changed 6 of 6

KL across six prefix-free words is **0.0150** nats against a pre-registered bar of 0.05, so D1 FAILS and
the bar is not moved after the fact. The same run shows state cosine **0.0743** and **6 of 6 distinct
top-5 sets** - the model now prefers different next tokens per input while most of the probability mass
still sits on frequent tokens. Honest headline: ranking changed, distribution shape has not. Source:
`docs/EVIDENCE.md`.

## Direct path: first non-degenerate, prompt-dependent output

In a 12-step smoke test of the run-7 driver (not a trained result), four different prompts produced four
different strings with no repetition loop: `truffles sarcastic west escaped...`, `cos constitutes
overweight 135 graham...`, `barcelona vietnamese titanium ahead...`, `fabrics defeat fatigue wondering...`.
Every earlier run collapsed to the identical loop `the , . the , .` for every prompt. No accuracy claim -
twelve steps is not training - but it is the first evidence that input reaches the output. Source:
`docs/EVIDENCE.md`.

## Batched direct path: both bugs found and fixed

The gate failure had two independent causes: `einsum("btd,de->bte")` contracted the wrong axis of
`Wd` (correct: `"bte,de->btd"`), and positions whose target is `<unk>` were scored even though
`<unk>` is banned from prediction. After both fixes, four independent seeds pass at **1.68e-10,
1.60e-10, 2.65e-10, 2.70e-10**. The batched direct-path core is verified and runs at **792** scored
tokens/s. Source: `docs/EVIDENCE.md`.

## Batched gradient: proven wrong, not noise

The batched embedding gradient is stable at **-0.00119500** across eps 1e-4 to 1e-8 while the analytic
value is **+0.00581498** - a different number with a different sign, so it is a real error, not finite-
difference noise, and no relu unit sits near the kink. That implementation stays unused; the single-chunk
direct-path core passes at 2.238e-10 and is the version available for a run. Source:
`docs/EVIDENCE.md`.

## Batched gradient bug narrowed

Per-parameter finite differences show the fault is isolated: with subset scoring only the embedding
term is wrong (**7.01e-03**), while Wx, Wd, Wh, Wa and ba agree to about **1e-10**. The suspect is the
tied-readout term `dlg.T @ dsel` when positions are masked. Tool: `tools/diag_batched_grad.py`.

## Batched direct path: gate FAIL, not used

The batched direct-path step fails its finite-difference check: worst abs **7.010e-03** at `emb[1,0]`
against a 1e-6 bar. No run uses it. The single-chunk direct-path core passes at **2.238e-10** and is
the version that may be trained. Source: `docs/EVIDENCE.md`.

## Direct path added to the state core

The published d512 model reaches 37.37% partly because it adds a direct token-to-logits path; the
state core did not. That path is now implemented (`lens/scan_direct.py`) with hand-written gradients
verified numerically to **2.238e-10** abs. It becomes run 7 once run 6 finishes, against the same bar:
KL across six prefix-free words above 0.05 nats. Source: `docs/EVIDENCE.md`.

## Frequency prior ruled out

Subtracting log token frequency at decode leaves input sensitivity unchanged: KL **0.0102 -> 0.0104**
nats across six different words, and the output simply loops on a rare token instead of a common one.
Four candidate causes are now eliminated by measurement - instability (fixed), capacity (d512 probe),
bias (1/33 of the logit term), frequency prior. The remaining gap is structural: the readout cannot
amplify 0.01 nats of state signal into different words. Source: `docs/EVIDENCE.md`.

## Why the output ignores the input

Three tests on run 5's model: **capacity ruled out** (d512, 9,426,402 params, changes KL 0.0102 to
0.0108), **bias ruled out** (input part of the logits is 33x the bias, and removing the bias leaves KL
identical), and the mechanism found: state differences are multi-dimensional but the readout maps them
to nearly one direction (logit cosine **0.88**) - a frequency-prior collapse. Source:
`docs/EVIDENCE.md`.

## Run 5: accuracy doubled, speech still broken

With the spectral fix, held-out accuracy went **2.34% -> 5.34%** (loss 7.2164 -> 7.5663) - the first
state-core gain that came from the model reading its input. Generations still loop, and the measured
cause is that the output distribution barely changes with input: cosine **0.991**, KL **0.0102** nats
across six different words. Source: `docs/EVIDENCE.md`.

## Run 5 drift: decelerating, plateau below the bar

Pairwise state cosine across six prefix-free words: **0.2068, 0.5029, 0.6878, 0.7680, 0.8088** at steps
120 / 250 / 500 / 750 / 1000. Increments shrink (+0.30, +0.19, +0.08, +0.04), so it plateaus under the
0.9 input-blindness bar instead of crossing it. My earlier prediction of a crossing near step 1000 was
wrong and is recorded as such. Source: `docs/EVIDENCE.md`.

## Run 5: state similarity climbing

Same six prefix-free words, three checkpoints: pairwise state cosine **0.2068 -> 0.5029 -> 0.6878** at
steps 120, 250, 500, with the radius held at 0.900 and tanh slope 0.975. The run-4 collapse is not
happening, but the drift is monotone and heading for the 0.9 input-blindness bar. Source:
`docs/EVIDENCE.md`.

## Run 5 mid-flight: fix holding

At step 250 the recurrent radius is **0.900**, state norms about **2.0**, tanh slope **0.984**, and
the pairwise state cosine across six prefix-free words is **max 0.5029** (run 4 was +1.0000). The
fix survives training, though similarity drifted up from 0.2068 at step 120, which is being watched.
Source: `docs/EVIDENCE.md`.

## Spectral fix, test 2: PASS

Holding the recurrent radius at 0.9 with prefix-free prompts gives max state cosine **0.2068**
(mean 0.1216) where run 4 gave **+1.0000** for every pair, with tanh slope **0.98964**. The
input-blindness is a consequence of an unconstrained recurrent matrix. Source:
`docs/EVIDENCE.md`. Accuracy from the 1500-step run with this fix is pending.

## Spectral fix, test 1

Holding the recurrent radius at 1.0 removed the saturation (state norm 15.96 to 3.5, tanh slope
0.005 to 0.954) but the pre-registered falsifier still fired: max pairwise state cosine **0.9885**
against a 0.9 bar. Test prompts shared a prefix, which is a confound. Source:
`docs/EVIDENCE.md`.

## State core: input-blind, cause found

Four different prompts produce the same state (pairwise cosine **+1.0000**), state norm **15.96**,
tanh slope **0.00516**, and the recurrent matrix has **spectral radius 3.406** - the loop saturates
and destroys the input. All four failed runs had one cause. Source: `docs/EVIDENCE.md`.

## State core, run 4

8x throughput (792 vs 99 scored tokens/s) produced held-out loss **7.2164**, accuracy **2.34%** -
slightly worse than run 3 (6.9472 / 2.58%). More tokens at this size is not the lever. Source:
`docs/EVIDENCE.md`.

## Release

`v0.1.0` is published: https://github.com/blueprint-ai-hq-git/glassnet/releases/tag/v0.1.0 - both checkpoints, the tested Colab, the checksums, and the
limits stated before the features. Verified against the GitHub API on 2026-11-09.

## 2026-11-07

Today's changes, each with its log line: `docs/EVIDENCE.md`.

# Where GlassNet is right now

Updated: 2026-11-06. Every figure below comes from a run recorded in `docs/history/` or `lens/runs/`.

## Honest completion

| Area | Done | Note |
|---|---:|---|
| Autograd engine | 100% | gradients verified to 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) |
| Our optimiser | 100% | built and measured. A 3-seed sweep at its own rung, scored best-masked-per-seed, puts it +0.78 points over SGD's best setting there (28.59% vs 27.81%, `docs/EVIDENCE.md`, `docs/EVIDENCE.md`) and level at matched epoch; it fails the stability bar (1.89 pp spread) - not the 4.3 points once published. That arm used a different rung and SGD at lr 1.0 with no decay, so it is not the ladder result: on the capped-window ladder with final VAL, our rule leads only at 500k and plain SGD is ahead from 2M up (`docs/EVIDENCE.md`) |
| KV cache | 100% | 2.29x faster (0.483 -> 0.211 ms/token), identical output |
| Exports (npz/safetensors/gguf) | 95% | safetensors export of the best checkpoint pending |
| Local server | 95% | serves, not yet load-tested |
| Live inspector | 85% | real weights on :8770, read-only |
| Measurement rig | 95% | every run logged to JSONL |
| Named concepts | 60% | 256 neurons, audit runs, coverage printed |
| Our own architecture | 40% | built and measured; **lost** to the plain shape |
| Safety layer | 10% | designed, not built |
| Coherent conversation | 15% | grammar real, content weak |
| Complete readability | 10% | partial coverage only |
| **Machine (engineering)** | **~90%** | |
| **Intelligence (model quality)** | **~22%** | |
| **Project ready to ship** | **~32%** | |

## What this means in one paragraph

The scaffolding around the model is genuine engineering: hand-derived gradients that pass numerical checks,
an optimiser that measurably beats SGD, a cache that is 2.29x faster with identical output, and a system that
stops and resumes without losing work. The model itself is not good yet - 37.37% held-out next-token
accuracy at best. Anyone claiming this is a working assistant would be lying, and the numbers in this
repository are the proof of that.

## Best numbers

| Metric | Value | Where |
|---|---:|---|
| Best held-out accuracy | 37.37% | d=512, full corpus, 6 epochs |
| d=256 full corpus | 35.26% | by position, `lens/runs/ATTN-GRADE-RUN.txt:28` (legacy by-id: 26.31%) |
| d=256, 2M windows | 24.20% | seed 1, by position, `lens/runs/GPU-BEST.txt:64` |
| d=512, 3 epochs, 2M windows (run twice) | 24.39% / 24.40% (ppl 109.90 both) | same script, 0.01 pp spread between two runs a day apart, `docs/EVIDENCE.md` |
| Parameters, shipped d=256 model | 5,908,686 | counted off the model object; `docs/EVIDENCE.md`, also `lens/runs/GPU-BEST.txt:57` |
| Optimiser vs SGD (matched sweep, 3 seeds) | 28.59% vs 27.81% best-epoch mean; 29.12% vs 27.15% at matched epoch | `docs/EVIDENCE.md` |
| Gradient check | 7.5e-10 abs / 1.8e-07 rel (live check, 9/9 PASS) max error | `tests/test_gradcheck.py` |
| KV cache speedup | 2.29x (0.483 -> 0.211 ms/token, 2,069 -> 4,747 tokens/s) | `lens/runs/KV-CACHE-MICRO.txt` |
| Arithmetic, unseen operands | 0.0% exact | digit 0 alone: 100%; two different runs, `lens/runs/ARITHMETIC.txt` and `lens/runs/ARITH_DIAG.txt` |

## Levers measured and closed

These were tested one change at a time against a matched reference, and lost on every seed.
They are listed so nobody has to re-run them.

| Lever | Result | Where |
|---|---|---|
| Attention (bare single head, no position) at 20k windows | 8.21% vs 11.02% reference, 2 seeds, -2.81 | `docs/EVIDENCE.md` |
| Window 64 (double tokens per example) at 20k windows | 9.45% vs 11.02% reference, 2 seeds, -1.57 | `docs/EVIDENCE.md` |

## Measurement bugs found and fixed

A wrong number is worse than a missing one, so these are listed with the correction.

| Bug | What it did to the numbers | Where |
|---|---|---|
| Talk decoder left-padded short prompts with token id 0 (`<unk>`) | 20-23 of 32 context slots were `<unk>` for the project prompts, so 95.1% (d256) / 96.5% (d512) of the reply tokens were `<unk>`; pad-free the same weights measure 26.4% / 21.5% | `docs/EVIDENCE.md` |
| Architecture table added the proposed recurrent core and counted the tied embedding twice | published "7,081,122 parameters"; the shipped model prints **5,908,686**, so the count was overstated by 1,172,436 (19.8%) | `docs/EVIDENCE.md` |
| Cost table added a stage labelled "runs 32x" exactly once | published 0.1446 ms/token; with the core counted the 32 times it runs, the same rows give 0.2283 ms | `docs/EVIDENCE.md` |
| KV cache speedup "9.7x" had no run file behind it | only the prose figure existed (0.217 vs 2.115 ms, `docs/EVIDENCE.md`); the recorded micro-benchmark is **2.29x** (0.483 -> 0.211 ms/token, identical to 5.3e-06) | `lens/runs/KV-CACHE-MICRO.txt` |
| `wall_ab_grade.py` P2 computed `on - off` but graded the pre-registered `off - on` band | the falsifier (`on - off < 0.5`) fired on a real 3.11 pp LOSS and called it "earns under 0.5 pp"; the -1.0..-6.0 band was unreachable, so a perfect result would still have printed "OUTSIDE". Numbers unchanged (-1.39 / -3.11), only the sentence was wrong | `docs/EVIDENCE.md` |
| A 200k-window arm ran at 19:05 on `data/chat/prepared` with plain SGD | its VAL 13.12% looks like progress next to the 20k numbers but is **not comparable**: 13.3M vs 294.2M cached tokens, gates OFF vs `ours`, 13,651 vs 87,035 val windows, 40 vs 36 concepts. Recorded as a run, never as evidence - it also had no queue entry and no pre-registered prediction | `docs/EVIDENCE.md` |
| Same command, same seed, three runs at the 20k-window rung | VAL acc 11.32 / 10.79 / 10.29 and ASST 11.96 / 11.42 / 10.92 - a **1.03 pp spread**, not the 0.01-0.02 pp measured on the d512 / 2M-window path. The small figure must not be carried to this rung, and a +/-1.25 band around one draw cannot serve as a control at n=1 seed | `docs/EVIDENCE.md` |
| `train_chat.py` printed `len(Xtr) / params` under the label "tokens per parameter" - but `len(Xtr)` counts **windows** | every such figure in every log is low by exactly 32x, and the "UNDER-TRAINED" verdict was computed on it: the d=512 / 4M-window rung printed **0.26 (UNDER-TRAINED)** when it truly had **8.22 (`adequate`)**, and the d=256 arms are 0.17 / 1.68 / 4.20 at 20k / 200k / 500k windows, not 0.01 / 0.05 / 0.13. Recorded once as a *prediction* error (`AUTOPILOT-LOG.md:4552`) and never fixed in code; now computes tokens and prints the old ratio beside it so old logs stay comparable. Second unit question under it: `chat_data.py:166` uses `stride = window // 2`, so the figure counts token INSTANCES - the headline's 34.74 is 18.90 distinct tokens/param per epoch, i.e. about half. Thresholds left alone | `docs/EVIDENCE.md`, `lens/scripts/train_chat.py:830` |
| A restarted run rewrote `<out>_best.npz` from its own epoch 1 | `wall_off_s0`'s epoch-3 checkpoint was replaced by an aborted second attempt's epoch 1. Repaired earlier; the **guard is now wired into the trainer and proven at the call site** (2 hunks, patch sha256 f10dc89e -> e5acd0aa, 4/4 unit tests, `--check` import=True call=True, and the end-to-end pair: run 2 printed 1 `CLOBBER GUARD` line, archived run 1's best.npz as sha256 b349a10b... with a receipt, and wrote its own 422ebc7e...) | `docs/EVIDENCE.md`, `lens/runs/BEST-CKPT-GUARD-E2E.txt` |

Every percentage on this page and in `README.md` that carries a `file:line` citation is now checked mechanically: `lens/scripts/check_public_cites.py` opens each cited line and fails if the quoted number is not on any cited line of that row (missing file, line past the end, blank line, or a number that appears nowhere it cites). Current state: **33 cited rows on these two pages and 56 across all of `docs/history`, 0 problems**. It has already earned its keep three times - it found a citation to a **blank** line (`STATE.md:2768`, the 37.37% claim, now `:2770`) and a table row that cited the `final VAL` line while also quoting that arm's ASST, a number only on the epoch line (`WALL-AB.txt:850` -> `:765`) - and a third time on this very page, catching a new 2M row that quoted a
best-checkpoint 22.82% cited only to the final-epoch line. It does not judge which column a number came from - that mistake (P1, VAL vs ASST) still needs a human eye.

**The 17.43% arm's checkpoint now verifies mechanically, and one arm has none.** `verify_wall_ab_ckpt.py` used to split a log at `----- off seed N` markers, so a run launched before that convention existed - the seed-0 200k arm behind 17.43% - was invisible to it and the tool answered about a stale block appended after it. With leading unmarked segments parsed, **20 arms across 11 logs** now verify: `datascale_honest_s0` has 3,811,534 params as announced, best checkpoint = epoch 2, and `sha256(best) == sha256(ep2)` with identical arrays. The single remaining problem is the 11.04% / 41-concept run on `data/chat/prepared`: it announces **3,912,509** params but the tag its save line names resolves to the honest seed-1 checkpoint at **3,811,534**, so that run has **no checkpoint of its own** - which matches `docs/EVIDENCE.md` (recorded as a run, never as evidence). Note: `docs/EVIDENCE.md`.

**Held-out sets are disjoint by construction, at every scale.** Above 200,000 train windows the trainer skips its exact content walk and prints the skip (`train_chat.py:783-787`); every data-scale arm here - 500k, 2M, 16.9M - exceeded that, so none printed a walk. It is redundant: the train/val split is assigned per conversation (`chat_data.py:133`), windows are selected per conversation by that flag (`chat_data.py:165`), and windows never cross a conversation boundary (`chat_data.py:166`), so a train and a val window cannot share tokens; the vocabulary is built from train text only (`chat_data.py:119`). What is still unchecked is corpus-level duplication (a val conversation that is a near-copy of a train one) - that would inflate every number here and has never been tested. Note: `docs/EVIDENCE.md`.

## Wall A/B (concept-wall bottleneck) - FINISHED 2026-11-06 17:51, six arms, P1 FAILs as coded (band mixes two columns)

Question: docs/EVIDENCE.md OPEN #1. Design + predictions written before any run:
docs/EVIDENCE.md (P1-P5, plus P6 "supervision must not move a number").
Full write-up with the grader's own words: `docs/EVIDENCE.md`.

| arm | seed | final val ppl | val acc | concepts | params | source |
|---|---|---:|---:|---:|---:|---|
| wall_off (no bypass) | 0 | 399.50 | **12.18%** | 20/36 | 3,811,534 | lens/runs/WALL-AB.txt:201-202 |
| wall_on (tied full-rank bypass, +2,097,152 params) | 0 | 575.00 | 10.79% | 20/36 | 5,908,686 | lens/runs/WALL-AB.txt:477-478 |
| wall_r64 (rank 64, +1,292,416 params) | 0 | 550.85 | 11.28% | 24/36 | 5,103,950 | lens/runs/WALL-AB.txt:664-665 |
| wall_off (no bypass) | 1 | 395.80 | **12.79%** | 26/36 | 3,811,534 | lens/runs/WALL-AB.txt:850-851 |
| wall_on (tied full-rank bypass) | 1 | 628.41 | 9.68% | 22/36 | 5,908,686 | lens/runs/WALL-AB.txt:1036-1037 |
| wall_r64 (rank 64) | 1 | 521.68 | 10.87% | 26/36 | 5,103,950 | lens/runs/WALL-AB.txt:1222-1223 |

Runner marker: `lens/runs/WALL-AB.txt:1226 ===== WALL-AB DONE 2026-11-06 17:51 arms_missing=0`.
Reference for the same 20k-window rung: 11.96 / 10.08 (mean 11.02%), lens/runs/CTX-ATTN.txt.

Verdicts as printed by `lens/scripts/wall_ab_grade.py` (output `lens/runs/WALL-AB-GRADE.txt`,
sha256 ac55e35b..., 6 rows in `lens/runs/WALL-AB-RESULTS.tsv`):

* **P1 FAIL as coded, but the band mixes two columns** - the 10.0-12.5 band was derived from `ASST` values
  (11.96 / 10.08 at `LEARNCURVE.txt:4773` and `:4960` - not in the file the predictions cite, which has no
  such numbers) and then applied to the `VAL` column. On the graded column `wall_on` s0 10.79% is inside
  the band and s1 9.68% is 0.32 pp below the floor. Compared on the SAME column the control reproduces
  within 0.55 pp on both seeds (VAL -0.53 / +0.15, ASST -0.54 / +0.22), and the trainer commands and the
  five config-header lines match field for field. So this FAIL is a metric mismatch, **not** evidence that
  the rung drifted. The gate keeps its coded verdict - re-deriving it after seeing the result is exactly
  what a gate must not do - so until the maintainer rules, no delta against the stored rung is quoted. Both readings
  with all numbers: `docs/EVIDENCE.md`.
* **P2 OPPOSITE SIGN** - off minus on = +1.39 (s0), +3.11 (s1). Removing the bypass did not cost the
  predicted 1-6 points; on both seeds it was ahead.
* **P3 MIXED** - r64 minus on = +0.49 (s0), +1.19 (s1): not below `on` on either seed, so "rank 64 is
  under-provisioned" is unsupported, but the dose is not clean at seed 1 either.
* **P4 PASS** - mean train-minus-val gap off +8.35 vs on +28.98, i.e. 20.63 pp smaller with 2.1M fewer
  parameters: at this rung the big gap follows capacity.
* **P5 CONTRADICTION CONFIRMED** - the no-bypass arm is above the bypass arm by more than 1.0 pp on both
  seeds, reported as a finding against the plan, not as noise.
* **P6 not triggered** - the resumed arms' recorded command lines match `run_wall_ab.sh` field-for-field
  (only shell quoting of the seed differs), so the resume is valid and its arms are poolable.

What that means: the A/B does **not** establish that the concept wall is lossy, and it does not license
the wider-residual build. It is a coherent lead - same direction on both seeds, 2.1M fewer parameters,
far less overfit - and its control reproduces on a matched comparison, but the pre-registered gate as
coded still says FAIL, so nothing is quoted against the stored rung until that is ruled on. If the maintainer
adopts the metric-consistent reading, the deltas become quotable and the plan's next lever is data scale;
if he keeps the gate as coded, the control has to be re-run before the block can be read at all.

A third run of the same command, same seed, launched at 18:07 and finished at 18:15, landed at VAL 10.29% /
ASST 10.92% (`lens/runs/REF-REPRO.txt`) - so three draws of one configuration span **1.03 pp**, and P1's
+/-1.25 band around a single draw cannot work as a control at n=1 seed. Full write-up:
`docs/EVIDENCE.md`.

### Replication on fresh seeds (WALL-REP) - finished 2026-11-06 18:56

A separate pre-registered block (`docs/EVIDENCE.md`), own log
`lens/runs/WALL-REP.txt`, own tags - its arms are **not** in the six-arm table above.

| arm | seed | VAL acc | ASST acc | source |
|---|---|---:|---:|---|
| wall_rep_off (no bypass) | 2 | **13.40%** | 13.83% | lens/runs/WALL-REP.txt:329 |
| wall_rep_on (tied bypass) | 2 | 9.98% | 10.65% | lens/runs/WALL-REP.txt:110 |
| wall_rep_off (no bypass) | 3 | **13.26%** | 13.72% | lens/runs/WALL-REP.txt:517 |
| wall_rep_on (tied bypass) | 3 | 11.21% | 11.99% | lens/runs/WALL-REP.txt:705 |

**R1 direction PASS** (+3.42 at seed 2, +2.05 at seed 3), **R2 magnitude PASS** (both inside the predicted
1.0-4.0 pp), **R3 noise bound FAIL** (the `on` arm spans 1.53 pp over four draws against a 1.5 pp bound -
missed by 0.03), **R4 not pooled PASS**. All four seeds now put the no-bypass arm ahead:
+1.39 / +3.11 / +3.42 / +2.05. Because R3's pre-registered falsifier says a span above 1.5 pp means "no
delta may be quoted at all", **no delta is quoted here and none is claimed as significant** - the numbers
are shown, the rule is not re-read. Checkpoints: 4 arms, 0 provenance problems
(`verify_wall_ab_ckpt.py lens/runs/WALL-REP.txt`). Full write-up: `docs/EVIDENCE.md`.

Every arm's checkpoint has been machine-checked against the epoch its log line named
(`lens/scripts/verify_wall_ab_ckpt.py`, output `lens/runs/WALL-AB-CKPT-VERIFY.txt`,
sha256 4e11965f...: **6 arms, 0 provenance problems**; the 5-arm version is kept, not deleted, at
`lens/runs/archive/ckpt_verify_5arms_2026-11-06/`, sha256 6b55befd...).
Supervisor: lens/scripts/run_wall_ab_resume2.sh via the chain (label wall_ab_resume2), log
`lens/runs/WALL-AB.txt` (1,226 lines).

## Data scale at d=256 - the first lever that moved this rung (2026-11-06 19:29)

One flag changed (`--max-windows 20000` -> `200000`), everything else identical: same corpus, same command,
same concept module (the auto-kept set is data-dependent: 37 kept at 20k windows, 36 at 200k), **same 3,811,534 parameters**, no bypass, seed 0 - and the **same 87,035-window
held-out set** in both runs (`n=44418` masked / `n=58191` assistant positions in each epoch-3 line, verified
from both logs), so the extra data cannot have come from resizing the scored set.

| run | seed | windows | VAL acc | ASST acc | final ppl | source |
|---|---:|---:|---:|---:|---:|---|
| `wall_off` | 0 | 20,000 | 12.18% | 13.00% | 399.50 | `lens/runs/WALL-AB.txt:116` |
| `datascale_honest_s0` | 0 | 200,000 | **17.43%** | 18.61% | 207.29 | `lens/runs/DATASCALE-HONEST.txt:230` |
| `wall_off` | 1 | 20,000 | 12.79% | 13.21% | 395.80 | `lens/runs/WALL-AB.txt:765` |
| `wall_off` | 2 | 20,000 | 13.40% | 13.83% | 384.39 | `lens/runs/WALL-REP.txt:329`, reproduced exactly by `DATASCALE-20K-S2.txt` |
| `datascale_honest_s1` | 1 | 200,000 | **19.71%** | **20.82%** | **184.32** | `lens/runs/DATASCALE-HONEST-s1.txt:262` |
| `datascale_honest_s2` | 2 | 200,000 | **20.50%** | **21.47%** | **180.30** | `lens/runs/DATASCALE-HONEST-s2.txt:256` |
| `datascale_500k_s0` | 0 | **500,000** | **20.81%** | **21.82%** | **171.47** | `lens/runs/DATASCALE-500K.txt:392` |
| `datascale_2m_s0` | 0 | **2,000,000** | **21.29%** | **22.60%** | **149.14** | `lens/runs/DATASCALE-2M.txt:1142`; best checkpoint ep2 ASST **22.82%** at `lens/runs/DATASCALE-2M.txt:817` |
| `datascale_500k_nogate_s0` (**gates OFF** control) | 0 | 500,000 | **22.05%** | **23.22%** | **140.27** | `lens/runs/DATASCALE-500K-NOGATE.txt:69` |
| `sgd_500k_s0` (**plain SGD**, gates OFF) | 0 | 500,000 | 19.81% | 20.38% | 221.78 | `lens/runs/SGD-500K.txt:66` (concepts 31/36) |
| `datascale_2m_nogate_s0` (**gates OFF**, 2M) | 0 | 2,000,000 | **23.12%** | **24.96%** | **104.71** | `lens/runs/DATASCALE-2M-NOGATE.txt:69` |
| `datascale_4m_nogate_s0` (**gates OFF**, 4M) | 0 | 4,000,000 | **23.74%** | **25.71%** | **95.99** | `lens/runs/DATASCALE-4M-NOGATE.txt:69` |
| `datascale_4m_nogate_ep6_s0` (**gates OFF**, 4M, 6 epochs) | 0 | 4,000,000 | **23.92%** | **25.92%** | **93.72** | `lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73` |
| `datascale_4m_nogate_s1` / `_s2` (**gates OFF**, 4M, seeds 1-2) | 1, 2 | 4,000,000 | **24.13% / 24.03%** | **25.84% / 25.84%** | **93.88 / 93.52** | `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` (seed 1) and `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` (seed 2) |
| `datascale_8m_nogate_s0` (**gates OFF**, 8M) | 0 | 8,000,000 | **24.51%** | **26.46%** | **87.55** | `lens/runs/DATASCALE-8M-NOGATE.txt:235` |
| `datascale_16m_ours_s0` (**our step rule**, 16M) | 0 | 16,000,000 | **24.58%** | **26.40%** | **84.47** | `lens/runs/DATASCALE-16M-OURS.txt:225` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS.txt:226` (final, concepts) |
| `datascale_4m_ours05_s1..s2` (**our rule, opt-lr 0.5**, 4M, seeds 1-2) | 0 | 4,000,000 | **26.10 / 25.84** (3-seed mean **26.00**, spread 0.26 pp) | — | **85.61 / 86.02** | `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:151` and `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:301` |
| `datascale_4m_ours025_s0` (**our rule, opt-lr 0.25**, 4M) | 0 | 4,000,000 | **26.78%** | — | **88.18** | `lens/runs/DATASCALE-4M-OPT025.txt:154` and `lens/runs/DATASCALE-4M-OPT025.txt:155` |
| `datascale_8m_ours05_s0` (**our rule, opt-lr 0.5**, 8M) | 0 | 8,000,000 | **26.63%** | **28.65%** | **77.86** | `lens/runs/DATASCALE-8M-OPT05.txt:154` (VAL/ASST/ppl) and `lens/runs/DATASCALE-8M-OPT05.txt:155` (concepts) |
| `datascale_4m_ours05_s0` (**our rule, opt-lr 0.5**, 4M) | 0 | 4,000,000 | **26.06%** | **27.98%** | **85.60** | `lens/runs/DATASCALE-4M-OPT05.txt:154` (VAL/ASST/ppl) and `lens/runs/DATASCALE-4M-OPT05.txt:155` (concepts) |
| `datascale_16m_ours05_s0` (**our rule, opt-lr 0.5**, 16M) | 0 | 16,000,000 | **26.76%** | **28.85%** | **71.69** | `lens/runs/DATASCALE-16M-OURS05.txt:225` (VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-OURS05.txt:226` (concepts) |
| `datascale_8m_sgd_s0` (**plain SGD**, 8M) | 0 | 8,000,000 | **27.58%** | — | **93.68** | `lens/runs/DATASCALE-8M-SGD.txt:151` |
| `datascale_2m_ours05_s0` (**our rule, opt-lr 0.5**, 2M) | 0 | 2,000,000 | **24.81%** | — | **99.79** | `lens/runs/DATASCALE-2M-OURS05.txt:154` |
| `datascale_500k_ours05_s0` (**our rule, opt-lr 0.5**, 500k) | 0 | 500,000 | **22.97%** | — | **139.47** | `lens/runs/DATASCALE-500K-OURS05.txt:155` |
| `datascale_16m_sgd_ep6_s0` (**plain SGD, 6 epochs**, 16M) | 0 | 16,000,000 | **28.61%** | — | **72.25** | `lens/runs/DATASCALE-16M-SGD-EP6.txt:227` |
| `datascale_16m_sgd_lr02_s0` (**plain SGD, lr 0.2**, 16M) | 0 | 16,000,000 | **27.90%** | — | **91.17** | `lens/runs/DATASCALE-16M-SGD-LR02.txt:151` |
| `datascale_2m_sgd_s0` (**plain SGD**, 2M) | 0 | 2,000,000 | **25.10%** | — | **133.41** | `lens/runs/DATASCALE-2M-SGD.txt:151` |
| `datascale_4m_sgd_s0` (**plain SGD**, 4M) | 0 | 4,000,000 | **26.51%** | — | **110.50** | `lens/runs/DATASCALE-4M-SGD.txt:151` |
| `datascale_16m_nogate_s0` (**plain SGD**, 16M) | 0 | 16,000,000 | **28.30%** | **30.42%** | **80.15** | `lens/runs/DATASCALE-16M-NOGATE.txt:136` (epoch line: VAL/ASST/ppl) and `lens/runs/DATASCALE-16M-NOGATE.txt:213` (final, concepts) |

**Three seeds, three matched pairs: mean gain +6.42 pp.** Val acc 17.43% / 19.71% / 20.50% against 20k
baselines of 12.18% / 12.79% / 13.40%, so the paired gains are **+5.25, +6.92 and +7.10 pp** (mean **+6.42**),
mean VAL **19.21%**. The baseline and the arm rise together - a seed that gives a stronger 20k run also gives a
stronger 200k run - which is why the paired gain spans only 1.85 pp while the 200k values themselves span
3.07 pp; the paired gain is the number to quote. Seed 1 passed every pre-registered check (band R1, ppl R2,
concepts C1); seed 2's band check **failed by 0.32 pp on the high side**, the direction its prediction doc had
pre-named as "the band was too conservative", while its mean and concept checks passed. Graders:
`lens/scripts/datascale_grade.py`, `lens/scripts/datascale_seed2_grade.py`; write-ups
`docs/EVIDENCE.md`, `DATASCALE-SEED2-RESULT.md`, `DATASCALE-20K-S2-RESULT.md`.

**The lever bends flat between 500k and 2M in the GATED method - and the ungated rung says that was partly the gates.** 2,000,000 windows at seed 0 reached
**21.29% VAL / 22.60% ASST / ppl 149.14**, concepts 36/36, and the pre-registered checks split: T1 (> 20.81%)
PASS at **+0.48 pp**, T3 and T4 PASS, **T2 FAIL** (landed 0.21 pp below its 21.5% floor) and **C4 FAIL** (epoch 3
at 21.29% is 0.40 pp *below* epoch 2, so the best checkpoint - ASST 22.82% - is epoch 2, not the final one). The
whole seed-0 ladder is now 12.18 -> 17.43 -> 20.81 -> 21.29%, i.e. **1.58, then 2.56, then 0.24 pp per
doubling**: 100x data bought +9.11 pp in total and the last 4x bought 5.3% of it, while perplexity kept falling
(399.50 -> 149.14) as accuracy stopped rising. So the earlier "not flattened at 200k" note is now completed by
the rung above it: the curve rose one more step and then flattened, and 0.24 pp/doubling is inside the 3.07 pp
seed spread, so the last rung is not distinguishable from noise on one seed. Source: `lens/runs/DATASCALE-2M.txt:1142`;
grader `lens/scripts/datascale_ladder_grade.py 2m`; write-up `docs/EVIDENCE.md`. Not
comparable to the older 2M result at batch 1024 (24.20%) - different batch and sampling path.

**Re-run at 2M with the gates off, the flat curve moves: 23.12% VAL / 24.96% ASST / ppl 104.71**, against the gated
twin's 21.29 / 22.60 / 149.14 - **+1.83 pp for the same data, and 8.75x faster** (249 s/epoch against 2,081-2,279).
All five pre-registered checks pass except U2, which missed *high*: I predicted at most +1.0 pp over the ungated
500k arm and it bought **+1.07 pp** (21.5-23.0 band, got 23.12) - the direction its own falsifier named as "above
23.0 reopens the ladder". So the two curves are 500k 20.81 -> 2M 21.29 gated (**0.24 pp/doubling**) and 22.05 ->
23.12 ungated (**0.54 pp/doubling**): both flatten, the gated one flattens hardest, and the ungated curve is still
climbing. **23.12% is the highest d=256 held-out number on record** (previous best 21.29% gated), and still 14.25 pp
below the d=512 headline. The gates-off effect has now been measured at two rungs, same direction, similar size
(+1.24 pp at 500k, +1.83 pp at 2M), with the mechanism visible in train accuracy (35.10% ungated vs 21.14% gated).
Source: `lens/runs/DATASCALE-2M-NOGATE.txt:69`; write-up `docs/EVIDENCE.md`. One seed per
configuration - the 3.07 pp seed spread at 200k is larger than either delta, so direction and speed are solid and
effect size is not.

**And it is still paying at 4M.** 4,000,000 windows, gates off: **23.74% VAL / 25.71% ASST / ppl 95.99**, concepts
36/36, 484-490 s per epoch (`lens/runs/DATASCALE-4M-NOGATE.txt:69`). Paired to the 2M ungated arm that is **+0.62 pp
for 2x data = 0.62 pp/doubling**, against 0.54 pp/doubling for the 500k->2M step: the ungated rate has not decayed,
which is the opposite of what the gated ladder reported. All five pre-registered checks passed (V1 > 23.12, V2 inside
23.4-24.6, V3 ppl < 104.71, V4 mechanical, V5 concepts), and the one generic check that failed is C4 - **epoch 3
equals epoch 2 to two decimals (23.75 -> 23.74)** while perplexity kept falling, so the arm plateaued on accuracy
inside the run. That plateau is the next question, not more data: the arm is limited by its 15,625 steps x 3 epochs,
so "more epochs at 4M" and "bigger batches" are different experiments and neither has been run. 23.74% is the best d=256 number in this **capped-window ladder** (<=4M windows, 3 epochs, honest mix) - 2.45 pp
above the gated 2M arm and 13.63 pp below the d512 headline - but it is **not** the best d=256 number in the
project: full-corpus d=256 runs reach **35.72% by position** (`docs/EVIDENCE.md`, 16.9M windows,
6 epochs), 10.01 pp higher on the comparable metric. Source for the 23.74% row: `lens/runs/DATASCALE-4M-NOGATE.txt:69`. One seed;
the 3.07 pp spread at 200k dwarfs the +0.62 pp. Write-up: `docs/EVIDENCE.md`.

**The step gain survives seeds, and the crossing is still unlocated.** At 4M with `--opt-lr 0.5`, seeds 1 and 2 give 26.10% and 25.84% (`lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:151`, `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:301`); with seed 0's 26.06% (`lens/runs/DATASCALE-4M-OPT05.txt:154`) the three-seed mean is **26.00%** over a 0.26 pp range (`docs/EVIDENCE.md` stands for that computed mean; this paragraph does not recompute it). That clears the best old-step seed (24.13) by 1.87 pp, so the +2.2 pp is not a single-seed artefact. It does not change the top of the table: nothing in this rule reaches plain SGD's 28.30% at 16M (`lens/runs/DATASCALE-16M-NOGATE.txt:213`), and no rung between 500k and 16M has both rules measured - the next arm is a 4M plain-SGD run to bracket where the ordering flips. S5 failed on one concept (seed 1: AUROC 0.9909, fires 0.016 vs a 0.02 floor) and is recorded as a miss, with the audit in `docs/EVIDENCE.md`.

**At matched settings the rule gap is a smooth monotone, and our rule only wins at 500k.** Filling every rung for both rules (our 0.5 arm against SGD's default) gives: 500k **+3.16** for ours, then -0.29, -0.51, **-0.95** and **-1.54** at 2M / 4M / 8M / 16M (`lens/runs/DATASCALE-8M-SGD.txt:151` is the 8M SGD cell; the rest are cited in `docs/EVIDENCE.md`). Five rungs, one configuration change, a monotone - not a step change at a particular rung, which is what my earlier 'level at 2M-4M' sentence implied. Individually the two smallest cells are inside one-seed noise; the monotone across five rungs is what carries the claim. The step effect over the same fill-in reads **+0.92 / +1.69 / +2.03 / +2.12 / +2.18** at 500k / 2M / 4M / 8M / 16M.

 Plain SGD at 4M reaches **26.51% / ppl 110.50** (`lens/runs/DATASCALE-4M-SGD.txt:151`) in **11 min 18 s** - against 28 min 23 s and 28 min 20 s for the two 4M arms under our rule tonight, both at similar load - so it is **level with our best (26.00, three seeds, `docs/EVIDENCE.md`) at 2.51x less compute**. I predicted it would land below 26.00 and it did not: T1 failed. The defensible sentence is not 'SGD wins at 4M' (0.51 pp is inside noise) but 'our rule's advantage is a small-data effect, gone by 4M', with the seed-proof losses at 16M. Note the trade: SGD's accuracy comes with far worse calibration (ppl 110.50 vs 85.60 / 85.61 / 86.02, `lens/runs/DATASCALE-4M-OPT05-SEEDS.txt:302`).

**At 16M windows plain SGD beats our step rule by 3.72 pp, reversing the 500k result.** Matched pair, identical corpus, windows, seed, epochs and batch, only the optimiser differs: **SGD 28.30% / ASST 30.42% / ppl 80.15** against **our rule 24.58% / 26.40% / 84.47** (`lens/runs/DATASCALE-16M-NOGATE.txt`, `lens/runs/DATASCALE-16M-OURS.txt`). I predicted our rule would still win, by less than the +2.24 pp it won at 500k; the pre-registered comparator returned PREDICTION FALSIFIED. **The +2.24 pp is therefore a small-data result** and must not be quoted as a property of the optimiser. Mechanism, written down before epochs 2-3 existed and then observed: our rule's accuracy stayed flat (24.65 -> 24.49 -> 24.58) while perplexity fell 3.80, the signature of a fixed-size unit step overshooting; SGD's decaying steps climbed 27.27 -> 27.90 -> 28.30 over the same epochs. This arm also lost one concept: 35/36, `mentions_a_name` fires on 1.8% of rows (AUROC 0.9982). One seed, and this rule's own seed spread is 1.89 pp - the 3.72 pp gap exceeds it, but two seeds would settle it.

**The 4M rung reproduces to 0.10 pp - and the 8M gain beats the whole range.** Three seeds of the same 4M ungated config: **23.74 / 24.13 / 24.03%**, mean **23.97%** (computed; recorded in `docs/EVIDENCE.md`), range 23.74-24.13, seed-to-seed spread **0.10 pp** (`lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:69` for seed 1, `lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt:219` for seed 2, and `lens/runs/DATASCALE-4M-NOGATE.txt:69` for seed 0); provenance 2 arms, 0 problems. Reproducibility improves with data: the 200k rung's three-seed spread is **3.07 pp** (`docs/EVIDENCE.md`), twenty times wider. So the 8M result (`lens/runs/DATASCALE-8M-NOGATE.txt:235`) is above every one of the three 4M seeds by 0.38-0.77 pp - the "one more doubling pays" claim no longer rests on one seed. The same fact cuts the other way at 200k, where the 3.07 pp spread is wider than the +1.24 pp gates-off delta at 500k, so that delta stays unpublishable until gated twins exist at those seeds.

**Six epochs is the ladder's best arm, and it buys calibration more than accuracy.** The same 16M plain-SGD rung run for six epochs gives **28.61% / ppl 72.25** (`lens/runs/DATASCALE-16M-SGD-EP6.txt:227`) against three epochs' 28.30% / 80.15 (`lens/runs/DATASCALE-16M-NOGATE.txt:136`): **+0.31 pp** (inside one seed) and **7.90 ppl better**. Accuracy stops moving after epoch 3 (28.17 -> 28.46 -> 28.46 -> 28.61) while perplexity falls at every one of the six epochs, so the extra passes bought confidence, not correctness. Note the decay confound: `--lr-decay` spreads over the run, so this is six passes at a gentler decay, not six passes alone. Full write-up: `docs/EVIDENCE.md`.

**And halving the step is our rule's trick, not a general one.** The same idea applied to the winning rule - plain SGD at 16M with `lr 0.2 -> 0.1` - gives **27.90% / ppl 91.17** (`lens/runs/DATASCALE-16M-SGD-LR02.txt:151`) against the default lr's (`lens/runs/DATASCALE-16M-NOGATE.txt:136`) 28.30% / 80.15: 0.40 pp down on accuracy (inside one seed, so 'buys nothing' rather than 'hurts') and **11.02 ppl worse**, which is not inside anything. So the +2.2 pp from smaller steps belongs to `--method ours` and must not be stated as a property of the architecture.

**Halving the step recovers 2.18 pp - and leaves the rule 1.54 pp behind SGD.** 16M windows, gates off, seed 0, our rule at `--opt-lr 0.5`: **26.76% VAL / ASST 28.85% / ppl 71.69**, concepts 36/36 (`lens/runs/DATASCALE-16M-OURS05.txt:225`). Against the same arm at `1.0` (24.58%, `lens/runs/DATASCALE-16M-OURS.txt:226`) that is **+2.18 pp**, Z1's PASS, and it confirms the overshoot mechanism written down before the arm ran: the 1.0 unit step was too large for 187,500 updates. Against plain SGD (28.30%, `lens/runs/DATASCALE-16M-NOGATE.txt:213`) the shortfall shrinks from 3.72 to **1.54 pp** - roughly 2.2 of the 3.7 was a fixable configuration error and the rest is the rule itself. The same signature reappears one level up: accuracy 26.74 -> 26.85 -> 26.76 while perplexity fell to 71.69, which is why C4 (ep3 >= ep2) failed. Because every ladder rung ran at 1.0, its absolute levels are probably depressed by about this much - deltas stand, ceilings do not.

**The step has a plateau below 0.5, not a peak.** At 4M, `--opt-lr 0.25` gives **26.78% / ppl 88.18** (`lens/runs/DATASCALE-4M-OPT025.txt:155`) against `0.5`'s 26.06% (`lens/runs/DATASCALE-4M-OPT05.txt:154`). The 0.72 pp gap is well inside this rule's **1.89 pp** seed spread, so the two settings are indistinguishable here and neither can be called the optimum. What is not in doubt is the gap to `1.0`: 23.74% (`lens/runs/DATASCALE-4M-NOGATE.txt:69`), about 3 pp below either. Practical reading: 0.5 is a safe setting, there is no demonstrated reason to halve it again, and every claim about *which* small step is best needs seeds this project does not yet have. My V1 predicted 0.25 would land below 0.5; it failed.

**One caveat that applies to every number in this table, measured at four rungs and smaller at the bottom.** Every gates-off runner at 4M and above passes `--opt-lr 1.0`, and there that step costs about **2.1-2.3 pp**: 4M 23.74 -> **26.06**, 8M 24.51 -> **26.63**, 16M 24.58 -> **26.76** (`docs/EVIDENCE.md`, `docs/EVIDENCE.md`, `docs/EVIDENCE.md`; the 1.0 twins on `lens/runs/DATASCALE-4M-NOGATE.txt:69`, `lens/runs/DATASCALE-8M-NOGATE.txt:235`, `lens/runs/DATASCALE-16M-OURS.txt:226`). The extremes are four times apart in update count and the effect is the same size, so it is a constant handicap, not a scale effect: every level here sits ~2.2 pp below what its configuration reaches, and none should be read as a ceiling. The curve's shape survives the correction - at 0.5 the steps are +0.57 then +0.13, at 1.0 they were +0.77 then +0.07 - so the flattening is real. The best number in this ladder is therefore **26.76%**, not 24.58%. The 3-seed optimiser sweep used 0.5 (`docs/EVIDENCE.md`), so it and this ladder were never the same configuration.

**At 500k the step is worth only half as much, and our small-data win gets bigger anyway.** Our rule at `--opt-lr 0.5` on 500k gives **22.97% / ppl 139.47** (`lens/runs/DATASCALE-500K-OURS05.txt:155`) against 22.05 at 1.0: **+0.92 pp**, not the ~2.2 seen from 4M up - and 500k is the noisiest rung in this project, so even that is inside its own spread. So the step effect is *not* flat across all rungs; it is ~2.2 pp where the ladder has data. The consequence that matters is the other one: at our better setting our rule still beats plain SGD at 500k, now by **3.16 pp** (22.97 vs 19.81, `lens/runs/SGD-500K.txt:66`), where the old comparison offered only 2.24. The one rung we win is won by more than the step correction explains.

**The data axis has not flattened: doubling to 8M paid more than the doubling below it.** 8,000,000 windows, gates off, seed 0, three epochs: **24.51% VAL / 26.46% ASST / ppl 87.55**, concepts 36/36 (`lens/runs/DATASCALE-8M-NOGATE.txt:235`; the concept audit 36/36 is on `:236`). The 4M rung was 23.74% (`lens/runs/DATASCALE-4M-NOGATE.txt:69`), so this doubling bought **+0.77 pp** against **+0.62 pp** for 2M -> 4M - the gain grew rather than decaying, which falsifies the expectation written into the prediction (X2 was centred below a repeat of 0.62). Path 23.46 / 24.03 / 24.51 with ppl 95.71 / 91.04 / 87.55. Two limits travel with it: the prepared mix holds ~18.4M windows, so only one more doubling is available before the axis runs out of data, and with one seed this says the axis has not turned down, not that 0.77 pp is precise (the 200k rung's three-seed spread was 3.07 pp). The machine rebooted mid-arm at 18:52; the arm resumed from its epoch-1 checkpoint (about 11 minutes lost, not 51).

**Doubling the steps was not the lever.** Six epochs at the same 4M windows reached **23.92% VAL / 25.92% ASST /
ppl 93.72** (`lens/runs/DATASCALE-4M-NOGATE-EP6.txt:73`) - **+0.18 pp** for doubling the updates from 46,875 to
93,750. The hypothesis behind that arm ("the plateau is step-limited") was falsified by its own pre-registered
threshold (W1 needed >= 24.04, got 23.92). At this rung, **doubling the data bought +0.62 pp and doubling the steps
bought +0.18 pp**, so data is worth ~3.4x what steps are worth per doubling; ppl fell the whole way (95.89 ->
93.72) while accuracy wobbled in a +-0.3 pp band. Neither closes the **11.80 pp** gap to the project's d256 record
(35.72% by position, full corpus, `docs/EVIDENCE.md`). 8M windows were then run: 24.51% (`lens/runs/DATASCALE-8M-NOGATE.txt:235`). One seed;
only the ratio of the two deltas is readable, not their sizes. Note: `docs/EVIDENCE.md`.

**The gates were part of that ceiling, and the headline was never gated.** Same 500k rung, same step rule, same
data, gates OFF: **22.05% VAL / 23.22% ASST / ppl 140.27** against the gated arm's 20.81 / 21.82 / 171.47 - and
**72 s per epoch instead of 558, 7.7x faster** (`lens/runs/DATASCALE-500K-NOGATE.txt:69`, twin at
`DATASCALE-500K.txt:392`). All four pre-registered checks passed (G1 +1.24 pp, G2, G3 mechanical 0 gate lines,
G4 36/36 concepts). The gates let the model fit the training set far harder (22.25% -> 35.20% train) and overfit
far more (train-minus-VAL gap 1.44 -> 13.15 pp) **while held-out accuracy rose** - so at this rung they were
suppressing learning that generalised. That matters beyond the rung: every ladder arm ran `--method ours` (gated),
while the run behind the **37.37%** headline prints `METHOD: standard (the old loop: plain SGD, our gates OFF)`
(`lens/runs/TIED-D512-EXT6.txt`). The project's best number therefore comes from the ungated old loop, and "our
method vs the old loop" has never been measured at a matched rung - that is the next arm. One seed, one rung, and
+1.24 pp is inside the 3.07 pp seed spread, so this is a direction, not a settled effect; the 7.7x is a measured
ratio. Write-up: `docs/EVIDENCE.md`.

**And the step rule is worth more than the gates.** Adding the third configuration - plain SGD, gates off, same
rung and seed - gives **19.81% / 20.38% ASST / ppl 221.78**, concepts only **31/36** (`lens/runs/SGD-500K.txt:66`).
The matched three-way at 500k is therefore **relgrad ungated 22.05 > relgrad gated 20.81 > plain SGD 19.81**: our
step rule is worth **+2.24 pp** over the loop this project used before **at this rung** (at 16M windows the ordering reverses - see the paragraph above), the gates cost **1.24 pp**, and plain SGD
is 2.3x cheaper per epoch while learning the least. That is awkward for the headline: **37.37% was produced by
plain SGD** on 16.9M windows in 6 epochs (`TIED-D512-EXT6.txt`), so the best number sits on the weakest-measured
step rule. The next test is that configuration at that scale - d=512, full corpus, 6 epochs, relgrad, gates off -
and it needs a **one-epoch cost probe** first, because relgrad is ~2.3x plain SGD per epoch while plain SGD already
took 1,720 s/epoch at that scale (`docs/EVIDENCE.md`). One seed per configuration and all three deltas
sit inside the 3.07 pp seed spread: the ordering is a direction, not a settled effect. Write-up:
`docs/EVIDENCE.md`.

**What one d512 epoch actually costs, and a confound that blocks the obvious conclusion.** A one-epoch probe in
the headline's own configuration (d=512, `--direct --direct-mode tied`, batch 1024, lr 0.3, 16.9M windows, our
step rule, gates off) took **2,854 s** - so six epochs scale to **~4.8 h** - and reached **25.18% VAL / 26.67%
ASST / ppl 110.24**, against the headline's epoch 1 at **31.52% / 33.46% / 46.99** (`lens/runs/PROBE-D512.txt`,
headline at `TIED-D512-EXT6.txt:83`). That is **-6.34 pp** and 1.66x slower per epoch. It is **not** evidence
against our step rule: the probe's train accuracy is **21.12%** against the headline's **42.01%**, and its log
shows the step size is still the constant calibrated at d=256 (`opt_lr 0.0441256`, `--lr` unused by this rule)
while the headline ran SGD at lr 0.3. A step that is too small explains both the deficit and the slower
convergence, and this arm cannot separate that from a weaker rule - the next step is to calibrate the step size at
d=512, not to run six epochs. One epoch, one seed, and 12.19 pp below the headline: nothing here changes 37.37%.
Note: `docs/EVIDENCE.md`.

The concept jump was not predicted by D1/D2, and it repeated: the audit went **20/36 at 20k windows ->
36/36 at 200k** on both seeds, counted from the AUROC rows rather than from the summary lines.

**Is the 3-epoch schedule truncating it? No.** A 6-epoch arm at 200k, seed 0, one knob changed
(`docs/EVIDENCE.md` written first), reached final VAL **18.43% / ppl 201.69** - a gain
of **+1.00 pp** over the 3-epoch 17.43, landing exactly on the predicted threshold, so E1 is graded **PARTIAL**
by its own wording (E2 ppl PASS). The path there was a swing, not a climb: 17.48, 17.90, 17.43, **15.13**,
**19.68**, 18.43 - and the best saved checkpoint is epoch 5 (ASST 20.83% vs 18.87% for the 3-epoch arm). So
doubling the schedule doubles the cost for +1.00 pp of endpoint and a better best-checkpoint, and this rung is
unstable epoch-to-epoch. Write-up: `docs/EVIDENCE.md`; the 3-epoch result is not averaged
with it.

Caveats, stated because they matter: **three seeds**, so no p-value and no invented interval - the three
values and their mean are what is published. The 20k rung's measured run-to-run spread is 1.03 pp
(`docs/EVIDENCE.md`), and the three seeds span 3.07 pp, i.e. seed-to-seed variation at this rung is
three times the measurement noise and is what limits precision here. The headline is unchanged - best held-out accuracy is still **37.37%**
(d=512, full corpus, 6 epochs). Checkpoints verified with `verify_wall_ab_ckpt.py`: the seed-1 arm is 1/1
clean, the parameter count off the stored arrays is 3,811,534 in both seeds, and `best.npz` equals the epoch
the log named. Write-ups: `docs/EVIDENCE.md` (seed 0) and
`docs/EVIDENCE.md` (seed 1).

## CORRECTION (2026-11-14 09:05): the three-seed paired number is +0.53, not +1.30 - my own falsifier triggered

The "FINAL VERDICT" above was **one seed per side**. Both sides now have three seeds at matched 9-epoch training,
and `tools/compare_arms.py --paired` prints:

```
paired by seed:
  seed 0: best +1.30   final +1.50   last-3 +1.71
  seed 1: best +0.35   final +0.16   last-3 +0.67
  seed 2: best -0.07   final -0.82   last-3 +0.26
across seeds:
  best    mean +0.53  sd 0.70  se 0.41  positive 2/3  t=1.3
  final   mean +0.28  sd 1.16  se 0.67  positive 2/3  t=0.4
  last-3  mean +0.88  sd 0.75  se 0.43  positive 3/3  t=2.0
```

Source: `lens/runs/PAIRED-VERDICT-9EP.txt`, generated by the published comparator with every arm reduced to its own
last run.

**Prediction P2 FAILED.** Before the last arms finished I wrote that the paired best-of-N mean would land between
**+0.8 and +2.5**. It is **+0.53**.

**My falsifier triggered.** I wrote: "if any seed pair is negative under best-of-N, or the paired mean falls below
+0.5, the matched-training width claim is NOT established." Seed 2's pair is **-0.07**. The claim is therefore not
established, and this is the **third correction** to this document.

What the record now says:

- Width is worth **+0.28 to +0.88 points** at matched 9-epoch training, depending on which endpoint is quoted, with a
  t-statistic between 0.4 and 2.0. Under the pre-registered best-of-N rule it is **+0.53, inside the noise (t=1.3)**,
  and in one of three seeds the wide model finishes *behind*.
- The only comfortably positive rule is the all-epoch mean (**+2.71**, t=24.8), and it must not be read as a ceiling
  effect: it is driven by the EARLY epochs. At epoch 1 the wide model is 24.46 / 24.52 / 24.56 and the narrow one is
  19.58 / 19.91 / 18.40 - about a five-point gap that has shrunk to under a point by epoch 9.
- That makes the two apparently conflicting results agree: width is worth **+3.95 at equal 3-epoch training** and
  **+0.53 at equal 9-epoch training**. **Width buys speed, not ceiling, on this task.** It reaches a usable level
  sooner and ends up in about the same place.

The equal-3-epoch number (**+3.95**, three paired seeds) remains a correct answer to the question "width when both
sides train for only three epochs". It is not an answer to "width in general", and this page previously implied it
was closer to one than the data now supports.

## Freezing the pointer protects memorised copy by +8.2 points (2026-11-14)

Base `early_s0_ep1000` (copy 83.0% taught / 71.0% held-out), 4000 steps of language-only batches, 2 seeds each:

```
arm                        held-out copy (mean)   language (mean)
freeze ptrW/Wq/Wk                 67.2               16.42
no protection (control)           59.0               17.66
difference                        +8.2               -1.24
```

**+8.2 points of held-out copying kept, for 1.24 points of language.** Same direction as the earlier five-seed +2.07.

The separate-memory arm meant to test the stronger claim was **void twice** and both times the tell was identical
numbers to the control: attempt 1 started from a language checkpoint (taught copy 0.0%, impossible), attempt 2 hit a
`--sep-emb` block nested inside `if a.pointer:` so the flag did nothing. Adding `--pointer` to fix that destroys the
base (83.0/71.0 becomes 19.2/0.0). The trainer now refuses to run that arm silently, and the corrected run is in
flight. `docs/MEMORY-PROTECTION-FREEZE-VS-SEPARATE.md`.

## The separate memory table does nothing; freezing is what works (two seeds, 2026-11-14)

Corrected, properly engaged arm (the trainer used to skip the flag silently - it now refuses to run instead):

```
                          held-out copy (mean)   language (mean)
separate table (idea)            58.5                17.70
control, no table                59.0                17.66
difference                       -0.5                +0.04
```

Half a point of copy and four hundredths of a point of language: **the separate table changes nothing.** The mechanism
that does work on this base is freezing the pointer matrices, **+8.2 points of copy for -1.24 language** (67.2 vs 59.0).
`docs/MEMORY-PROTECTION-FREEZE-VS-SEPARATE.md`.

## The model's confidence is real, just inflated by 14 points (2026-11-14)

`exposure_d1024_e9_s0_best`, 14,145 held-out assistant-masked positions, from the checkpoint's own logits:

```
accuracy                     29.34%
mean confidence              43.02%      (+13.68: over-confident)
ECE                           0.1368
AUC (confidence -> correct)   0.8069      (0.50 = noise)
wrong answers, mean conf      33.49%      (quiet, not loud)
```

Sorted by confidence, accuracy climbs monotonically 4.95% -> 86.13%. So the confidence **carries information** (AUC 0.81)
and the model already signals its own uncertainty when it is wrong. The defect is a uniform inflation of ~14 points, which
is a calibration fix, not an architecture problem. Two of my three predictions failed: I expected ECE > 0.15 and expected
wrong answers to be confidently wrong (>50%), and they are not. `docs/CONFIDENCE-AUDIT.md`.

## One temperature removes 62% of the over-confidence, free (2026-11-14)

Fit on the first half of held-out windows, scored on the second half, `tools/calibrate.py`:

```
                  accuracy   mean conf      ECE      AUC
BEFORE (T=1)        27.71%      41.41%     0.1370   0.7955
AFTER  (T=1.132)    27.71%      32.83%     0.0521   0.7980
```

**ECE 0.1370 -> 0.0521 with accuracy exactly unchanged**, because a temperature cannot move an argmax. The model's
confidence was ordered correctly all along and only needed to be scaled down. My 0.05 bar was missed by 0.0021, so a
single parameter removes most of the over-confidence, not all of it. `docs/CALIBRATION.md`.

## Where the 29% comes from: two thirds of its successes are copying (2026-11-14)

`tools/failure_attribution.py`, 14,145 held-out assistant-masked positions:

```
overall accuracy                 29.34%
target IS in the input window    43.90%  (45.2% of positions)
target is NOT in the window      17.32%  (54.8% of positions)   ratio 2.54x
predictions copied from window   60.23% of all, 67.66% of correct ones
```

**Two thirds of its correct answers are tokens retrieved from the window.** When the answer has to be *produced* rather
than found, accuracy is **17.32%**. That, not width (+0.53 paired) or training length (plateau at epoch 6), is where the
next lever is. All three pre-registered predictions held. `docs/FAILURE-ATTRIBUTION.md`.

## The model over-predicts `<unk>` by 4x, and 86% of those predictions are wrong (2026-11-14)

```
overall accuracy                     29.34%
accuracy, <unk> targets excluded     27.79%   (-1.55: the headline overstates token prediction this much)
model PREDICTS <unk> on              21.73%   (true <unk> rate is 5.40%)
  of those predictions, correct      14.02%
```

`<unk>` is where the model puts its uncertainty, it uses it four times too often, and it is wrong almost every time. That
is a **data and vocabulary** defect, not a scaling one - and it is the most actionable finding of the day. All three
pre-registered predictions held. `docs/UNK-AUDIT.md`.

## Our own RL: the variance gate works, the objective is the blocker (2026-11-14)

Three attempts, warm start, 3 seeds each, equal updates (`lens/runs/RL-V2-VERDICT-EQUPD.txt`):

```
                       acc_in (in-domain)   acc_gen (never-trained)   dead groups
v1  group  8, 800 steps   47.72 +/- 3.39           0.00                  51.2%
v2  group 16, 800 steps   50.42 +/- 5.62           0.00                  36.6%
```

The variance gate **roughly halves the batches that carry no learning signal** (51.2% -> 36.6%, and 59.9% -> 28.9% in an
earlier pair), which is what it exists for. The +2.70 points on in-domain accuracy is **not** attributed to the machinery:
v2 draws twice the samples, and the standard error is about 3.2 points.

**`acc_gen` is 0.00 for the 13th time.** Verifier-only learning has now failed 13 measurements in a row, which points at the
**objective, not the estimator** - there is no gradient to estimate in a reward that only says right or wrong.
`docs/RL-METHOD-V2.md`.

## The 13 RL zeros are explained: zero correct samples in 12,800 draws (2026-11-14)

`tools/fuel_check.py`, warm start 400 steps on sums <= 9 then 200 generalising prompts (a+b >= 10) x G=64 samples:

```
answers drawn              12,800
correct answers                 0   (0.0000%)
prompts with >=1 correct    0 of 200
```

**No self-training method could have started.** The 13 consecutive `acc_gen = 0.00` measurements are not evidence about our
advantage estimator, curriculum or KL anchor - the reward was always wrong, so the gradient was always empty. The task is
not unreachable in principle (supervised training reaches 34.66% +/- 6.16 on the same split), so the gap is
**reachable-when-taught, unreachable-when-sampled at this capacity**. The next RL question is how much capacity the fuel
needs, not how to improve the estimator. `docs/RL-FUEL-CHECK.md`.

## Our RL was training on an empty support, not losing to supervised (2026-11-14)

Thirteen arms measured `acc_gen = 0.00` on the generalising split. Four measurements now explain why
(`docs/RL-EMPTY-SUPPORT.md`):

```
fuel at the shipped model (12,800 draws)          0 correct
fuel with an 8,000-step warm start                0 correct
fuel at 103,054 parameters (14x the model)        0 correct
fuel per target sum 10..18 (34,560 draws)         0 correct at every sum
```

**The support is empty, not thin** - so group-relative advantage, dense credit, KL anchors and curricula were all
irrelevant: a reward the model never attains carries no gradient. The task is still reachable **when taught** (supervised
34.66% +/- 6.16), but not by sampling at any capacity or schedule tried. Our RL needs a task where a plausible wrong sample
can actually pass the checker, not more tuning on this one.

## With fuel present, our RL still learns nothing - 12 settings, one outcome (2026-11-14)

On `(a+b) mod 10`, where fuel was measured to exist, LENS-PG v2 was run across sampling T 1-8 and RL lr 0.0005-0.05, 3 seeds each:

```
greedy generalising accuracy   0.00% -> 0.00%   at EVERY setting
in-domain accuracy             damaged in proportion to lr (-51 at 0.01, -6 at 0.0005)
dead groups                    21% at the best setting (so the variance gate is doing its job)
```

Exploration was raised until 93.5% of prompts could produce a correct sample (T=4) and 99% at T=8, and reward mean reached
0.095. **Every precondition was satisfied and the policy still did not learn**, which eliminates reward density, support,
capacity and estimator design as the cause.

Sharpest diagnostic: on the generalising split the answer is **never 9** (max sum is 18), and the model emits **9 for 59.5%**
of those prompts. Its strongest habit points at the one digit that can never be right. `docs/RL-TEST-BED-WITH-FUEL.md`.

## Rejection sampling: the first method here to create generalising ability from nothing (2026-11-14)

On the fuel-bearing task, after twelve policy-gradient configurations failed to move greedy generalising accuracy off 0.00%:

```
4 rounds of: sample at T=4 -> keep correct answers -> supervised train on them
seed 0   0.00% -> 40.75% -> 60.00% -> 77.50% -> 88.00%
seed 1   0.00% -> 39.25% -> 54.75% -> 66.25% -> 82.75%
seed 2   0.00% -> 38.25% -> 51.00% -> 70.50% -> 78.25%
```

It costs in-domain accuracy (70.5-72.5% -> 2-3%): textbook forgetting. Mixing in-domain rows into each batch (`--replay`)
trades smoothly:

```
replay 0.0   gen 78-88%   in-domain  2-3%
replay 0.5   gen 34-40%   in-domain 62-72%
replay 0.8   gen 10-14%   in-domain 81-85%   (above the 70.5-72.5% baseline)
```

**Verified self-training beats policy gradient decisively in this setting:** +78 to +88 points in four rounds, same model and
same warm start, with less machinery. `docs/RL-REJECTION-SAMPLING.md`.

## Refusing to train on unanswerable targets raises real token accuracy by +1.62 points (2026-11-14)

Same d1024 reference recipe, ONE flag added (`--drop-unk`, which zero-weights training targets whose token is `<unk>` - 83,211
of 2,000,000 targets, 4.161%). Measured with the same tool and the same 20,000 held-out windows as the baseline:

```
                          reference      drop-unk     change
overall accuracy            29.34%        27.83%      -1.51
accuracy, <unk> excluded    27.79%        29.41%      +1.62   <- the real test
model predicts <unk> on     21.73%         0.00%     -21.73
```

All three pre-registered predictions held. The raw headline falls 1.51 points - necessarily, since `<unk>` targets were scoring
56.41% and are now untrained - while accuracy on tokens that actually exist **rises 1.62 points** and `<unk>` emissions go to
**exactly zero**. `docs/UNK-DROP-RESULT.md`.

## Readability: readable model is 1.4-2.2 points behind plain at matched parameters (2026-11-14)

8-arm sweep (4 configs x 2 seeds), every arm printing READABLE vs PLAIN at matched parameter counts in the same run:

```
arm                    gap (2 seeds)      concept honesty
shipped  d64 patch 7   -2.06 / -1.54      100.0% / 92.3%
wide     d96 patch 7   -2.23 / -1.91      100.0% / 100.0%
softlam  lam 0.15      -1.63 / -1.36       84.6% / 92.3%
fine     patch 5       crashed, no result
```

**This is NOT a smaller gap than the published 4.96.** That figure is post rule-pruning at 3 seeds; this sweep is unpruned,
`--extra none`, 2 seeds. Different measurement, different configuration - not a before-and-after.

Softer honesty pressure buys accuracy and costs honesty, as predicted (lam 0.15 gives the smallest gap with honesty below the
85% bar). Wider concepts do **not** close the gap. `docs/READABILITY-SWEEP.md`.

**Update (same run):** the `patch 5` arm could never run - 28 is not divisible by 5, so the front end asks for 625 pixels per
image where there are 784. Valid patch sizes are 1, 2, 4, 7, 14, 28, and `lens/vision.py` now refuses others with a message
naming them. Re-run correctly as **patch 4**, the finer layout is **worse**: gap **-3.81 / -3.57** against patch 7's
**-2.06 / -1.54**. Finer concepts cost 1.75-2.03 points, so the readable model's price is paid in concept count, not concept
granularity. `docs/READABILITY-SWEEP.md`.

## Correction: the "4.96 points worse" readability headline compares a 24k model to a 102k model (2026-11-14)

Both published numbers are right; the comparison is unmatched. Arithmetic:

```
plain Phase-1 baseline 784->128->10   101,770 params, 10 epochs
readable model                          24,458 params, 24 epochs
                                        4.2x size difference
```

So "the readable model does not beat the baseline" attributes to **readability** what is at least partly **size**. With
parameters matched inside the same run (readable 24,458 vs plain 24,467, same data, same epochs, same seed):

```
shipped  d64 patch 7   -2.06 / -1.54 (2 seeds)   honesty 100.0% / 92.3%
wide     d96 patch 7   -2.23 / -1.91             honesty 100.0% / 100.0%
softlam  lam 0.15      -1.63 / -1.36             honesty  84.6% / 92.3%
```

**At matched parameters the readable model costs 1.4-2.2 points, not 4.96.** Direction unchanged, magnitude roughly halved.

Also flagged, not papered over: the phase-2 document's "+5.63 from pruning" implies its unpruned readable model scored ~76.1%,
while the current trainer measures 84.36% unpruned - so those are not the same implementation, and the ~76% is not quoted
anywhere until that is located. `docs/READABILITY-GAP-CORRECTION.md`.

**Confirmed at three seeds (same run):** matched gap **-1.53 +/- 0.44** points (seed values -2.06 / -1.54 / -0.98), concept
honesty **97.4%** mean, with the plain control trained at a matched parameter count inside each seed's run. All three
pre-registered predictions held. The honest headline for the readable image model is **1.53 +/- 0.44 points behind a plain MLP
of the same size**, not 4.96 - the readable net still does not win, but the penalty is about a third of the quoted figure and
now carries a standard deviation. `docs/READABILITY-GAP-CORRECTION.md`.

**And now decomposed.** The missing half of that comparison - a plain net at the readable model's own size - has been measured
by reusing the Phase-1 trainer itself with only the hidden width changed (10,345 params vs the readable 10,173, same 10 epochs,
batch 128, lr 0.1, 3 seeds):

```
101,770-param plain (Phase-1 baseline)      86.69%
 10,345-param plain (matched control)       84.37% +/- 1.04    -2.32  <- CAPACITY
  10,173-param readable, post-prune         81.73%             -2.64  <- READABILITY
```

**Of the published 4.96-point gap, 2.32 points (47%) is capacity and 2.64 points (53%) is readability itself.** The readable
model still loses to a same-size plain net, so that part of the original claim stands - but half the quoted penalty was never
readability's. `docs/READABILITY-GAP-DECOMPOSED.md`.

Two related corrections, both mine: pruning is worth **+0.64**, not the +5.63 I inferred from the phase-2 document, so the
"~76% unpruned readable model" I flagged as missing never existed - it was my own arithmetic error, withdrawn. And my
prediction that the readable model would land within 2 points of the matched control **failed** (-2.64).

## Phase-2 gate item 5 closed: the rule layer costs +4.09 points, and the concept names are taught (2026-11-14)

The one open Phase-2 gate item - "remove one piece and show what changes" - is now measured, 3 seeds, 10 epochs, one schedule:

```
variant                        test acc (post-prune)     concept AUROC      conns/rule
A0 full (supervisor+L1+prune)      79.04% +/- 1.97        79.75 +/- 1.61        3.00
A1 no concept supervision          77.80% +/- 1.41        48.61 +/- 3.63        3.00
A2 no sparsity (no L1, no prune)   81.71% +/- 0.76        77.36 +/- 1.01       14.00
A3 no rule layer                   83.13% +/- 0.42        72.26 +/- 1.89          -
```

- **Rules removed: +4.09 points.** The rule layer is the single most expensive piece of the readable architecture.
- **Sparsity removed: +2.67 points.** Forcing 3 connections per rule is not free in accuracy - my "sparsity only buys
  readability" premise is refuted.
- **Supervision removed: -1.24 accuracy AND -31.14 AUROC** of concept honesty (79.75 -> 48.61). The names are taught by the
  supervisor, not decorative - that is the honesty check, with a number.
- **Pruning is configuration-dependent**: -0.45 here, +0.64 in the published pipeline. Not assumable either way.

With the matched control at 84.37% +/- 1.04, the price is now attributed piece by piece: ~1.2 points for the concept layer,
2.7 more for the 3-connection constraint, 4.1 more for the rule layer in the strict configuration.
`docs/PHASE2-ABLATION-CLOSED.md`.

## Scaled up: 96.15% generalising on its own, or 77.03% generalising with the old task IMPROVED by 19 points (2026-11-14)

The rejection-sampling pilot went from 4 rounds / 400-prompt evaluation to **8 rounds / 2,000-prompt held-out evaluation / 3
seeds**, on a task the model starts at **0.00%**:

```
                       generalising    in-domain      what it is
no replay (8 rounds)      96.15%          2-3%        max new task, old task gone
replay 0.5 (8 rounds)     77.03%         90.33%       both tasks, old task +19.0 over its warm start
```

Seed 0 of the no-replay arm scores **100.00% - 2,000 of 2,000 held-out prompts** - from 0.00%, with no labelled generalising
example, in a 7,342-parameter model.

**With replay there is no trade-off at all**: in-domain accuracy ends **19.00 points above** where it started (71.33% ->
90.33%) *while* the new task reaches 77%. The pilot's apparent trade-off was a 4-round artefact. All three predictions held; the
wider evaluation did not flatter the pilot's number. `docs/RL-REJECT-SCALED.md`.

Not claimed: this is `(a+b) mod 10` with a 7k model, not the 47M chat line; and the method needs a free verifier, which this
task has and language does not.

## Where the readable rule layer's cost comes from: not the rule count, not the sparsity (2026-11-14)

Two 3-seed sweeps at K=14 named concepts, alpha 3.0, lam 0.005, 10 epochs, with predictions written before each:

```
rules (connections fixed at 3)          connections per rule (R=32)
R=16   76.41% +/- 6.35                  keep=1    65.44% +/- 3.06
R=32   78.57% +/- 3.87                  keep=3    77.76% +/- 4.20   <- shipped
R=64   79.42% +/- 1.53                  keep=6    78.60% +/- 4.34
                                        keep=12   78.45% +/- 4.55   <- no sparsity at all
```

Reference: **A3 no rule layer 83.13% +/- 0.42**, matched plain control **84.37% +/- 1.04**.

- **Rule count is not the constraint**: R=64 beats R=32 by only **+0.85** (bar was +1.0). But it collapses the spread **4x**
  (+/-6.35 -> +/-1.53), so more rules buy **reliability**, not average accuracy.
- **Connectivity is not the constraint either**: even at ~**12 connections per rule** - no sparsity at all - the readable model is
  still **4.68 points** below removing the rule layer. The written falsifier triggered.
- **The sharpest positive result: keep=3 is already the sweet spot.** 1 -> 3 connections is worth +12.32 points, 3 -> 6 only
  +0.84, 6 -> 12 is flat. **The project did not pay accuracy for the sparsity it advertises.**
- So the ~4-point cost of `explain()` is **structural**, sitting in having a small bottleneck between concepts and classes.

Confound stated, not hidden: the earlier "no sparsity" arm (81.71%) also had `lam 0` while these arms keep `lam 0.005`, so the
L1 penalty and the rule layer are **not yet separated**. Isolating `lam` is the next run.
`docs/RULE-LAYER-COST.md`.

## The concept-supervision knob, measured on the HONEST instrument (2026-11-14)

```
alpha   accuracy (3 seeds)   per-seed spread   AUROC
1.0     82.68%               0.30              0.805
3.0     78.46%               9.26              0.879   <- shipped
6.0     74.34%               2.55              0.896
10.0    63.51%               15.26             0.918
```

AUROC says alpha 1.0 is "slightly less honest" (0.805 vs 0.879). The project's own label audit says something much worse:

```
                                    alpha 1.0        alpha 3.0 (shipped)
mean |rank corr| with own label       0.561              0.732
concepts below 0.50                   7 of 14            2 of 14
concepts tracking ANOTHER feature     5 of 14            2 of 14
dead rules                            1 of 32            0 of 32
```

**At alpha 1.0 half the concept layer (7 of 14) does not track the name attached to it.** AUROC was the flattering instrument,
exactly as `tools/phase2_reaudit.py` warns. So honesty costs **4.22 accuracy points** and even the shipped alpha 3.0 still leaves
**2 of 14 concepts mislabeled** (`symmetry` and `vertical_edges` track other features at both settings).

Two structural facts for free: **concept interaction is 0.0%** in both arms (purely additive - good for `explain()`, since
contributions really do add up), and **12 of 32 rules combine concepts that rarely co-occur** (median rule-pair lift 1.12), which
is hard for a human to check.

Path forward is better concept DEFINITIONS, not more pressure: more alpha collapses accuracy faster than it fixes labels.
`docs/CONCEPT-ALPHA-HONESTY.md`.

## Dropping the redundant concepts: 17 of 18 neurons now mean their name, and accuracy went UP (2026-11-14)

The label set was the problem, not the neurons. Measured on labels alone: **8 concept pairs correlate above 0.80** (max 0.916),
`ink_amount` correlates 0.87+ with four quadrant "concepts", and `symmetry` is 82.1% positive - so a neuron could be 82%
"accurate" by always saying yes. When two names describe the same measurement, no pair of neurons can both mean their own name.

`tools/select_concept_set.py` greedily selected **18 of the 36 features the library already defines**, keeping only
|corr| <= 0.60: pairwise correlation now **median 0.000, max 0.587**. Every feature has a documented plain-English definition.

```
                               shipped 14 concepts      18 de-duplicated concepts
accuracy (3 seeds)               78.46% +/- 4.63            79.69% +/- 1.13
mean |rank corr| with own label      0.732               0.844 / 0.845 / 0.845
concepts below 0.50                  2 of 14                    0 of 18
concepts tracking ANOTHER feature    2 of 14                    1 of 18
```

**17 of 18 concept neurons track their own name better than any other feature** (the holdout is `vertical_edges`). Honestly
flagged: the 18-concept model has ~28% more parameters, so part of the +1.23 accuracy may be capacity and needs a
parameter-matched control; and **rule plausibility got worse** (median rule-pair lift 1.12 -> 0.92, 12 -> 19 of 32 rules combine
concepts that rarely co-occur) - decorrelated concepts make honest *concepts* easier and honest *rules* harder.
`docs/CONCEPT-SET18.md`.

## Every concept neuron now means its name: 0 of 14 mislabeled in all three seeds, +1.67 accuracy at IDENTICAL parameters (2026-11-14)

My own capacity caveat from the 18-concept result was tested and **refuted**. The same de-duplication rule truncated to 14 features
gives a model with **precisely the shipped parameter count (11,800)**, so only the labels differ:

```
                              shipped 14           14 de-duplicated        change
parameters                     11,800                  11,800             identical
accuracy (3 seeds)         78.46% +/- 4.63         80.13% +/- 0.81        +1.67
mean |rank corr|, own label     0.732           0.846 / 0.847 / 0.848     +0.114
concepts below 0.50             2 of 14               0 of 14             none left
concepts tracking ANOTHER       2 of 14               0 of 14             none left
```

**In all three seeds every one of the 14 concept neurons tracks its own name better than any other feature.** First configuration
in this project where "each middle neuron means what its label says" holds with no exceptions - and it cannot be a capacity
artefact because the parameter count is identical. The 14-set also **beats the 18-set** (80.13 vs 79.69) with 3,268 fewer
parameters.

Against the matched plain control (84.37% +/- 1.04) the full readable stack now costs **4.24 points**, in exchange for honest
concepts, a purely additive concept layer (interaction 0.0%), and rules a human can read. `docs/CONCEPT-SET14-MATCHED.md`.

**Resolved same day:** the "~28% more parameters" caveat on the 18-concept result was tested and **withdrawn**. At identical 11,800
parameters, 14 de-duplicated concepts score **80.13% +/- 0.81** with **0 of 14 concepts mislabeled** - so the gain is label
quality, not capacity, and 14 clean concepts beat 18 clean ones. `docs/CONCEPT-SET14-MATCHED.md`.

## Re-tuning alpha on clean labels: the readable stack is now 1.61 points behind a same-size plain net (2026-11-14)

alpha=3.0 was tuned when the concept labels correlated up to 0.916 with each other. With the de-duplicated 14-concept set, accuracy
falls monotonically as alpha rises and honesty becomes much cheaper:

```
alpha 0.5   83.13%   own-label corr 0.711   1 of 14 mislabeled     rule lift 0.87
alpha 1.0   82.76%   own-label corr 0.772-0.786   0/0/1 mislabeled  rule lift 0.79
alpha 3.0   80.20%   own-label corr 0.846-0.848   0/0/0 mislabeled  rule lift 0.92   <- shipped
```

**alpha 1.0 buys +2.56 accuracy and keeps 0 concepts below 0.50 in every seed**, at the cost of a lower mean own-label
correlation and one mislabeled concept in one seed. alpha 3.0 remains the only setting with 0 mislabeled in all three seeds - so
this is a genuine choice, not a winner:

```
matched plain control (10,345 params)                     84.37% +/- 1.04
readable, 14 clean concepts, rules, alpha 1.0 (11,800)    82.76% +/- 0.35    -1.61
readable, 14 clean concepts, rules, alpha 3.0 (11,800)    80.20% +/- 0.78    -4.17
```

**1.61 points is now the honest price of explain() on this task**, down from 4.24 before the labels were fixed.

Still open, and independent of alpha: **19-20 of 32 rules wire together concepts that co-occur less than chance** (median rule-pair
lift 0.79-0.92). Honest concepts do not guarantee sane rules. `docs/ALPHA-CLEAN-SWEEP.md`.

## The rules are not really 3-concept rules - effective connectivity is 1, and the "implausible rules" finding was measuring noise (2026-11-14)

Audited the recommended alpha 1.0 checkpoint at `runs/phase2_sweep/CLEAN14_A1p0/final_seed0.npz`.

**1. The old audit ignored weight polarity.** 22 of 32 rules have opposite-signed top-2 weights, and a rule weighted (+A, -B) means
"A high AND B low", so below-chance P(A and B) is *expected*. Correcting polarity flips rule 24 from **0.53 to 1.47** and rule 26
from **0.59 to 1.41**, but only moves the below-chance count 19 -> 18, so it was necessary and not sufficient.

**2. The real problem: the weights, not the wiring.**
```
nonzero kept weights per rule      min 3   median 3   max 3    <- the advertised "2-3 concepts"
effective concepts at |w| > 0.50   median 1   rules with <=1: 27 of 32
share of |weight| in the top concept: median 0.71, 11 of 32 above 0.90
```
Real examples: `rule 23: outline_ratio +1.14, light_region_count -0.01, diag_edges +0.01`. Most rules are **single-concept rules
wearing two near-zero weights**, and `explain()` would faithfully print three concepts when two contribute nothing.

**3. The "implausible rules" claim is withdrawn.** The 13 rules that genuinely read two concepts (both weights > 0.40) have a naive
co-occurrence lift median of **1.25 - above chance, i.e. plausible**. The below-chance numbers came from rules whose second weight
is noise, where a pair test measures nothing (correlation between |second weight| and lift is +0.494).

**Honest statement of the architecture's effective structure: at |w| > 0.50 the median rule reads ONE concept, and 27 of 32 rules
read one or fewer.** 13 of 32 rules are genuine two-concept rules and they behave plausibly. The fix is a minimum-contribution
floor during pruning, and reporting multi-concept and single-concept rule counts separately.
`docs/RULE-EFFECTIVE-CONNECTIVITY.md`.

## Enforcing a minimum contribution: the honest rule structure is ~1.5 concepts per rule (2026-11-14)

The advertised "each rule connects to 2-3 concepts" was true in span and false in effect. Adding a **minimum contribution floor**
(after the top-`keep` cut, zero any weight below it) measures the real structure:

```
floor   accuracy (3 seeds)   avg conns/rule   rules reading >= 2 concepts (|w| >= 0.50)
none    82.76% +/- 0.35         3.00 reported        12 / 11 / 9  (multi / single / dead)
0.25    82.22% +/- 0.35         1.55                  5 / 24 / 3
0.50    78.95% +/- 0.88         1.06                  5 / 24 / 3
```

**The real structure is ~1.55 concepts per rule**, and only **4-5 of 32 rules** genuinely read two or more - the rest are
single-concept rules wearing a 3-connection costume. A 0.25 floor costs just **-0.54 points** and makes the number honest; a 0.50
floor costs **-3.81 points without reducing the multi-concept count at all**, which triggers my own written falsifier: the weights
in the 0.25-0.50 band are **collectively load-bearing**. The rules are **sparse in span, dense in effect**.

**Recommended: floor 0.25**, and `explain()` must report effective connectivity at that floor plus the disclosed tail, because the
tail is worth ~3.8 points and dropping it silently would misreport the model.

Next: try making sparsity true by **training** (higher L1) instead of by amputation. `docs/WFLOOR-MINIMUM-CONTRIBUTION.md`.

## Stronger L1 buys SINGLE-concept rules, not multi-concept ones - and that is the better readable model (2026-11-14)

I predicted a stronger L1 penalty would concentrate each rule onto two concepts. It did the opposite:

```
lam      accuracy (3 seeds)    rules by REAL connectivity (|w| >= 0.50)
0.005    82.76% +/- 0.35       12 multi / 11 single / 9 dead
0.01     82.06% +/- 0.69        0-2 multi / 30-32 single / 0-1 dead
0.02     81.71% +/- 0.78        0 multi / 32 single / 0 dead
0.05     80.53% +/- 0.89        0 multi / 32 single / 0 dead
```

**At lam >= 0.02 every one of the 32 rules reads exactly ONE named concept and no rule is dead.** L1 shrinks the minor weights
harder than the dominant one, so a rule ends up concentrated on its single best concept instead of split across two.

This replaces a false complex claim with a true simple one. Instead of "each rule connects to 2-3 concepts" (measured false in
effect: 1.55 real concepts per rule), the architecture can now honestly say: **32 rules, each reading exactly one named concept
with one signed weight, no dead rules, concept interaction 0.0%** - a statement a 12-year-old can check by reading 32 numbers. The
concept layer stays honest at this setting (0 of 14 below 0.50, 0 mislabeled).

Cost: **-0.70 points at lam 0.01, -1.05 at lam 0.02**, putting the fully single-concept readable model **2.66 points behind the
matched plain control** (84.37 +/- 1.04).

Also recorded: the audit's `rule-pair lift` metric (27 of 32 below 1.0 at lam 0.02) is **provably meaningless for single-concept
rules** - it tests a rule's second-strongest weight against noise - and the honest connectivity count supersedes it.
`docs/LAM-SINGLE-CONCEPT-RULES.md`.

## explain() is causal, but 5 of the 14 named concepts are read by no rule - and trimming them costs 1.55 points (2026-11-14)

`tools/explain_verified.py` prints a receipt and then **removes the rules it named** to check the claim:

```
image 2  true trouser  predicted trouser  margin +5.083
   rule #25 reads light_region_count (-0.76) -> +0.969 to class trouser
   rule #26 reads top_heavy (-0.85)          -> +0.812 to class trouser
   remove top-2 rules -> margin +3.192 (37% gone)
   remove 2 RANDOM    -> margin +4.110 (19% gone)
```

Across images, removing the top-2 **named** rules removed a mean **346.7%** of the margin (sign flips on some images) against
**10.8%** for 2 random rules - **the receipt is causal, not decoration.**

**But 32 rules read only 9 of the 14 concepts.** Five concepts - `center_ink`, `border_ink`, `outline_ratio`, `symmetry_center`,
`tall_shape` - are read by no rule. They are honest (they track their own labels) and **idle**: nothing downstream reads them, so
they cannot affect a prediction.

The coverage penalty cannot fix it: `--cov` is genuinely wired into the loss, but at `lam 0.02` the L1 penalty shrinks exactly the
weights coverage tries to grow, and coverage stayed at exactly 9 of 14 for cov 0.05 and 0.2.

Trimming the set to those 9 costs **80.16% +/- 2.13 against 81.71% +/- 0.78** - 1.55 points and a tripled spread - which triggers my
own written falsifier: the idle concepts help through the joint objective, not through rule weight. So they stay, and the honest
burden becomes **disclosure**: `explain()` names only rules that actually contributed and never presents the idle five as reasoning.
`docs/EXPLAIN-CAUSAL-IDLE-CONCEPTS.md`.

## The idle concepts act only through supervision - excluding them from the rules is FREE (2026-11-14)

Last run left a contradiction: 5 of 14 concepts are read by no rule, yet trimming them from the concept set costs **1.55 points**.
New trainer flag `--rule-idle` keeps the full 14-concept supervision but **forbids the rule layer** from reading those five:

```
                                                     accuracy (3 seeds)   spread
14 concepts, rules free (baseline)                   81.71% +/- 0.78      0.78
14 concepts, rules FORBIDDEN the 5 idle concepts     81.58% +/- 0.50      0.50
9 concepts only (trimmed entirely, not recommended)   80.16% +/- 2.13      2.13
```

**-0.13 points, inside the noise floor, with a tighter spread.** So the five concepts matter purely through the concept-supervision
term and never through a path a prediction can take.

That makes the readable claim stronger than disclosure: **14 named concepts, 8 read by the 32 single-concept rules, and the other 6
training-time only - excluded from the rules by construction, so they cannot appear in any explanation at all.** A reader cannot be
misled by a concept the architecture physically denies the rules.

Recommended config: 14 de-duplicated concepts, alpha 1.0, lam 0.02, 32 single-concept rules, keep 3, `--rule-idle` for the 5 →
**81.58% +/- 0.50** (matched plain 84.37 +/- 1.04), 0 of 14 concepts mislabeled, and `explain()` verified causal.
`docs/RULE-MASK-CHANNEL.md`.

## Rule count is saturated: 64 rules do not beat 32 (81.27% +/- 0.65 vs 81.58% +/- 0.50), so the gap is the concept bottleneck (2026-11-14)

Also closed last run's open gap - concept honesty is now audited **at the final configuration**: **0 of 14 concepts mislabeled in both
seeds**, mean own-label correlation 0.768-0.773.

Doubling the rules changed nothing, and my saturation falsifier fired (P1 predicted >= 82.08, got 81.27). The decomposition of the
readable model's 2.79-point gap is now measured rather than assumed:

```
matched plain MLP (10,345 params)                     84.37% +/- 1.04
readable, concepts -> classes only (no rules)         83.13% +/- 0.42   -1.24  <- concept bottleneck
readable, 14 concepts -> 32 single-concept rules      81.58% +/- 0.50   -2.79  <- +1.55 for the rules
```

**The largest remaining lever is the concept layer, not the rules:** 14 hand-computed pixel statistics cannot represent Fashion-MNIST
as well as 784 raw pixels. That is the honest price of a readable middle layer, now quantified.

Also recorded so nobody wastes a run: **`--keep` is now a dead knob.** At lam 0.02 every rule carries exactly one weight above 0.50,
so keep 3/6/12 describe a structure training already produces on its own. `docs/RULE-COUNT-SATURATED.md`.

## More concepts is not the fix: +8 concepts bought 0.57 accuracy, cost 3 mislabeled labels, and widened the penalty (2026-11-14)

The concept bottleneck (1.24 points) - is it the *number* of statistics or their *kind*? Tested with 22 features instead of 14, each
against a plain control of the same parameter count:

```
                                params    accuracy (3 seeds)   vs its own matched plain
14 concepts (recommended)       11,800    81.58% +/- 0.50      84.37 +/- 1.04  -> -2.79
22 concepts                     18,336    82.15% +/- 0.38      85.28 +/- 0.46  -> -3.13
```

**The count does help a little (+0.57), but the readable penalty gets WORSE (2.79 -> 3.13)** because extra capacity helps the plain
network half again as much (+0.91 vs +0.57). And the 22-set costs honesty: **3 of 22 concepts mislabeled** against **0 of 14** for
the clean set, because relaxing the correlation limit put overlapping labels back.

So the 22-concept set is **not adopted** - it is a worse model on the project's own priority order. The finding that matters is a
scaling statement: **the readable architecture falls further behind as it grows**, which points at the *kind* of statistic
(whole-image summaries discard spatial detail) rather than the count.

Next: the same measurements computed **per image region**, so concepts keep spatial information while remaining nameable. Prediction
rule pre-registered: closing more than 1 point of the 3.13-point gap means resolution was the constraint; otherwise ~3 points is the
honest, irreducible price of a small readable middle layer. `docs/SET22-MORE-CONCEPTS.md`.

## Per-region concepts: resolution is real but small - the readable penalty is ~2.4 points (2026-11-14)

The pre-registered test ("if per-region concepts close more than 1 point of the gap, resolution was the constraint") **failed its own
threshold**, and that is the result:

```
                                           params    accuracy (3 seeds)    gap vs matched plain
14 whole-image concepts                    11,800    81.58% +/- 0.50       -2.79
12 per-region concepts                     10,166    81.95% +/- 0.53       -2.42
```

Twelve new per-region concepts (`ink_TL`, `hedge_TR`, `vedge_BL`, ... - the same measurements computed separately on each 14x14
quadrant) recovered **0.37 of the 2.79-point gap, about 13%**, far short of the 1-point bar. So **~2.4 points is close to the
irreducible price of a small readable middle layer on this task.**

They are still worth keeping as an efficiency win: **better accuracy with 14% fewer parameters**, and better coverage (**10 of 12
concepts read by rules** against 8 of 14), at the cost of one mislabeled concept.

**Best honest config: 12 per-region concepts -> 32 single-concept rules -> 10 classes, 81.95% +/- 0.53, 10,166 params, 1 of 12
mislabeled, `explain()` verified causal.** Every lever on this line - concept count, concept kind, rule count, connections,
sparsity, coverage, trimming, masking - has now been measured rather than assumed. `docs/REGIONAL-CONCEPTS.md`.

## The chat concepts are not redundant - the defect that cost the image model does not exist there (2026-11-14)

Image side, the central defect was redundant labels (8 pairs above 0.80 correlation, 2 of 14 concepts mislabeled until they were
de-duplicated). Chat concepts are token lists, so the same defect would show as **overlapping vocabulary**. Measured directly:

```
chat_concepts.py (12 concepts, the ones train_chat.py uses)
  no two concepts share a single token - disjoint by construction
chat_concepts_wide.py (43 concepts)
  highest overlap anywhere: mentions_education ~ mentions_science at jaccard 0.08 (the word "study")
  pairs above 0.50: 0     above 0.20: 0
```

**So de-duplication is not needed on the chat side**, and one hypothesis about its poor concept honesty is eliminated before
spending GPU time. What is NOT established: uncorrelated labels *in practice* - two concepts can share no words and still co-occur
constantly (greeting/farewell by position, food/request by topic). That check needs the 1.1 GB token file loaded, so it waits for a
free machine. Prediction written in advance: no pair above 0.50, at most two above 0.30.
`docs/CHAT-CONCEPT-OVERLAP.md`.

**Correction (same run):** my first version said the 12-concept module is the one `train_chat.py` uses. It is not -
`train_chat.py:860` defaults `TC_WIDE=1`, so the **live set is `chat_concepts_wide`, 43 defined and 36 kept** after the support
floor (confirmed in every training log). The finding is unaffected and applies to the set that matters: the overlap check covered
all 43 defined concepts, a superset of the 36 kept, and the worst overlap anywhere is **jaccard 0.08** with **0 pairs above
0.20**. So the chat labels are still not redundant and image-side de-duplication is still unnecessary there.

## Chat concept labels are not redundant in effect either - the one real correlation is a single ambiguous word (2026-11-14)

Disjoint vocabulary does not imply disjoint events, so the follow-up measured pairwise phi between every pair of the live chat
concept labels over **60,000 sampled training windows**:

```
595 pairs    above |phi| 0.50: 0    above 0.30: 1
strongest: is_an_agreement ~ mentions_law +0.415
next:      is_a_request ~ mentions_the_user +0.203
commonest concept (mentions_the_user): max |phi| with any other = 0.203
```

All three pre-registered predictions held. **The only correlation above 0.30 is explained and fixable:** `is_an_agreement` and
`mentions_law` share exactly one token - the word **"right"** ("you're right" vs "right to remain silent") - so one ambiguous word
manufactures the only meaningful co-occurrence in the whole set. Give "right" to one of them and the set is essentially orthogonal.

**Conclusion: redundant labels are not why chat concept neurons fail to mean their names** - ruled out in both forms (vocabulary
max jaccard 0.08, effect max phi 0.415), with numbers, so the remaining suspects are training pressure and the rule wiring, in that
order. `docs/CHAT-CONCEPT-CORRELATION.md`.

## Concept supervision sets chat concept honesty - 19 -> 31 of 36 concepts pass their own-name audit, at no accuracy cost (2026-11-14)

Two label-based explanations for chat concept dishonesty were already eliminated (vocabulary jaccard 0.08; max pairwise phi 0.415,
one shared token). That left training pressure. Swept `--lam-concept` on 2-epoch d512 arms over the full 294M-token corpus:

```
lam-concept   concepts passing own-name audit   VAL accuracy
0.1                 19 of 36  (52.8%)              18.86%
0.3  (default)      25 of 36  (69.4%)              18.88%
0.6                 31 of 36  (86.1%)              18.89%
```

**+12 concepts become name-honest between the default and 0.6, and accuracy moves by +0.03 points - i.e. not at all.** P1 held
strictly; P2's predicted direction was wrong (accuracy rose instead of falling); the falsifier (flat pass counts) did not trigger, so
chat honesty **is** governed by supervision pressure, agreeing with the image line.

**The default `--lam-concept 0.3` is leaving 6 concepts per run of honesty for nothing.** Not yet claimed for the shipped model: these
are 2-epoch, single-seed, d512 arms, and the curve has not plateaued (31/36 at 0.6), so the next arm is lam 1.2 to find where
honesty saturates and where accuracy finally pays. `docs/CHAT-LAMCONCEPT-SWEEP.md`.

## Correction: the concept honesty audit passes degenerate concepts and fails accurate rare ones (2026-11-14)

The fidelity check (`lens/api.py:176`) passes a concept when **AUROC >= 0.55 and fires >= 0.02**. So "31 of 36 concepts mean their
name" means "31 of 36 beat barely-above-chance". The full ladder from the same three runs:

```
lam        >=0.55*   >=0.60   >=0.70   >=0.80   >=0.85
0.1           19       13        4        2        0
0.3           25       19        8        4        3
0.6           31       25       13        4        3
```

At a bar that means something, lam 0.6 is **13 of 36 (36%)**, and 3 of 36 reach 0.85.

The instrument also inverts its own purpose:

- `mentions_a_number` is **well detected (AUROC 0.84468)** and is FAILED only because it fires on 1.7% of windows against a 2% floor.
- Seven concepts pass **only** because the bar is 0.55, and all seven fire on **90-97%** of windows - a nearly constant detector can
  take a slightly-above-chance AUROC for free.

Fix queued: quote the ladder, drop the positive-rate penalty, and add a constant-detector guard. `docs/CONCEPT-AUDIT-INSTRUMENT-FLAW.md`.

### Follow-up measured the same night: what survives the corrected verdict

With the strict verdict wired in (`lens/api.py`, old `pass` untouched; self-test `tools/test_strict_audit.py` 7/7):

```
lam     old pass   strict pass   median AUROC
0.1       19/36      20/36         0.56490
0.3       25/36      18/36         0.61657
0.6       31/36      21/36         0.67186
```

- **What survives:** concept pressure genuinely raises ranking quality - median AUROC **0.56490 -> 0.61657 -> 0.67186**.
- **What does not:** the "19 -> 25 -> 31 of 36 pass" improvement. Under the corrected instrument it is **20 -> 18 -> 21**, so most of the
  gain at the 0.55 bar was near-constant detectors (11 concepts fire on more than 90% of windows at lam 0.6) becoming slightly less
  arbitrary, not concepts becoming honest.
- **New structural finding:** **11 of 36 chat concepts (31%) fire on >90% of windows.**

### Fourth arm: lam-concept 1.2 - the curve keeps rising and accuracy still does not fall

```
lam      old pass   strict pass   median AUROC   fires > 0.90   VAL acc
0.1        19/36      20/36         0.56490          ?           18.86%
0.3        25/36      18/36         0.61657          ?           18.88%
0.6        31/36      21/36         0.67186         11           18.89%
1.2        36/36      30/36         0.76235          6           18.92%
```

- P1 held (36/36 old-bar, above the predicted 33); **P2 failed** - accuracy did not pay, it rose to 18.92%, the best of four arms.
- Strict pass climbs **18 -> 21 -> 30 of 36** and median AUROC **0.61657 -> 0.67186 -> 0.76235** from lam 0.3 -> 0.6 -> 1.2.
- **Near-constant concepts halve, 11 -> 6**, so the flaw that made the old bar misleading is itself reduced by pressure.
- Not claimed for the shipped model: four 2-epoch, single-seed, d512 arms. Full-training confirmation is the next step.

## RETRACTION (same night): every full-training run already passes, so lam-concept is not a lever for the shipped model

Checked the audit logs of the six full-length runs already on disk - free, no GPU:

```
log                epochs  median AUROC   old pass   STRICT pass   fires > 0.90
d1024_e9_s0..s2      9      0.968-0.973     36/36       36/36            0
d512_e9_s0..s2       9      0.871-0.874     36/36       36/36            0
dropunk_d1024_s0     9      0.97111         36/36       36/36            0
short arms (2 ep)    2      0.565-0.762     19-36/36    18-30/36       0-16
```

**Withdrawn:** that the default `--lam-concept 0.3` leaves concepts on the table (it already delivers 36/36 strict at 9 epochs), and
that 11 of 36 concepts are near-constant (all six full-length runs have **zero** - it is an under-training artefact).

**Survives:** the instrument flaw was real and is fixed (strict verdict + 7-case self-test), and at low compute concept pressure does
raise concept ranking (median 0.565 -> 0.762).

The lesson, now written into the docs: **a short arm answers a question about short training, never about the shipped model.**

**Next open question:** full training scores 36/36 against *the same deterministic label function the neurons were trained to predict*,
so a high AUROC partly measures "it learned its own target". The honest follow-up is an independent-label audit plus a causal
explain() test, not another knob sweep.

## Do chat concept neurons mean their names, or only their keyword lists? (2026-11-14)

The trainer's own fidelity audit scores each concept against **the same label the neuron was trained to predict**, so 36/36 at AUROC
0.87-0.97 cannot tell "learned the meaning" from "learned its own target". Scored instead against **held-out synonyms** chosen
disjoint from every concept's trained token list (22 concepts, d512 9-epoch checkpoint, 4,000 held-out windows, inference only):

```
median trained-label AUROC   0.9097
median held-out AUROC        0.6980     (chance 0.50)
concepts below 0.60         6 of 22
worst: mentions_law 0.9515 -> 0.4217  (below chance)
biggest drops: work -0.39, science -0.34, number -0.34, weather -0.32, explanatory -0.31
best: time 0.90, family 0.90, food 0.90 held-out
spearman(trained-list length, held-out AUROC) +0.2772
```

**So the concept layer is not a pure keyword detector - it transfers to unseen synonyms at median 0.70 - but it is far short of its
own audit number, and some concepts collapse.** My P1 (median <= 0.65) and P2 (half below 0.60) both FAILED; P3 held; the falsifier
did not trigger. Caveats stated in the doc: only 7-234 held-out positives per concept, synonym lists are my own choice, one seed, and
the two instruments differ on the trained-label number (0.9097 here vs 0.87449 in the trainer log - different val sampling).
`docs/CHAT-CONCEPT-MEANING-VS-KEYWORDS.md`.

### Doubling the width widens the meaning gap (same audit, d1024 checkpoint)

```
checkpoint        trained-label   held-out   gap      below 0.60
d512  e9 s0         0.9097        0.6980   0.2117      6 of 22
d1024 e9 s0         0.9754        0.7178   0.2576      3 of 22
```

Width buys **+0.0657 on the concepts' own trained labels** but only **+0.0198 on meaning transfer**, so the gap **widens** by +0.0459
with scale. The model improves at what the audit measures faster than at what it is supposed to measure - **concept honesty on the
chat line is not a scaling problem**, and an independent-label instrument has to be part of the gate.

## Is chat explain() causal? Yes - but it accounts for only ~10% of the decision (2026-11-14)

Ablating the concepts explain() names vs random concepts, 200 held-out windows (d512, 9 epochs):

```
margin removed, NAMED top-3 concepts : 17.51%
margin removed, RANDOM 3 concepts    :  2.08%      ratio 4.13x
prediction flipped, named 22.50%  |  random 5.83%
median share of the predicted logit from the concept path: 9.74%
explain() residual: max 18.7734     (its docstring says ~0)
concepts needed for 90% of the margin: 21 of 36
```

- **explain() is causal, not a story** (4.13x random; P1 held).
- **But the named concepts account for a median 9.74% of the predicted logit.** About 90% comes from the direct pathway
  (`logits += relu(x @ dmix) @ E.T`, `lens/model.py:427`), which explain() does not decompose and never mentions.
- **explain()'s own claim of an exact decomposition is false on direct-path checkpoints**: residual up to 18.7734 against a
  documented 1e-4 bar (P3 failed).

**So the chat explanation is correct and incomplete, and the incompleteness is invisible unless the residual is read.** That is the
sharpest defect found on the chat line so far. Next options, to be measured not assumed: decompose the direct term too and show both
shares, or make the concept path carry the decision and pay whatever accuracy cost that has.

My first run of the tool was invalid - it reported a negative margin removed (-56.55%), which is impossible, and that is what exposed
that I had omitted the direct term. `docs/CHAT-EXPLAIN-CAUSAL.md`.

### Fixed in the same wake: explain() now names the direct pathway

`lens/concepts.py` explain() now returns `direct_term`, `concept_share` and `direct_share` (shares sum to 1), so the ~90% that the
concept layer does not explain is stated in every explanation rather than left in an unnamed residual. Verified on three held-out
windows: concept share **37.6% / 14.1% / 7.5%**, direct share **62.4% / 85.9% / 92.5%**.

### Verified across consumers, and measured over 200 windows instead of 3

The added explain() fields broke nothing (`api.explain` keys PASS, shares sum to 1 PASS, `explain_text()` still returns 884 chars).
Distribution over 200 held-out windows:

```
concept_share   median 18.00%  (q25 11.48, q75 29.54, min 3.90, max 66.98)
direct_share    median 81.99%
concept layer is the MAJORITY contributor : 10 of 200  (5.0%)
concept layer contributes under 25%       : 136 of 200 (68.0%)
explain() exact on this checkpoint        : 0 of 200
```

Two share numbers, both recorded with their definition: **9.74%** net-signed (ablation tool, decision-relevant) and **18.00%**
absolute (explain()'s own reporting). **On 95% of windows the concept layer is not the majority reason for the prediction, and
explain() is not exact on any of the 200.**

### Doubling the width halves the concept layer's share of the decision (pre-registered, all 3 predictions held)

Same instrument, same 200 windows, only the checkpoint changed:

```
checkpoint   concept_share median   direct_share   concepts majority   concepts <25%   exact
d512  e9 s0       18.00%              81.99%         10 of 200          136 of 200    0 of 200
d1024 e9 s0        8.89%              91.11%          1 of 200          174 of 200    0 of 200
```

Two independent instruments now agree in direction: **the name-generalisation gap WIDENS with width (0.2117 -> 0.2576) and the concept
share of the decision SHRINKS (18.00% -> 8.89%)**, with majority-contributor windows falling 10 -> 1. The bigger model routes more of
the answer around the named layer while its own audit improves to 0.9754.

**So "chat readability is not a scaling problem" is a two-metric finding: width makes the readable story weaker on both meaning and
coverage.** Getting a bigger model will not fix it; the direct pathway has to be constrained by design, and every future width
increase should carry this measurement.

## What the direct pathway is worth: the concept layer alone keeps ~18% of the accuracy (2026-11-14)

Evaluated each checkpoint with the direct pathway removed (`logits = h @ W + b`), 2,000 held-out next-token positions:

```
checkpoint      full acc   concept-only acc   loss        concept path keeps   agreement
d512  e9 s0      28.25%          5.25%        -23.00 pts         18.58%          18.75%
d1024 e9 s0      29.30%          5.25%        -24.05 pts         17.92%          19.70%
```

**The layer the audit rates 36/36 name-honest reproduces less than a fifth of the model's accuracy on the model's own task.**

Four instruments now agree:

```
instrument                                  d512      d1024     direction
own-label audit                             0.9097    0.9754    concepts look BETTER
held-out synonyms                           0.6980    0.7178    meaning lags (gap widens)
share of the predicted logit                18.00%     8.89%    coverage shrinks
accuracy retained without the direct path   18.58%    17.92%    performance collapses
```

**Consequence: removing the direct pathway is not a cheap cleanup, it is a research bet** (P2 and P3 held; my prediction that full
accuracy would be 15-25% FAILED at 28-29%). This is an upper bound on removal cost, since the checkpoint was trained with the path.

Recommendation: **report both shares now** (truthful today, costs nothing) while a constrained-path arm runs as an experiment.
`docs/DIRECT-PATH-COST.md`.

## The concept readout is degenerate: it answers `<unk>` every time (2026-11-14)

The concept-only accuracy was **5.25% at both widths**, which is suspicious, so it was checked instead of quoted:

```
checkpoint        concept-only acc   bias-only acc   distinct tokens predicted   top-1 share
d512  e9 s0            5.25%             5.25%                 1                    100.0%
d1024 e9 s0            5.25%             5.25%                 1                    100.0%
targets that are <unk> : 105 of 2000 = 5.2500%
```

**The concept path predicts `<unk>` on 2,000 of 2,000 positions, so its accuracy is exactly the `<unk>` base rate.** Cause, measured:
`bias[<unk>] = 1.2886` against `0.3276` for the next token (3.9x), and the concept-to-answer column for `<unk>` averages `0.3197`
against `0.1879` for all tokens (1.7x). The concepts are not too weak - median contribution 2.618 vs bias 1.289 at d512 - they are
simply all funnelled into the same winning column.

**This corrects the previous number:** the "concept path keeps 18.58% / 17.92% of the accuracy" figure is arithmetically true and
misleading. The retained accuracy is the `<unk>` base rate, not retained skill.

**So there are two separate defects - coverage (direct path carries 82-91%) and a degenerate readout - and fixing one without the
other changes nothing.** A constrained-path arm must fix the readout too, or it just becomes a `<unk>` machine. P1 and P2 held; P3
failed (my magnitude hypothesis was wrong). `docs/CONCEPT-READOUT-DEGENERATE.md`.

## The concept activations carry almost no generalisable information (2026-11-14)

Two tests, both inference-only, both predicted before running:

**1. Can the model's own readout be repaired post-hoc? No.**
```
baseline 5.25% | no_unk 3.15% | centred 0.60% | no_unk+centred 0.65% | zrow 5.25%
```
Every repair fails; three are worse than doing nothing, and it drops BELOW the `<unk>` base rate once `<unk>` is excluded - there is
no hidden ranking under that column.

**2. Is the information in the activations at all? Almost none.** My own softmax-regression probe from the 36 concept activations to
the next token, fitted on 20,000 train positions:

```
probe TRAIN accuracy      18.32%    <- it can fit, so the probe is not broken
probe HELD-OUT accuracy    4.45%    <- below the <unk> base rate of 5.25%
```

Train 18.32% vs held-out 4.45% is memorisation, not representation. P4 (>=15%) failed, P5 (<8%) held.

**So no readout fix can rescue the chat concept layer - the information is not there to read.** The readable layer is decorative for
prediction: direct pathway 82-91% of the decision, concept path answers a constant `<unk>`, held-out probe 4.45% against a full model
of 28.25%.

**Consequence: do not spend a GPU hour on a readout tweak.** The next readable-chat arm must force the concepts to carry the answer -
a bottleneck like the image model's concepts -> rules -> classes (81.95% +/- 0.53 at 10,166 parameters) - and until it trains, every
public statement must say the concepts do not yet carry the answer. `docs/CONCEPTS-CARRY-NOTHING.md`.

Two of my own bugs were caught and fixed before any number was trusted: an underfitted 400-step probe (2.25%, thrown away) and a
weight-gradient shape error.

## A fully readable chat checkpoint exists: forcing the concepts to carry the answer costs ~4 points, not 23 (2026-11-14)

Two matched 3-epoch d512 arms, identical except `--direct --direct-mode tied`:

```
epoch      no-direct VAL   control VAL   gap
1             9.94%          16.63%      -6.69
2            13.53%          18.97%      -5.44
3            14.84%          19.51%      -4.67
held-out      15.63%          21.53%      -5.90

next-token metric, 2000 positions:  no-direct 17.25% (28 distinct tokens) | control 21.20% (121) | 9-epoch direct 28.25% (132)
```

- **`explain()` is exact on 100 of 100 windows on the no-direct checkpoint** (mean residual 0.000000, max 0.000003, concept_share
  100.00% by construction), against exact on none of 200 with concepts carrying 8.9-18% on the direct-path model.
- **The price is ~4 points of validation accuracy, not the 23-24 the inference ablation suggested** - that number was labelled an
  upper bound on a model trained *with* the path, and this matched pair shows a retrained model recovers most of it.
- **`<unk>` degeneracy disappears: 0.0% of predictions in both 3-epoch arms** versus 18.75% in the 9-epoch direct checkpoint. So
  `<unk>` dominance was a product of the direct pathway plus `--drop-unk`, not the concept layer.
- **The real cost accuracy hides: output diversity collapses to 28 distinct tokens against 121.** At 3 epochs the no-direct model
  is a narrow talker.

P1 failed (I predicted <=12%, got 14.84%) and P2 failed (predicted >100 distinct tokens, got 28); P3 held. **Readable chat is not
hypothetical and needs no new architecture - it needs a full-length arm of the arm that already works, plus diversity work.**
`docs/READABLE-CHAT-NODIRECT-ARM.md`.

## Why the readable chat arm is narrow: 36 saturated concepts cannot cover 12,002 tokens (2026-11-14)

Decomposing the readable arm's predictions against its matched control (2,000 held-out positions, both 3-epoch d512):

```
                              no-direct   control
concept-only accuracy           16.45%      3.25%
concept agreement with full     84.8%       6.6%
bias agreement with full        19.5%       7.2%
distinct tokens, full              28        121
distinct tokens, concept only      27         12
mean entropy (nats)            6.3085     5.1881
concepts firing (>0.5)        18 of 36    7 of 36
```

- **In the readable model the concepts really do decide** - concept-only agreement 84.8% and accuracy 16.45% against the full
  17.25%. The readable path is faithful, unlike the direct-path model (6.6% agreement, 3.25%).
- **The bias is NOT the cause** (19.5% agreement, 2.60% alone, top two entries nearly tied).
- **It is not over-confidence either - it UNDERTALKS**: entropy is HIGHER (6.3085 vs 5.1881) while it emits only 28 words.

**Cause: 18 of 36 concepts fire on a median window, so the code is half-saturated and the 36 -> 12,002 projection can only learn
about 28 output modes.** That is a capacity problem in the concept layer. Fixing it means widening the concept set (the trainer has
no `--k` flag; K comes from the concept module), which is a change to the concept definitions and is being raised with the owner.

All three of my predictions failed and the falsifier fired, which is the only reason the conclusion landed on capacity instead of the
bias. `docs/history/DIVERSITY-COLLAPSE-CAPACITY.md`.

## At 9 epochs the readable chat model matches its control: -0.17 points VAL, and the only deficit left is vocabulary (2026-11-14)

Matched pair, identical except `--direct --direct-mode tied`, both 9 epochs, d512, seed 0, full 294M-token corpus, each arm trained
fresh in the same run:

```
                                   readable   control    gap
final VAL accuracy                   19.95%    20.12%   -0.17
final VAL ppl                        292.65    352.11       -
held-out best (by position)          21.45%    23.67%   -2.22
next-token metric, 2000 positions    21.90%    22.40%   -0.50
distinct tokens predicted               44       191      -147
concept fidelity                     34 of 36  36 of 36     -2
explanations                     EXACT 100/100  not exact    -
```

- **The accuracy cost of readability has almost vanished: -0.17 points at the final epoch.** The same comparison at 3 epochs was
  **-4.67**, so **4.5 of those 4.67 points were under-training, not readability.**
- **The readable arm's validation perplexity is better (292.65 vs 352.11)**: the control overfits (train acc 36.9 -> 44.3 while VAL
  ppl rises 252 -> 352) and the 36-concept bottleneck regularises.
- **`explain()` is exact on 100 of 100 windows** on this checkpoint (mean residual **5.634e-07**, max **2.295e-06**,
  `concept_share` **99.999996%**), so readability is not a 3-epoch artefact (P3 held).
- **P1 held** (within 3 points), **P2 failed** (predicted >60 distinct tokens, got 44).

**The model is now credible: within 0.17 points of its own control on accuracy, better validation perplexity, exact explanations,
34 of 36 concepts honouring their names. The one remaining deficit is vocabulary breadth - 44 distinct words against 191 - which is
the concept-capacity bottleneck, and widening the concept set is the owner's call because it changes the definitions every audit
compares against.** `docs/READABLE-CHAT-9EP-MATCHED.md`.

## Doubling the width changes almost nothing for the readable chat model (2026-11-14)

Readable arms only (d512 and d1024, both 9 epochs, both forced through the concept vector). The d1024 control is still training, so no
gap claim is made yet:

```
                          readable d512   readable d1024
final VAL accuracy             19.95%          19.95%
next-token, 2000 positions     21.90%          22.45%
distinct tokens predicted          44              49
concepts firing (>0.5)       15 of 36        15 of 36
concept fidelity audit        34 of 36        34 of 36
explain() exact              100 of 100      100 of 100
```

**Final validation accuracy is identical to two decimals at double the width.** The concept layer was diagnosed as the bottleneck (15
of 36 concepts fire on a median window), and if the concept layer is the choke point then hidden width above it should buy nothing -
which is what happened.

Two independent facts now agree: on a **direct-path** model, width buys accuracy by routing *around* the concepts (concept share
shrinks 18.00% -> 8.89%); on a **readable** model, where routing around is impossible, width has nowhere to go (19.95% -> 19.95%).
**The 36-concept layer, not the hidden width, is the binding constraint.**

**Widening the hidden size is measured out as a route to a better readable chat model; widening the concept layer is the remaining
lever** - and that is the owner's decision. `docs/WIDTH-DOES-NOT-HELP-READABLE.md`.

## At double width the direct-path model overfits and the readable model catches it (2026-11-14)

Matched d1024 pair, both arms 9 epochs, seed 0, identical except `--direct --direct-mode tied`. All three endpoints, per this
project's own rule:

```
endpoint                     readable    control    difference
final epoch VAL accuracy       19.95%     18.57%     +1.38  readable
last-3-epoch mean VAL acc      19.59%     19.51%     +0.08  readable
held-out best (by position)    21.57%     23.46%     -1.89  control
final VAL perplexity           291.04     987.76     readable 3.4x better
```

The trajectories are the finding:

```
readable VAL acc  10.02 13.84 15.82 17.38 18.12 18.64 19.21 19.60 19.95   monotonic, still improving at the end
control  VAL acc  17.12 19.41 20.07 20.32 20.54 21.05 20.19 19.76 18.57   peaks at epoch 6 then falls
control  VAL ppl    318   263   262   276   318   363   512   679   988   3.8x explosion
```

**The direct-path model overfits catastrophically at this width** - training accuracy 55.46% while validation accuracy falls and
perplexity goes 263 -> 988. **The readable model, forced through its 36-concept bottleneck, improves monotonically for all nine
epochs and is still improving when it stops.**

So the readable model is ahead on the final epoch and last-3 mean and behind on best-of-N, because best-of-N rewards the control's
single lucky epoch before it collapsed. **P1 FAILED** - I predicted the readable arm would fall >2 points behind at d1024; it did
not. **P2 and P3 held.**

Honest limits: the readable arm had **not converged** (still improving at epoch 9), one seed per arm, and the ordering is
endpoint-dependent. "Readable is better at width" is **not** supported; "at double width the direct-path model overfits and the
readable model catches it" is. `docs/WIDTH-PAIR-D1024-OVERFIT.md`.

## Trade-off found: forcing the concepts to carry the answer makes explanations faithful but concept names WORSE (2026-11-14)

Same instrument, same 22 concepts, same 4,000 windows as the published direct-path audit; only the checkpoint changed (both d512,
9 epochs, seed 0):

```
checkpoint                         own-label median   held-out synonym median   gap     held-out below 0.60
direct path                          0.9097               0.6980             0.2117       6 of 22
READABLE (no-direct)                 0.7193               0.5392             0.1801      16 of 22
```

**The readable model's concepts are markedly worse at both jobs**: -0.19 against their own trained labels, -0.16 against unseen
synonyms. **P1 and P2 both FAILED** (I predicted the readable arm would be MORE name-honest); the falsifier fired.

So the project has two kinds of honesty that pull against each other:

```
                explanation faithfulness                        concept name semantics
direct path     NOT exact (0 of 200, concepts carry 8.9-18%)     BETTER (0.9097 / 0.6980)
readable        EXACT (100 of 100, concepts carry 99.999996%)    WORSE  (0.7193 / 0.5392)
```

Mechanism: 36 concepts must now encode everything needed to predict 12,002 tokens, so each neuron is pressed into service for
information unrelated to its declared name - which is what a drop in own-label AUROC measures.

**For the "100% honest" goal: the readable chat model is honest about the arithmetic of its own decision, and its concept names are
less trustworthy than the direct-path model's.** Reporting only the exactness would be a half-truth.
`docs/HONESTY-TRADEOFF-READABLE.md`.

## The readable model's ceiling is above what 9 epochs showed: 21.48% VAL, 78 distinct tokens, still improving (2026-11-14)

Same readable config at **18 epochs** instead of 9 (config-only change, predictions pre-registered):

```
                          9 epochs    18 epochs    change
final VAL accuracy         19.95%       21.48%      +1.53
final VAL perplexity       291.04       253.02      -38.02
held-out best              21.57%       23.67%      +2.10
distinct tokens predicted      49           78        +29
concept fidelity           34 of 36     36 of 36       +2
concept own-label AUROC     0.7193       0.8384      +0.12
concept held-out synonyms   0.5392       0.5676      +0.03
```

**All three predictions held** (P1 >=20.95% got 21.48; P2 ppl <291.04 got 253.02; P3 diversity 50-190 got 78), and the falsifier did
not trigger. The held-out best was set at the **last** epoch, so this arm had not converged - **quoting the 9-epoch numbers as the
readable model's ability understates it.**

**No head-to-head claim against the 9-epoch control** (18.57% final, ppl 987.76): different epoch counts, so it is not a matched
comparison. The matched 18-epoch control is the outstanding experiment, queued rather than launched.

Concept honesty narrows the trade-off found earlier but does not close it: own-label recovers 0.7193 -> 0.8384, held-out synonyms only
0.5392 -> 0.5676 against the direct-path model's 0.6980. `docs/READABLE-CEILING-18EP.md`.

### Follow-up: the 2x2 at matched epochs proves the trade-off is not a width artefact

Matched 9-epoch audit, 22 concepts, 4,000 windows:

```
                    own-label median            held-out synonym median
                 d512      d1024   change      d512      d1024   change
direct path     0.9097    0.9754   +0.0657    0.6980    0.7178   +0.0198
readable        0.7193    0.7256   +0.0063    0.5392    0.5436   +0.0044
```

- **Width helps the direct-path concepts and does almost nothing for the readable ones** - the same shape as every other width result
  tonight, because the readable model's constraint sits *upstream* of the width.
- **The readability cost to concept names is large at BOTH widths** (-0.18/-0.25 own labels, -0.16/-0.17 held-out), so it is not a
  d512 artefact. **P1, P2 and P3 all held**; the falsifier did not trigger.

**The single sentence: the 36-concept bottleneck caps the readable chat model on accuracy, vocabulary breadth AND the meaning of its
own concept names simultaneously, and none of the three can be relieved with more width.** The only untested lever is widening the
constraint itself.

## Prepared, not launched: a measured concept-widening design, 36 -> 60 concepts (2026-11-14)

Every other lever has been measured and come back negative on the readable chat model (width +0.006 on concept names, four readout
repairs all failing, bias exonerated, more training helping but not closing the gaps). The constraint that explains all of it is that
**13-18 of 36 concepts fire on a median window**.

So the widening design is measured rather than guessed: 26 candidate concepts, support measured on 20,000 held-out windows of the full
corpus against the trainer's own 0.005 floor.

```
24 of 26 candidates survive -> the set goes from 36 to 60 concepts
strongest: code_refs 0.2623, modality 0.2261, help_ask 0.0850, sequence 0.0846, commands 0.0663
dropped at the floor: emotion_neg 0.0016, weather_sky 0.0016  (dropped, not forced)
```

Predictions for the widening arm are written down before it runs: **W1** diversity above 110 distinct tokens (78 now), **W2** median
concepts firing below 13 of 60, **W3** held-out concept honesty above 0.60 (0.5676 now), **W4** accuracy no worse than 20.98% (21.48
now). **Falsifier**: if diversity stays under 90 and honesty under 0.58, then 60 concepts is not enough either and the bottleneck needs
a different capacity mechanism, which is a finding about the architecture rather than the concept list.

**Nothing is trained by this document and no existing concept definition is changed.** It waits on the owner, because adding concepts
changes the definitions every audit compares against. `docs/CONCEPT-WIDENING-DESIGN.md`.

## Matched 18-vs-18: the readable chat model wins on accuracy, perplexity and explanation - and still loses on vocabulary (2026-11-14)

The arms are now matched (both 18 epochs, d1024, seed 0, full corpus, identical except `--direct --direct-mode tied`):

```
endpoint                          readable    control    advantage
final epoch VAL accuracy           21.48%     17.73%    +3.75  readable
last-3-epoch mean VAL accuracy     21.41%     17.71%    +3.70  readable
held-out best (by position)        23.67%     23.43%    +0.24  readable
final VAL perplexity              253.02   54,751.20    readable 216x better
next-token, best checkpoint        23.15%     22.05%    +1.10  readable
distinct tokens, best checkpoint       78        226      -148  control
concept path carries the answer     83.4%       3.8%    readable by construction
concept-only accuracy              22.75%      2.45%    readable 9.3x
```

**The direct-path control memorises**: training accuracy 70.29% while validation accuracy falls from 21.07% (epoch 6) to 17.73% and
validation perplexity explodes **263 -> 54,751** (208x). The readable arm improves monotonically and sets its held-out best at the
**last** epoch.

**So on every accuracy endpoint and on perplexity the readable model is ahead of its matched control, and it is the only one whose
named concepts explain the answer (83.4% vs 3.8%).** Its single remaining deficit is vocabulary breadth: **78 distinct tokens against
226.**

All three control-side predictions held (P1: <=18.57% and ppl >=987.76; P2: held-out best within 0.5 of 23.46%; P3: training far past
55.46% while validation degrades), so the falsifier did **not** trigger - the 9-epoch control had not been stopped early in a way that
flattered the readable arm.

**Not claimed**: that the readable model is better in general. It speaks in 78 words against 226, it is one seed, and its concept
neurons remain less name-aligned than the direct-path model's (held-out synonyms 0.5676 vs 0.7178). The open lever is diversity, and
the measured widening design (36 -> 60 concepts) is ready and waiting on the owner.
`docs/MATCHED-18EP-READABLE-WINS.md`.

## The vocabulary deficit is capacity, not convergence (27-epoch arm, 2026-11-15)

Diversity had climbed 49 -> 78 from 9 to 18 epochs, so the obvious question was whether 78 distinct tokens was simply under-training.
It is not. Repeat at 27 epochs, same config:

```
epoch    ppl      VAL        epoch    ppl      VAL
  1    663.45   10.02%         15    253.45   21.52%   <- VAL peak
  9    273.94   20.72%         19    253.77   21.43%   <- held-out best 23.69%
 12    258.97   21.31%         27    273.68   20.91%   <- final
```

**The readable arm peaks at epoch 15 and then decays** (validation perplexity 253.08 -> 273.68 while training accuracy keeps climbing),
and **50% more training bought +5 distinct tokens (78 -> 83, +6%) and cost 0.57 points of accuracy**. The direct-path control reaches
**226** tokens. So breadth is not a function of training length.

All three predictions failed honestly (D1 wanted >90, got 83; D2 wanted >21.48% final, got 20.91%; D3 wanted ppl <253.02, got 273.68),
and the falsifier did **not** fire in its written form - it required diversity to plateau *while accuracy still rose*, and accuracy
peaked instead. What the data supports, stated directly: **a 50% increase in training bought 6% more vocabulary and no accuracy.**

**Two things this settles**: the widening question is now evidence-based rather than a hunch, and the useful epoch count is **15-19, not
27** - future arms should spend their budget on architecture rather than epochs. `docs/DIVERSITY-CONVERGENCE.md`.

## Widening the concept set to 53: best readable accuracy yet, vocabulary barely moved (2026-11-15)

One change from the 36-concept arm: the concept table grows to **53 kept concepts** (17 measured additions, every added token disjoint
from the 283 tokens already in use and from each other; no existing definition altered). 18 epochs.

```
endpoint                     36 concepts    53 concepts    change
final VAL accuracy              21.48%        22.04%       +0.56
held-out best                   23.67%        24.22%       +0.55
final VAL perplexity            253.02        240.55       better
concept path carries answer      83.4%         86.9%       +3.5
distinct tokens                    78            86          +8 (+10%)
```

**Every accuracy endpoint improved, the best checkpoint again came at the LAST epoch, and the decision now flows through named concepts
86.9% of the time.** Predictions scored honestly: W4 passed, **W1 failed** (>110 tokens predicted, 86 got), **W2 failed in the wrong
direction** (firing rose to 19, which is what more concepts should do - the prediction was written without thinking that through),
**W3 failed** (0.5884 vs the 0.60 target, though up 0.021), falsifier did not fire but missed by 0.0084.

**The measured lesson: named concepts have sharply diminishing returns for vocabulary.** +47% concepts bought **+10%** breadth (+8
tokens). Extrapolating, matching the control's 226 tokens would need hundreds of concepts - no longer a set a human can read. So more
named concepts is a reliable route to **accuracy**, not to **breadth**.

**Also found**: a no-shared-token self-check caught four **pre-existing** pairs sharing a token - `later`, `study`, `law`, `country` -
the same defect class as the `right` collision, invisible to the phi audit because they do not co-fire above 0.30. Inherited unchanged
here, to be fixed in a separate arm.

This is **one seed**; a paired 3-seed check is running before the accuracy gain is treated as established.
`docs/CONCEPT-WIDENING-53.md`.

## The concept-widening gain is real: +0.57 VAL across three paired seeds (2026-11-15)

The widened set beat the 36-concept baseline by +0.56 at seed 0. Because a width claim in this project had already reversed on a later
seed, the pair was retrained at seeds 1 and 2 - four arms, 18 epochs each.

```
seed  base36 VAL   wide60 VAL   paired delta   base ppl   wide ppl
 0      21.48%       22.04%        +0.56        253.02     240.55
 1      21.37%       21.98%        +0.61        251.90     240.91
 2      21.46%       21.99%        +0.53        253.10     235.37

final VAL      mean +0.5667, sd 0.0404, 3 of 3 seeds positive
last-3 mean    +0.5378
held-out best  +0.5733
```

**All three predictions held and the falsifier did not trigger** (S1 mean delta above zero, S2 wins on 3 of 3 seeds, S3 keeps 53
concepts). The widened arm is also better on perplexity on every seed, and raises the share of decisions flowing through named concepts
from 83.4% to 86.9%. **This is the first seed-robust accuracy win in the readable chat line, and it survived exactly the test that
killed the earlier width claim.**

**Not established**: that widening fixes vocabulary breadth. It bought **+8 distinct tokens (78 -> 86)** for +47% more concepts - a
sharply diminishing return. Named concepts are a route to accuracy, not to breadth. Concept-name honesty also remains the weakest
property at **0.5884** on held-out synonyms, missing its 0.60 target by 0.0116. `docs/WIDENING-3SEED.md`.

## Why the readable model talks narrowly: the rank explanation is dead, activation collapse is real (2026-11-15)

The readable model uses 86-135 distinct tokens where its direct-path control uses 226, and widening the concept set 36 -> 53 bought only
+8 tokens. This measures why.

**Hypothesis 1 - the answer map's rank is the ceiling - REFUTED.** The readable answer layer is (53, 12002), so all its logits lie in a
53-dimensional family, and the control adds a rank-1024 direct term. That looked like the answer. Then I sampled the *same* 53-dim family
at random:

```
observed predictions, 4,000 held-out windows :  135 distinct tokens
random draws inside the same family          :  341 distinct tokens
uniform inside the observed range            :  212 distinct tokens
```

**The same family reaches 341 tokens; the model uses 135.** So rank is not the ceiling, and the "rank per concept" surgery I was about to
build was not built.

**Hypothesis 2 - the activations collapse - SUPPORTED.**

```
model                     concept dims   effective dimensionality   variance in top-5 directions
readable 53-concept            53             10.37 of 53 (20%)                 54%
direct-path 36-concept         36              2.97 of 36  (8%)                 82%
```

**53 concept numbers move through about 10 effective directions, with 54% of their variance in five.** No dimension is dead, which is
why the concept audits still pass - but collectively they describe far less than their count. That is why 17 more concepts bought 8 more
tokens.

**What this changes**: neither more concepts nor more rank per concept - both target a ceiling measured not to exist. The target is the
*spread* of the concept activations. The supervision lever is config-only and runs first.
`docs/ACTIVATION-COLLAPSE.md`.

## Naming the collapse, and a redundancy in my own widening (2026-11-15)

53 concept numbers move through about **10.4 effective directions**; this names which ones. Measured with
`tools/chat_concept_entanglement.py` on 4,000 held-out windows.

```
direction 1: 26.6% of variance | mentions_self 0.50, mentions_the_user 0.31, cp_thinking 0.26, is_a_question 0.24
direction 2: 10.1% of variance | mentions_a_number 0.40, is_a_question 0.36, mentions_self 0.27
effective dimensionality: 10.03 of 53 (19%)
```

**A quarter of all concept variance is a single axis: who is speaking and whether it is a question.** Topic concepts barely move. So
the concept vector is mostly a **register detector**, which is why changing the concept list does not change how much the layer tracks.

**The correction**: `cp_certainty` (r = +0.671 with `cp_emotion_pos`) and `cp_thinking` (r = +0.61 with `mentions_self` and
`mentions_the_user`) are concepts **I added** in the widening arm. Both pass my "no shared token with anything already in use" rule -
but **token disjointness is not non-redundancy**: `sure/certain/probably` and `love/like/enjoy` co-occur in hedged emotional chat, so
the neurons learned overlapping signals from disjoint words. **The +0.57 accuracy win is solid and seed-robust, but part of it was
bought with redundant concepts, and the self-check I reported as a clean pass was measuring the wrong property.** The check that catches
this is activation correlation, and it should have run before the arm launched, not after.
`docs/CONCEPT-ENTANGLEMENT.md`.

## Breadth follows concept supervision, not activation spread - second explanation dead (2026-11-15)

One change from the 53-concept readable arm: concept supervision weight 0.3 -> 0.6, 18 epochs, same seed and corpus. All checkpoints
measured on one instrument, 4,000 held-out windows:

```
checkpoint                    effective dims   top-5 var   distinct tokens
direct-path control (36c)      2.97 of 36 (8%)     0.80          539
readable 53c, lam 0.3          10.37 of 53 (20%)   0.54          135
readable 53c, lam 0.6           9.69 of 53 (18%)   0.53          227
```

- **Breadth rose 68% (135 -> 227)** and **concept names got more honest (0.5884 -> 0.6298)**, the first time held-out honesty crosses
  the 0.60 target; trained-label fidelity rose 0.8058 -> 0.9257.
- **Accuracy fell 0.36 points (22.04% -> 21.68%).**
- **The activations did NOT spread: effective dimensionality FELL 10.37 -> 9.69.**

**That kills my second explanation in a row.** `docs/ACTIVATION-COLLAPSE.md` argued breadth was capped by the concentration of the
activation cloud; vocabulary nearly doubled while the cloud got *more* concentrated. The first explanation (the answer layer's rank)
died earlier the same way - sampling the same 53-dim family at random reached 341 tokens where the model used 135.

**What the data says**: the quantity that moved WITH breadth is **label fidelity** - stronger supervision makes each concept fire more
selectively on its own trigger, and a more selective set of 53 numbers takes more distinct combinations, so more distinct tokens, even
though the cloud concentrates. Breadth tracks how decisively concepts fire on their names.

Predictions scored honestly: **L2 and L4 passed, L1 and L3 failed** - and L3 was the prediction that carried the theory. The falsifier
did not trigger (it needed diversity <= 86; got 227), which is exactly what stopped me writing the "needs decorrelation" conclusion
anyway.

**A caveat now stated plainly**: the same control checkpoint gives **226** distinct tokens under `tools/diversity_diagnosis.py` and
**539** under `measure_activation_spread.py`. Both real, not interchangeable - **every breadth claim must name its instrument.**
`docs/BREADTH-IS-SUPERVISION.md`.

### Breadth is now a dial: lam 0.3 -> 0.6 -> 1.0 gives 135 -> 227 -> 364 tokens (2026-11-15)

```
arm                            distinct tokens   effective dims   final VAL   final ppl
direct-path control (36c)            539           2.97 of 36        17.73%     54,751
readable 53c, lam 0.3                135          10.37 of 53        22.04%     240.55
readable 53c, lam 0.6                227           9.69 of 53        21.68%     252.31
readable 53c, lam 1.0                364          10.09 of 53        21.08%     269.30
```

**Breadth is monotonically controlled by the supervision weight** - it is a dial, not a mystery - costing roughly half an accuracy
point per step. **The readable model stays ahead of its matched control on accuracy at every point**, including lam 1.0 where it has
68% of the control's breadth and is **+3.35 points more accurate** (21.08% vs 17.73%) with 52 of 53 concepts still answering to their
names.

The remaining question is no longer "can it talk widely" but **"what happens at equal breadth"** - at lam ~1.5-2.0 the readable curve
should cross the control's 539, which makes the fair fight possible.
`docs/BREADTH-IS-SUPERVISION.md`.

### Three paired seeds: the breadth gain is solid, my honesty claim was not (correction, 2026-11-15)

```
seed   lam0.3 VAL   lam0.6 VAL   delta     lam0.3 tok   lam0.6 tok   delta
  0      22.04%       21.68%     -0.36        135          227        +92
  1      21.98%       21.67%     -0.31        131          238       +107
  2      21.99%       21.61%     -0.38        131          207        +76

accuracy delta : mean -0.350, sd 0.036  (3 of 3 seeds)
breadth  delta : mean +91.7, sd 15.5    (3 of 3 seeds)
```

Both solid. But I published "held-out concept honesty 0.5884 -> 0.6298, first time above 0.60" from **one seed**, and paired across
three seeds it does not hold up:

```
held-out honesty : mean delta +0.0185, sd 0.0229, positive on 2 of 3 seeds   <- WEAK, not established
trained-label    : mean delta +0.137,  positive on 3 of 3 seeds               <- SOLID
```

**The "first time above 0.60" claim is withdrawn as a finding.** What is established is narrower and more useful: stronger supervision
makes concepts much better detectors of **their own trained keyword lists** while their transfer to **unseen synonyms** barely moves -
which sharpens the project's central worry rather than resolving it. Supervision buys fidelity to the words a concept was trained on,
not to the concept's meaning, so a reader of `explain()` should not be told "the concepts got more honest" on the strength of the
trained-label number.

Also noted: I wrote P3 against a fixed baseline value instead of as a paired delta, which was a weaker test than the data allowed.
`docs/BREADTH-IS-SUPERVISION.md`.

### The dial saturates: breadth caps at ~422, and the falsifier fired (2026-11-15)

```
arm                        tokens   effective dims   final VAL   concepts passing
direct-path control          539       2.97 of 36       17.73%         --
readable lam 0.3             135      10.37 of 53       22.04%       52/53
readable lam 0.6             227       9.69 of 53       21.68%       53/53
readable lam 1.0             364      10.09 of 53       21.08%       52/53
readable lam 1.5             414      10.60 of 53       20.36%       50/53
readable lam 2.0             422      11.02 of 53       19.93%        49/53
```

**Breadth rises steeply then stops: +92, +45, +50, then only +8.** Doubling supervision from 1.0 to 2.0 bought **8 tokens** and cost
**1.15 accuracy points**. The readable model's breadth on this lever is **capped at ~422, which is 78% of the control's 539** - and
**the falsifier I registered fired exactly as written** (below 450 tokens with accuracy below 20.50%). The honest conclusion is the
pre-registered one: **the readable design trades breadth for accuracy inside a band it cannot leave by this lever alone.**

Worth keeping in perspective: even at its worst point (19.93%) the readable model is still **+2.20 points more accurate** than its
control, and 49 of 53 concepts still answer to their names.

**Why the readout still is not the ceiling** - the reachability test repeated on every checkpoint:

```
checkpoint     tokens used   reachable in the same family
lam 0.3             135                 341
lam 1.0             364               1,583
lam 2.0             422               1,859
```

**The answer layer can reach 1,859 tokens while the model uses 422**, so the rank explanation stays dead on every checkpoint. The
constraint is upstream: how many distinct patterns the 53 concept numbers take across real text. Effective dimensionality stays between
9.7 and 11.0 on every checkpoint while breadth varies 3x - so supervision raises breadth by making concepts fire more sharply inside a
cloud of constant dimension. **A mechanism that raises that dimensionality has not been tested, and it is the only remaining candidate
with evidence behind it.**
`docs/BREADTH-IS-SUPERVISION.md`.

## The shipped readable configuration: lam 1.0, stable across three seeds (2026-11-15)

```
seed   final VAL   final ppl   concepts passing   distinct tokens   effective dims
  0      21.08%      269.30         52/53               364           10.09 of 53
  1      21.15%      267.04         51/53               352           10.64 of 53
  2      21.17%      262.88         52/53               326           10.14 of 53

final VAL : mean 21.13%, sd 0.05        breadth : mean 347 tokens, sd 19
```

**P1 and P2 passed** (every seed inside the predicted band, spread only 0.09 points; breadth above 300 on all three). **P3 failed
because I wrote it wrong** - it named the paired lam 0.3 arm while claiming to test the direct-path control, and lam 0.3 is the
higher-accuracy readable arm, so the prediction as written required lam 1.0 to beat something it cannot. That is my second
prediction-writing error today; the intended comparison is stated separately instead of retro-fitted.

**The comparison that matters**: readable lam 1.0 is **21.13% +/- 0.05** across three seeds against the matched direct-path control's
**17.73%** - **+3.40 points while being fully explainable**. Deliberate caveat: the control side is a single seed, so the honest form is
"the readable model's three seeds all sit 3.2 to 3.4 points above the one matched control run".

**Shipped point**: lam 1.0 - the widest stable readable configuration that keeps accuracy above 21% with 51+ concepts honest. lam 1.5
and 2.0 reach 414 and 422 tokens but lose 0.8-1.2 points and drop concepts below the name audit, so the ~422 breadth ceiling stands as
documented. `docs/READABLE-SHIPPED-CONFIG.md`.

## Decorrelation works and buys no breadth - the fourth explanation is dead (2026-11-15)

The term was verified before launch: analytic gradient checked against finite differences (max abs error **1.3e-10**), registered,
smoke-tested for a full epoch, and proven to enter the loss only at non-zero weight.

```
arm (53 concepts, lam 1.0)   effective dims   pairs |r|>0.50   tokens   final VAL   concepts
baseline                        10.09 of 53          4           364      21.08%      52/53
+ decorrelation 0.05            11.13 of 53          1           366      21.13%      52/53
+ decorrelation 0.20            12.72 of 53          1           366      21.19%      52/53
```

**The term does exactly what it was built for**: dimensionality **+26%** (10.09 -> 12.72) and concept pairs above |r| = 0.50 fall
**4 -> 1**, with the targeted `cp_certainty`/`cp_emotion_pos` redundancy leaving the top list. Accuracy unchanged, 52/53 concepts honest.

**And breadth does not move: 364 -> 366.** A 26% increase in the dimensionality of the concept cloud bought **two tokens**.

**Four explanations for the ~422-token ceiling are now all dead**: answer-layer rank (the same family reaches 1,859 tokens at random),
activation spread via supervision (breadth 3x while dims stayed 9.7-11.0), concept count (36 -> 53 bought +8), and effective
dimensionality (+26% dims bought +2). The remaining structural difference is the **width of the interface to the answer**: 53 scalars
for the readable model against 1024 dimensions for the control.

That is now testable **without new code** - the model already has a low-rank direct path that adds exactly r scalars - so the next arms
measure how much breadth each added interface scalar buys. It is a diagnostic, not a ship candidate: a direct path costs explanation
purity, and the measurement decides whether that price is worth paying at all. `docs/DECORRELATION-NEGATIVE.md`.

## Correction: the narrow-vocabulary ceiling was partly my measuring instrument (2026-11-15)

Two results in one block.

**1. Interface width is dead too, and the direct path hurts.** Readable arm plus a low-rank direct path (53 interface scalars vs the
control's 1024):

```
arm                        tokens   final VAL   final ppl
readable, no direct path     364      21.08%      269.30
+ direct rank 8              359      18.75%      421.39
+ direct rank 32             297      18.79%      558.07
+ direct rank 128            347      19.50%      907.10
direct-path control          539      17.73%     54,751
```

All three breadth predictions failed, accuracy fell 1.6-2.3 points, and **the falsifier fired** (347 tokens at rank 128, threshold 380).
The explanation-cost step produced no output at all - the `explain_verified.py --ckpt` invocation I flagged as unverified does not take
that argument, so that number is missing rather than assumed.

**2. My breadth metric was measuring confidence, not vocabulary.** Every breadth number today used distinct **argmax** predictions.
Sampling the same distributions on the same windows:

```
checkpoint          argmax   sampled T=1.0   sampled T=1.2   entropy (nats)
direct-path control   395        1,543            1,961            5.02
readable lam 1.0      364          944            1,120            2.05
data itself           971 distinct target types occur in this sample
```

**The readable model produces 944 distinct tokens when sampled against 364 at argmax - close to the 971 types the data actually
contains.** It is not short of words; it is unwilling to commit, because its distribution is far sharper than the control's (2.05 vs 5.02
nats).

**Withdrawn**: any claim that the readable model "cannot talk widely" or that its vocabulary is capped near 422 words.
**Survives**: the accuracy advantage, the shipped lam 1.0 configuration, and the supervision trade-off - which must now be described as
"willingness to commit to a rare token", not "how many words it knows".
**Open**: whether that sampled output is actually good. Diverse tokens are not sensible text, and nothing here measured generation
quality. `docs/ARGMAX-IS-NOT-BREADTH.md`.

### Correction, an hour later: the 944 tokens were junk breadth (2026-11-15)

I published that the readable model "samples 944 distinct tokens, close to the 971 target types the data contains" as evidence it is
not short of vocabulary. Then I measured what those tokens ARE, by generating from held-out prompts:

```
checkpoint / mode        repetition   self_repeat   coverage(top-571 real)   KL(gen||targets)   stop_rate
readable lam 1.0 greedy      0.339        0.229             0.141                  22.5             0.417
readable lam 1.0 sample      0.321        0.260             0.083                  24.0             0.250
direct control  greedy       0.006        0.062             0.042                  24.2             0.000
direct control  sample       0.000        0.021             0.047                  24.2             0.000
```

**The readable model's sampled tokens are 8-14% realistic**, and its text degenerates into symbol runs and repeated turn-end tokens:

```
readable sample : "<|im_end|><|im_end|>youwantyouransweranswer:########"
readable greedy : "################"
control  sample : "moretooandknowledgeabout1peoplebecauseanalyzinganalyzingbefactory.then"
```

**So the 944-token figure was long-tail junk, not vocabulary** - the argmax set was narrow partly because argmax avoids that junk. The
honest statement is the reverse of what I published an hour ago. Both models generate badly, in different ways: the readable one loops
symbols and stops early, the control produces repetitive word salad.

Unchanged by this correction: the accuracy advantage, the shipped lam 1.0 configuration, and explanation exactness. What it establishes
is that **generation quality is the real remaining gap for both models**, and that distinct-token counts are the wrong instrument -
coverage, repetition and KL are the right ones, and both models score badly on all three.
`docs/ARGMAX-IS-NOT-BREADTH.md`.

## Generation measured properly: the readable model loops, but its words are MORE realistic than the control's (2026-11-15)

Third correction of the day, and the first in the readable model's favour. My "junk breadth" reading an hour ago used 12 prompts; this
grid uses 24 and reverses the emphasis. `lens/runs/DECODING-GRID.txt`.

```
checkpoint            decoder              repetition   self_repeat   realistic coverage   KL    stop_rate
control               greedy                  0.040        0.091            0.123        22.05    0.000
control               all controls            0.000        0.003            0.146        21.11    0.083
readable lam 1.0      greedy                  0.555        0.518            0.351        18.11    0.292
readable lam 1.0      all controls            0.000        0.159            0.229        19.99    0.250
```

1. **The readable model's generated words are MORE realistic than the control's on every decoder** - coverage 0.351 vs 0.123 at greedy,
   0.229 vs 0.146 even at the most constrained decoder - and its unigram distribution is much closer to real text (**KL 18.11 vs
   22.05**). So "junk breadth" was wrong in emphasis: the tail exists, but the body of what it generates is more text-like.
2. **Its defect is LOOPING, and that is a decoder problem**: 0.555 repetition under greedy, **0.000** with controls, at a cost of
   coverage 0.351 -> 0.229. The control barely loops (0.040).
3. **Both are bad in absolute terms** - coverage 0.12-0.35 against the 0.5+ a working model should reach. Neither is usable text, and
   generation quality, not vocabulary count, is where this stands.

**Scorecard**: X1 passed (repetition to 0.000 both models), X2 failed narrowly (0.229 vs the 0.10-0.20 band), X3 held, falsifier not
triggered (0.229 below the 0.30 threshold).

**Method note**: the earlier "8-14% realistic" figure came from 12 prompts, this from 24, and coverage moved 0.141 -> 0.351 on the same
checkpoint. Generation metrics here are prompt-count sensitive, so every generation claim now states its prompt count - the same lesson
as the argmax correction, one level down: the instrument has to be specified, not just named.

**Next**: the readable model is not vocabulary-poor and its words are not junk - it loops, and it is under-trained (the trainer prints
"tokens per parameter: 1.31 (UNDER-TRAINED)"). So the next arm is more data at the same recipe.
`docs/GENERATION-MEASURED.md`.

## More data improves prediction and DEGRADES generation - the objective is misaligned (2026-11-15)

One change against the shipped arm: 600,000 -> 2,000,000 training windows (19.2M -> 64.0M tokens). The trainer's own warning moves from
"1.31 tokens per parameter (UNDER-TRAINED)" to "4.36 (adequate)".

```
arm                    tokens/param   final VAL   final ppl   repetition   coverage   KL   
shipped (600k)             1.31        21.08%      269.30       0.555       0.351    18.11
more data (2M)             4.36        23.32%      209.74       0.659       0.243    20.86
```

**3.3x more data bought the largest accuracy gain of the day (+2.24 points, perplexity 269 -> 210) and made the generated text worse on
every quality measure**: more looping, fewer realistic tokens (0.351 -> 0.243), further from the real distribution (KL 18.11 -> 20.86),
narrower argmax set (364 -> 278).

Predictions: **M3 passed** (23.32% vs the 21.50% floor), **M1, M2 and M4 failed** - I predicted generation would improve with data and it
went the other way. The falsifier did not trigger: "more data buys nothing" is wrong. More data buys a great deal; it buys the wrong
thing.

**What this establishes, and it matters more than the numbers**: the metric this project has been optimising is not the metric the
product needs. Next-token accuracy rose 2.24 points while every generation-quality measure fell - a clean demonstration at fixed seed,
fixed recipe, one changed variable. Any arm that reports only VAL accuracy is now assumed to be optimising the wrong thing until it also
reports repetition, coverage and KL on a stated prompt count.

**Next**: context length, config-only, because a model that is more accurate yet more loopy is plausibly leaning on a locally repetitive
solution inside a 32-token window. `docs/ACCURACY-VS-GENERATION.md`.

### Context 64: generation improves slightly, accuracy falls, all four predictions miss (2026-11-15)

```
arm                              final VAL   final ppl   greedy repetition   greedy coverage   greedy KL
2M windows, window 32              23.32%      209.74          0.659              0.243          20.86
2M windows, window 64              22.00%      241.26          0.532              0.290          19.42
```

- **W1, W2, W3 and W4 all FAILED** (predicted repetition <0.40 got 0.532; coverage >0.35 got 0.290; VAL >=23.00% got 22.00%; KL <19.0
  got 19.417).
- **Falsifier not triggered, but only just**: it needed repetition >=0.55 **and** coverage <=0.28; got 0.532 and 0.290, both missing by
  tiny margins. Reported as numbers rather than as either conclusion.

**Directionally, longer context helps generation and costs accuracy** - the same trade the supervision dial made, in the opposite
direction, which confirms the accuracy/generation dissociation is not a one-arm artefact.

**A defect found and fixed**: the generation tool hardcoded a 32-token window, so both measurement calls crashed on the window-64
checkpoint - and the runner still printed COMPLETE. That is the false-completion defect this project has hit twice before. The tool now
derives context length from the checkpoint, the runner checks every measurement's exit code and prints MEASUREMENT FAILED instead, and
the numbers above were produced after both fixes.

**Across three arms**: 21.08% accuracy / 0.555 repetition / 0.351 coverage; 23.32% / 0.659 / 0.243; 22.00% / 0.532 / 0.290. **No arm is
good on both axes, and the best accuracy arm is the worst on generation.** The consistent reading is that this recipe buys accuracy with
confidence, and confidence is what makes the text loop - so the next lever attacks confidence directly, with a training-time signal
against repeated tokens. `docs/ACCURACY-VS-GENERATION.md`.

## Repetition is invariant to every knob - the diagnosis is exposure bias (2026-11-15)

The more-data recipe at 12 and 9 epochs:

```
arm              final VAL   final ppl   greedy repetition   greedy coverage
2M, 18 epochs      23.32%      209.74          0.659              0.243
2M, 12 epochs      23.47%      211.49          0.614              0.247
2M,  9 epochs      22.92%      222.87          0.653              0.349
```

Predictions: **E3 passed** (23.47%, the best accuracy of any arm today), **E1, E2 and E4 failed**, falsifier not triggered (0.614 vs the
0.65 bar).

**The finding that matters: repetition does not respond to any knob we have.**

```
lever                          effect on looping
concept supervision 0.3 -> 2.0       none at any setting
training data 600k -> 2M             worse (0.555 -> 0.659)
context length 32 -> 64              0.659 -> 0.532, cost 1.32 accuracy points
epochs 18 -> 12 -> 9                 0.659 -> 0.614 -> 0.653, flat
concept widening 36 -> 53            none
decorrelation term                   none
```

**Six independent changes, none of which moves it.** That is strong evidence the cause is not any of them - and the one thing all six
share is that they preserve **teacher forcing**. The model trains on true text and generates from its own output, drifting into contexts
it never saw and falling back on local repetition. That is the exposure-bias signature, and it is testable with no training at all: feed
the model its own generated prefix and measure the collapse against a true prefix of the same length.

**Also useful**: 12 epochs at 2M is the best accuracy point found today (**23.47%**, +2.39 over the shipped arm), and 9 epochs gives
22.92% with coverage 0.349 - matching the 600k arm's 0.351 while being 1.84 points more accurate. So most of the accuracy gain is
available without giving up coverage; only the looping survives everything.
`docs/EXPOSURE-BIAS.md`.

### Exposure bias CONFIRMED - and it hurts the control MORE than the readable model (2026-11-15)

Two branches over 24 prompts x 24 steps, scored against the same true target: `teacher` conditions on true text, `self` on the model's
own output. No training.

```
checkpoint                  gap: self NLL vs teacher NLL, by step bucket
2M/12ep (best accuracy)     1-2 +3.6%   3-4 +13.4%   5-8 +26.1%   9-16 +19.8%   17-24 +27.1%
2M/18ep                     1-2 +5.1%   3-4 +11.1%   5-8 +25.8%   9-16 +16.2%   17-24 +21.3%
direct-path control         1-2 +14.1%  3-4 +38.2%   5-8 +41.3%   9-16 +42.5%   17-24 +57.6%
```

**X1 and X2 PASSED** (gap above 15% at steps 9-16; gap grows with horizon on every checkpoint). **X3 failed on two of three** - self
accuracy does collapse to 0-2% but by 3-4 points rather than the 5 predicted. **Falsifier not triggered.**

**The result I did not expect: exposure bias is WORSE in the control (+38% to +58%) than in either readable model (+11% to +27%).** So
it is a property of the training protocol, not of the concept bottleneck - and the readable architecture is the more robust of the two
when fed its own text. First comparison today where the readable model wins on something other than accuracy and explanation.

**Two defects in my instrument**: (1) FIXED - the column I called "perplexity" was mean negative log-likelihood in NATS, not
perplexity; the tool now says NLL. (2) NOT YET FIXED - the model trains `--assistant-only` but this test scores every position including
user turns, which inflates all absolute NLL values (validation perplexity is 211 while the test shows 23.4 nats). The **between-branch
comparison is unaffected** since both branches score identical positions, so X1/X2/X3 stand, but no absolute NLL here may be quoted as
the model's perplexity.

**Seven levers fail to move repetition; the eighth - the training protocol - is now confirmed as a real and large effect.** Both models
collapse on their own output, so the fix must change what the model sees during training: scheduled sampling, or an unlikelihood term
against repeated tokens. `docs/EXPOSURE-BIAS.md`.

## Withdrawal: my per-token NLL is not a sound measurement (falsifier fired, 2026-11-15)

The reconciliation was pre-registered with a falsifier: if the two instruments still disagree by more than 2x with no cause found, the
per-token NLL is withdrawn. They do.

```
checkpoint                        instrument          accuracy   mean NLL
2M/12ep (assistant-only)          my exposure tool     6.25%     20.38-22.07 nats
2M/12ep (trainer's own VAL)       train.py            23.47%     5.35 nats (= ppl 211.49)
```

**R1 passed** (assistant masking moved NLL only -5.7%, so masking was not the explanation). **R2 failed** - same checkpoint, my tool reads
21 nats where the trainer reads 5.35, and 6.25% accuracy where the trainer reads 23.47%. **R3 not satisfied** - I have candidate causes
but no identified one. **Falsifier fired at 4x disagreement.**

**So the per-token NLL from both exposure-bias tools is WITHDRAWN, and the quantitative claim that the control suffers exposure bias more
severely (+38-58% vs +11-27%) is withdrawn with it**, because that comparison was NLL-based.

**What survives, re-derived from the same-instrument comparison that does not depend on the withdrawn number:**

```
checkpoint   teacher accuracy (5 buckets)          self accuracy (5 buckets)
2M/12ep      6.25 / 6.25 / 6.25 / 6.25 / 6.25%     8.33 / 0.00 / 0.00 / 1.04 / 2.08%
control      8.33 / 2.08 / 5.21 / 7.81 / 5.73%     2.08 / 0.00 / 1.04 / 2.08 / 0.00%
```

**Self-conditioned accuracy collapses to 0-2% in both models while teacher-forced accuracy stays flat.** Exposure bias is confirmed on a
comparison that does not use the withdrawn measure. What is NOT established is which architecture suffers more - that claim is gone.

**Next action is not another model arm.** Seven levers fail to move repetition and the training-protocol explanation is supported but
unmeasured, so the next task is to reconcile the two instruments, because every future generation-quality claim depends on which one is
right. This is the fourth correction I have made against my own work today, and the second where the fix was to withdraw a number rather
than add one. `docs/NLL-WITHDRAWN.md`.

## The instrument was broken: my hand-built forward differs from the real model by 131 logits (2026-11-15)

Predictions were pre-registered with an inverted falsifier ("if the two paths agree, the withdrawal is withdrawn"). That is not what
happened.

```
path                       accuracy   mean NLL
hand-built numpy path       12.50%      35.433
REAL model forward          29.17%       5.033
max |logit difference| : 131.8675
```

**My hand-built path - `emb[window] @ concept.W + concept.b -> answer.W` - is not the model.** It differs by up to 131 logits, halves
accuracy and inflates NLL sevenfold. The real model reads **29.17% / 5.03 nats**, matching the trainer's **23.47% / 5.35 nats**.

**So every number produced by that hand-built path today is withdrawn**: all of `chat_generate_compare.py` (repetition, coverage, KL,
every generated sample), both exposure-bias tools, `measure_activation_spread.py`, the 135/227/364/414/422 breadth curve and its ceiling,
and the "944 vs 364 tokens" correction. **The conclusions they supported - looping, junk tokens, a breadth ceiling - are not supported**,
because they were measured through a forward pass that is not the model.

**Why it survived for hours**: the broken path uses the right parameter names and shapes and produces a valid distribution over the right
vocabulary, so nothing crashed and every metric looked plausible. The only tell was a 4x disagreement with the trainer, and it surfaced
only because a pre-registered falsifier forced a reconciliation. The library had already met this exact defect class - `lens.load()`'s
own docstring records a loader bug that produced "max |logit difference| 24.96" - and the project's written lesson was to load through
the library's loader. I built my own path anyway.

**What stands**: everything the trainer measured - validation accuracy and perplexity, the three-seed widening win (+0.57 +/- 0.04),
the shipped lam 1.0 configuration, concept-name honesty, explanation exactness. **What is withdrawn**: the entire generation-quality and
argmax-breadth line, pending re-measurement through `lens.load()` + `net.forward()`. Until that runs, the honest statement about this
model's generation quality is "unmeasured". `docs/INSTRUMENT-WAS-BROKEN.md`.

## Generation re-measured through the real forward: more data helps everything (2026-11-15)

All numbers here come from `lens.load()` + `net.forward()` after the hand-built path was shown to differ by 131 logits. 24 prompts x 24
tokens, greedy.

```
arm                                   repetition   coverage   KL(gen||targets)   VAL accuracy
2M windows, 12 epochs (best accuracy)     0.271       0.316         17.845          23.47%
600k windows, 18 epochs (shipped)         0.475       0.260         19.748          21.08%
direct-path control                       0.030       0.240         19.049          17.73%
```

Samples are made of **real words on every arm** - e.g. "thenumberofthenumberofthenumberofthenumberis" - so the withdrawn "junk token"
narrative was entirely an artefact of the broken forward.

Predictions: **R1, R2 and R3 failed** (repetition 0.271 vs <0.20; coverage 0.316 vs >0.50; KL 17.845 vs <10), **R4 passed**, falsifier not
triggered.

**The important result - it reverses a published finding.** `docs/ACCURACY-VS-GENERATION.md` claimed 3.3x more data improved accuracy
while degrading generation. Through the real forward it is backwards:

```
                              accuracy    repetition   coverage    KL
600k windows, 18 epochs        21.08%       0.475       0.260     19.748
2M windows, 12 epochs          23.47%       0.271       0.316     17.845
change                        +2.39 pts    -0.204      +0.056     -1.903
```

**More data improved accuracy AND generation on every measure at once. There is no dissociation and no accuracy-versus-quality
trade-off** - that conclusion is withdrawn as a measurement artefact.

**What is real**: the readable model loops more than the control (0.271 vs 0.030) but its distribution is much closer to the corpus
(coverage 0.316 vs 0.240, KL 17.85 vs 19.05) and it is +5.74 points more accurate. The honest trade is **looping versus corpus fidelity**,
not accuracy versus quality. `docs/GENERATION-REAL.md`.

### Four million windows: accuracy and fidelity keep climbing, looping does not follow (2026-11-15)

```
arm                  tokens/param   final VAL   final ppl   repetition   coverage   KL
600k windows, 18 ep      1.31        21.08%      269.30       0.475       0.260    19.748
2M windows, 12 ep        4.36        23.47%      211.49       0.271       0.316    17.845
4M windows, 12 ep        8.72        24.57%      181.83       0.318       0.339    17.385
```

- **M1 PASSED**: **24.57%** validation, the best accuracy measured all day.
- **M2 FAILED**: repetition rose to **0.318** instead of falling.
- **M3 PASSED**: coverage 0.339 and KL 17.385, both better than the 2M arm.
- **Falsifier not triggered**: it needed repetition >=0.30 (yes, 0.318) **and** accuracy <=23.47% (no, 24.57%).

**Three of four axes improve monotonically with data - accuracy +3.49 points in total, coverage 0.260 -> 0.339, KL 19.748 -> 17.385 -
while looping does not (0.475 -> 0.271 -> 0.318).** So the data lever is not exhausted for prediction and fidelity, but it is **not a
looping lever**. Sample from this arm: `"thewaytodotheworldoftheworldoftheworldoftheworld"` - real words, still a loop.

Looping has now survived eight changes, so it needs the training-protocol work rather than more tokens. Before that, **24.57% is one
seed** and this project has twice seen a verdict reverse on a later seed, so the seed check runs first.
`docs/GENERATION-REAL.md`.

### Three seeds at 4M: stable, and my "looping got worse" verdict was noise (2026-11-15)

```
seed   final VAL   final ppl   repetition   coverage   KL
 0      24.57%      181.83       0.318       0.339    17.385
 1      24.53%      181.21       0.252       0.373    16.423
 2      24.64%      180.08       0.256       0.316    17.825

mean    24.58%      181.04       0.275       0.343    17.211
sd       0.055       0.88        0.037       0.029     0.71
```

**S1, S2 and S3 all passed**; the falsifier did not trigger. Validation spread is **0.055 points** across three seeds.

**The correction inside this result**: last wake I reported that repetition "rose to 0.318 instead of falling" and scored it as a failed
prediction. Across three seeds the 4M repetition is **0.275 +/- 0.037** and seed 0's 0.318 is the outlier - against the 2M arm's
single-seed 0.271 that is no difference at all. **So "more data made looping worse" was noise.** With this noise level, any single-seed
repetition difference below about 0.08 is uninterpretable, and from here every repetition comparison is multi-seed or explicitly labelled
provisional.

**The configuration this establishes**: **4M windows / 12 epochs is the best measured configuration in the project** - 24.58% +/- 0.06
validation, perplexity 181.04, coverage 0.343, KL 17.21, 50-52 of 53 concepts honest. Against the previously shipped 600k model at 21.08%
that is **+3.50 points** with coverage up from 0.260 to 0.343.

Looping at this configuration is **0.275 +/- 0.037**: real, repeatable, and untouched by data. It is the only defect left.
`docs/GENERATION-REAL.md`.

## The loop was the decoder: sampling eliminates it, and the cost is fidelity (2026-11-15)

Three seeds of the best configuration, three decoders, all through `lens.load()` + `net.forward()`:

```
decoder                          repetition   coverage   KL(gen||targets)
A greedy, no controls               0.275       0.343        17.21
B sample T=0.8                      0.0013      0.217        19.51
C sample + top-k 50 + rep 1.3       0.000       0.222        19.07
```

**D1 passed** (repetition 0.000-0.002, far below the 0.10 bar). **D2 failed** (coverage 0.217-0.222 vs the 0.30 bar). **D3 failed narrowly**
(KL 19.07 vs 19.0). **Falsifier not triggered.**

**The finding that closes the looping question: looping is a property of GREEDY decoding, not of the model.** The same weights that repeat
themselves 27.5% of the time under argmax produce **0.13%** repetition under plain sampling. Every attempt today to fix looping by
changing the model - data volume, context length, epochs, supervision, concept count, decorrelation, interface width - was aimed at a
defect living in the decoder. The training-protocol arm is therefore **not required** for repetition, and its own falsifier says so.

**The trade that remains**: greedy is faithful but loops; sampling escapes the loop by wandering into the long tail, which is why coverage
falls 0.343 -> 0.217 and KL rises. The next arm is config-only - find the temperature/truncation knee that keeps looping near zero
without giving up that 0.12 of coverage.

**Correction to my own scorecard**: the "looping survived eight changes" line was measured under greedy decoding through a broken forward.
It was wrong twice over. What is true: **under greedy the best configuration loops 0.275 +/- 0.037; under sampling it does not loop at
all.** `docs/DECODING-CLOSES-LOOPING.md`.

## Temperature sweep: fidelity is best at T=0.5-0.6, and my runner dropped the metric that mattered (2026-11-15)

**A defect in my own tooling first**: the launched command filtered its own output with `grep -vE '...|^  repetition'`, which deleted
**the exact metric the sweep existed to measure**. So the result file has coverage and KL and no repetition; the filter is fixed and the
sweep is re-running. Nothing about repetition is reported here because I do not have it.

What it did capture, three seeds, real forward:

```
temperature   seed 1 coverage/KL   seed 2 coverage/KL   seed 0 coverage/KL
T=0.5            0.312 / 17.511       0.351 / 16.498         (filtered)
T=0.6            0.337 / 16.858       0.316 / 17.366         (filtered)
T=0.7            0.281 / 17.932       0.302 / 17.525       0.292 / 17.719
T=0.9            0.247 / 18.701       0.240 / 18.930       0.259 / 18.452
```

**K3 passed**: coverage falls and KL rises monotonically with temperature on every seed.

**The useful result**: `T=0.5-0.6` keeps corpus coverage at **0.316-0.351**, matching greedy's 0.343 - so a mild temperature does **not**
pay the fidelity bill that `T=0.8` did (0.217). **If repetition is also low there - exactly the number my runner discarded - then the
knee is at T=0.5-0.6, and the fidelity-versus-diversity trade is far cheaper than the numbers I published an hour ago implied.** K1 and
K2 remain unscored until the re-run lands.

**Method note**: this is the third time today that a defect in my tooling, not the model, produced a wrong or missing result - after the
hand-built forward (131 logits wrong) and the single-seed repetition comparison (noise read as effect). The pattern is consistent: the
measurement path deserves the same verification as the model. Concrete rule going forward: a runner must print the metric list it
intends to capture and assert each one is present, rather than filtering by pattern and trusting that what remains is complete.
`docs/TEMPERATURE-KNEE.md`.

### The decoding knee is T=0.7: ten times less looping at 85% of greedy's fidelity (2026-11-15)

Corrected numbers (the earlier run's repetition column was destroyed by my own grep), three seeds, real forward:

```
temp   mean repetition   mean coverage   mean KL
0.5        0.050            0.340          16.79        greedy: rep 0.275, cov 0.343, KL 17.21
0.6        0.065            0.311          17.49        T=0.8 : rep 0.001, cov 0.217, KL 19.51
0.7        0.027            0.292          17.73
0.9        0.002            0.249          18.69
```

- **K1 PASSED**: `T=0.7` gives repetition **0.027** with coverage **0.292**.
- **K2 passed under my +/-0.02 tolerance with a caveat**: 0.050 -> 0.065 -> 0.027 -> 0.002 is not strictly monotone; T=0.6 is slightly
  worse than T=0.5, so "monotone" is the wrong word. Repetition falls overall with one small inversion.

**The result: `T=0.7` is the knee.** Against greedy it cuts repetition **0.275 -> 0.027** (ten times less looping) while keeping coverage at
**0.292** - 85% of greedy's 0.343 and far above T=0.8's 0.217. **The fidelity-versus-diversity trade I described yesterday was the wrong
framing**: at the right temperature there is no meaningful trade, just slightly lower coverage for a tenfold reduction in looping.

**Shipped configuration**: 4M windows, 12 epochs, 53 concepts, lam 1.0, **sampling at T=0.7** - with 24.58% +/- 0.06 validation.
`docs/TEMPERATURE-KNEE.md`.

## Shipped-model verification: the explanation is exact, the names are keyword detectors (2026-11-16)

All three shipped checkpoints (4M windows, 12 epochs, 53 concepts), through `lens.load()` and the library's own `explain()`:

```
seed   max |residual|   concept share    trained-label AUROC   held-out synonym AUROC   concepts passing
 0       2.027e-06       1.000000              0.9990                 0.6398                 52/53
 1       2.354e-06       1.000000              0.9992                 0.6184                 50/53
 2       2.921e-06       1.000000              0.9994                 0.5930                 52/53
```

**V1, V2, V3 and V4 all PASSED; falsifier not triggered.** Worst residual across 300 windows is **2.9e-06**, three orders of magnitude
inside the bar, and **concept share is 1.000000 on every window** - the named concepts account for the prediction to within rounding. **So
`explain()` is not a story: the arithmetic it prints is the model's arithmetic, to six figures.**

**But the hardest number in the project is now measured cleanly.** Trained-label fidelity is **0.999**, transfer to unseen synonyms is
**0.59-0.64** - a gap of **0.36 to 0.41**, the largest yet, and it grew as the model got better data. **The concepts have become
near-perfect detectors of the specific words they were trained on and only modest detectors of the idea those words stand for.**

**Two claims that must never be conflated again**: (1) *the explanation is exact* - verified, may be claimed without qualification;
(2) *the names mean what they say* - about 60% true, not 99.9%. Anyone reading `explain()` output has to be told that, or the
transparency promise becomes a different opacity.

Shipped model, documented end to end: 24.58% +/- 0.06 validation, perplexity 181.04, greedy coverage 0.343, T=0.7 knee at 0.292
coverage with 0.027 repetition, explanation exact to 2.9e-06, naming 0.59-0.64 honest.
`docs/SHIPPED-VERIFY.md`.

## What the concept neurons actually promote: 51 of 53 are partly code detectors (2026-11-16)

A concept's effect on the logits is its own row of `answer.W`, so this needs no ablation and no reconstruction - the matrix is the model's
own numbers. For each of the 53 concepts, the 12 tokens it promotes most, classified with a fixed stated heuristic:

```
is_a_request     promotes: 'sqrt' 'be' 'solve' 'frac' 'pi' 'get' '*' '^'      own token: help
is_a_question    promotes: 'python' 'so' '00' 'import' 'in' '?'               own token: ?
is_an_agreement  promotes: ']' 'right' ',' '^' '$' 'left' '\\' '!'           own token: right
mentions_self    promotes: 'i' 'j' 'my' '#' 'me' '[' 'print' 'arr'           own: i, me, my
mentions_a_person promotes: 'she' 'he' 'her' 'frac' 'his' 'people'            own: he, her, him, people, she
```

```
concepts promoting at least one of their OWN trained tokens : 23 of 53
concepts with at least one code/math token in the top 12    : 51 of 53
median code/math share of a concept's top 12                : 0.417
```

**C1 failed** (23 vs the 25 predicted), **C2 passed** (51 vs 20 predicted), **C3 passed** (0.417, inside the 0.08-0.50 band), falsifier not
triggered.

**This explains the whole honesty picture.** The model's vocabulary mixes conversation, Python and LaTeX; a concept layer that is one
linear map over a shared window embedding cannot separate "this is a request" from "this is maths", so each neuron carries a code-or-math
co-factor beside its name. That fits every number already published: **trained-label fidelity 0.999** (the listed words really are
present), **held-out synonym transfer 0.59-0.64** (the mass is on code tokens, not semantic neighbours), and **`explain()` exact to
2.9e-06** (the arithmetic is right; what it is arithmetic about is partly something else).

**So `explain()` can truthfully report "I answered because `is_a_request` fired 0.8" while `is_a_request` is in large part pushing
`sqrt`, `frac` and `pi`.** That is exactly the failure this project exists to catch, and it is now measured and public rather than hidden
behind a passing audit.

**More data cannot fix this** - it is a property of how the concept layer is supervised against a mixed corpus. The three fixes with
evidence behind them: supervise concepts on paraphrases rather than listed tokens; penalise the code/math co-factor contrastively; or
supervise on a code-free corpus slice and test on the full one. `docs/CONCEPT-CONTENT.md`.

## Data-driven check: the code co-factor is 0.204, not 0.417 - and own-word mass is 0.2% (2026-11-16)

The earlier co-factor number used a **hand-written** keyword list; this run derives the nuisance set from the corpus (top 2,000 tokens by
code-versus-prose frequency ratio across 184,041 code-quarter and 186,834 prose-quarter conversations).

```
hand list tokens present in vocab               : 51
those inside the data-driven nuisance set       : 47  (92%)
median share of a concept's promoted mass on
  the DATA-DRIVEN nuisance set                  : 0.204
median share on the concept's OWN trained words : 0.002
```

**C4 passed** (92% agreement - my hand list was not idiosyncratic). **C5 failed** (predicted >0.30, got **0.204**). Falsifier not
triggered.

**Correction**: the **0.417** figure was the hand list applied to each concept's *top-12 promoted tokens*; the whole-row figure - the
honest one for "how much of this neuron's push is code" - is **0.204**. Both are now stated with their definitions.

**The sharper fact**: **a concept's promoted mass on its own trained words is 0.002.** Across all 53 concepts, the words a concept was
trained to detect are a rounding error of the force it exerts. About 20% of its mass is code-flavoured; the other ~80% is spread over the
whole vocabulary.

**So the honest description of the shipped model is**: the answer is provably a weighted sum of 53 named directions, and each named
direction is **a diffuse vocabulary-wide push rather than a concentrated detector of its name**. The name is a **handle, not a
description of the direction's content**. "A concept detects questions" is over-claiming; "the answer is exactly attributable to 53 named
handles whose content is diffuse" is what the measurements support. `docs/NUISANCE-DATA-DRIVEN.md`.

## LOOK INSIDE: the concept name describes what the neuron DETECTS, not what it PUSHES - and the two are decoupled (2026-11-16)

Read-only measurement, predictions written first (`lens/runs/INOUT-PREDICTIONS.txt`). For each concept: input quality = trained-label
AUROC; output content = share of its absolute `answer.W` row landing on its own trained word ids.

```
concepts                                             : 53
median output own-word mass                          : 0.0008
concepts at or above 0.05 output own-word mass        : 0 of 53
rank correlation (input quality, output content)      : -0.307   over 22 audited concepts
input range 0.914-1.000   |   output range 0.0002-0.0024

mentions_law      input 0.9999   output 0.0010
mentions_weather  input 0.9998   output 0.0008
is_a_question     input 0.9997   output 0.0002
```

**M1, M2 and M3 all passed; falsifier not triggered.**

**The mechanism, plainly: a concept neuron is a near-perfect detector on the way in and a diffuse, almost contentless knob on the way out,
and nothing in the training ever required the two to agree.** `lam_concept` supervises the concept *activations* (the input side), which
is why AUROC reaches 0.9997-0.9999. The *answer matrix* is never supervised against the concept's own words - its rows are learned only by
next-token prediction - so each row becomes a broad, mildly code-flavoured direction holding 0.08% of its own words at the median.
`explain()` then attributes the answer to those rows, so it is **exact about the arithmetic and empty about the content**: it correctly
reports which handle was turned, on a knob whose label was never attached to what the knob does.

**This single mechanism explains the 0.999/0.59 honesty gap, the 0.002 own-word mass, the 0.204 code co-factor and the 2.9e-06 exactness
together.**

**The fix is not more data.** Supervise the **output side**: require concept `c`'s row of the answer matrix to prefer its own words and
demote the nuisance set - `-log softmax(answer.W[c] . onehot(own words))` averaged over concepts. Built verify-first like the
decorrelation term: analytic gradient, finite-difference check, `OPS` registration, smoke test, and proof it enters the loss only at
non-zero weight. `docs/LOOK-INSIDE-INOUT.md`.

## The meaning penalty is retired by a 60-second measurement (2026-11-16)

Before spending ~70 GPU-minutes on the fix arm, its central question was asked directly: **if a concept's answer row is forced to prefer
its own words, how much prediction does it lose?** Only the answer matrix is optimised - trained rows, the model's own activations, 400
steps - measured on **20,000 held-out windows the optimisation never saw**.

```
baseline: own-word mass 0.0014 | held-out next-token accuracy 0.1565
weight   own-word mass   held-out acc   cost (points)
2.0         0.0014          0.1559          0.06
8.0         0.0016          0.1537          0.28
30.0        0.0041          0.1266          2.99
```

**P2 passed; P1 and P3 failed; the falsifier fired in the exact form written.** A weight of 30 buys own-word mass **0.0041 for 2.99
points** - about 700 points of accuracy per unit of meaning, so the target 0.05 can never be afforded. **The penalty approach is
retired.**

**Why**: the row is asked to do two opposing jobs. Its training signal says "produce the next-token distribution", which needs mass spread
over thousands of continuations; this term says "concentrate it on a handful of words". The second is nearly orthogonal to the first.

**What replaces it**: the falsifier named the alternative before the numbers arrived - **make meaning structural instead of penalised**, by
**deriving each concept's answer row from the embeddings of its own words**. Nothing to pay for, nothing to drift. That is an architecture
change and it must pass the same verify-first gate. The queued penalty arm is not launched; the term, its verified gradient (3.7e-10) and
this curve stay in the tree as the record of why the cheap-looking fix was wrong.
`docs/ROW-COST-NEGATIVE.md`.

## Meaning is free if it is structural, and prediction is dead if it is not: three measurements (2026-11-16)

All on the shipped 4M/12ep checkpoint, the model's own activations, 20,000 held-out windows. Predictions written before each run.

```
1. PENALTY (retired)      weight 30: own-word mass 0.0014 -> 0.0041 for 2.99 points   (~700 points per unit of meaning)
2. STRUCTURAL ROWS        row = mean(E[own words]) @ E^T, norm-matched:
                          median own-word mass 0.9200, held-out accuracy 0.0073  (baseline 0.1565, -14.92 points)
3. POST-HOC BLEND         row = a*structural + (1-a)*trained:
     a      own mass   accuracy   cost
    0.0      0.0014     0.1565     0.00
    0.1      0.0030     0.1500     0.65
    0.2      0.0079     0.1326     2.39
    0.3      0.0189     0.1009     5.56
    0.5      0.1082     0.0309    12.55
    1.0      0.9200     0.0073    14.92
```

**S1 and S2 passed, and pure tying is dead** (0.0073 is below the 0.03 the falsifier set). **B1 failed, B2 passed, falsifier not
triggered** at a=0.1, which is nearly free but buys almost no meaning.

**What the three say together**: paying for meaning costs ~700 points per unit; getting it free from the embedding geometry costs the whole
output path (0.92 mass, 0.7% accuracy); blending after training does not rescue it, because the useful-meaning region (mass above 0.10)
begins where accuracy has already been destroyed. **The direction "built from this concept's own words" is almost orthogonal to the
direction "predicts the next token" in the geometry this model happened to learn** - and nothing ever pressured the two to agree.

**So meaning cannot be added after training by any means.** The one remaining option is to **train the constraint in from the start**: the
answer row is a structural term from the concept's own word embeddings plus a learned low-rank correction, letting the embeddings
reorganise so the structural direction becomes predictive. That is the next arm, and these three curves are why.
`docs/MEANING-BRACKET.md`.

## 8M windows: +1.07 accuracy, better perplexity, and the meaning gap does not narrow (2026-11-16)

One change from the shipped 4M arm: twice the training windows. Numbers from `lens/runs/DATA8M.txt`.

```
                              4M (shipped)      8M        change
final VAL accuracy               24.58%        25.65%      +1.07
final VAL perplexity             181.04        157.89      better
greedy repetition                 0.275         0.208       better
greedy coverage                   0.343         0.356       +0.013
T=0.7 repetition                  0.027         0.053       within noise
T=0.7 coverage                    0.292         0.271       -0.021
trained-label AUROC               0.9990        0.9996      saturated
held-out synonym AUROC            0.6398        0.6336      unchanged
explain() max |residual|         2.9e-06       2.4e-06      exact
concept share                    1.000000      1.000000     exact
```

**D1 passed** (25.65% against a 24.58% bar). **D2 failed** (T=0.7 coverage 0.271, not above 0.292). **D3 failed narrowly** (0.053 against a
0.05 bar) - and under this project's own noise rule a 0.026 difference is uninterpretable, so that is not evidence data made looping
worse, only that it did not make it better. **D4 passed, and it was the prediction I expected to fail: the meaning gap did not narrow** -
held-out synonym transfer **0.6336** against trained-label **0.9996**, a gap of **0.366**, essentially unchanged from 4M's 0.359.
Falsifier not triggered.

**Twice the data buys about one point of accuracy and better perplexity, and buys nothing for meaning.** Held-out concept honesty moved
0.6398 -> 0.6336 across a doubling of training text - noise, not progress. That is consistent with the measured mechanism: answer rows are
diffuse directions learned from next-token prediction, and more text sharpens the same machinery rather than changing what it does.

**Best model in the project is now 8M/12ep: 25.65% validation (single seed), perplexity 157.89, explanation exact to 2.4e-06 with concept
share 1.000000, concept naming about 0.63 honest on unseen wording.** `docs/DATA8M.md`.

## Structural answer rows: accuracy survives at 24.33%, meaning does not arrive (own-word mass 0.0008) (2026-11-16)

One change from the shipped arm: `row_c = (mean E over concept c's own words) @ E^T + A_c B`, correction rank 32, zero-initialised, so
`answer.W` is not a parameter. Numbers from `lens/runs/STRUCTURAL-ANSWER.txt`.

```
final VAL accuracy        24.33%    (shipped 4M: 24.58%, so -0.25)
final VAL perplexity     167.38     (shipped 181.04, better)
concepts passing          53/53
explain() max |residual| 1.927e-06  | concept share 1.000000
trained-label AUROC       0.9991    | held-out synonym 0.6423
own-word mass             0.0008    | 0 of 53 concepts at or above 0.05
```

**R1 failed** - the one that mattered. Own-word mass is **0.0008**, identical to the free answer matrix, against a 0.10 bar.
**R2 passed** (24.33% against a 20.58% bar: the structural architecture costs **0.25 points**, not four). **R3 passed** (explanation exact
to 1.9e-06, concept share 1.000000). **R4 passed** (input honesty unchanged at 0.6423). Falsifier not triggered.

**Settled: the architecture is not the obstacle.** Replacing the free answer matrix with rows built from the concepts' own word embeddings
costs a quarter of a point. If meaning were going to be cheap anywhere, it would be here.

**Measured twice now: the correction eats it.** Own-vocabulary mass stays at 0.0008 because the learned correction grows to dominate -
median norm **87.42** against the structural term's **1.06** at epoch 6, carrying **0.9999** of every row's energy, up from 46.89 at
epoch 1. A zero-initialised low-rank correction is an **escape hatch**: nothing stops it growing until it supplies the entire answer.

**One question remains**: can the correction be restrained without destroying accuracy? Two soft-price attempts diverged to NaN (weights
30 and 1.0, both far too large; the measured stable weight is ~2e-4) and a third is running. If no stable price works, the next step is a
hard per-row normalisation; if that fails too, the honest conclusion is that at 16M parameters a free low-rank correction is load-bearing
and meaning cannot be forced into the rows. `docs/STRUCTURAL-ARM.md`.

## Pure structural rows: falsifier fired at 13.74%, and the meaning is missing even with nothing to blame (2026-11-16)

One change from the structural arm: no learned correction at all, so the answer row is exactly its own vocabulary's geometry.

```
final VAL accuracy        13.74%     (predicted 20.0-23.5%, falsifier 18.0%)  -> FALSIFIER FIRED
final VAL perplexity     392.84
concepts passing          46/53
median own-word mass      0.0019     (S1 predicted above 0.10)               -> FAILED
explain() max |residual| 2.089e-06   | concept share 1.000000                -> PASSED
```

**Removing the correction costs 10.6 points** (24.33% -> 13.74%), so it is load-bearing. **And removing it does not buy meaning**: own-word
mass is 0.0019 against a 0.10 bar.

**The uncomfortable part.** This arm has the most coherent own-word embedding geometry of any model here:

```
checkpoint                       alignment   coherence
shipped 4M (free answer.W)         0.3822      0.0034
structural, correction carried     0.3826      0.0064
pure structural, no correction     0.4893      0.0969
```

25% better alignment and **28x more pairwise coherence** than the shipped model - because its structural rows were the only answer path - and
its trained rows still put only **0.19%** of their mass on the concepts' own words.

**So the correction was never the villain.** With no correction to occupy the row, next-token training still produces rows that do not favour
their own vocabulary: the objective rewards evidence spread across thousands of plausible continuations, and a concept's handful of words
is not where that evidence goes.

**Ruled out as routes to concept meaning**: embedding geometry (present and better, worth nothing), row magnitude (destroys accuracy), row
support restriction (meaning 0.63, accuracy 0.0022), removing the free correction (costs 10.6 points, buys 0.0011 mass). **What remains** is
supervising the row directly against its own words at a measured weight - the term this morning was retired at 2000% of the CE gradient,
and the fair test is queued at 13%.

If that fails, the honest conclusion is that this project's guarantee is **exact attribution** (delivered to 2.4e-06 with concept share
1.000000) and **not concept content** - and the public documentation will say exactly that. `docs/PURE-STRUCTURAL.md`.

## Priced correction: the price works, it is not enough, and its falsifier fired at share 0.9968 (2026-11-16)

Third attempt at restraining the learned correction; the first two diverged to NaN at weights 30 and 1.0, and this weight was **measured**
(about 11% of the cross-entropy gradient) rather than guessed. Numbers from `lens/runs/CORR-NORM.txt`.

```
final VAL accuracy        23.66%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     172.06
concepts passing          53/53
median correction share   0.9968     (N2 predicted 0.50-0.99)     -> FAILED, narrowly
median ||A_c B||         27.1553     (unpriced at its end: 87.42) -> the price cut growth 3.2x
median own-word mass      0.0013     (N3 predicted above 0.002)   -> FAILED
explain() max |residual| 1.697e-06   | concept share 1.000000      -> PASSED
no NaN in any epoch                                             -> PASSED
```

**It bought real restraint**: the correction's median norm ended at **27.16** instead of **87.42**, a **3.2x** reduction in growth, for
**0.67 accuracy points**.

**It did not buy meaning, and the falsifier fired in its written form**: the correction still carries **0.9968** of every row's energy,
above the 0.99 bar set in advance. Own-word mass is **0.0013** - barely above the free matrix's 0.0008 and below the 0.002 N3 required.

```
arm                                  corr norm   corr share   own-word mass   VAL
unpriced                             87.42       0.9999       0.0008          24.33%
priced at 11% of the CE gradient     27.16       0.9968       0.0013          23.66%
```

Each intervention moves its own quantity and none of them moves meaning. **Pricing cuts the correction 3.2x and the meaning share by
0.003** - the third independent confirmation that the share cannot be bought down by acting on the correction, because what is missing is
the structural term's own mass.

**What the falsifier demands next**: a HARD per-row normalisation instead of a soft price - renormalise the correction after each optimiser
step so `||A_c B||` cannot exceed a fixed multiple of `||structural_c||`. A projection has no weight to fail at. `docs/CORR-NORM.md`.

## Scale 10 on the structural term: 23.58% accuracy, correction still carries 0.9877 (2026-11-16)

Re-aimed from 70 to 10 after a post-hoc pre-test showed 70 destroys accuracy. Numbers from `lens/runs/SCALE.txt`.

```
final VAL accuracy        23.58%     (shipped 4M: 24.58%, so -1.00)
final VAL perplexity     160.87      (shipped 181.04, better)
concepts passing          52/53
median correction share   0.9877     (S1 predicted below 0.10)   -> FAILED
median ||structural||     2.1871     (unscaled arm: 1.5466)
median own-word mass      0.0012     (S2 predicted above 0.10)   -> FAILED
explain() max |residual| 1.260e-06   | concept share 1.000000     -> PASSED
```

**A fixed multiplier of 10** did what it was asked geometrically - structural norm 1.5466 -> 2.1871 - and moved the correction's share by
**0.012** (0.9999 -> 0.9877) for **0.75 accuracy points**. Own-word mass went 0.0013 -> 0.0012: nowhere.

**The fourth and last row-level lever is measured and also fails.** A stronger structural term does not become the carrier of the row, because
the correction grows to match whatever the structural term does. The share is a ratio and both sides are under the optimiser's control.

```
arm                                   corr share   own-word mass   VAL
unpriced structural                     0.9999        0.0008       24.33%
priced at 11% of the CE gradient        0.9968        0.0013       23.66%
priced + structural scale 10            0.9877        0.0012       23.58%
pure structural, no correction          0.0000        0.0019       13.74%
content term at 13% of the CE gradient             0.0006 at ep3
```

Four interventions, each moving its own quantity, none moving meaning above 0.002. **The pattern is the finding**: in this architecture a
concept's answer row is whatever predicts best, and any pressure aimed at the row is absorbed by the other side of the ratio.
`docs/SCALE10.md`.

## The limit, measured: five mechanisms fail to give concept rows their meaning; exact attribution is what this project delivers (2026-11-16)

The row-content term ran twelve epochs at a measured weight (~13% of the cross-entropy gradient, after being retired that morning at 2000%
of it and given a fair re-test). Numbers from `lens/runs/CONTENT-ARM.txt`:

```
final VAL accuracy        23.73%     (shipped 4M: 24.58%, so -0.85)
median own-word mass      0.0008     (Z1 predicted above 0.002)   -> FAILED, falsifier fired
explain() max |residual| 2.980e-06   | concept share 1.000000     -> PASSED
```

**The term verified correct (3.749e-10), ran at a sane weight, and changed the quantity it directly optimises by zero.** It cost 0.85
accuracy points for nothing.

```
mechanism                                        own-word mass   VAL      verdict
free answer matrix (shipped 4M)                      0.0008       24.58%   baseline
structural rows + correction (unpriced)              0.0008       24.33%   no change
structural rows + priced correction                  0.0013       23.66%   +0.0005 for 0.67 pts
structural rows + structural scale 10                0.0012       23.58%   no change
pure structural rows, no correction                  0.0019       13.74%   +0.0011 for 10.6 pts
row-content supervision at 13% of the CE gradient    0.0008       23.73%   nothing
```

**Five mechanisms, six measurements, none moves own-word mass above 0.002 without destroying accuracy.**

**So the honest statement of what this project delivers**: **exact attribution** - `explain()` residual 2.4e-06 to 2.98e-06, concept share
**1.000000**, no hidden path, verified on the shipped model - and **not concept content**. A concept's answer row puts **0.08%** of its mass
on the words that name it; it is a diffuse, mildly code-flavoured direction whose input-side detection is 0.9997-0.9999 while its output
side barely contains the name (decoupled at rank correlation **-0.307**).

**The accurate sentence is**: the model's answer is provably a weighted sum of 53 named handles, and each handle is a diffuse
vocabulary-wide direction rather than a concentrated detector of its name. **Not** "the concepts mean what they say".
`docs/THE-LIMIT.md`.

## The share WAS fixed and meaning still did not appear: hard cap flips 0.9999 -> 0.2087, own-word mass stays 0.0016 (2026-11-16)

Predictions written before the run (`lens/runs/HARD-CAP-PREDICTIONS.txt`); numbers from `lens/runs/HARD-CAP.txt`. This is the decisive version of
the experiment because the intervention **worked exactly as designed on the quantity it targeted**.

```
final VAL accuracy        14.48%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     373.25
concepts passing          49/53
median correction share   0.2087     (unpriced: 0.9999)   -> H1 PASSED, the share flipped
median ||structural||     4.4949     (unpriced: 1.5466)
median ||A_c B||          2.6950     (unpriced: 87.42; capped)
rows with correction >50% : 10 of 53  (unpriced: 53 of 53)
median own-word mass      0.0016     (H2 predicted above 0.05)   -> FAILED
explain() max |residual| 1.048e-06   | concept share 1.000000     -> PASSED
```

**H1 passed for the first time in this family**: the correction's energy share fell from **0.9999 to 0.2087**, the structural term became the
dominant carrier (norm 4.49 against a capped 2.70), and rows where the correction carries more than half the energy fell **53 of 53 -> 10 of
53**. **And meaning still did not appear: own-word mass 0.0016**, three orders of magnitude below the 0.10 bar. Cost: **10 accuracy points**.

**This refines the earlier conclusion.** I had said "the share is a ratio and cannot be won by acting on the correction". Too weak: **the share
CAN be won - this arm won it decisively - and winning it does not produce meaning.**

```
arm                                   corr share   own-word mass   VAL
free answer matrix (shipped 4M)          --           0.0008       24.58%
structural + correction (unpriced)      0.9999        0.0008       24.33%
structural + priced correction          0.9968        0.0013       23.66%
structural + structural scale 10        0.9877        0.0012       23.58%
structural + HARD CAP (share flipped)   0.2087        0.0016       14.48%
pure structural, no correction          0.0000        0.0019       13.74%
row-content supervision at 13% of CE      --           0.0008       23.73%
```

Six row-level mechanisms spanning the **entire** range of correction share from 0.0000 to 0.9999, and own-word mass stays inside
**0.0008-0.0019** in all of them. The mass does not track the share, the magnitude, the scale, or the support of the row.

**Established, as narrowly as the evidence allows**: in this architecture at 16M parameters a concept's answer row can be made structurally
dominant without becoming meaningful, and every knob that changes what the row *does* leaves what it *contains* untouched. **Meaning is not a
control on this quantity.** `docs/HARD-CAP.md`.

## Second confirmation: the absolute cap flips the share to 0.1830 and meaning stays at 0.0018 (2026-11-16)

Predictions in `lens/runs/CAP-ABS-PREDICTIONS.txt`, written before the run. Numbers from `lens/runs/CAP-ABS.txt`. This is an independent
mechanism from the hard cap - a cap expressed in ABSOLUTE Frobenius units rather than as a multiple of the structural norm - and it lands on
the same answer.

```
final VAL accuracy        15.35%     (unpriced structural arm: 24.33%, shipped 4M: 24.58%)
final VAL perplexity     361.24
concepts passing          47/53
cap decisions logged      11,250
median correction share   0.1830     (A1 predicted 0.30-0.90)   -> below the band, i.e. the cap bit harder than predicted
median ||structural||     5.6159
median ||A_c B||          2.5393
rows with correction >50% : 4 of 53   (unpriced: 53 of 53)
median own-word mass      0.0018     (A3 predicted below 0.01)  -> PASSED
explain() max |residual| 1.245e-06   | concept share 1.000000    -> PASSED
```

## Two independent caps, one answer

```
cap mechanism                    corr share   rows corr>50%   own-word mass   VAL
none (unpriced structural)         0.9999        53 of 53        0.0008       24.33%
HARD cap at 0.5 x structural       0.2087        10 of 53        0.0016       14.48%
ABSOLUTE cap at 8.0 Frobenius      0.1830         4 of 53        0.0018       15.35%
```

**Both caps succeeded on their own terms** - shares of 0.2087 and 0.1830, rows dominated by the correction cut from 53 to 10 and 4 respectively -
and **both left own-word mass at 0.0016-0.0018**, three orders of magnitude below the 0.10 bar, for 9-10 accuracy points each.

**A3 was the prediction I wrote in advance**: *"own-word mass stays below 0.01 - I expect this, because five mechanisms have already failed and a
cap in different units should not change what the row contains."* It passed. The falsifier (mass above 0.05 with the cap holding) did not fire.

## What two independent confirmations establish

The correction's dominance of a concept's answer row is **not** what suppresses the row's meaning. Two different caps, expressed in different
units, both made the structural term the dominant carrier, and neither produced a row that favours its own words. Combined with the earlier
arms spanning correction shares from 0.0000 to 0.9999 and leaving mass inside 0.0008-0.0019 throughout, the conclusion holds:

**In this architecture at 16M parameters, what a concept's answer row CONTAINS is not controlled by what the row does, how large it is, what
supports it, or what dominates it.** The row is a prediction device; its name is a handle attached to it afterwards.

The one thing still standing is the supervision change: predicating a concept on its own word being the next token reached **25.7%** validation
at epoch 8 - the only arm to beat the shipped 24.58% - with its own mass creeping 0.0006, 0.0007, 0.0008. Its endpoint is the last open question
of the day.

## The supervision change buys accuracy and does not buy meaning: next-token concepts reach 25.97% VAL, the best in the project, with own-word mass back at 0.0008 (2026-11-16)

Predictions in `lens/runs/NEXT-TARGET-PREDICTIONS.txt`, written before the code changed. Numbers from `lens/runs/NEXT-TARGET.txt` and
`logs/exposure/nodirect1024_nexttarget_data4m_e12_s0.log`. This was the last mechanism standing, because it acts on what a concept is asked to
PREDICT rather than on its answer row.

```
final VAL accuracy        25.97%     (ep 11: 26.01% - best in the project)
final VAL perplexity     169.67      (shipped 4M: 181.04)
train accuracy           29.81%
own-word mass            0.0008      (N1 bar 0.10; free-matrix baseline 0.0008)
concepts at or above 0.05 0 of 53     (highest: is_a_request 0.003)
explain() max |residual| 2.384e-06   | concept share 1.000000   -> N4 PASSED
```

## Both halves of the prediction

**N2 was exceeded on the good side.** I predicted 15.0-22.0% because the label becomes a rarer, sharper event; the model reached **25.97%**,
**1.39 points above the shipped 24.58%** and better perplexity. Asking a concept whether its own word is the token being predicted is a *better*
training signal than asking whether its words appear anywhere in the window.

**N1 failed and the falsifier fired.** Median own-word mass is **0.0008** - exactly the free-matrix baseline. The mid-run creep (0.0006 at
epoch 2, 0.0007 at epoch 5, 0.0008 at epoch 8) **plateaued at the baseline and went no further** in epochs 9-12. So even supervision that
couples a concept's firing to its own word being imminent does not make its row favour that word.

## The concept-content question, closed

Every mechanism this project can bring to bear has now been tested:

```
mechanism                                  own-word mass   VAL
free answer matrix (shipped 4M)               0.0008       24.58%
structural rows + unpriced correction         0.0008       24.33%
structural rows + priced correction           0.0013       23.66%
structural rows + structural scale 10         0.0012       23.58%
structural rows + HARD CAP (share 0.2087)     0.0016       14.48%
structural rows + ABSOLUTE cap (share 0.1830) 0.0018       15.35%
pure structural, no correction                0.0019       13.74%
row-content supervision at 13% of CE          0.0008       23.73%
next-token concept supervision                0.0008       25.97%
```

Nine arms. Correction shares from 0.0000 to 0.9999. Supervision targets changed from "present in the window" to "is the next token". Row
structure changed from free to structurally derived to capped in two independent units. **Own-word mass never leaves 0.0008-0.0019.**

**The conclusion, stated as narrowly as nine arms allow**: in this architecture at 16M parameters, a concept's answer row is a prediction device.
It becomes meaningful only if training rewards it for being meaningful, and next-token prediction does not - because at any position the model
wants mass on whatever token is likely, which is not the concept's own word. Making the row structurally derived, dominant, small, large, or
supervised on a sharper label changes what the row does; none of them changes what the row contains.

**What this does not say**: that readability failed. `explain()` is exact to 2.4e-06 with concept share 1.000000 in every arm - the explanation
faithfully reports which concepts drove each decision. What is not yet true is that a concept's NAME predicts the words it promotes.

## The accuracy result stands on one seed and is not yet claimable

**25.97% is one seed.** The shipped 24.58% carries a 3-seed bar with sd 0.055, so seeds 1 and 2 are running now with the prediction that all
three land above the shipped upper edge of 24.64%, and the falsifier that any seed at or below 23.6% means the 1.39-point gain was seed luck.
No accuracy claim will be made public before those land.

## Next-token concept supervision lifts validation accuracy by +1.54 points on three seeds, and leaves concept-row content at baseline (2026-11-16)

Decision rule `lens/runs/NEXT-TARGET-DECISION-RULE.txt` was written BEFORE any seed 1 or 2 endpoint existed. Predictions
`lens/runs/NEXT-TARGET-PREDICTIONS.txt` were written before the code changed. Numbers from `lens/runs/NEXT-TARGET-3SEED.txt`,
`lens/runs/NEXT-TARGET-3SEED-MASS.txt` and `logs/exposure/nodirect1024_nexttarget_data4m_e12_s*.log`.

## The accuracy half - R1 was satisfied on every clause

```
seed   next-target VAL   shipped VAL   paired delta
 0        25.97%            24.57%        +1.40
 1        26.24%            24.53%        +1.71
 2        26.16%            24.64%        +1.52
mean     26.12% (sd 0.139)  24.58%       +1.54 (sd 0.156)
```

**All three seeds land above the shipped best seed of 24.64%**, the mean is 26.12%, and the comparison is **paired on seed** (the rule this
project set after unpaired arm means misled it once). Three of three paired deltas are positive. Perplexity improves on every seed as well
(169.67 / 169.13 / 168.10 against the shipped 181.04).

## The meaning half - unchanged, and it is published with the same weight

```
seed 0 own-word mass 0.0008
seed 1 own-word mass 0.0008
seed 2 own-word mass 0.0010
            baseline 0.0008      concepts at or above 0.05: 0 of 53 on every seed
```

## What this means, both ways

**A real gain, on a properly seeded comparison.** Labelling a concept 1 only when one of its own words is the *token being predicted* is a better
training signal than labelling it when its words merely appear in the window: **+1.54 points of validation accuracy, 3 of 3 seeds, paired, with
better perplexity on all three.**

**And it buys nothing for concept-row content.** Median own-word mass sits at 0.0008, 0.0008 and 0.0010 - the free-matrix baseline, with zero
concepts above 0.05 on any seed. So the supervision change improves prediction while the answer rows remain prediction devices that do not favour
their own words. This is the tenth mechanism measured on that question, and it is the one that improves accuracy while confirming the wall.

The two facts belong in one sentence, which is why the timing of the mass measurement was fixed before the third seed landed: a publication that
quoted +1.54 without the mass numbers next to it would have been a half-truth, and the half it dropped is the one this project exists to answer.

## What remains open

`explain()` stays exact (residual 2.384e-06 at the seed-0 endpoint, concept share 1.000000), so the system still truthfully reports which
concepts drove each decision. What is not true is the stronger claim that a concept's NAME predicts the words it promotes. Ten mechanisms,
correction shares from 0.0000 to 0.9999, two supervision targets, and own-word mass never leaves 0.0008-0.0019.

## A 15x price change moves own-word mass by nothing: the row-content penalty is inert under training, at every weight tested (2026-11-16)

Predictions `lens/runs/MASS-CURVE-PREDICTIONS.txt` and decision rule `lens/runs/MASS-CURVE-DECISION-RULE.txt` were both written before the third
weight finished. Numbers from `lens/runs/MASS-CURVE.txt`.

```
weight   VAL accuracy   VAL ppl    own-word mass   concepts >= 0.05
 2.0       23.16%       222.07        0.0007            0 of 53
 8.0       23.12%       229.00        0.0007            0 of 53
30.0       23.07%       236.79        0.0007            0 of 53
unweighted baseline                    0.0008
```

## Every pre-registered prediction failed except the one bounding it

- **C1** (every weight at or above the 0.0008 baseline): **FAILED at all three.** Mass is 0.0007 at every weight - *below* the baseline, not
  equal to it.
- **C2** (mass rises with weight): **FAILED.** Fifteen-fold increase in price, zero change in mass.
- **C3** (accuracy falls with weight): technically yes, 23.16 -> 23.12 -> 23.07, which is **0.09 points across a 15x price change** - flat, so the
  price is not biting on anything.
- **C4** (mass stays below 0.05): **PASSED**, trivially and uninformatively.
- **FALSIFIER** ("if mass does not exceed baseline at ANY weight, the anti-correlation is mechanical rather than a price effect and this line
  closes"): **FIRED IN FULL.**

## The conclusion, which is sharper than "the term is too weak"

**On a frozen model this exact penalty raises own-word mass.** One gradient step moved it **0.1207 → 0.1237**, measured through the term's gradient,
verified against finite differences at **3.7e-10**. The term works on the model it is applied to.

**But training under it for six epochs returns the model to baseline** - 0.0007 at weight 2.0, and still 0.0007 at weight 30.0.

So the objective is not *resisting* the penalty; **the training process erases its effect**. A term can measurably move the quantity on a fixed
model and still produce no such model when you train with it. That distinction is the real result here, and it also explains why the earlier
frozen-model cost curve - which priced this penalty against held-out accuracy on fixed checkpoints - pointed the wrong way. Pricing a term on a
frozen model does not tell you what the term will do when trained with.

## What this closes, and what it does not

**Closed**: the row-content penalty as a route to concept-row meaning. It is inert under training at 2.0, 8.0 and 30.0, and fifteen-fold more weight
buys nothing.

**Still open, unchanged**: `explain()` remains exact (residual 2.4e-06, concept share 1.000000), so the system truthfully reports which concepts
drove each decision. The unresolved claim remains the stronger one - that a concept's NAME predicts the words it promotes. Ten mechanisms,
correction shares from 0.0000 to 0.9999, two supervision targets, and a 15x price sweep all leave own-word mass between 0.0007 and 0.0019.

## The faster learner does not write better: next-token supervision gains +1.54 accuracy and LOSES generation quality, so it is not promotable (2026-11-16)

Predictions `lens/runs/NT-DECODE-PREDICTIONS.txt` written before the run. Numbers from `lens/runs/NT-DECODE.txt`, 12 prompts, 40 generated tokens,
checkpoint `nodirect1024_nexttarget_data4m_e12_s0_best.npz`.

```
decoder            repetition   coverage   KL
next-target greedy    0.075       0.190     18.44
next-target T=0.7     0.002       0.173     20.19
shipped    greedy     0.275       0.343     17.21
shipped    T=0.7      0.027       0.292     17.73
```

## The falsifier fired

I wrote, before running: *"if coverage drops below 0.20 or repetition exceeds 0.10 at T=0.7, the +1.54 accuracy gain came WITH a generation
regression, and the honest headline becomes: more accurate, worse text - which would mean the new config is NOT promotable and the shipped model
stays."*

Coverage at T=0.7 is **0.173**, under the 0.20 bar. So: **not promotable.** The prediction that failed is the one I most wanted to pass.

## What the accuracy gain actually bought

- **coverage 0.292 -> 0.173**, a 41% loss of vocabulary breadth
- **KL 17.73 -> 20.19**, a 2.46 increase in divergence from the target distribution
- **accuracy 24.58% -> 26.12%**, +1.54 paired on seed
- repetition improves dramatically, 0.275 -> 0.075 greedy and 0.027 -> 0.002 sampled

So sharper concept labels make the model **more accurate, less repetitive, and less varied** - it learns to say the likelier thing more often and the
range of what it says narrows. That is a coherent trade, not a defect, and it is the opposite of what the accuracy number alone suggested.

## Why this is the most useful negative of the day

It is the project's accuracy-versus-generation dissociation firing **in the direction that would have flattered me**: the arm that won on accuracy lost
on generation. Had I promoted the config on the accuracy number - which R1 of the decision rule explicitly permitted, and which would have been easy to
stop there - the shipped model would have become the narrower, worse-text model with a better benchmark score.

Two things saved it: the standing rule that VAL accuracy alone cannot promote a configuration, and a falsifier with a number in it, written before the
run. Neither is clever; both had to be there.

## What ships, and what stays open

**Shipped configuration is unchanged: 4M / 12 epochs, d1024, 53 concepts, lam_concept 1.0, next-token supervision OFF, decoding at T=0.7**, carrying
VAL 24.58% +/- 0.055, coverage 0.292, repetition 0.027, KL 17.73.

The +1.54 accuracy result stands as a measured fact and stays published (`docs/NEXT-TARGET-3SEED.md`); what does not follow from it is promotion. If a
future arm buys accuracy **without** losing coverage, that is a different and promotable result - and now there is a baseline curve to judge it against.

## The accuracy/breadth trade is not a dial: halving supervision strength raises accuracy to 26.52% and still costs coverage (2026-11-16)

Prediction file `lens/runs/LAM-NEXT-PREDICTIONS.txt` and its promotion criterion were written before either arm ran. Numbers from
`lens/runs/LAM-NEXT.txt` and `lens/runs/LAM05-DECODE.txt` (12 prompts, 40 generated tokens).

```
arm                        VAL acc   VAL ppl   coverage T=0.7   repetition T=0.7   KL
next-target lam 1.0        26.12%    169.67        0.173             0.002        20.19
next-target lam 0.5        26.52%    158.85        0.167             0.083        20.65
shipped (window labels)    24.58%    181.04        0.292             0.027        17.73
```

## Every number moved the wrong way for promotion

- **accuracy and perplexity improved**: 26.52% and 158.85 are the best this project has measured on both
- **coverage fell again**: 0.167, below the lam 1.0 arm's 0.173 and far below the shipped 0.292
- **repetition got worse**: 0.083 against 0.002 at lam 1.0 and 0.027 shipped
- **KL got worse**: 20.65 against 17.73 shipped

The promotion criterion fixed in advance required **coverage >= 0.27 AND VAL >= 25.2**. Accuracy clears it with margin. Coverage fails at 0.167.

## The falsifier fired, and it answers the question the curve was built to answer

I wrote: *"if coverage stays below 0.22 at BOTH strengths, the breadth cost is not a strength dial - it is carried by the label change itself - and
promoting accuracy-with-breadth requires a mechanism that changes WHAT is supervised per position rather than how hard."*

Two strengths, 0.173 and 0.167, both under 0.22. **The trade is not a dial.** Supervision strength sets accuracy (26.12 -> 26.52 when halved) and
leaves breadth roughly where it was. Lowering the price buys a little accuracy and no breadth at all, which is the opposite of the middle setting I
hoped for.

**So this direction closes.** Next-token concept supervision is a real and repeatable accuracy gain - +1.54 paired on three seeds at lam 1.0, and
higher still at lam 0.5 - and it cannot be shipped, because what it buys in accuracy it takes out of vocabulary breadth at every strength tested.

## What is now established about the two halves

- **accuracy** responds to the supervision target and to its strength
- **breadth** responds to neither, in this family
- **concept-row meaning** responds to neither, across ten mechanisms

Three quantities, one lever that moves only the first. That is the cleaned-up statement of this project's central difficulty, and it is more precise
than anything available this morning.

## Accuracy rises as concept supervision weakens: 26.69% at quarter strength, the best this project has measured (2026-11-16)

Predictions and promotion criterion for these arms: `lens/runs/LAM-NEXT-PREDICTIONS.txt`, `lens/runs/LAM0-PREDICTIONS.txt`. Numbers from
`lens/runs/LAM-NEXT.txt`, `lens/runs/LAM05-DECODE.txt`.

```
arm                     concept supervision   VAL acc   VAL ppl   coverage T=0.7
shipped (window labels)       1.0 (window)      24.58%    181.04       0.292
next-target lam 1.0           1.0 (next)        26.12%*   169.67       0.173
next-target lam 0.5           0.5 (next)        26.52%    158.85       0.167
next-target lam 0.25          0.25 (next)       26.69%    150.64       pending
                                          * 3-seed mean; the others are one seed each
```

## What is measured, and what is not

**Measured**: at a fixed 12-epoch budget on the same data, the same split and the same seed, weakening the concept supervision raises both validation
accuracy and perplexity, monotonically across three settings. The quarter-strength arm is the best pair of numbers this project has recorded.

**Not yet measured**: whether that continues to zero. A **lam 0 control with no concept supervision at all** is running now, with its prediction written
before the launch (`lens/runs/LAM0-PREDICTIONS.txt`): I predict 26.4-27.0%, i.e. the trend continues, because a loss that constrains the hidden state
toward 53 fixed directions should cost a little prediction. The falsifier is recorded too: **lam 0 below 26.12% falsifies the monotone reading entirely**
and the whole "supervision costs accuracy" story gets withdrawn rather than reinterpreted.

**And a caveat in the same breath**: these are one seed per strength. If the lam 0 arm lands inside 26.1-26.4%, all four arms are within seed noise of
each other and the honest statement becomes the coarse one - supervision strength does not move accuracy on this budget at all, while labels still cost
breadth and still buy an exact explanation.

## Why this is uncomfortable for the project's own claim

The front page says readability has a price of **-1.53 +/- 0.44 points at matched parameter budget** - measured on the image task against a plain
control. If the chat side behaves the same way, then for chat that price is **understated**, and the correct statement is not "readability costs about a
point and a half" but "the concept term costs accuracy at this budget, and the readable architecture pays for itself in explanation, not in prediction."

That is a correction to a public claim. It will be published when the control is measured, not before, and it will be published whether or not it flatters
the readable architecture - which is the entire reason the control was launched the moment the trend appeared.

## The concept supervision costs accuracy: removing it entirely gives 26.98%, the best number this project has measured (2026-11-16)

Predictions written before the arm ran: `lens/runs/LAM0-PREDICTIONS.txt`. Numbers from `lens/runs/LAM0.txt`, `lens/runs/LAM-NEXT.txt`,
`lens/runs/LAM05-DECODE.txt`, `lens/runs/LAM0.txt`.

```
arm                        concept supervision   VAL acc   VAL ppl   coverage T=0.7
no supervision (lam 0)            0.0            26.98%    147.81       pending
next-target lam 0.25              0.25           26.69%    150.64       0.150
next-target lam 0.5               0.5            26.52%    158.85       0.167
next-target lam 1.0               1.0            26.12%*   169.67       0.173
shipped (window labels)      1.0 (window)       24.58%*   181.04       0.292
```
`*` = three-seed mean; the others are one seed each. Same data, same split, same seed, same 12-epoch budget; only the supervision differs.

## The prediction passed and the falsifier did not fire

I predicted 26.4-27.0 for the no-supervision arm and recorded the falsifier: *"lam 0 below 26.12% falsifies the monotone reading entirely."* It landed
at **26.98%**, the top of the predicted band, with the lowest perplexity in the project at **147.81**.

**Verified that the control was real**: the concept loss printed **0.0000** at the endpoint, so the arm genuinely trained with no concept supervision
rather than a flag that silently did nothing. That check exists because this project has already shipped one silent-no-op.

## What this means

**Monotone across four settings**: every step of concept supervision added buys a little accuracy loss. Removing supervision entirely is worth **+0.86
points** over the strongest supervised arm (26.98 vs 26.12, one seed each).

**And breadth moves the other way**: coverage falls monotonically as supervision weakens, 0.173 -> 0.167 -> 0.150, with the no-supervision arm's coverage
still measuring. So supervision strength is a single dial that **trades accuracy against vocabulary breadth**, in both directions, across four points.

## The correction this forces on the front page

The front page states readability's price as **-1.53 +/- 0.44 points at matched parameter budget**, measured on the image task against a plain control. On
the chat side the concept term's cost is now measured from the other end: switching it off entirely is worth **+0.86 points** and better perplexity, while
switching it on is what produces the vocabulary breadth (0.173 vs 0.150 at quarter strength).

So the honest chat-side statement is: **the concept layer buys an exact explanation and vocabulary breadth, and it pays for both in accuracy.** That is a
trade, not a tax - and it is a different, more precise claim than "readability costs about a point and a half". The public page is being updated to say so.

## Caveat, unchanged and load-bearing

**One seed per strength.** The no-supervision arm has not been run on three seeds, and until it is, 26.98 is a single-seed number. It is also not a
promotable configuration on its own: it has no concepts, so it cannot explain anything, and the shipped model is the one that can.

## The family is complete: accuracy is monotone in supervision, coverage is NOT (2026-11-16)

Numbers from `lens/runs/LAM0.txt`, `lens/runs/LAM0-DECODE.txt`, `lens/runs/LAM-NEXT.txt`, `lens/runs/LAM05-DECODE.txt`. Predictions for every arm were
written before it ran; the 3-seed check on the no-supervision arm is running now (`lens/runs/LAM0-3SEED-PREDICTIONS.txt`).

```
arm                     supervision   VAL acc   VAL ppl   coverage T=0.7   repetition
no supervision (lam 0)       0.0        26.98%    147.81       0.190          0.013
next-target lam 0.25        0.25        26.69%    150.64       0.150          0.079
next-target lam 0.5         0.50        26.52%    158.85       0.167          0.002
next-target lam 1.0         1.00        26.12%*   169.67       0.173          0.002
shipped (window labels)     1.00        24.58%*   181.04       0.292          0.027
                                        * three-seed mean; others one seed
```

## Accuracy is monotone. Coverage is not, and the correction matters

**Accuracy** rises perfectly monotonically as supervision weakens: 26.12 -> 26.52 -> 26.69 -> 26.98. That is four points, same data, same split, same
seed, same 12 epochs, with the concept loss printing exactly **0.0000** on the no-supervision arm so the control is verified rather than assumed.

**Coverage does not follow that pattern.** Among the supervised arms it rises with strength (0.150 -> 0.167 -> 0.173), which is the direction I reported
earlier today - but the **no-supervision arm comes in at 0.190**, above all three of them. So coverage has no monotone relationship to supervision
strength: it is lowest at quarter strength, higher at half and full, and highest with supervision switched off entirely.

**This corrects a claim I made an hour ago.** I wrote that breadth moves monotonically the other way as supervision weakens (0.173 -> 0.167 -> 0.150). With
the no-supervision number in, that is false, and it is corrected here rather than quietly dropped. The honest version is narrower: **accuracy is a
monotone function of supervision strength; coverage is not.**

## What the concept supervision actually buys, measured from both ends

- Removing it entirely gains **+0.86 accuracy points** (26.98 vs 26.12) and **+0.017 coverage** (0.190 vs 0.173)
- The shipped configuration, which uses *window* labels rather than next-token labels, holds **0.292 coverage** - the only configuration in this project that
  reaches that level - at **24.58%** accuracy, i.e. **2.40 points below** the no-supervision arm

So the concept layer's cost is now measured from both ends, and the benefit is concrete: it is the only thing in this family that produces the vocabulary
breadth of the shipped model, and it buys an exact explanation that no other arm here can give. It pays for both in accuracy.

## Nothing in this family ships

`lam 0` fails the same promotion test as every other arm: coverage **0.190** is below both the shipped 0.292 and the 0.27 bar fixed before these runs. It also
has no concepts, so it cannot explain anything. The shipped model stays the shipped model.

The remaining claim - that the concept term costs ~2.4 points on chat - rests on one seed for the no-supervision arm. Seeds 1 and 2 are running with their
predictions recorded: **S1 all three above 26.12**, **S2 mean 26.8-27.1 with sd at or below 0.20**, **falsifier: any seed at or below 26.12 withdraws the
+0.86 and I say so.**

## Confirmed on three seeds: no concept supervision reaches 27.01% +/- 0.03, +2.43 over the shipped model (2026-11-16)

Predictions written before the run: `lens/runs/LAM0-3SEED-PREDICTIONS.txt`. Numbers from `lens/runs/LAM0-3SEED.txt`.

```
seed   VAL acc   VAL ppl   concept loss
 0      26.98%    147.81       0.0000
 1      27.03%    148.38       0.0000
 2      27.03%    145.87       0.0000
mean    27.01%    147.35
sd       0.03       1.34
```

## All three predictions passed

- **S1** all three seeds above 26.12 (the full-supervision 3-seed mean): **passed**, worst seed 26.98
- **S2** mean 26.8-27.1 with sd at or below 0.20: **passed**, mean 27.01, sd 0.03
- **S3** perplexity mean at or below 150.0: **passed**, 147.35
- **Falsifier** (any seed at or below 26.12 withdraws the +0.86): **did not fire**

**Paired on seed**, which is this project's rule: against the full-supervision arm the deltas are **+1.01 / +0.79 / +0.87, mean +0.89 sd 0.11, three
of three positive**. Against the shipped configuration they are **+2.41 / +2.50 / +2.39, mean +2.43 sd 0.06**.

The control is verified on every seed by its own instrument: the concept loss printed exactly **0.0000** at all three endpoints, so none of these runs
silently trained the ordinary objective.

## The claim, now publishable

**Removing concept supervision from this recipe is worth +0.89 points of validation accuracy (paired, 3 seeds, 3/3 positive) and 2.43 points against the
shipped configuration.**

That is a cost, not a defect: the concept layer is what makes the model explainable, and it is what produces the vocabulary breadth of the shipped
model (coverage 0.292, the only configuration in this family to reach it). The readable architecture pays for both in accuracy.

## What is still single-seed and still labelled as such

The per-strength points - lam 0.5 at 26.52% and lam 0.25 at 26.69%, and their coverage figures 0.167 and 0.150 - remain one seed each. The monotone
accuracy ordering across all four settings is therefore suggestive, not established; only the two endpoints (no supervision 3 seeds, full supervision 3
seeds) carry a three-seed bar. Stated here so the tidy four-point curve is not mistaken for a measured one.

## Nothing in this family ships

No arm here beats the shipped model on both halves. Coverage remains the binding constraint: 0.190 for this arm against 0.292 shipped, and the
promotion bar of 0.27 was fixed before any of these runs. The shipped model stays the shipped model.

## Breadth does not survive weaker window supervision: coverage 0.292 -> 0.133 at half strength (2026-11-16)

Predictions written before the run: `lens/runs/WINDOW-LAM-PREDICTIONS.txt`. Numbers from `lens/runs/WINDOW-LAM.txt`.

```
arm                        VAL acc   VAL ppl   coverage T=0.7   repetition   KL
window labels, lam 1.0      24.58%*   181.04       0.292          0.027     17.73   (shipped)
window labels, lam 0.5      25.38%    168.98       0.133          0.127     21.48
                                        * three-seed mean
```

## The falsifier fired, and harder than the bar

I predicted coverage would stay at or above 0.25 and fixed the falsifier at below 0.20. It came in at **0.133** - not merely under the bar, but the **lowest
coverage of any arm measured in this project**, and **halving window supervision more than halved breadth**.

Scored: **V1 FAILED** (0.133 vs 0.25), **V2 PASSED** (25.38 is inside 24.9-25.7), **V3 FAILED** (repetition 0.127 vs a 0.08 limit). The arm is not promotable:
coverage 0.133 is nowhere near the 0.27 bar fixed before the run.

## What this establishes

**Breadth in the window-label family is sharply sensitive to supervision strength**: 0.292 at full strength, 0.133 at half. Compare the next-token family,
where coverage barely moved with strength (0.173 / 0.167 / 0.150). So the two label definitions respond to strength completely differently, and neither
responds the way I predicted at first.

**The shipped configuration is the only arm in the project that reaches 0.292 coverage**, and every weaker setting of either label definition lands between
0.133 and 0.190. Whether 0.292 is a ceiling or a point on a rising curve is now the open question, and the arm running answers it by testing strength **2.0**:
if coverage exceeds 0.292 the shipped setting was never tuned for breadth, and if it does not, the shipped configuration is a local optimum in both
directions - a stronger statement than "we picked a setting".

## The honest summary of the whole day on this axis

Six configurations, one per setting, and **not one of them beats the shipped model on both accuracy and breadth**. Accuracy can be bought with weaker
supervision (up to 27.01% +/- 0.03 with none at all); breadth cannot be bought back at any setting tested. The shipped model remains the shipped model
because it is the only one that keeps the vocabulary, and the concept layer's price is now measured from both ends.

## Coverage is a spike, not a dial: the shipped setting sits on top of it (2026-11-16)

Predictions before the run: `lens/runs/WINLAM2-PREDICTIONS.txt`. Numbers from `lens/runs/WINLAM2.txt`.

```
window-label strength   VAL acc   ppl      coverage T=0.7   repetition
0.5 (half)               25.38%   168.98       0.133          0.127
1.0 (shipped)            24.58%*  181.04       0.292          0.027
2.0 (double)             23.23%   204.13       0.156          0.042
                                 * three-seed mean; others one seed
```

## The falsifier fired

I wrote: *"coverage at or below 0.292 means lam 1.0 is a CEILING, not a point on a rising curve."* Doubling supervision gave **0.156** - not higher than 0.292, but
roughly half of it.

So the shape of this relationship is neither rising nor monotone: **0.133 at half strength, 0.292 at the shipped strength, 0.156 at double.** Coverage forms a
**narrow peak at exactly the shipped setting**, and moving the weight in either direction falls off it.

Scored: **W1 FAILED** (0.156 vs the 0.292 bar), **W2 PASSED** (23.23 below 24.58), **W3 PASSED** (repetition 0.042).

## What this settles

**The supervision-strength axis is closed.** Seven configurations measured today, and no weight above or below the shipped one recovers the vocabulary breadth of
0.292. The shipped configuration is a **local optimum in both directions** - which is a stronger statement than "we picked a setting": it means breadth is not a
tunable dial at all, and further weight settings are not worth a GPU hour.

What remains open is a different question entirely: **what is supervised per position**, not how hard.

## The complete day, one table

```
config                          VAL acc   coverage
no supervision                  27.01%*    0.190
next-token, quarter             26.69%     0.150
next-token, half                26.52%     0.167
next-token, full                26.12%*    0.173
window, half                    25.38%     0.133
window, full (SHIPPED)          24.58%*    0.292
window, double                  23.23%     0.156
                                * three-seed means
```

**Accuracy falls monotonically with supervision strength in both label families** - the one quantity that behaved consistently across every arm. **Breadth does
not**: its maximum is a single point, at the shipped setting, and every neighbour loses roughly half of it.

Nothing in this table beats the shipped model on both. The shipped model stays the shipped model, and the reason is now measured rather than argued.

## Depth: one certified block on the shipped model (2026-11-17)

Paired over 200 held-out windows, bootstrap 95% intervals, five checkpoints:

```
resume point      -0.02623 nats   CI [-0.04997, -0.00305]
+1 pass           -0.03502 nats   CI [-0.06437, -0.00782]
+2 passes         -0.04021 nats   CI [-0.07520, -0.00575]
+3 passes         -0.03989 nats   CI [-0.07505, -0.00318]
+4 passes         -0.04127 nats   CI [-0.07931, -0.00271]  <- endpoint
```

Adding one certified 15.2M-parameter block to the shipped model improves held-out loss by **0.041 nats**, resolvable, and the gain **plateaus after two passes**.
That is **0.67% of the 6.0379 baseline loss for 51% of the parameter budget** - a real effect and a poor trade, reported as both.

Source: `lens/runs/WARMSTART-EXTEND.txt`, `lens/runs/WARMSTART-PASSES.txt`, `tools/eval_resolution.py`, `docs/history/DEPTH-PLATEAU.md`.

**Method note:** all paired numbers use per-window differences against a baseline re-measured in the same run. Per-window losses have sd 3.3764; per-window
differences have sd 0.11696, so pairing removes a factor of 29 of the noise. Earlier arms in this project compared arm means, which is adequate for large
effects and inadequate for small ones.

## Step speed: 8.55x from one keyword, bit-identical (2026-11-17)

All four `np.einsum` calls in `lens/scan_direct_batched.py` were made without `optimize=True`, so NumPy used its generic looping contraction instead of a BLAS path.
Measured on the real training step, same inputs and seed:

```
BEFORE  loss 9.9041865856  gnorm 1.1704698742  step 32.47s
AFTER   loss 9.9041865856  gnorm 1.1704698742  step  3.80s     (8.55x faster)
```

Loss and gradient norm identical to ten decimals; no arithmetic changed. A predicted matmul rewrite turned out to be exactly this same fix, at 1.09x against it.
Source: `tools/einsum_opt_test.py`, `tools/ab_step.py`, `docs/history/EINSUM-OPTIMIZE.md`.

### Confirmed on a real training arm

Run 9 restarted on the fixed code reproduces the old run's loss **exactly** (step 25: 9.4038 both, step 50: 8.5875 both, step 75: 8.7967 both) while 25 steps drop from
**222 s to 45 s - 4.96x** - and the new run uses **4 cores instead of 12**, with Roblox running. Source: `lens/runs/SCAN-RUN9.txt`, `docs/history/OPTIMIZE-ON-ARM.md`.

## Run 9 scored against its pre-registration (2026-11-17)

```
F1  KL across six prefix-free words     1.1702   bar >0.05                       PASS
F2  state cosine                         0.1736   bar <0.50                       PASS
F3  held-out accuracy                   16.93%    bar 5.34%                       PASS
F4  trained vs random-noise readout      ratio 1.0036x  (trained 1.1702, noise 1.1660)  FAIL
G1  readout gain bounds                  0.978 .. 1.119   bounded, not saturating
G2  mean predictive entropy              8.835 = 94.1% of uniform   no collapse
```

**CORRECTION 2026-11-17: F4 was blind - see below. The trained readout is NOT noise (16.78% vs 0.0000% next-token accuracy, 859x).** Original text follows: the trained readout is *indistinguishable from noise* at separating inputs (1.0036x against a required 2x). F1 therefore passes for a reason that has nothing to do with training - the noise control clears the same bar - which is exactly what F4 was added to detect. Guards rule out gain saturation and entropy collapse, so this is a clean negative: our readout normalisation is not the lever. Source: `tools/eval_run9.py`, `lens/runs/chat/scan9_ep1500.npz`, `docs/history/RUN9-SCORED.md`.

### Correction: F4's metric was blind, the readout works (2026-11-17)

F4 compared KL across six one-word prompts (trained 1.1702, noise 1.1660). That measures how much a projection's *response varies* with input - which a random projection does by construction - not whether it predicts anything. Measured directly on next-token accuracy over 5,120 held-out tokens:

```
TRAINED readout  16.7773%
NOISE   readout   0.0000%
chance            0.0083%      ratio 859x
```

**The trained readout is doing real work.** The earlier conclusion that it was indistinguishable from noise is withdrawn, and the inference that the failure sits between state and answer layer is withdrawn with it. Source: `tools/f4_instrument_check.py`, `docs/history/F4-CORRECTION.md`.

### RETRACTED 2026-11-17: the readout is NOT saturated (entropy is 6.954 nats, 74% of uniform, top-1 prob 0.1106). See docs/SATURATION-RETRACTED.md. Original text kept below.

```
held-out real text (5,120 tokens):  entropy 0.007 nats = 0.1% of uniform  |  accuracy 16.78%
six one-word probes:               entropy 0.148 nats = 1.6% of uniform
uniform reference:                 9.393 nats
```

The softmax is effectively one-hot: the model commits to one word and is **wrong 83% of the time**. That also explains the looping generation seen at the end of every scan run.
The probes are saturated too, so "little context" does not explain the earlier figure - and **G2's reported 8.835 nats cannot be reproduced from either slice**, so that metric is not usable until `tools/eval_run9.py`'s entropy path is audited (the second metric defect in that script, after F4). Source: `tools/g2_entropy_check.py`, `docs/history/SATURATION.md`.

### RETRACTED 2026-11-17: saturation was never shown (it rested on a bad entropy figure). Logit-range 11.6->12.1 stands; the conclusion does not. See docs/SATURATION-RETRACTED.md.

```
ckpt     logit_rng   entropy   %unif    acc%
ep250         11.6    0.0108    0.12%    7.10
ep1500        12.1    0.0079    0.08%   17.97
```

Logit spread moves **11.6 -> 12.1 (1.05x)** across 1,250 steps while accuracy climbs 7.10% -> 17.97%. The readout is **born one-hot**; training never softens a single distribution, it only improves *which* word it is certain about. The failure is **calibration, not capacity** - states decorrelate, the readout beats noise 859x. Source: `tools/saturation_trajectory.py`, `docs/history/SATURATION-FROM-START.md`.

### Corrected entropy of the run-9 readout (2026-11-17, verified with sum(probs) = 1.0)

```
T=1   entropy 6.954274 nats = 74% of uniform   top-1 prob 0.110566
T=4   entropy 9.353526 nats = 99.6% of uniform  top-1 prob 0.000608
```

The earlier one-hot reading (0.0079 nats) was a bug in my own entropy expression; the corrected value is an ordinary moderately-peaked distribution, consistent with 17.97% accuracy. Temperature scaling raises entropy with accuracy unchanged, as it must. Source: `tools/temp_debug.py`, `docs/history/SATURATION-RETRACTED.md`.

### G2 is correct; my disagreeing figures were the broken ones (2026-11-17)

```
six-probe mean entropy  direct=False  8.838
six-probe mean entropy  direct=True   8.835   <- reproduces the eval's G2 exactly
sum(probs) = 1.000000 in every case
```

My earlier 0.148 nats came from encoding probes **without masking `<unk>`**; my 0.0079 nats came from a broken entropy expression (retracted). The eval script was right; I was wrong three times in a row for three different reasons - a blind proxy (F4), a missing mask, and a broken expression. G2's 8.835 nats on context-free probes means the model is near-agnostic with no context, which is expected and is the opposite of the saturation G2 was written to detect. Source: `tools/path_reconcile.py`, `docs/history/G2-IS-CORRECT.md`.

**Still not established:** held-out entropy under the full forward path (direct term + UNK masked) has not been measured correctly yet.

### Definitive calibration of the run-9 readout (2026-11-17, 5,120 held-out tokens)

Forward path `readout(state) @ Wa + ba + relu(emb[last] @ Wd) @ emb.T`, `<unk>` masked, `sum(probs)` verified:

```
TRAINED (full path)    entropy 6.6505 nats (70.80% unif)  top1 0.1256  acc 16.70%  sum(pr)=1.000000000000
NOISE READOUT control  entropy 8.8683 nats (94.42% unif)  top1 0.0027  acc  0.00%  sum(pr)=1.000000000000
chance 0.0083%
```

The model is **slightly under-confident, not saturated**: right 16.70% of the time, mean top-1 probability 0.1256 (a calibrated model at that accuracy would sit near 0.167). The trained readout is **47x sharper** than a noise readout at the same scale and actually predicts, where noise never does. The looping generation is therefore **not** a calibration or saturation defect. Source: `tools/calibration_final.py`, `docs/history/CALIBRATION-FINAL.md`.

### Why generation loops: exposure bias, measured (2026-11-17)

Same model, same starting context, 40 steps, only the input differs:

```
teacher-forced accuracy   22.19%   (fed the true next token, as in training)
free-running accuracy      4.06%   (fed its own argmax, as in generation)
free-running repetition    0.93     (1 - unique/40; 1.0 = same token every step)
```

Accuracy **collapses 5.5x** purely from consuming its own output, and it emits roughly **3 distinct tokens in 40 steps**. That is the loop, quantified: one wrong token moves the recurrent state somewhere training never took it. **Every scan run 1-9 was scored teacher-forced** - the one condition where this defect is invisible. Source: `tools/decode_path.py`, `docs/history/EXPOSURE-BIAS.md`.

### The exposure-bias lever failed (2026-11-17, pre-registered)

```
                        baseline   rollout arm (300 steps)
teacher-forced accuracy   22.19%        21.25%
free-running accuracy      4.06%         2.19%
free-running repetition    0.93          0.93
```

Training on the model's own rollouts **halved** free-running accuracy and left repetition untouched. **The pre-registered falsifier fired:** free-running stays under 6%, so exposure bias is not the binding constraint. The detectors were clean (D1 0.9743 of rolled positions genuinely changed, D3 never fired), so this is a real negative result, not a void arm. Source: `lens/runs/ROLLOUT1.txt`, `tools/train_rollout.py`, `docs/history/ROLLOUT-LEVER-FAILED.md`.

### CORRECTED 2026-11-17: the state IS load-bearing (ablation: -16.68 points). The "carries almost no information" reading below was wrong - extra context past ~16 tokens adds nothing, but the state itself does the work. See docs/STATE-IS-LOAD-BEARING.md.

```
  ctx   teacher-forced   free-running   repetition
    4          16.25%          5.00%        0.89
   16          15.62%          4.69%        0.91
   32          15.62%          4.38%        0.92
   64          15.62%          4.38%        0.92
   96          15.62%          4.38%        0.92
  128          15.62%          4.38%        0.92
```

Free-running accuracy is 4.38-5.00% across a **32x** range of context, and the shortest context is marginally best - so the generation loop is not a context-window mismatch. The sharper result: **teacher-forced accuracy is flat at 15.62% from 16 to 128 tokens**, i.e. 8x more context buys **0.00 points**. A model that gains nothing from more context is not using its context. That one fact is consistent with the loops (repetition 0.92), with why rollout training could not help (nothing to teach), and with a readout that predicts frequent continuations while being unable to generate. Source: `tools/context_length_sweep.py`, `docs/history/CONTEXT-LENGTH-REFUTED.md`.

### State ablation: the recurrence is load-bearing and order-sensitive (2026-11-17)

```
arm            acc%     entropy    sum(pr)     top1
full         16.70%      6.6505   1.000000   0.1256
zero          0.02%      9.3894   1.000000   0.0001
shuffled      5.66%      6.7721   1.000000   0.1056

cost of zeroing the state   : 16.68 points
cost of destroying the order: 11.04 points
```

**The recurrence is not decorative.** Zeroing the carried state collapses accuracy to 0.02% (chance 0.008%), and destroying token order costs 11.04 points - so it reads a sequence, not a bag. Combined with the flat context-length sweep: the model has a **short but real effective horizon**, extracting a great deal from the first handful of tokens and then saturating. Source: `tools/state_ablation.py`, `docs/history/STATE-IS-LOAD-BEARING.md`.

**Reconciled 2026-11-17:** the earlier 0.1951 was one chunk leaking out of a loop; accumulated over all 5,120 tokens it is 0.1256, matching `calibration_final.py` exactly. See docs/TOP1-RECONCILED.md.

### The state forgets exponentially: effective memory is single-digit tokens (2026-11-17)

```
   k    state shift        (perturb the token at t-k, measure ||dh_t|| / ||h_t||)
   1       47.97%
   2       29.98%
   4       12.05%
   8        3.485%
  16        0.885%
  32        0.081%
  48        0.010%

state norm stable ~2.96 | cosine(h_t, h_t+1) 0.5711
```

A token **eight steps back moves the state by 3.5%**; sixteen back by **0.9%**. The influence decays with a half-life of ~1.5 tokens, so the **effective memory is single-digit tokens** despite a 256-dim state and 128-token windows. That settles four separate puzzles at once: flat accuracy past 16 tokens, why rollout training could not help (nothing older to recover), why generation loops (near-n-gram behaviour, repetition 0.89-0.93), and why the readout and concept instruments looked healthy (they measure instantaneous prediction only). Source: `tools/state_memory.py`, `docs/history/STATE-FORGETS.md`.

**Gate for the fix:** perturbation shift at k=16 must exceed **10%** (today 0.885%) while held-out accuracy holds ≥16%. **Falsifier:** if k=16 passes 10% and free-running accuracy is still under 6%, memory length was not the binding constraint either.

### CORRECTED 2026-11-17: the memory is CLAMPED by the trainer, not spent by the gradient. `renorm_recurrent(p, 0.9)` runs after every step and forces Wh to spectral radius 0.9, which contracts perturbations by construction. The measurement below stands; the causal explanation does not. See docs/RADIUS-CLAMP.md.

```
   k        init      trained
   1     48.662%      48.530%
   2     46.023%      31.116%
   4     40.622%      13.093%
   8     28.917%       3.231%
  16     17.812%       0.858%    <- the gate was 10%
  32      5.711%       0.091%

Wh largest singular value: init 1.9753 -> trained 1.3188
Wh Frobenius norm        : init 15.996 -> trained 10.497
```

**At initialisation the k=16 perturbation shift is 17.81%, above the 10% gate; after 1,500 steps it is 0.858% - a 20.8x loss of memory.** `Wh`'s spectral norm moves 1.98 -> 1.32, so the transition went from expansive to contractive while `Wx` and the embeddings were essentially untouched. So the fix is **not** a better initialisation: gradient descent had no reason to keep memory and a cheaper short-horizon solution available. Source: `tools/memory_init_vs_trained.py`, `docs/history/MEMORY-KILLED-BY-TRAINING.md`.

### Spectral-radius sweep on the existing checkpoint (2026-11-17, no training)

```
radius     k=16 shift        acc%    entropy
   0.9         0.703%       17.84%     6.6152
   1.0         1.864%       18.68%     6.7658     <- best accuracy
   1.2         1.241%       11.26%     8.1006
   1.5         0.759%        4.88%     8.1733
   2.5         3.185%        2.73%     8.4368
   3.4        15.380%        1.50%     8.6826     <- memory gate passed, accuracy destroyed
```

**Radius 1.0: confirmed on 5,120 tokens it is NOT an accuracy win** (+0.53 points against SE 1.28 - within noise; the earlier +0.84 was a 1,536-token slice and is withdrawn). It does give a **resolvable free-running gain: 5.00% -> 6.75%, paired CI [+0.44, +3.38]**, with repetition unchanged (0.915 vs 0.917) and the k=16 memory shift still only 1.864% against the 10% gate. See docs/RADIUS-1-CONFIRMED.md. And **the memory gate cannot be reached by radius alone**: at 3.4 the k=16 shift passes 10% (15.380%) but accuracy collapses to 1.50% and entropy rises to 8.68 nats, which is the tanh saturation this function's docstring already documented. Memory without saturation needs a mechanism that separates *keeping* information from *amplifying* everything - which a gate does and a global rescale cannot. Source: `tools/radius_sweep.py`, `docs/history/RADIUS-CLAMP.md`.

### Radius 1.0 confirmed: small real generation gain, fails the memory gate (2026-11-17)

```
r      acc%    entropy      top1   free-run acc%      rep
0.9   16.70%     6.6505    0.1256          5.00%     0.915
1.0   17.23%     6.7473    0.1138          6.75%     0.917

accuracy: +0.53 pts (SE 1.28) WITHIN NOISE   |   free-run: +1.75 pts, paired CI [+0.44, +3.38] RESOLVABLE
```

The clamp explains the forgetting, but relaxing it modestly is **not** the fix: 0.9 -> 1.0 buys 1.75 points of free-running accuracy and nothing on repetition, while 3.4 (where the memory gate passes) destroys accuracy to 1.50%. A mechanism that keeps information **selectively** is what the evidence points at. Source: `tools/radius_confirm.py`, `tools/freerun_confirm.py`, `docs/history/RADIUS-1-CONFIRMED.md`.

### A naive carry path cannot buy memory (2026-11-17, feasibility probe)

```
  beta     k=16 shift       acc%    entropy state norm
   0.0         0.788%     14.69%     6.6466       2.93
   0.2         1.719%     10.86%     7.9453       6.12
   0.4         0.948%      3.24%     8.0835      14.06
   0.9         0.443%      4.26%     8.1123     126.20
```

Adding an identity carry to the recurrence, `h + beta*h`, **never lifts the k=16 shift above 1.719%** while the state norm explodes from 2.93 to 126.20. The carry makes the state grow, growth saturates `tanh`, and a saturated `tanh` squashes the perturbation regardless of how much old state is added back - the same failure as radius 3.4, reached from another direction. **Memory cannot live inside a saturating nonlinearity**: it needs a separate linear channel with learned gates. No accuracy claim is made - these weights were trained with `beta = 0`. Source: `tools/skip_feasibility.py`, `docs/history/SKIP-FEASIBILITY.md`.

### CERTIFIED 2026-11-17: the gated cell's gradients now pass at T = 1, 2, 3, 8, 16, 32 (worst relative error 2.0e-06). The earlier failure was a single row-vector transposition, `Wf.T @ gf` where `gf @ Wf.T` was required, hidden by T=1 because that line is never reached at the final step. See docs/CELL-CERTIFIED.md.

`lens/cell.py` implements a separate linear cell state (never through `tanh`) with learned forget/input gates. Both pre-training checks fail:

```
T=1  worst abs diff 2.429e-12   <- exact, per-gate derivatives correct
T=2  worst abs diff 2.903e-02   <- carry-through-time path is wrong
T=3  worst abs diff 1.004e-01

memory probe (random weights): k=1 50.17%, k=8 10.50%, k=16 3.10%   (gate is 10%)
```

The T=1-exact / T>=2-wrong split localises the fault to the carry path through time. **Nothing may be trained on an uncertified gradient**, so no arm was launched. The memory falsifier has **not** fired and is not being claimed: it was written for a trained arm, and these gates have had no opportunity to learn to hold anything - although the 30x decay from k=1 to k=16 with gates initialised open does need explaining. Source: `lens/cell.py`, `tools/cell_selftest.py`, `docs/history/CELL-UNCERTIFIED.md`.

### Gated cell certified (2026-11-17)

```
   T    worst err   where      verdict
   1    8.977e-09   Wxi(0,2)   PASS
   2    5.651e-07   Wf(3,3)    PASS
   3    1.336e-06   Wi(1,1)    PASS
   8    1.431e-07   Wxf(0,0)   PASS
  16    1.993e-06   Wxf(3,0)   PASS
  32    3.758e-07   Wf(3,4)    PASS

CELL CERTIFIED at every required length (relative error vs finite differences, float64)
```

`lens/cell.py` implements a separate **linear** cell state (`f*c + i*v`, never through `tanh`) with learned forget/input gates. Memory on untrained weights is k=16 **3.10%** against the 10% gate - better than the tanh carry's 1.719%, but untrained gates prove nothing about trained ones. Next: train with the spectral clamp **off** and re-measure. Source: `lens/cell.py`, `tools/cell_gradcheck_all.py`, `docs/history/CELL-CERTIFIED.md`.

### The trained-gate question is not answered: the harness does not learn (2026-11-17)

```
attempt 1 (readout 0.02, lr 3e-3, 100 steps): loss 9.4054 -> 9.3936, acc 0.27% -> 0.00%
attempt 2 (readout 0.2,  lr 1e-2,  60 steps): loss 10.0356 -> 9.3083, acc 0.00%
ln(12002) = 9.3936 - the loss sits at uniform, so nothing is being learned

memory after those steps: k=16 4.40% (untrained 3.10%, gate 10%)
```

A model that predicts nothing cannot have its trained gates judged, so **the pre-registered question stays open and the falsifier has not fired.** The lesson is recorded: the pre-registration called for an end-to-end composition check before training, and skipping it produced a harness that runs, yields numbers, and measures nothing. Next work is that single finite-difference test of the whole cell+readout+loss stack - not another training run. Source: `tools/cell_train.py`, `docs/history/CELL-HARNESS-DOES-NOT-LEARN.md`.

### Composition certified: cell + readout + loss agree with finite differences (2026-11-17)

```
Wf 1.972e-07  Wi 2.417e-07  Wv 2.078e-08  Wxf 6.284e-08  Wxi 3.857e-08
Wxv 1.964e-09  bf 1.360e-08  bi 1.119e-08  bv 7.643e-10  Wc 6.765e-09  bc 5.738e-10
ALL PASS - 11 of 11 parameters, whole stack
```

The check that was skipped before the last training attempt is now done: every cell parameter **and** the readout agree with finite differences through the composed loss. So the harness's failure to learn is **not** an unverified gradient. What is left is the cell state's magnitude - with the loss pinned at exactly `ln(12002)` (uniform), the readout's input is near zero or near constant. Next measurement: `||c_t||` and consecutive-state cosine **during** training, before another run. Source: `tools/cell_compose_check.py`, `docs/history/COMPOSITION-CERTIFIED.md`.

### Why the cell harness sat at uniform: the state is nearly constant, not collapsed (2026-11-17)

```
step 0             ||c|| 2.5140   cosine(c_t,c_t+1) 0.9414   logit sd 0.5937
step 60 loss 9.230 ||c|| 2.1931   cosine            0.9650   logit sd 0.4498
```

The state's norm is healthy (~2.2, no collapse) but **consecutive states barely differ: cosine 0.9414, rising to 0.9650 during training.** The readout therefore sees virtually the same vector at every position and emits the marginal distribution - uniform by construction - which is exactly why the loss stuck at `ln(12002)`. **Training makes it more constant, not less.** A high consecutive-state cosine with a healthy norm means the carry is a slowly drifting running average: memory of the *average*, not response to the *present* - the mirror image of the earlier ~8-token decay finding. Design implication: the readout must see the current input alongside the state. Source: `tools/cell_state_probe.py`, `docs/history/STATE-TOO-CONSTANT.md`.

### Cell harness abandoned by its own pre-registered rule (2026-11-17)

```
HELD-OUT at start  loss 9.4722  acc 0.00%  cosine 0.9575
step 200           loss 9.3625  acc 0.00%  cosine 0.9701
GATE (>5% acc AND cosine <0.90): FAIL
```

The readout was given the current token embedding as well as the state (`logits = [c_t ; x_t] @ Wc + bc`) - the pre-registered change. **Accuracy stayed at 0.00%**, so the falsifier fired and the harness is **abandoned rather than re-scaled**, as the pre-registration required. The loss does fall monotonically (9.4722 -> 9.3625, `ln(12002)` = 9.3936), so this is slow learning, not a dead pipeline - but three attempts have now failed to reach non-zero accuracy and a falsifier obeyed only when it agrees is not a falsifier. `lens/cell.py` remains certified and reusable. Next: train the real scan model with `TARGET_RADIUS=1.0` from the start, the one mechanism this week with a confirmed cause. Source: `tools/cell_input_readout.py`, `docs/history/CELL-HARNESS-ABANDONED.md`.

### Raising the concept count did not close the matched gap; what a concept costs does (2026-11-18)

```
READABLE: 24,945 params  16 patches x 64  ->  21 concepts  ->  10
PLAIN   : 24,655 params  784 -> 31 hidden ->  10   (1.16% size difference)
delta -1.47 points   concepts passing 20/21   (13-concept run: -1.53)
```

**The gap did not move (-1.53 -> -1.47, noise).** The plain control is sized to match *parameter count*, so adding concepts also grew the control's hidden layer. The arithmetic is the mechanism: one concept unit costs **1034** params (16 patches x 64 dims + 10 answer weights) against a plain unit's **794** (784 pixels + 10), a ratio of **1.30**, so at matched parameters the readable model gets **68-77% of the units** (measured 21 vs 31). A named concept sits on a wider input than a plain hidden unit. Next arm: `d=49` makes 16x49=784 concept features - exact per-unit parity - with P1 unit parity, P2 gap within +-0.5, and a falsifier of still <=-1.0. Source: `lens/runs/WIDTH-MATCH-21.txt`, `docs/history/UNIT-COST-NOT-WIDTH.md`.

### The readability cost is in the concept representation; width, unit cost and the answer head are ruled out (2026-11-18)

```
13 concepts, d=64                    -1.53   (3 seeds)
21 concepts, d=64                    -1.47
21 concepts, d=49 (unit-cost parity) -1.46   <- falsifier FIRED

answer head on FROZEN concepts:  linear 83.23% | 40 pair rules 83.49% | ALL 78 pairs 84.18% | plain 86.42%
                                 -> +0.26 / +0.95 points, still 2.2 under plain  <- falsifier FIRED
```

Raising concepts 13->21 moved the gap 0.06 points; equalising per-unit parameter cost moved it 0.01. Giving the answer layer **rules** ("concept A and concept B" - still fully readable) recovered at most **+0.95** of the deficit. So the cost is not width, not unit cost and not the linear head: **it is the concept representation itself** - a layer forced to match one named formula per unit is a weaker encoder than a free hidden layer of equal size. The number this project reports is therefore **~1.5 points at matched parameters, in exchange for 100% of the middle layer being human-checkable** (21 of 21 honest; zeroing the layer drops 84.15% -> 10.17%, shuffling -> 9.55%, so it is load-bearing). One arm left on this line: give readable a 3x unit advantage (patch 14 x d 56) and see whether capacity, not the constraint, was the story. Source: `lens/runs/UNIT-COST-PARITY.txt`, `tools/rule_answer_probe.py`, `docs/history/READABILITY-COST-LOCATED.md`.

### Readability: parity in the MEAN, and a 3-seed spread result that is only a hypothesis (2026-11-18)

```
READABLE: 15,977 params   4 patches x 56  ->  21 concepts  -> 10
PLAIN   : 15,910 params   784 -> 20 hidden -> 10   (0.42% size difference)
delta -0.08 points      concepts passing 20/21 (95.2%)
```

Every configuration so far measured ~-1.5 points. This one measures **parity**, and the difference is shape, not capacity: `patch 7 x d 64` gave 1024 features per concept unit (gap -1.47/-1.53), `patch 7 x d 49` gave 784 (gap -1.46), and **`patch 14 x d 56` gives 224 features per concept unit for a gap of -0.08**. Coarse 14x14 patches match the concept formulas, which are region-shaped (`top_half_heavier`, `centre_heavier`), and unit cost falls from 1034 to 234 parameters so the concept layer is no longer starved. **P1 failed** (I predicted a 30+ unit advantage; the readable model fielded 21 against 20), P2 and P3 hold, and the falsifier did not fire - a weak pass, not a strong one. **One seed: the 3-seed replication is running and this page gets the retraction in place if the mean lands at or below -1.0.** Source: `lens/runs/CAPACITY-ADVANTAGE.txt`, `docs/history/READABILITY-PARITY.md`.

### 3-seed replication: parity in the mean, and the plain control is the unstable one (2026-11-18)

```
seed   readable    plain      gap
0       84.78%    84.86%    -0.08
1       84.21%    85.80%    -1.59
2       83.96%    81.52%    +2.44

readable: mean 84.32  sd 0.34  range 0.82
plain   : mean 84.06  sd 1.84  range 4.28
gap     : mean +0.26  sd 1.66

P1 mean parity HOLDS | P2 gap sd <= 0.6 FAILED (1.66) | P3 honesty>=85% HOLDS | FALSIFIER did not fire
```

**Parity holds for the mean (+0.26 points) and fails for the spread**: the gap is a coin flip per seed. But the readable model is tightly clustered (spread 0.82) while the plain control jumps **4.28 points** between seeds - **5.4x more seed-to-seed variation**. Same data, same recipe, so this is the architectures: one wide free hidden layer lands in different local minima, while 21 units pinned to named formulas are constrained in what they can learn. **With n=3 that spread difference is a hypothesis, not a result**, and an 8-seed run is in flight with a falsifier that retracts it if the plain sd drops below 1.0. Source: `lens/runs/CAPADV-SEEDS.txt`, `docs/history/PARITY-IN-MEAN-NOT-SPREAD.md`.

### The recurrent clamp is causally responsible for the forgetting: 5.4x more memory, at 3 points of accuracy (2026-11-18)

```
                       radius 1.0     radius 0.9
Wh spectral radius      1.0000         0.9000      <- clamp verified to do what it claims
held-out (teacher-forced)  13.23%         16.27%     <- 3.04 points WORSE unclamped
free-running             5.81%          5.00%      paired CI [-1.38,+3.00] NOT RESOLVABLE
memory k=16              4.730%         0.878%     <- 5.4x better unclamped
memory k=32              0.587%         0.079%     <- 7.4x
memory k=1              39.144%        50.383%     <- 0.8x: the clamped model is reactive and forgets fast
```

Two arms, same recipe, same seed, 1500 steps, **one number changed**. Unclamping the recurrent matrix multiplies long-range memory by **5.4x at k=16** and **7.4x at k=32**, so the clamp is the *cause* of the forgetting, not a correlate. It costs **3.04 points** of teacher-forced accuracy, does **not** make the model generate (free-running difference unresolvable), and still **fails** the 10% memory gate at 4.73%. Next, pre-registered: two matched 3000-step arms to test whether the accuracy cost is training-length or the radius itself (falsifier: gap >= 2.5 points at 3000 steps means the trade is permanent). Source: `lens/runs/EVAL-R10.txt`, `lens/runs/SCAN-R10-RADIUS1.txt`, `docs/history/CLAMP-IS-THE-CAUSE.md`.

### The readable image model, scored on four questions at 8 seeds (2026-11-18)

```
                                readable      plain
mean accuracy (8 seeds)          84.14%      84.44%    gap -0.30 points (SE 0.43)
seed-to-seed sd                  0.49        1.26      readable 2.6x tighter
spread                           1.42 pts    4.28 pts
epochs to 80%                    3.2         2.9       0.88x (no speed advantage)
epochs to 84%                   14.9        10.7       0.72x
concepts honest                 20-21 of 21  --        95-100% every seed
```

**Two wins, one parity, one loss.** Parity on mean accuracy (-0.30, under one standard error from zero). A real but smaller-than-first-measured stability advantage: **2.6x tighter across seeds** (the 3-seed estimate of 5.4x did not survive; P2 failed). **No learning-speed advantage at all** - faster at 75%, slower at 80% and 84%, no trend. And full transparency: 20-21 of 21 concepts honest, zeroing the layer drops 84.62% -> 9.98%, and all 21 formulas ship in the published model's `config.json`. Source: `lens/runs/CAPADV-SEEDS8.txt`, `docs/history/READABLE-SCORECARD.md`.

### The spectral radius was the wrong dial: memory dies on the AVERAGE singular value (2026-11-18)

```
Wh spectral radius        1.0000
Wh largest singular value 1.9601
Wh MEAN singular value    0.8424
memory operator diag(1-tanh^2) @ Wh, mean singular value   0.7618

0.7618^16 = 1.29% predicted retention at k=16   (4.73% measured)
tanh saturation: 0.00% of units beyond |preactivation| 2.0   <- not happening at all
```

Sphere radius 1.0 still forgets, and this says why: the memory operator's **average singular value is 0.7618**, so the average direction is multiplied by 0.76 every step and sixteen steps of that is 1.29% predicted retention (4.73% measured). **Tanh saturation is 0.00%** - the activations are tiny (mean 0.269), so the saturating-nonlinearity premise that motivated the gated-cell detour was wrong. `renorm_recurrent` constrained the one number that does not govern memory. New `ortho_recurrent()` projects `Wh` onto an orthogonal matrix (every singular value exactly 1.000000, verified), with two arms running: **P1 k=16 above the 10% gate at scale 1.0**, P2 1.0 trains worse than 0.98, P3 operator mean singular value above 0.90; falsifier k=16 below 6% means the decay is in the embedding or readout path instead. Source: `tools/saturation_probe.py`, `lens/runs/EVAL-R10.txt`, `docs/history/SPECTRAL-RADIUS-WRONG-DIAL.md`.

### explain() now exists for the vision model, and it is exact (2026-11-18)

The readable architecture's promise is that its middle layer is human-readable. That promise needs the model to be able to say **which named concepts produced an answer**, and the text, audio and speech models each had an `explain()` while the vision model had none. `lens/vision_explain.py` adds it, and it is not an attribution approximation - the model computes `logits = h @ answer.W + answer.b`, so each term `h_k * W[k, class]` **is** one named concept's exact contribution.

```
I answer Dress (score 4.453).
  vertical_edges_dominate  fires 1.00 x weight +2.895 = +2.887   formula: mean|dI/dx| > 1.15 * mean|dI/dy|
  bottom_half_heavier      fires 0.85 x weight +1.963 = +1.659   formula: mean(bottom 14) - mean(top 14) > 0.05
  AGAINST: left_right_symmetric 0.85 x -1.548 = -1.313
bias +0.227 + all 21 concept contributions +4.226052 = 4.453275 | measured 4.453275 | residual 6.99e-07

identity verified on 500 images: max |bias + all contributions - logit| = 1.689e-06, 0 of 500 fail
```

`examples/explain_image.py` in the public tree reproduces it from the published weights in plain numpy, including the per-concept formulas, so a reader can recompute every concept by hand. Source: `lens/vision_explain.py`, `tools/vision_explain_check.py`, `examples/explain_image.py`.

### The one dishonest concept: a claim 96.9% of images satisfy cannot be read (2026-11-18)

```
centre_heavier   AUROC 0.3196 (BELOW chance)   fires 39.2%   truth 96.8%   <- the single failure of 21
has_top_edge_ink    0.9601 | has_bottom_edge_ink 0.9651 | has_diagonal_structure 0.8003   (all added today, all honest)
the other 17 concepts: 0.70 - 0.98
```

"20 of 21 passing" hid which one fails. `centre_heavier` scores **AUROC 0.3196 - below chance**: its neuron fires *more* often where its claim is **false**, because at threshold 0.05 the concept fired on **96.9%** of training images and left almost nothing to rank against. A claim nearly every image satisfies is not a property of an image. Same defect class as the near-constant concepts that could pass an audit for free on the text side. **The three concepts added today are all honest** (0.9601 / 0.9651 / 0.8003), so the widening bought real neurons. Fix queued: threshold **0.05 -> 0.40** (support 96.9% -> **23.45%**), P1 AUROC above 0.80, P3 honesty 21 of 21; falsifier AUROC below 0.60 means **drop the concept** rather than keep a neuron that cannot mean what it says. Source: `tools/concept_honesty_detail.py`, `docs/history/ONE-DISHONEST-CONCEPT.md`.

### The measured-best configuration is now the default, and the model card was corrected (2026-11-18)

**Shipped default changed.** `lens/scripts/train_image.py` used to default to `patch 7 / d 64 / support floor 0.02` - the shape that measured **-1.47 to -1.53 points** against its matched plain control. It now defaults to `--preset parity`: **patch 14, d 56, floor 0.002**, which resolves to **4x56 patches -> 21 named concepts -> 10 classes, 15,977 parameters** and measured **-0.08 points**. The old shape stays reachable as `--preset legacy`, and any explicit `--d / --patch / --min-support` still overrides the preset. Both paths were smoke-tested after the change and print the same shapes they did before.

**Published model card corrected.** `glassnet-readable-vision-21` said *"3 seeds: this architecture's spread 0.82 points against the plain control's 4.28"* - the smaller-sample estimate. With 8 seeds the honest figure is **standard deviation 0.49 against 1.26, a 2.6x tighter spread**, and the card now states that and says the 3-seed figures are superseded. A published artifact is not allowed to keep a number the project has outgrown. Source: `lens/scripts/train_image.py`, `lens/runs/CAPADV-SEEDS8.txt`, `tools/publish_vision_hf.py`.

### The safetensors file hash is not rebuild-stable; artifacts now carry a canonical hash (2026-11-18)

```
build 0: header fdbecdd9703d613a | DATA 853dfd0ad0b6999b (63908 B) | canonical 9ea18a1b41897ce9
build 1: header fdbecdd9703d613a | DATA 853dfd0ad0b6999b (63908 B) | canonical 9ea18a1b41897ce9
build 2: header 112ac390d0b4880b | DATA 853dfd0ad0b6999b (63908 B) | canonical 9ea18a1b41897ce9
```

Republishing a model kept changing the `.safetensors` `sha256` with identical tensors (three pushes: `c83f8410`, `d6955049`, `d853b88e`). The **tensor data is byte-identical** every time; only the **512-byte JSON header** varies, and metadata key ordering was tested and is not the cause. A manifest hash that changes when nothing changed is decoration, not an integrity check. Every artifact now carries **two labelled hashes**: a **canonical** hash over sorted name+dtype+shape+bytes (stable - verify this) and the container file hash (documented as rebuild-unstable). Verified from the Hub: canonical **MATCH: YES**, and the model rebuilt from the **downloaded bytes only** in plain numpy scores **85.50% over 2,000 test images**. The explainer `explain_image.py` now ships inside the model repo too. Source: `tools/verify_hf_vision.py`, `docs/history/SAFETENSORS-HASH-STABILITY.md`.

### Both published artifacts verify from the Hub bytes alone (2026-11-18)

```
glassnet-scan-r9-recurrent
  published canonical 6dce6567eefa0c6b386fa97101834fb86aa5ebef26631dfd98f34ceeab9aacc4
  recomputed          6dce6567eefa0c6b386fa97101834fb86aa5ebef26631dfd98f34ceeab9aacc4   MATCH
  held-out accuracy recomputed from the DOWNLOADED weights: 16.27% over 756 positions
  published figure: 16.27%   <- exact
```

The chat model's published accuracy reproduces **to the digit** from the downloaded file alone, and the readable vision model rebuilt from its downloaded bytes scores 85.50% over 2,000 images. Both models now carry a **canonical weights hash** (stable across rebuilds) alongside the container file hash, because the safetensors header is not byte-stable. `tools/verify_hf_scan.py` and `tools/verify_hf_vision.py` re-run these checks and fail loudly on a canonical mismatch. Source: `docs/history/SAFETENSORS-HASH-STABILITY.md`, `docs/history/PUBLISHED-ARTIFACTS-VERIFIED.md`.

### CORRECTION: the published "32-token context" is wrong - the model uses 2 tokens (2026-11-18)

```
     L   accuracy   vs L=1
     1      9.27%   +0.00    <- only the current token
     2     16.73%   +7.47
     4     17.40%   +8.13
     8     17.40%   +8.13
    16     17.40%   +8.13
    32     16.73%   +7.47    <- the figure the cards used to claim
    64     17.93%   +8.67
   128     16.40%   +7.13

SE 0.98 points (n=1500, p~0.174); a difference needs ~1.9 points to be resolvable at 95%
  1 -> 2 tokens   : +7.47 points = 7.6 SE   RESOLVABLE
  2 -> 128 tokens : -0.33 points = -0.3 SE  NOT RESOLVABLE
  spread over L>=2 : 1.53 points = 1.6 SE   NOT RESOLVABLE
```

Both chat cards said **context window 32 tokens** - inherited from an older README and never measured. Measured now, by feeding exactly L tokens and scoring the next one over the same 1,500 held-out chunks: context buys **7.6 standard errors** of accuracy going from 1 to 2 tokens (9.27% -> 16.73%), and **nothing resolvable** from 2 to 128. **The effective context is 2 tokens**, and the published figure overstated it by 16x. This is the same finding as the state-side memory measurements (a token 16 back moves the state 0.878%) seen from the outside instead: nothing past two tokens changes the prediction. `glassnet-scan-r9-recurrent` now reports **effective context (measured): 2 tokens**, and `config.json` carries `effective_context_tokens` and `trainer_context_length` separately so a trained length can never be mistaken for a usable one. Source: `tools/context_curve.py`, `docs/history/EFFECTIVE-CONTEXT-IS-TWO.md`.

### 84% is a fair-weather number: a 2-pixel shift costs 27.8 points (2026-11-18)

```
clean 84.60% | brightness +0.20 78.13% | contrast x0.70 81.83% | noise sd 0.10 83.30% | blur 82.73%
1px shift 76.83% (-7.77) | 2px shift 56.77% (-27.83)   <- collapse
```

Every image accuracy this project published was measured on the **clean** test set. Under the shifts a user actually produces, **translation is the killer**: noise sd 0.10 costs 1.3 points and blur 1.9, but a **two-pixel shift costs 27.8**, dropping 84.60% to **56.77%**. Brightness is intermediate (-6.5 at +0.20) because the concept set is built on brightness/contrast formulas. The cause is structural and predicted by the architecture: the model reads a **fixed 14x14 patch grid**, so a shift moves content across patch boundaries, and concepts like `top_half_heavier` are alignment-sensitive by construction. **Honest disclosure: the readable image model expects a centred, fairly-exposed input.** Fix under test (`run_shift_aug.sh`): 2px training augmentation, P2 requiring 2px accuracy to reach 75%; falsifier below 65% means alignment is structural, not augmentable. Source: `tools/vision_robustness.py`, `docs/history/ROBUSTNESS-FAIR-WEATHER.md`.

### The radius dial trades memory for accuracy monotonically; the 10% memory gate sits past the collapse (2026-11-18)

```
 radius   k=16 memory   held-out acc   free-running   repetition   gnorm sd
   0.90      0.878%        16.27%         5.00%         0.915       0.81
   1.00      4.730%        13.23%         5.81%         0.908       1.25
   1.05      6.695%        11.90%         5.50%         0.893       1.38
   1.10      8.511%        10.05%         5.06%         0.881       1.43
```

Four arms, one number changed, every row a finished checkpoint. **P1, P2 and P3 all HOLD and the falsifier did not fire.** Each +0.05 of radius buys **+1.8 points of k=16 memory** and costs **1.3 points of held-out accuracy**; extrapolating, the 10% memory gate needs radius ~**1.14**, where accuracy would be ~**9%** against 16.27% at 0.9. **The memory gate is not reachable by tuning the clamp at any accuracy worth having** - it would cost 44% of the shipped accuracy. Free-running never moves (5.00 / 5.81 / 5.50 / 5.06%, every paired CI spanning zero), so memory and generation are separate problems and this dial touches only the first. The clamp line ends here; the structural answer is a separate linear state channel, and the orthogonal-recurrence arms are running. Source: `lens/runs/SCAN11-R105.txt`, `lens/runs/SCAN11-R110.txt`, `docs/history/RADIUS-LADDER-TRADE.md`.

### "Stop and start without losing anything" now exists and is tested (2026-11-18)

A standing requirement was **not met** until now: no trainer accepted a resume, so every invocation re-initialised from scratch and stopping a 3000-step arm discarded everything.
`lens/scripts/train_scan9.py` now takes `RESUME=<checkpoint.npz>`: it loads the weights, reads the step from `__meta__`, and continues from **step + 1** on the same learning-rate
decay curve. `CK_PREFIX` still keeps runs separate and an existing checkpoint is still **refused, never overwritten**.

```
RESUMED from lens/runs/chat/scan11_r100_ep250.npz at step 250
then: step 275/275 loss 7.1772 acc 7.45%      <- resumed
a fresh run at step 25:   loss 9.5173 acc 0.42%   <- a model that knows nothing
source checkpoint mtime unchanged (09:04:47 before and after)
```

That contrast is the proof the weights loaded rather than being silently re-initialised. **My first version died instantly with `NameError: name 'log' is not defined`** - it called `log()` before its definition further down the file, and the patch parsed fine; the smoke test caught it. **Honest limitation, written in the source:** the sampler's rng state is not stored, so a resumed run continues the weights exactly but sees a different shuffle of the same corpus. Saving the rng state would make it bit-exact and is not done yet. Source: `lens/scripts/train_scan9.py`, `docs/history/RESUME-IMPLEMENTED.md`.

### Resume is now BIT-EXACT; the earlier version was only weight-exact (2026-11-18)

The resume added earlier today continued the weights but not the sampler, so a resumed run saw a different shuffle. `__meta__` now carries `rng_state` (PCG64 state, JSON round-tripped) plus `ortho`/`ortho_scale`, and the state is restored before the first batch is drawn.

```
meta keys: ['D','H','V','ck_prefix','ortho','ortho_scale','rng_state','step','target_radius']

two independent resumes from the same checkpoint, 25 steps each:
step 275/275 loss 8.3962 ema 8.4815 acc 2.54% gnorm 3.46
step 275/275 loss 8.3962 ema 8.4815 acc 2.54% gnorm 3.46   <- identical to every printed digit
```

**Two bugs of mine were caught by running the code, not by reading it:** `NameError: name 'log' is not defined` (the first patch called `log()` before its definition, so every *resumed* run died), and then `NameError: name 'seen' is not defined` (the second patch matched a line part-way and buried `t0/ema/seen` inside the `if _resume_msg:` block, so every *fresh* run died). Both failure modes are recorded, because a patch that appends to a line that already has statements is exactly where they live. Source: `lens/scripts/train_scan9.py`, `docs/history/RESUME-BIT-EXACT.md`.

### The chat model is 83% readable, not 100% - and the audit names the cause (2026-11-18)

```
readable d512   30/36 concepts pass   mean AUROC 0.6886   mean FIRE RATE 47.4%
copy-path d512  36/36 concepts pass   mean AUROC 0.8695   mean FIRE RATE 28.3%

failing in the readable model only: is_a_list, mentions_a_name, mentions_a_quantity,
                                    mentions_body, mentions_family, mentions_sport
failing in BOTH models: none
```

Two facts matter more than the pass count. **Every failure is the readable model's alone** - a matched model with a direct path passes all 36 with the same formulas, so the six failures are architectural, not wrong definitions. And **the readable units fire 5-20x too often**: concept supports run from 0.58% to 13.9%, but the neurons average a **47.4%** firing rate, so a unit that fires on nearly half of all windows cannot sharply detect a 0.6% property - which is why AUROC sits at 0.6-0.7 while the name is in its own supervision. This is the same defect class found in the image model this morning (`centre_heavier`, AUROC 0.3196 at a 96.9% firing rate). `--lam-act`, the penalty on mean concept activation, has been **0.0 in every run this project has done**; two arms now try 0.3 and 1.0 with P2 requiring mean AUROC above 0.80 and a falsifier that sends the next attempt at the concept definitions instead. Source: `tools/concept_label_audit.py`, `docs/history/CHAT-READABILITY-AUDIT.md`.

### Concept honesty across 8 seeds: one unreliable concept, no others - and a correction to my own claim (2026-11-18)

```
seed       pass    centre_heavier AUROC   failing
0         20/21          0.2889         centre_heavier
1         21/21          0.6336         none
2         20/21          0.5095         centre_heavier
3         20/21          0.2903         centre_heavier
4         21/21          0.5999         none
5         21/21          0.7741         none
6         20/21          0.4643         centre_heavier
7         20/21          0.4282         centre_heavier

centre_heavier fails in 5 of 8 seeds; NO other concept fails in any seed (168 concept-seed pairs)
```

**`centre_heavier` is the only concept that ever fails**, and it is **unreliable rather than always inverted**: 5 of 8 seeds fail, AUROC ranges 0.2889 to 0.7741 - it learns something real at seed 5 (0.77) and inverts at seeds 0 and 3 (0.29). **CORRECTION to the earlier report in this same day**, which said "AUROC 0.3196, below chance, the neuron fires more often where the claim is false": that was measured at **seed 0** and stated as a property of the concept. It is true at seeds 0 and 3 only. The honest statement is **seed-dependent unreliability**, not constant inversion - the stronger claim came from one checkpoint, which is the mistake the three-seed rule exists to prevent, caught here by applying that rule to my own result. The fix (threshold 0.05 -> 0.40, support 96.9% -> 23.45%) is now tested at **three seeds**, with a falsifier that drops the concept if it still fails in 2 or more. Source: `tools/honesty_all_seeds.py`, `docs/history/HONESTY-ACROSS-SEEDS.md`.

### WITHDRAWN: the "resolvable free-running gain" was training length, not the radius (2026-11-18)

```
                              r=1.0 @1500   r=1.0 @3000   r=0.9 @1500 (reference)
held-out teacher-forced          13.23%        16.80%           16.27%
free-running                      5.81%         8.44%            5.00%
k=16 memory                       4.730%        4.434%           0.878%
paired free-running vs reference  NOT RESOLVABLE   RESOLVABLE      -
                                  CI [-1.38,+3.00]  CI [+1.75,+5.44]
```

**The 3.04-point accuracy penalty of unclamping was mostly under-training**: at 3000 steps radius 1.0 reaches **16.80%** against the reference's 16.27%, and the free-running difference becomes **resolvable for the first time in this project** (8.44% vs 5.00%, paired mean +1.375 tokens per 40, CI [+1.75, +5.44], excludes zero). Memory does not improve with length (4.434% at 3000 vs 4.730% at 1500) and still fails the 10% gate. **CAVEAT: this comparison is unmatched** - 3000 steps against a 1500-step reference - so the gain could come from training length rather than the radius. The matched control (radius 0.9 at 3000 steps) is at step 2700 and will separate them. Source: `lens/runs/SCAN11-R100-3K.txt`, `docs/history/R100-3K-RESOLVABLE.md`.

### CORRECTION: the resolvable free-running gain was TRAINING LENGTH, not the radius (2026-11-18)

```
                          radius 1.0     radius 0.9     (both 3000 steps)
held-out teacher-forced      16.80%         21.03%
free-running                  8.44%          6.88%
k=16 memory                   4.434%         1.312%

paired free-running difference (1.0 - 0.9): +0.625 tokens of 40, CI [-0.38, +3.56]  -> NOT RESOLVABLE
```

The claim published last wake - "the first resolvable free-running gain in this line", 8.44% vs the 1500-step reference's 5.00% with CI [+1.75, +5.44] - is **WITHDRAWN**. The comparison was unmatched (3000 steps against 1500) and the matched control has now run: **the free-running difference vanishes into a CI that includes zero**. The gain was training length. **P1, the prediction written against my own result, held.** P2 held (radius 0.9 is 4.23 points more accurate at matched length) and P3 held (radius 1.0 keeps 3.38x the memory, still failing the 10% gate). At equal training, unclamping buys memory that still fails and an unresolvable generation gain, at a cost of 4.23 points of accuracy - **so the shipped radius 0.9 stands.** Meanwhile at radius 0.9, doubling training moved held-out from 16.27% to **21.03%**, memory 0.878% to 1.312% and free-running 5.00% to 6.88%: **the lever that actually moved this model was more training, not the clamp.** Source: `lens/runs/MATCHED-3K.txt`, `lens/runs/SCAN11-R090-3K.txt`, `docs/history/CORRECTION-MATCHED-3K.md`.

### The returns-to-training curve at radius 0.9, measured on all 12 checkpoints (2026-11-18)

```
step  250    500    750   1000   1250   1500   1750   2000   2250   2500   2750   3000
acc  5.29  12.70  13.49  16.80  16.27  17.20  17.33  19.31  19.71  20.24  20.90  21.03
loss 7.451  7.064  7.034  6.828  6.811  6.698  6.653  6.590  6.537  6.495  6.454  6.442
```

**The climb is real** - 5.29% to 21.03%, a 15.7-point gain, loss falling monotonically with one wobble. **But the standard error on a single point is 1.47 points**, so the fine structure is noise: the last four increments (+0.40, +0.53, +0.66, +0.13 per 250 steps, mean +0.43) are **all inside the error bar**, which means **"the curve is decelerating" is NOT established** - that is what increments smaller than a 1.47-point error look like. What is resolvable is coarser: step 1500 to 3000 gains **3.83 points = 2.6 SE**. Linear fits over the later half extrapolate to **26.7-29.6%** at 6000 steps, which contains the 6000-step arm's prediction of >= 24%, but the fit rests on 1.47-point points, so 24% is **plausible and unproven** and the arm is the test. Source: `tools/learning_curve.py`, `docs/history/LEARNING-CURVE-R090.md`.

### CORRECTION: the held-out eval was too small - every absolute accuracy was ~3.2 points too low (2026-11-18)

```
checkpoint                    12 chunks (768)   104 chunks (6552)   difference
scan9_ep1500   (r=0.9)            16.27%            19.67%           +3.40
scan11_r100_ep3000 (r=1.0)        16.80%            20.83%           +4.03
scan11_r090_ep3000 (r=0.9)        21.03%            23.95%           +2.92
```

The protocol scored **12 chunks = 768 positions**, SE **1.47 points** - it could not resolve anything under ~2.9 points, and it was being used to argue about 1-point effects. The power calculation: to resolve 1 point at 95% needs **~6,636 scored positions = 104 chunks** (SE 0.50). Scored that way, **all three checkpoints are 2.9-4.0 points higher** - not because the estimator is biased but because every model shared the *same* unrepresentative first twelve chunks. **What survives:** the ordering and every paired comparison (same sample). **What is corrected:** radius 0.9 at 3000 steps is **23.95%** not 21.03%; radius 1.0 is **20.83%** not 16.80%; scan9 is **19.67%** not 16.27%. **What got sharper:** the radius-0.9 advantage is **3.12 points at ~4.4 SE**, resolvable, where it was previously a 2-SE call. Also: the 6000-step arm's P1 is restated - its baseline is 23.95%, not 21.03%, so the honest prediction is **>= 26%**, and 24% should not be counted as success. `tools/eval_radius_arm.py` now defaults to **EVAL_CHUNKS=104**. Source: `docs/history/EVAL-PROTOCOL-TOO-SMALL.md`.

### The activation penalty works: 30/36 -> 34/36 concepts honest (2026-11-18)

```
lam-act   pass     mean AUROC   mean fire rate   failing concepts
0.0      30/36      0.6886         47.4%         is_a_list, mentions_a_name, mentions_a_quantity, mentions_body, mentions_family, mentions_sport
0.3      32/36      0.7038         36.2%         mentions_a_name, mentions_a_quantity, mentions_body, mentions_sport
1.0      34/36      0.7370         20.1%         mentions_body, mentions_sport
```

`--lam-act` had been **0.0 in every run this project has ever done**; raising it takes label-honest concepts from **30/36 (83.3%) to 34/36 (94.4%)** and cuts the mean firing rate from **47.4% to 20.1%** (supports are 0.6-14%, so the neurons were firing 5-20x too often). **P1 and P3 held; P2 FAILED** (mean AUROC 0.7370 against a 0.80 target), and the falsifier did not fire but only just - 0.7370 is 0.017 above its 0.72 line. The two remaining failures are `mentions_body` (1.32%) and `mentions_sport` (1.43%), but **rarity is not the whole story**: `mentions_a_name` has the *lowest* support (0.82%) and now passes. Next test (lam-act 1.5 and 2.0, pre-registered) checks whether the dial saturates; if it does, the remaining two need per-concept positive weighting rather than more uniform pressure. Source: `lens/runs/CONCEPT-ACT.txt`, `docs/history/CONCEPT-ACT-WORKS.md`.

### Orthogonal recurrence: the best memory/accuracy point found, still 7.7 points behind (2026-11-18)

```
                        ortho s=1.0     radius 0.9 (same 1500 steps)
Wh largest singular value   1.0000        1.6245    <- exactly orthogonal, as designed
held-out teacher-forced     13.34%        21.02%
free-running                 4.06%         6.31%   (paired CI [-4.62,+0.00] NOT RESOLVABLE)
k=16 memory                  7.888%        1.322%
memory operator mean sv      0.8969        0.7618 (radius 1.0)
```

**P1 FAILED** (7.888% against the 10% gate - the most memory any configuration has produced, and not enough), **P3 FAILED narrowly** (operator mean sv 0.8969 against a 0.90 line: the projection is exact, and `diag(1-tanh^2)` pulls it under), **falsifier did not fire** (7.888% > 6%), so orthogonalising genuinely buys memory - just not enough. The result worth keeping: **ortho dominates the frontier**, giving **3.29 points more accuracy for 0.62 points less memory than radius 1.10** (8.511% / 10.05%). The probe's mean-singular-value model **over-predicts by 2.2x** (17.53% predicted vs 7.888% measured) because it uses the mean direction and the real perturbation finds contracting ones. **A bug in my own probe** printed a hardcoded `4.730%` from a *different* checkpoint as if measured on the one being probed - the measured value must now be supplied, and the tool says `NOT SUPPLIED` instead of inventing one. Memory line status: nothing reaches 10% at usable accuracy; a recurrent weight matrix alone cannot deliver it. Source: `lens/runs/SCAN11-R090-1500` reference, `docs/history/ORTHO-FRONTIER.md`.

### The accuracy plateau is measurable: flat from step 4500 while loss keeps falling (2026-11-18)

```
                             104 chunks (6,552 scored)      loss
step 3000  (resume point)          23.95%                  6.3261
step 4500                          25.35%                  6.1303
step 5000                          25.24%                  6.1148
step 5500                          25.76%                  6.0826
```

**Accuracy has plateaued.** From step 4500 to 5500 the readings are 25.35 / 25.24 / 25.76 - a spread of **0.52 points against a 0.50-point standard error**, i.e. flat - and the 2500 steps from 3000 to 5500 bought **+1.81 points** against the **+3.83** that 1500-to-3000 bought. **But loss has not plateaued**: it falls monotonically, **-0.24 nats** over the same stretch. **The model is getting more confident, not more correct.** Scoring: the arm's **original P1 (>= 24% at 6000) is MET**, already at step 4500; the **restated P1 (>= 26%)**, after the eval correction moved its baseline to 23.95%, reads **25.76% at step 5500** and must be judged on the final checkpoint, not extrapolated. **This closes the 'just train longer' line with a measured floor**: more training stopped buying accuracy around step 4500 at ~25.5%, and that was only visible because the eval was made four standard errors tighter than the one that started the line. Source: `lens/runs/chat/scan12_r090_6k_ep*.npz`, `docs/history/PLATEAU-MEASURED.md`.

### Every concept is honest on every seed, and the accuracy gap is not resolvable (2026-11-18)

```
seed   READABLE   PLAIN    delta        concepts    centre_heavier AUROC   fire vs truth
  0    15,977     15,910   -0.14 pts     21/21            0.9676            28.0% vs 22.9%
  1    15,977     15,910   -1.75 pts     21/21            0.9412            34.0% vs 22.9%
  2    15,977     15,910   +2.26 pts     21/21            0.9327            37.9% vs 22.9%
```

**21 of 21 concepts label-honest on all three seeds, 0 failing** - the first time this project has measured that. `centre_heavier` had scored **AUROC 0.3196 (below chance)** because its formula fired on **96.8%** of images; moving its threshold 0.05 -> **0.40** (support 96.9% -> 22.9%) took it to a **mean 0.9472** and made its firing track its support (28-38% vs 22.9%). The defect was in the *definition*, not the model.

**And the readable-vs-plain gap is NOT resolvable:** gaps -0.14 / -1.75 / +2.26 give **mean +0.12 points, sd 2.02, 95% interval [-3.11, +3.36]**, which includes zero. This puts the earlier single-seed claims in context: **-1.47 (width21), -1.46 (parity49) and -1.53 all sit inside this arm's seed spread.** A single-seed gap of -1.5 is an ordinary draw from a distribution averaging +0.12, so the readable model is **not demonstrably worse - and not demonstrably better**, and the mechanism questions asked about the "gap" were asked about a quantity with no resolvable sign. Settling it needs 8-12 seeds, where the SE of the mean falls to 0.6-0.7 points. Source: `lens/runs/FIX-CENTRE.txt`, `docs/history/ALL-CONCEPTS-HONEST.md`.

### Shift augmentation removes the 27-point collapse, at a measured cost of 4.4 points of clean accuracy (2026-11-18)

```
input                           no augmentation   2px shift aug    difference
clean                             84.47%            80.03%          -4.44
brightness +0.20                  77.63%            71.60%          -6.03
gaussian noise sd 0.10            83.63%            79.50%          -4.13
blur 3-tap binomial               83.20%            80.10%          -3.10
shift 1 px right                  76.93%            79.80%          +2.87
shift 1 px down                   78.33%            79.60%          +1.27
shift 2 px right                  57.00%            78.10%         +21.10
```

Matched control: **same parity config, same seed, no augmentation** (`fixc3_s0_ep24`, 15,977 params) against the augmented model (`shiftaug_s0_ep24`). **Bought:** under a 2-pixel shift the augmented model scores **78.10%** against the control's **57.00%** - a **+21.10-point** improvement, and the shift penalty falls from **-27.47 to -1.93 (14x)**. **Cost:** **4.44 points of clean accuracy**, and every other perturbation moves down 3.1-6.0 points. That is the honest shape of the trade: augmentation does not add robustness on top of accuracy, it **moves accuracy from pristine inputs to realistic ones**. One asymmetry: brightness +0.20 got **1.60 points worse**, because training on shifted copies teaches nothing about global intensity. The arm's own readable-vs-plain delta of **-1.01 at one seed** is **inside the sd 2.02 spread** the fixcentre arm just measured, so it is recorded and not interpreted. Source: `lens/runs/SHIFT-AUG.txt`, `lens/runs/vision_robustness.py` on `fixc3_s0_ep24.npz`, `docs/history/SHIFT-AUG-MATCHED.md`.

### 6000 steps: the plateau holds, and a silent no-op bug in the resume path (2026-11-18)

```
step  3,000   4,500   5,000   5,500   5,750   6,000      (104 chunks, SE 0.50)
acc   23.95%  25.35%  25.24%  25.76%  26.01%  25.89%
loss  6.3261  6.1303  6.1148  6.0826  6.0662  6.0640
```

**Original P1 (>= 24% at 6000) MET** at 25.89%. **Restated P1 (>= 26%) MISSED by 0.11 points** - recorded as missed, because a gate stated as a number and not reached is missed, and redefining it afterwards is what this project forbids. **Plateau band 4500-6000: 25.24-26.01%, a 0.77-point spread against a 0.50-point SE.** The 3000->4500 span gained **+1.40 points (2.8 SE)**; 4500->6000 gained **+0.54 points (1.1 SE)**, an interval including zero. **Conclusion: at radius 0.9 with a 32-token window this recipe plateaus at ~25.5-26% after ~4,500 steps (~9M tokens).** Not data exhaustion - one epoch is 294M tokens and 12.3M were seen, ~4% of a pass.

**Bug found and fixed in the same wake:** `STEPS` was an absolute end step, so the queued memory arm (`RESUME` at step 3000, `STEPS=1500`) built `range(3001, 1501)` - an **empty loop**. It would have loaded the resumed weights, trained **zero** steps, and printed held-out numbers belonging to the model it started from, with a `COMPLETE` line. A larger `START_STEP` would also have produced a **negative learning rate**. Now `STEPS` means **additional** steps when resuming, both the loop and the schedule are relative to the resume point, a fresh run is unchanged (`steps=2 (from 0 to 2)`), a resume reports `steps=1500 (from 3000 to 4500)`, and a guard **refuses to run** if the resume point is at or past the end. Source: `lens/scripts/train_scan9.py`, `docs/history/PLATEAU-AND-RESUME-BUG.md`.

### Training length DID teach the model to use context: ~2 tokens -> ~16 (2026-11-18)

```
L       6000-step    old 1500-step        (1,500 chunks per length, SE ~1.05)
1        13.07%          9.27%
2        20.33%         16.73%
16       24.13%         17.40%
32       22.47%         16.73%
128      22.07%         16.40%
```

**The 6000-step model gains +3.80 points going from 2 to 16 tokens of context (2.5 SE, RESOLVABLE). The 1500-step model gained +0.67 points (0.5 SE, nothing)** - and that null result is what produced the published "effective context = 2 tokens". So **that figure was true of the model it was measured on and is false as an architectural claim**: for `scan9_ep1500` it is 2 tokens, for `scan12_r090_6k_ep6000` it is **at least 16** in a 32-token window, with no resolvable gain past that (L=32 reads 22.47% and L=128 22.07%, both below L=16's 24.13% at -1.1 and -1.9 SE, so the dips are noise and the honest reading is a plateau, not a peak). **Training length fixed something the accuracy number badly understates** - the model learned to use context it previously ignored - and it bounds the ceiling: more steps stopped helping while context use was still improving, so the remaining limit is **capacity or mechanism, not the schedule and not the amount of text**. `effective_context_tokens` must be **per-model**, since it changes with training. Source: `tools/context_curve.py` on `scan12_r090_6k_ep6000.npz`, `docs/history/CONTEXT-DEPENDS-ON-TRAINING.md`.

### The activation dial is characterised: pass count peaks at lam-act 1.0-1.5, then falls (2026-11-18)

```
lam-act   pass     mean AUROC   mean fire   failing concepts
0.0      30/36      0.6886        47.4%     is_a_list, mentions_a_name, mentions_a_quantity, mentions_body, mentions_family, mentions_sport
0.3      32/36      0.7038        36.2%     mentions_a_name, mentions_a_quantity, mentions_body, mentions_sport
1.0      34/36      0.7370        20.1%     mentions_body, mentions_sport
1.5      34/36      0.7573        14.2%     mentions_body, mentions_sport
2.0      32/36      0.7767        10.5%     mentions_a_quantity, mentions_body, mentions_sport, mentions_time
```

**AUROC rises monotonically across the whole range, but the pass count peaks and then falls** (30 -> 32 -> 34 -> 34 -> 32). That is the mechanism the arm predicted before it ran: **uniform pressure that keeps climbing stops helping the rare concepts and starts suppressing the common ones** - at 2.0, `mentions_a_quantity` (**13.9% support**, not rare) and `mentions_time` drop out because the pressure is so strong a genuinely-present concept cannot make its neuron fire. Scoring: **P1 HELD** (AUROC 0.7767 >= 0.76), **P2 FAILED** (two new failures at 2.0), **P3 FAILED** (34/36 at 1.5, 32/36 at 2.0), falsifier did not fire. **Honest operating point: lam-act 1.0-1.5, 34/36 (94.4%)**; 1.5 dominates 1.0 because it reaches the same pass count with a higher AUROC (0.7573 vs 0.7370) and lower fire rate (14.2% vs 20.1%). **Uniform activation pressure is finished as a mechanism** - it bought 30/36 -> 34/36 and then began costing what it had bought. Next: `TALK_BOOST` per-concept positive weighting aimed only at `mentions_body` (1.32%) and `mentions_sport` (1.43%), with the doctrine's fallback pre-registered - if they still fail at 3x, they are **DROPPED, never forced**, and the report becomes 34/34 = 100% honest. Source: `lens/runs/CONCEPT-ACT.txt`, `lens/runs/CONCEPT-ACT2.txt`, `docs/history/ACT-DIAL-CHARACTERISED.md`.

### A zero-initialised component is a no-op in the FORWARD pass, not in the training trajectory (2026-11-18)

The memory-channel arm's safety claim - "with `mem_Wo = 0`, step 0 reproduces the resumed model exactly" - was testable for the first time in the live run, and it is **true of the forward pass and false of the training run**. What is exactly true: **loss 9.396832162837e+00 with and without the channel at step 0 (lr=0), and every pre-existing gradient bit-identical**. What is not: the optimiser clips by the **global** gradient norm, and the new parameters sit inside it - **norm 6.0425e-01 off vs 6.2892e-01 on (+4.08%)**, giving a clip scale of 1.654949 vs 1.590031, so the **first update is 3.92% shorter for every shared parameter**. Visible at matched schedule: `channel off loss 6.1269 / acc 17.43% / gnorm 2.25` against `channel on loss 6.1375 / acc 17.27% / gnorm 3.44`. The bigger mechanism is by design: `mem_Wo` starts at zero but **receives a 4.870e-03 gradient on step 1**, so from step 2 the forward genuinely differs. **Refined rule: a component that must be a no-op for the whole run needs its output-projection GRADIENT masked, not just its weights zeroed.** Source: `tools/scan_identity_check.py`, `lens/runs/IDENTITY-CONTROL2.txt`, `docs/history/NO-OP-SCOPE.md`.

### The 8-slot memory bank was one slot in disguise, provably (2026-11-18)

Early read of the memory-channel arm, 500 steps in: **k=16 memory 1.605%** (resumed baseline 1.312%, +0.293 points against a 10% gate), **held-out 21.03% vs 21.03%** (no accuracy cost), **loss 6.3199 vs 6.4417** (better). P1 met trivially, P2 holding. Then P3 exposed the defect:

```
mem_decay: 0.8580580949783325  0.8580580949783325  0.8580580949783325 ...  (all 8 identical, bit for bit)
```

**The slots can never differentiate**, and it is exact symmetry rather than convergence: `M` starts at **zero** for every slot, so every slot scores `q.M = 0`, the softmax over slots is **uniform**, every slot receives the **identical** write, and `d_decay[k]` is therefore identical for every `k`, forever. **The 8-slot bank is mathematically ONE slot with 8x the write rate** - which is what a memory gain of 0.293 points looks like. Fix: `MEM_SYM=1` initialises per-slot decay from `uniform(0.85, 0.95)` (from all-0.9), `MEM_SYM=0` restores the old behaviour bit-for-bit. `run_memchan_sym.sh` re-runs the identical recipe with that one change: **P1** max-min across decays **> 0.005** (symmetric arm scored 0.000000), **P2** k=16 above 1.605%, **P3** no accuracy cost; **falsifier** - decays still identical or memory at/below 1.605%, which would mean the limiter is the **training signal** (a 32-token window cannot teach what must survive 16 steps), not the bank. Two real bugs fixed while editing: `Mm` possibly unbound in the held-out path, and `ema` being `None` in the final log line (any zero-step run would have crashed) - verified by a complete 2-step smoke run. Source: `lens/runs/MEMCHAN-ARM.txt`, `docs/history/MEMBANK-SYMMETRY.md`.

### Memory channel: all three predictions met, the gate 3.3% of the way, and the reason proven (2026-11-18)

```
                            channel arm    resumed baseline    change
held-out teacher-forced       24.30%          23.95%           +0.35 pts  (~0.5 SE, NOT resolvable)
held-out loss                 6.1754          6.3261           -0.15 nats
free-running                   6.06%           6.88%           -0.82 pts  (NOT resolvable)
repetition                     0.875           0.923           -0.048     (less repetitive)
k=16 memory                    1.642%          1.312%          +0.330 pts (gate is 10%)
```

1500 steps from `scan11_r090_ep3000`, scored at the corrected 104-chunk protocol. **P1 MET** (memory 1.642% > 1.322%), **P2 MET** (held-out 24.30% > 21.02%, and above the baseline's own 23.95% though not resolvably), **P3 MET** (decay moved off 0.9), **falsifier did not fire**. The channel is installed, gradient-certified, a proven forward no-op at injection, and it moves every number the right way: accuracy +0.35 (inside noise), **loss -0.15 nats**, **repetition -0.048**, memory **+0.33 points** - which is **3.3% of the way to the 10% gate**. **And P3 exposed why:** all eight slots read the *same* decay to the bit (`0.8580580949783325`), because `M` starts at zero, every slot scores `q.M = 0`, the softmax is uniform, every slot receives the identical write, so `d_decay[k]` is identical forever. **The 8-slot bank was mathematically one slot with 8x the write rate** - exactly what +0.33 points looks like, found by reading the weights instead of tuning. `MEM_SYM=1` now randomises the per-slot start and `run_memchan_sym.sh` re-runs the identical recipe with that one change. If that arm does not beat 1.605%, the limiter is the **training signal** (a 32-token window never asks the model to remember for 16 steps). Source: `lens/runs/MEMCHAN-ARM.txt`, `docs/history/MEMCHAN-ENDPOINT.md`.

### Re-warming the learning rate buys 1.31 points: the schedule was part of the plateau (2026-11-18)

```
step 6000 (plateau)   25.89%   loss 6.0640     <- flat band 4500-6000 was 25.24-26.01
step 6250             26.53%   loss 5.9681
step 6500             27.27%   loss 5.9166
step 6750             27.18%   loss 5.8918
step 7000             27.32%   loss 5.8783
step 7250             27.69%   loss 5.8653
step 7500             27.20%   loss 5.8903     <- ENDPOINT
```

By step 6000 the rate had decayed **0.03 -> 0.0030**, so the plateau was measured while the model took increasingly tiny steps. Resuming `scan12_r090_6k_ep6000` with the rate **re-warmed to 0.03** for 1500 steps, **one variable changed**: **P1 MET** (27.20% > 26.01% band top), **P2 MET** (loss 5.8903, -0.174 nats), **falsifier did not fire**, and **6 of 6** curve points sit above the old band. The effect is **+1.31 points = 1.85 SE** - just under the 2-SE bar on a single comparison - so the honest statement is that the schedule explains **part** of the plateau, worth **1.3-1.8 points**, not all of it. The shape matters more than the size: accuracy climbed for 500 steps then **flattened at ~27.3% while loss kept falling**, the same confidence-without-accuracy pattern at a higher level, so a re-warm is a **one-time lift, not a new regime**. **This exposed an n=1 problem:** the trainer hardcoded **both** `SD.init(..., seed=0)` and `default_rng(0)`, so a second independent draw was impossible. Both now follow **`SEED`** (default 0, existing arms bit-identical, recorded in checkpoint metadata), and `run_rewarm_seed1.sh` re-runs at `SEED=1` with its falsifier pre-registered: **if seed 1 stays inside 25.24-26.01%, this claim is withdrawn and the plateau stands as capacity-limited.** Source: `lens/runs/LR-REWARM.txt`, `docs/history/LR-REWARM.md`.

### 36 of 36 concepts pass the label-honesty gate - first time (2026-11-18)

```
                                   lam-act 1.5    lam-act 1.5 + boost x3 on the two
readable-only pass                  34/36               36/36      <- zero failing
mean AUROC                          0.7573              0.7703
mean fire rate                      14.2%               15.6%
failing in readable only      mentions_body,      none
                                mentions_sport
```

The dial had stalled at 34/36 and raising it further (lam-act 2.0) started failing *common* concepts, so this arm aimed the pressure at the two failures only - `TALK_BOOST="mentions_body:3,mentions_sport:3"` at the best uniform point. **P1 MET** (36/36, zero failing), **P2 MET** (failing set shrank to empty, mean AUROC 0.7573 -> 0.7703), **P3 MET in direction** (the two now fire on 62.3% and 47.5% of windows), **falsifier did not fire** - so per-concept positive weighting **is** the mechanism and the doctrine's 'drop the concept' fallback was not needed. **This is the first time zero concepts fail.**

**What it does not say:** the gate is AUROC >= 0.55 AND firing >= 2% - a minimum firing floor with **no precision requirement**. A **calibration check was attempted twice this wake and both attempts were wrong**: comparing firing against token-level support gave a fake **43.7x over-firing**, and comparing against `1-(1-p)^32` flipped it to **29 of 36 under-firing, 0 over-firing**. Neither is the right denominator - the neuron is scored at the scored position, not 'anywhere in the window' - and the audit's own per-window target vector is the only correct basis, which the script does not expose. **So no calibration claim is made**, and the script now prints that instead of a number. Stronger label-honesty - a neuron that fires *because* its concept is true - remains **unmeasured** and is the honest next question. Source: `lens/runs/BOOST-RARE.txt`, `docs/history/ALL-36-PASS.md`.

### QUALIFICATION: 36/36 measures RANKING, not detection - median F1 is 0.18 (2026-11-18)

```
                                              boosted readable model, 800 windows
median F1 at the shipped 0.5 cut                       0.081
median BEST achievable F1 (any threshold)              0.181
concepts with precision >= 0.5 AND recall >= 0.5
   at the shipped 0.5 cut                               0 of 36
   at each concept's own best cut                       3 of 36
median positives per concept                       32 of 800 windows
```

The previous entry's **36/36** is correct on its **stated** gate and it is **not** the claim a reader would take from it, so it is qualified here. `lens/api.py#audit` now computes **precision and recall from the same score and target vectors** it already used for AUROC (the only correct basis, after two failed attempts against token-level support), and `tools/concept_pr_curve.py` sweeps the firing threshold per concept. **The gate asks a weak question**: AUROC >= 0.55 with a firing floor asks whether a neuron *ranks* positives above negatives, which it can do while being a poor detector. **Detection is weak**: even after per-concept calibration the median achievable **F1 is 0.181**, and only **3 of 36** concepts reach both precision and recall above 0.5. A neuron named `mentions_sport` that fires mostly where sport is not mentioned does not mean what its label says, however good its ranking. **So the accurate sentence is: 36 of 36 pass the AUROC-0.55 ranking gate; as detectors their median best F1 is 0.18 and 3 of 36 reach precision and recall above 0.5.** "All concepts are label-honest" is **not supported**. The next question is not *which concept is dishonest* but **how to make a concept neuron a detector rather than a ranker** - a loss with precision in it, or a readout that reports PR at a chosen operating point. Both are testable. Source: `tools/concept_pr_curve.py`, `lens/api.py`, `docs/history/RANKING-NOT-DETECTION.md`.

### Breaking the memory bank's symmetry pays, but late (2026-11-18)

The 8-slot bank had a proven defect - every slot ended with a **bit-identical** decay, so it was one slot with eight times the write rate. Per-slot decay init (`MEM_SYM=1`) fixes it. **The benefit is delayed, which is why one checkpoint would have given the wrong answer:**

```
                                  symmetric      symmetry-broken
slot decay spread (max-min)       0.000000          0.049954   at step 3250
                                                    0.064555   at step 3750
k=16 memory, step ~3250             1.550%             1.512%   <- no benefit yet
k=16 memory, step ~3750             1.605%             2.254%   <- +0.649 pts, +40% relative
held-out teacher-forced            21.03%             21.16%
loss                              6.3199             6.3322
```

At step 3250 the differentiated bank was marginally **worse**; by 3750 it is clearly better, **+0.649 points at +40% relative**, with held-out slightly up. So the symmetry defect was **suppressing** memory and the slots keep diverging (0.0500 -> 0.0646). **But the memory gate is 10% and the best measurement is 2.254%, so the gate still FAILS by 4.4x** - symmetry was a real but small part of the gap. Caveat: ep3750 is compared against the symmetric arm's ep3500 because that arm stopped there, so 250 steps of training length sit inside the gain; the clean number is the endpoint at 4500. Two real defects have now been fixed in this mechanism (symmetry collapse, missing `dM` gradient path) and k=16 moved **1.312% -> 1.605% -> 2.254%** with the gate unmoved - the signature of a mechanism that is right but **starved** of training signal, exactly what the pre-registered falsifier for this arm predicted. Source: `lens/runs/MEMCHAN-SYM.txt`, `lens/runs/MEMCHAN-SYM-RESUME.txt`, `docs/history/MEMORY-SYMMETRY-PAYS-LATE.md`.

### Memory bank endpoint: the symmetry fix is real, durable, small, and not free (2026-11-18)

```
                        symmetric bank     symmetry-broken bank
                          (step 3500)          (step 4500)
k=16 memory                 1.605%               2.214%        +0.609 pts, +38% relative
k=32 memory                 0.093%               0.112%
held-out teacher-forced     24.46%               24.65%        +0.19 pts   (104 chunks, each trainer's own eval)
held-out loss              9.1529               9.0575
slot decay spread          0.000000             0.068527       (8 of 8 distinct)
```

**P1 MET** (spread **0.068527**, 8 of 8 distinct, still growing 0.0500 -> 0.0646 -> 0.0685, so the slots never collapse back). **P2 MET** (k=16 **2.214%**, **+0.609 points, +38% relative**). **P3 NOT met in direction**: held-out **20.37% vs 21.03%**, **-0.66 points** - about 1 SE against the 0.60-point same-step spread measured here, so not resolvable, but on the wrong side and stated as such. Falsifier did not fire. **The trend is the finding:** step 3250 **1.512%** (the broken bank was *worse* there), 3750 **2.254%**, 4000 **2.164%**, 4500 **2.214%** - delayed, then flat. A single early checkpoint would have concluded the fix did nothing. **Three fixes have now paid and the gate is still 4.5x away:**
```
radius-only recurrent state        1.312%
+ symmetric 8-slot bank            1.605%   (+0.293)
+ broken symmetry                  2.214%   (+0.609)
gate                              10.000%   still 4.5x away
```

**The gate is not reachable by fixing the bank.** Each defect was real, each fix paid +0.3 to +0.6 points, and the required move is +7.8 - the pre-registered falsifier's branch, now measured rather than predicted: **the limiter is the training signal, not the mechanism.** A 32-token window never asks the model to retain anything for 16 steps, so a better bank mostly stores better noise. The next attempt must change what the model is **asked to do** before touching the bank again. Source: `lens/runs/MEMCHAN-SYM.txt`, `lens/runs/MEMCHAN-SYM-RESUME.txt`, `docs/history/MEMORY-BANK-ENDPOINT.md`.

### Re-warm, two draws: +5.21 and +5.51 points on the reliable protocol - and my 12-chunk probe is withdrawn from endpoint decisions (2026-11-18)

The second draw finished, and comparing it to seed 0 exposed a **conflict between two instruments on identical checkpoints**:

```
                                       12-chunk probe   104-chunk (each trainer's own HELD-OUT)
base, 6000 steps                          25.89%              21.74%     (loss 9.4149)
seed 0, +1500 steps at lr 0.03            27.20%              26.95%     (loss 9.3506)
seed 1b, +1500 steps, new data order      24.47%              27.25%     (loss 9.3389)
gain, seed 0                               +1.31              +5.21
gain, seed 1b                                  -              +5.51
do the two draws agree within 2 points?   NO (-2.73)         YES (+0.30)
```

**They disagree in sign.** The 12-chunk probe scores 756 samples (SE ~1.5) against the trainer's 6,656 (SE ~0.55), and it has now produced **two wrong-signed verdicts in one day**: it said the symmetry-broken bank **cost** 0.66 accuracy points when the 104-chunk numbers show it **gained** 0.19, and it says these two draws disagree by 2.73 when they agree within 0.30. An instrument that flips the sign twice is **unfit for endpoint decisions**, so the verdict rests on the 104-chunk evaluation - same code path for every arm, 8.8x the samples.

**Claimed:** on that protocol the re-warm lifts held-out from **21.74% to 26.95% (seed 0, +5.21)** and **27.25% (seed 1b, +5.51)**, the two draws agreeing to **0.30 points**; the pre-registered falsifier does **not** fire. **Withdrawn:** the earlier **+1.31 points at 1.85 SE**, which understates the effect ~4x because it came from the 12-chunk probe, and with it the framing that this result is marginal. **New rule:** every arm report states which protocol produced each number, and endpoints are decided on the trainer's 104-chunk evaluation; any 12-chunk conclusion must be re-measured at 104 before publication. Source: `lens/runs/LR-REWARM.txt`, `lens/runs/LR-REWARM-S1-NEWSHUFFLE.txt`, `lens/runs/SCAN12-R090-6K.txt:135`, `docs/history/REWARM-TWO-DRAWS.md`.

### The memory bank CAN hold 16 steps - the gate was never about capacity (2026-11-18)

The gate (k=16 recall above 10% on real text) failed after every mechanism fix - **1.312% -> 1.605% -> 2.214%**, gate unmoved. This line's pre-registered falsifier said the limiter was **demand**, not capacity. That is now measured on a synthetic task with exactly that demand (fact at position 0, filler, must output it at position D), using the certified memory module with **one bank per sample**. The probe's own backward pass was **certified first**: worst relative error **6.789e-08** against central finite differences across `Wo`, `Wq`, `Wk`, `decay`, embeddings and readout.

```
delay   bank arm   control arm
  1      100.00%       3.12%
  4      100.00%       3.32%
  8      100.00%       3.03%
 16       81.25%       3.22%      <- FULL budget (12000 steps)
chance     3.12%      (vocab 32)
```

The **control** (same readout, memory removed, sees only the final token) sits **exactly at chance**, so the result is attributable to the bank and not to a leak in the wiring. **The falsifier did not fire: the bank can hold a fact for 16 steps when a task demands it.** So the k=16 gate on real text fails **not because the bank cannot hold, but because ordinary text never asks it to** - capacity was never the binding constraint, the **training signal** is. Three defects were found and fixed (each paid +0.293, +0.609) and none of them was the wall.

**Honest limits:** the synthetic task is much easier than text (vocab 32, one deterministic fact, single-token recall) - it bounds what the bank *can* represent, not what it will reach on natural text at this scale; delay 16 is now **confirmed at the full 12000-step budget: 81.25% with its control at 3.22%**, agreeing with the half-budget read (81.93%) within 0.68 points, so 16-step recall **plateaus near 81%** rather than reaching the 100% seen at 1, 4 and 8. The next experiment is not another bank fix but a training objective with real cross-step demand (`TGT_DELAY`: target at position t becomes `tok[t+1-16]`), in flight now, with its own falsifier. Source: `tools/delayed_recall_probe.py`, `tools/delayed_recall_gradcheck.py`, `lens/runs/DELAYED-RECALL-16.txt`, `docs/history/MEMORY-CAPACITY-NOT-THE-LIMITER.md`.

### 16-step demand on real text: the bank converts it (2026-11-18)

Two arms from the same base checkpoint, both 1000 steps with the target at position t being `tok[t+1-16]`, differing in exactly one thing - the memory bank:

```
                          delayed acc   next-token   same-sample constant
td16   (demand + bank)       8.67%       16.54%        6.59%
td16nb (demand, no bank)     5.22%       14.32%        6.59%
mem8sym (no demand, bank)    7.02%       23.50%        6.59%
6552 scored positions, 104 held-out chunks
```

**The bank arm is above a trivial predictor - 8.67% vs 6.59%, +2.08 points, ~6 SE - while the bankless arm is below it** (5.22%). So the bank earns its keep: **+3.45 points (~7 SE)**. Demand itself adds **+1.65 points (~3 SE)** on top of the bank. The pre-registered falsifier did **not** fire on the valid measure.

**An instrument error hid this for several wakes and is recorded here:** the k=16 probe scores **next-token** logits, and these arms were trained to emit `tok[t+1-16]`. `td16` trades next-token accuracy away (16.54% vs 23.50%), so a next-token probe reads ~1.1% while the arm's own task reads 8.67%. `tools/eval_delayed.py` now scores the offset the arm was trained on, against a **same-sample** constant baseline rather than chance (0.008%, which would make 5.22% look like success). **What this is not:** 'memory solved' - 8.67% is small, the 10% gate is scored by the next-token probe and therefore does not apply unchanged to these arms, and each arm is one seed, 1000 steps. Source: `tools/eval_delayed.py`, `lens/runs/TGT-DELAY16.txt`, `lens/runs/TGT-DELAY16-NOBANK.txt`, `docs/history/DEMAND-MAKES-MEMORY.md`.

### CORRECTION: the demand result survives, its interval was overstated (2026-11-18)

The entry above quotes **+2.08 points, ~6 SE**. That SE treated 6552 scored positions as independent; the 64 positions inside a window share state, so the resampling unit is the **window** and there are only 104. Window-level bootstrap (10,000 resamples, `tools/eval_delayed_ci.py`):

```
vs FAIR constant (most frequent corpus token, 6.59%):
  td16   (bank)      8.67%   +2.08 pts   95% CI [+0.93, +3.27]   RESOLVABLE above
  td16nb (no bank)   5.22%   -1.37 pts   95% CI [-2.24, -0.44]   RESOLVABLE below
vs WINDOW-ORACLE constant (per-window most frequent target, 14.41%):
  td16               8.67%   -5.74 pts   95% CI [-6.75, -4.81]   RESOLVABLE below
  td16nb             5.22%   -9.19 pts   95% CI [-10.30, -8.06]  RESOLVABLE below
bank vs no-bank:    +3.45 pts   95% CI [+2.43, +4.43]            RESOLVABLE
```

**Stands:** the bank arm is above the fair constant predictor, the bankless arm is below it, and the bank is worth **+3.45 points** with an interval excluding zero. **Wrong:** "~6 SE" - the true spread is about **3.5 SE**, because the effective sample size is 104 windows, not 6552 positions; the margin was inflated ~1.7x. **Was missing and cuts against the result:** a **window-oracle** constant scores **14.41%** - it sees the future so it is not a fair competitor, but against it both arms are far below (the banked arm by 5.74 points). So "the bank gives measurable 16-step recall" is true against a fair baseline while "this recall is far weaker than the per-window target concentration allows" is also true. Source: `tools/eval_delayed_ci.py`, `docs/history/DEMAND-MAKES-MEMORY.md`.

### The missing "before" numbers: demand takes 16-step recall from below trivial to above it (2026-11-18)

Every earlier comparison measured the arms against each other and against constants - **never against the starting point.** The same base checkpoint, no demand training:

```
base checkpoint, 104 chunks, 6552 scored:
  at delay 16:   5.83%   vs fair constant 6.59%   <- BELOW a trivial predictor
  at delay  4:  11.65%   vs fair constant 7.13%   <- well ABOVE it
```

**1. The real effect size is +2.84 points, 5.83% -> 8.67%.** One thousand steps of 16-step demand took the model from **worse than a constant predictor to better than one**, and the no-bank control stayed at 5.22% (still worse) - so the bank is what makes the difference. **2. The architecture already handles short delays**: at delay 4 the base scores **11.65%**, higher than the delay-16 arm reaches after training, consistent with this model's measured effective context of about 2-4 tokens. **3. My own falsifier for the delay-4 arm was framed against the wrong number** - it said "at or below 8.67% means no dose-response", but that arm's correct bar is **its own starting point, 11.65%**; a delay-4 arm that merely reproduces the base has demonstrated nothing. The corrected bar is recorded and will be applied when its endpoint lands. Current honest summary: **demand lifts 16-step recall from 5.83% (below a fair constant) to 8.67% (above it), the bank is worth +3.45 points, short delays were already handled, and all of it remains below a window-informed constant at 14.41%.** Source: `tools/eval_delayed.py`, `tools/eval_delayed_ci.py`, `docs/history/DEMAND-MAKES-MEMORY.md`.

### Dose-response at 250 steps: delay 4 goes from below a trivial predictor to above it (2026-11-18)

Measured against its **own starting point** - the bar written down before this result existed - with the window as the resampling unit:

```
delay 4, per-window constant = 14.68%
  scan12_r090_6k_ep6000 (base)   11.65%   -3.04 pts  CI [-4.20, -2.01]   BELOW trivial
  td4_r090_ep6250 (250 steps)    19.57%   +4.88 pts  CI [+3.54, +6.03]   ABOVE trivial
  arm vs its own base                     +7.92 pts  CI [+7.20, +8.64]   RESOLVABLE
```

**The gain is +7.92 points and it is not marginal** - the arm crosses from 3.04 points below a constant predictor to 4.88 points above it in 250 steps. Compare delay 16: base 5.83% -> arm 8.67%, **+2.84 points**. **So the gain is larger at the SHORT delay**, the opposite of what was predicted: demand training helps most where the base model was already capable, and least at the long delay that motivated the experiment.

**Why this is not the win it looks like:** at a 4-step delay the target is 4 tokens back and was in the model's recent input, so much of 19.57% is recent-token carry rather than memory - this model's measured effective context is only ~2-4 tokens. The result says the shifted-target objective can **reshape the readout where the information is already present**, which is narrower than "memory works". **Caveats:** 250 of 1000 steps, one checkpoint, one seed, one arm per delay - the endpoint decides. Source: `tools/eval_delayed_ci.py`, `docs/history/DEMAND-MAKES-MEMORY.md`.

### WITHDRAWN: "demand training creates 16-step memory" - refuted by a frequency split (2026-11-18)

**The interpretation behind three entries today does not survive a control that splits accuracy by target-token frequency, and it is withdrawn.** Frequency ranks computed on the training region only:

```
delay 4    (250 steps)     top-20    r21-500    tail (888)
  td4 arm                  35.16%     4.83%      0.00%
  base                     19.61%     4.75%      0.00%
  same-window constant     24.87%     5.56%      0.56%

delay 16   (1000 steps)    top-20    r21-500    tail (892)
  td16 arm                 16.15%     1.60%      0.00%
  base                     10.09%     2.14%      0.00%
  same-window constant     23.65%     6.85%      0.56%
```

**All the gain is in the frequent band** (+15.55 points on top-20 at delay 4, +0.08 at ranks 21-500, **0.00** on the tail), and **rare-token recall is exactly zero at both delays** - 0 correct out of 888 and 892 positions against a constant that gets 0.56%, where a single hit would show as 0.11%. At delay 16 the arm is below a per-window constant predictor in **every band**. So the delayed-target objective changed **how probability mass is distributed across frequent tokens** - a readout/prior shift - and did **not** create token-specific memory. The earlier "above a fair constant predictor" framing was measuring that prior shift.

**Why it looked positive:** the comparisons were **pooled**, and pooling is dominated by the top-20 tokens - 3325 of 6552 scored positions. The band where memory would have to show sat at zero the whole time. A pooled comparison hides exactly the failure this project exists to catch.

**What still stands:** the memory module holds a fact for 16 steps *in isolation* (81.25%, control at chance, gradient certified 6.789e-08, one bank per sample) - a different measurement with a different control, untouched by this. The bank's +3.45 accuracy points are real as an accuracy difference but sit in the same band, so they are not memory evidence either. **New rule:** a pooled accuracy gain is not evidence of the capability the task is named after; any recall claim is split by target frequency first, because the frequent band alone can carry a pooled number. This control should have run before the first publication, not after three. Source: `tools/eval_delayed_split.py`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### Dose-response endpoint: 47.30% pooled, still ZERO rare-token recall (2026-11-18)

The delay-4 arm finished 1000 steps. Frequency-split, ranks computed on the training region only:

```
delay 4 endpoint        top-20    r21-500    tail (888)    pooled
  td4_r090_ep7000       81.74%     16.29%      0.00%       47.30%
  base (no demand)      19.61%      4.75%      0.00%       11.65%
  same-window constant  24.87%      5.56%      0.56%        7.13%
```

**The pooled number quadrupled, 11.65% -> 47.30%, and rare-token recall is still exactly zero** - 0 of 888 positions, at every checkpoint this arm produced (250, 500 and 1000 steps); one correct rare token would have read 0.11%. **Top-20 81.74% against a 24.87% constant**, and **ranks 21-500 16.29% against 5.56%** (a genuine 2.9x, so not pure frequency) - but on the **tail the model is worse than a constant** (0.00% vs 0.56%), because a constant at least guesses one occasionally.

So the arm learned the local statistics of tokens it can afford to guess, with no ability to retrieve *which* rare token occurred four steps back - a target that was sitting in its input four tokens earlier, which a copy mechanism would catch some of. **The line's conclusion now has a mechanism:** the synthetic probe proved the memory bank holds a fact for 16 steps (81.25%, control at chance, gradient certified), while the real model trained with genuine delayed demand still cannot retrieve a rare token across 4 steps. The gap is neither the bank nor the objective - it is the base model's ability to represent token identity in its recurrent state at 4.7M parameters. **Both levers this line could pull have now been pulled and measured, and neither moved rare-token recall off zero.** This is also the strongest test yet of the rule published a wake earlier: a 47.30% pooled score with 0/888 rare-token recall would have read as a success if not split by frequency. Source: `tools/eval_delayed_split.py`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### Why the memory-module result survives the frequency test that killed the others (2026-11-18)

The real-text delayed numbers were withdrawn because all their gain sat in the frequent band with **0 rare-token recalls of 888-892**. The surviving result - the 8-slot bank holding a fact for 16 steps in the synthetic probe (**81.25%**) - is immune, and that is now measured:

```
per-token share     min 0.0298   max 0.0328   (uniform = 1/32 = 0.03125)
chi-square vs uniform   30.5    df 31    5% critical ~43.8   -> consistent with uniform
```

**There is no frequent band to exploit** - with only 32 tokens all equally likely, a prior over frequent tokens is a uniform prior. **The best constant predictor scores exactly 1/32 = 3.125%**, and the control arm (same readout, memory removed) measures **3.03-3.22%** across delays 1, 4, 8 and 16 - exactly where a constant must sit. So the bank's **81.25% at 16 steps is 26x chance with no frequency structure available to inflate it**: it is carrying token identity. This does not improve the real-text result, which stays withdrawn; it establishes that the module capability is real, which is what makes the real-text failure informative - the bank can hold a token for 16 steps and the real model cannot, so the limiter is the base model, not the memory mechanism. Source: `tools/delayed_recall_probe.py`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### WITHDRAWN: extended training at a 16-step delay - the "erosion" claim and its z=3.35 (2026-11-18)

**RETRACTED.** This section claimed a resolvable erosion with z = 3.35. It is withdrawn for two defects, both mine: the test treated 2379 """positions""" as independent when 64 positions share a window's recurrent state and bank, and the low-N band figure came from a draw so small it is unusable - **not** by bias, as I first wrote, but by over-dispersion: the evaluator's positions are i.i.d. uniform, so the estimate is unbiased yet hits cluster inside a minority of windows (same checkpoint, ranks 21-500: first 104 windows 38 hits / 0.365 per window, remaining 936 windows 62 hits / 0.066 per window = 18.1% of what the first-104 rate predicted). See the correction entry below. Corrected high-N numbers below. Original text kept for the record:

```
delay 16, ranks 21-500 band, n = 2379 positions per checkpoint
  td16  ep7000    1.60%   = 38 hits
  td16x ep7500    1.47%   = 35 hits
  td16x ep8250    0.59%   = 14 hits

ep7000 vs ep8250:  1.60% vs 0.59%   z = 3.35   95% CI [+0.42, +1.60] points   RESOLVABLE decay
```

So 1,250 further steps of the same delayed-target objective **actively removed about a point of delayed ability from the middle-frequency band**, while the frequent band stayed flat (16.15% -> 15.60% -> 15.79%) and the tail stayed at **0 of 892**. That is not "needs more training": the objective converges onto the frequency prior and away from whatever partial delayed information it briefly held.

**Bound I am not overstating:** the tail result is `0 of 892`, which only implies a true rate below **0.34%** - one in 300 rare targets could exist unseen. The high-N run (1,040 windows, ~8,900 tail positions) is in flight to tighten that to **0.034%**; until it lands, "zero" means "below 0.34%". Source: `tools/eval_delayed_split.py`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### WITHDRAWN: delay-16 extension endpoint - the "monotone erosion" reading (2026-11-18)

```
delay 16       POOLED   next-tok   top-20   ranks 21-500   TAIL(892)
ep7000          8.67%     16.54%   16.15%      1.60%        0.00%
ep7500                              15.60%      1.47%        0.00%
ep8250                              15.79%      0.59%        0.00%
ep8500          8.26%     11.86%   16.09%      0.55%        0.00%
```

The middle band declines **monotonically** across all four checkpoints (1.60 -> 1.47 -> 0.59 -> 0.55) while the frequent band is flat and the tail never moves off zero. The pooled number - the only one that had been improving - also falls at the endpoint (8.67 -> 8.26). **RETRACTED:** all four band figures in this section came from the 104-window protocol, which the seeded RNG makes a strict *prefix* of the 1040-window draw - and prefix windows are ~5x easier on the middle band (1.60% vs 0.43% on the same checkpoint). The monotone decline was measured on a biased instrument, so it is withdrawn. What still stands is only the tail: **no training length produces rare-token recall** (0 of 8727 at high N, true rate < 0.034%). The all-four-checkpoint re-measurement at 1040 windows is running. Source: `lens/runs/TD16-EXTEND.txt`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### High-N tail test: the zero is solid, and one published magnitude was overstated (2026-11-18)

The rare-token band had only ~890 positions per checkpoint, which bounds a zero at merely **< 0.34%**. Re-measured on **1,040 windows (10x)**:

```
delay 4, 8728 tail positions
  td4_r090_ep7000       top-20 82.37%    ranks 21-500 13.11%    TAIL 0.00%
  same-window constant  top-20 25.17%    ranks 21-500  3.89%    tail 0.38%

95% upper bound, 0 hits in 8728:  true rate < 0.034%
```

**P1 MET** (0 of 8728, bound 0.034%). **P2 PARTIALLY MET - and this is the correction:** top-20 reproduced (82.37% vs 81.74%), but the **middle band did not** - **13.11% at high N against the 16.29% published from low N**, a **3.18-point move**, outside the stability window I set in advance. So the published *"ranks 21-500 16.29% against a 5.56% constant"* is corrected to **13.11% against 3.89%**: the qualitative claim survives (still **3.4x its constant**, so real mid-frequency delayed learning exists) but the magnitude was **overstated by ~3 points** because 104 windows was too few for that band. Falsifier did not fire. **The lesson generalises:** only the *frequent* band was stable at 104 windows - anything derived from the middle or tail bands at that sample size is an estimate with ~3-point uncertainty and must carry that caveat or be re-measured at 10x. What now stands at the available resolution: frequent 82.37% vs 25.17% constant, middle 13.11% vs 3.89%, tail **0 of 8728 (true rate < 0.034%)**. Source: `lens/runs/TAIL-HIGHN.txt`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### Delay-16 at high N: the model is BELOW the constant predictor on every band (2026-11-18)

```
delay 16, td16_r090_ep7000, 1040 windows
  band                 model    same-window constant
  top-20               16.53%          24.94%
  ranks 21-500          0.43%           4.26%
  tail (>500)           0.00%           0.19%   (0 of 8727, true rate < 0.034%)
```

At the larger sample the delay-16 arm is **below** the trivial same-window constant on **every** band, the middle band by about **10x** (0.43% vs 4.26%). The earlier reading of "8.67% vs a 6.59% constant, so delay 16 slightly beats trivial" was propped up by **prefix-selected windows**: the evaluator draws from a seeded RNG, so 104 windows is a strict prefix of the 1040-window draw, and that first chunk is ~5x easier (1.60% vs 0.43% on the same checkpoint). **What survives:** the tail zero (0 of 8727, true rate < 0.034%) and the delay-4 frequent band (82.37% vs 25.17%, reproduced within 0.63 points). Source: `lens/runs/TAIL-HIGHN.txt`, `docs/history/DEMAND-MEMORY-WITHDRAWN.md`.

### Correction to my own retraction: it is over-dispersion, not prefix bias (2026-11-18)

Ten minutes after retracting the erosion claim I re-read the evaluator and found **my stated cause was wrong**. The positions are drawn **i.i.d. uniform** (`rr = np.random.default_rng(4242); i = rr.integers(...)`), so the first 104 windows are a valid uniform random subset - **not** a biased prefix. The arithmetic:

```
mid band (ranks 21-500), delay 16, td16_r090_ep7000, same checkpoint
  first 104 windows :  38 hits /  104 windows = 0.365 per window
  remaining 936     :  62 hits /  936 windows = 0.066 per window

if the first-104 rate had held, the remaining 936 windows would give 342 hits - they gave 62 (18.1%)
```

An unbiased draw cannot be 5x off by bias. It is off because **hits cluster inside a minority of windows**, so the effective sample size is the *window count*, not the *position count* - **over-dispersion**, the same window-level clustering already established for pooled numbers. **The retraction stands** (the erosion claim used independent-position arithmetic, and 104 windows is far too few for a band whose true rate is ~0.1%), but **the corrected language matters**: those figures are **not systematically optimistic**, they are unreliable by a factor of several in either direction. "The same checkpoint reads 4x differently at 10x sample" has two candidate causes - a biased instrument or an over-dispersed quantity - and they license different claims. Naming the wrong one turned a correct retraction into an incorrect accusation of my own tooling. **Check the draw before blaming the draw.** Source: `docs/history/DEMAND-MEMORY-WITHDRAWN.md`, `tools/eval_delayed_split.py`.

### BUG FOUND AND FIXED: every resume silently wiped the trained memory bank (2026-11-19)

The most consequential defect found so far, because it invalidates the **interpretation** of several arms rather than one number. The trainer restores weights from a checkpoint and then, further down, initialises the memory bank - so `mem_Wq`, `mem_Wk`, `mem_Wo` and `mem_decay` (**trained** parameters, the resumed checkpoint's `mem_Wo` has |sum| = 111.06) were reset to fresh random values, with `Wo` back to **zero**, while the log said "weights loaded".

```
step 8025          loss      ema      acc      gnorm    lr
original          4.7316   4.0703   44.12%   2.95   0.0116
offline replay    4.7316   3.9774   44.12%   2.95   0.0116
resumed (buggy)   4.7916   4.0391   42.43%   3.46   0.0116
resumed (FIXED)   4.7316   3.9774   44.12%   2.95   0.0116
```

Found by a resume disagreeing with its own checkpoint: the learning rate was proven equal and the weights were loaded, so an offline replay of the same 25 steps was run - it reproduced the **original** exactly, proving the checkpoint faithful and the *process* wrong. Fixing it turned up a **second** defect: guarding the init block on the set of restored tensors did nothing, because that set was built by iterating `p`, which does not yet contain `mem_*`; the "fixed" run produced numbers **identical to the buggy one**. The restore loop now iterates the checkpoint's keys. A resume is now a true continuation, and `ema` is saved and restored as well.

**Blast radius, 8 runner scripts.** `td16` and `td4` resumed from a **bankless** base, so a fresh bank was correct - those arms are clean. **`td16x` and `td4x` resumed from banked checkpoints and wiped the bank those arms had just trained for 7000 steps**, so the "extension" arms were not extensions: they were a new bank trained from scratch for 1500 steps on carried-over language weights. **The earlier conclusion "more delay-16 training erodes the middle band" is therefore confounded and is withdrawn pending the corrected re-run** (`td16xf_r090`, running now, predictions written before launch). Source: `docs/history/RESUME-WIPED-THE-BANK.md`, `lens/scripts/train_scan9.py`.

### The model is 1.3% computation and 93% lookup table (2026-11-19)

A parameter audit of the shipped checkpoint, measured rather than assumed:

```
parameter  what it is                size        share
Wa         answer table H x V        3,072,512   62.3%
emb        token embeddings V x D    1,536,256   31.1%
Wh         THE recurrent core           65,536    1.3%
Wx         input projection              32,768    0.7%
Wd         direct path                   16,384    0.3%
mem_*      memory bank                  196,616    4.0%
```

**Wh is the only tensor that mixes information between tokens: 1.3% of the model computes, 93.4% is a table mapping a state to a vocabulary entry.** That is why held-out accuracy stalls near 25% however long it trains - it is a shape problem, and no training length fixes a shape problem. The repair is quantified: tying the answer table to the token embeddings through a small H->D bottleneck (32,768 params) deletes the H x V table, taking the model from **4,735,970 to 1,684,224 parameters**, and the freed budget is worth **46 recurrent layers** of the current size. Both paths pass their finite-difference gates (untied 2.576e-10, tied 2.142e-10). Source: `docs/ARCHITECTURE-FLAWS.md`.

### tanh saturation DISMISSED by measurement; the real memory limiter found (2026-11-19)

The audit listed tanh saturation as a flaw. Measured over 2,560 positions on a trained checkpoint: **0.00%** of preactivations exceed 2.0 and **0.00%** exceed 3.0, mean |preactivation| 0.248. **Saturation is not real and the activation is untouched.**

The same probe found the actual limiter:

```
Wh spectral radius 0.9000 | largest singular value 2.1763 | mean singular value 0.6153
memory operator diag(1-tanh^2) @ Wh : mean singular value 0.5613
0.5613^16 = 0.000097  ->  0.01% retention at 16 steps
```

The operator carrying information from one step to the next has a **mean singular value of 0.5613**, so the average direction loses **44% of its perturbation every step**. Wh's largest singular value is **2.1763** against a spectral radius of 0.9000 - strongly non-normal, a few directions explode while the average shrinks. **This is the mechanism behind the whole memory line failing**: not saturation, capacity or training length, but a recurrence that physically cannot carry a fact 16 steps. It also explains why clamping the radius to 0.9 helped the original forgetting while still leaving the 16-step gate at 4.73%: radius governs the largest eigenvalue, retention depends on the average of the singular values. Repair: drive the **mean** toward 1.0 - measured achievable at **0.9500** with the largest capped at 2.0. Source: `tools/saturation_probe.py`, `docs/ARCHITECTURE-FLAWS.md`.

### Tied readout: 64% fewer parameters, no measurable loss (2026-11-19)

The audit said the H x V answer table was 62.3% of the model. Head-to-head, same trainer, same seed, same data order, same 600 steps, one variable:

```
arm        params        ep600 HELD-OUT        ep500 overall   ep500 top-20   train time
untied     4,735,970     5.27% (loss 6.3339)       5.10%          10.04%      3599 s
tied       1,684,224     4.93% (loss 6.9018)       5.71%          11.24%      1848 s
```

**The difference is not resolvable.** At ep600 tied is 0.34 points behind; at ep500 it is **0.61 points ahead** - the sign flips between two checkpoints of the same run. So the correct statement is that **at 600 steps a 64%-smaller tied model is statistically indistinguishable from the untied one**: the separate answer table bought nothing measurable and cost 3,051,746 parameters and 1.95x the training time. Predictions scored: P1 tied >= 90% of untied **MET** (93.5%), P2 bands close **MET**, P3 tied faster **MET** (1.95x), falsifier (below 80%) **did not fire**. **What this does not establish:** both arms are under-trained (5% against a 14.70% trivial constant), so it tests the *shape* at a small budget, not final quality - the claim is "the answer table is redundant at this scale", not "tied is better". Two defects found while measuring: the driver saved only on multiples of 250 so the 600-step endpoint left **no weights on disk** (now fixed to save at the last step), and the band evaluator only knew the untied readout and raised `KeyError: 'Wa'` on the tied model it was supposed to judge (now understands both). Source: `docs/history/TIED-READOUT.md`, `lens/runs/TIE-untied.txt`, `lens/runs/TIE-tied.txt`.

### The memory bank failed, and the readable window bank that replaces it (2026-11-19)

**The failure, measured.** Token-space bank against its no-bank control, same seed/data/600 steps: bank **5.33%**, no bank **5.27%** - **+0.06 points, nothing** - and **all 16 slots decoded to the same word**. The diagnosis on the trained weights: `slot-to-slot cosine mean 0.864, max 1.000`, **effective distinct slots 1.3 of 16**, each slot 0.934 cosine to the mean slot direction. **Sixteen slots held 1.3 slots' worth of information.** My previous fix (content-addressed writes plus a per-slot bias) missed the real cause: the write was still `softmax` over slots, so every token went a little into *every* slot, all slots converged to the corpus average, and the decoder mapped that average to whichever word sits nearest the embedding table's centre. Predictions: P1 (both above the 14.70% trivial constant) **FAILED**, P2 (within 1 point of control) **MET** (+0.06), P3 (distinct words per slot) **FAILED**.

**The rebuild** (`lens/winmem.py`): the bank IS the last K tokens. Slot j holds the embedding of exactly one real token and is replaced **wholly**, never blended; the addressing is a ring buffer, so it is discrete, needs no write policy, and cannot collapse. Reading is learned (`softmax(K q / sqrt(H))` attention), so the model chooses which past tokens matter while being unable to smear them in storage.

```
tokens fed:       the captain sailed west storm harbour again sailed captain
window contents:  storm | harbour | again | sailed
slot-to-slot cosine: -0.081      (was 0.864)
finite-difference: worst abs 4.334e-10   GATE: PASS
```

**Three gradient bugs were found by that one check:** the embedding gradient for slot contents was never wired (numeric 1.423356 vs analytic 0.0); attribution was gated on the slot being filled *before* the write, skipping the first K steps (off by 1.40); the pre-write slot was zeroed when this step's read had touched its old content (off by 0.63). Each was invisible in the code and obvious in the difference. **Honest status:** a working, gated, readable window bank - **not yet a result**. It has not been trained at scale or compared against a control; the trainer wiring is next. Source: `docs/history/READABLE-WINDOW-BANK.md`, `lens/winmem.py`.

### Rarity-weighted loss: REFUTED by its own falsifier (2026-11-19)

The first training method that follows from a measured failure (11,002 of 12,002 vocabulary entries beyond rank 1,000, occupying 14.41% of positions, with the rare bands at **0.00%**). The loss reweighted every scored position by inverse frequency. Predictions and the falsifier were written down **before** the run:

```
band                 alpha=0 (plain)   alpha=1 (rarity)   change
top-20                    32.74%            30.52%        -2.22
ranks 21-500               9.67%             8.26%        -1.41
ranks 501-2000             0.00%             0.00%         0.00   <- the band it was aimed at
TAIL (rank > 2000)         0.00%             0.00%         0.00   <- the band it was aimed at
overall                   20.07%            18.44%        -1.63

held-out (trainer line)   22.91%            21.08%        -1.83   (loss 9.3447 vs 9.6752)
```

**P1 (gain >= 3 points on the tail and 501-2000) FAILED. P2 (top-20 loss <= 3) MET at -2.22. P3 (overall loss <= 2) MET at -1.63. The pre-registered FALSIFIER fired: "if the tail band does not improve, the method is dead."** Rarity weighting made **every** band worse, including both bands it was designed to help, which stayed at exactly zero. The weighting is implemented correctly - the gradient check passed at alpha 0, 1 and 2 (worst 5.9e-10) and was proven inert at alpha 0 - it simply does not change what the model can represent: at this budget there is **no** rare-token ability to reallocate toward, so redistributing loss weight only distorts the frequent band. **The binding constraint is representational capacity or training budget, not loss weighting.** Honest scope: 2,500 steps, 4.7M parameters, one seed - which cannot rule out a small effect, but does rule out the >= 3-point gain that was predicted. Source: `docs/history/RARITY-LOSS-REFUTED.md`, `lens/runs/RARE-0.txt`, `lens/runs/RARE-1.txt`, `tools/eval_next_bands.py`.

**Update, third arm:** the comparison is complete and the effect is a **monotone dose-response in the wrong direction** - `alpha=0` **22.91%**, `alpha=1` **21.08%**, `alpha=2` **17.61%** held-out (loss 9.3447 / 9.6752 / 9.7392). More rarity weighting is consistently worse, which is stronger evidence than a single point: not noise around zero, but a cost that grows with the dose while the targeted bands stay at **0.00%**. Note also that `alpha=2` has the smallest held-out-minus-train gap (+0.77 against +2.87 for plain) purely because it fitted **less** - a shrinking generalisation gap is not automatically a better model, and this is a clean case of a metric moving the "good" way for a bad reason. Source: `lens/runs/RARE-2.txt`, `docs/history/RARITY-LOSS-REFUTED.md`.

**Authoritative band table (1040 windows, supersedes the 104-window figures):**

```
checkpoint       top-20   ranks 21-500   ranks 501-2000   TAIL(>2000)   overall   constant
rare0 (plain)    33.54%      7.55%           0.00%          0.00%        19.80%    14.29%
rare1 (alpha=1)  31.00%      6.37%           0.00%          0.00%        18.08%    14.29%
rare2 (alpha=2)  26.99%      3.43%           0.00%          0.00%        14.99%    14.29%
```

The monotone degradation holds at 10x the sample, and both targeted bands are **0.00% at ~8,700 positions each** - genuinely zero, not zero-at-a-small-sample. Source: `lens/runs/chain_rare.out`, `docs/history/RARITY-LOSS-REFUTED.md`.

### The readable window bank: readable, and useless at this budget (2026-11-19)

Predictions and falsifier were written before the run:

```
arm            params      HELD-OUT              train ema   gap
winoff (K=0)   1,684,224   6.07%  (loss 6.5013)   7.0492    -0.5479
winon  (K=8)   1,848,320   6.04%  (loss 6.5862)   7.0475    -0.4613

window at the end: ` | class | for | a | library | management | ` | book
```

**P1 FAILED** (both far below the 14.70% trivial constant), **P2 FAILED** (window 0.03 points *below* control), **P3 MET** (eight distinct real words - the readability problem is solved). **The pre-registered falsifier fired.**

**The diagnosis this forces:** both arms sit at ~6% while the per-window constant scores **14.70%**, so the model is *worse than guessing the most frequent token* - and a mechanism that supplies context cannot help a model that has not learned to use the context it already has. **The experiment was mis-designed, by me:** at 1,200 steps this trainer produces a model below trivial, so no memory mechanism could have shown anything. The base models that reach 21.74% needed **6,000** steps, and every memory arm today used 600-1,500. **The memory question has never been tested at a budget where the model can answer it.** Established: the window is readable, its gradient exact (4.334e-10), its batched path equivalent to the certified one (1e-16), and it does not collapse (slot-to-slot cosine -0.081 vs 0.864). Not established: that it helps. **Next:** window vs no-window at 3,000-4,000 steps where models clear the constant, plus the delayed-recall probe as a sharper instrument. Source: `docs/history/WINDOW-BANK-NO-BENEFIT.md`, `lens/runs/WIN-winon.txt`, `lens/runs/WIN-winoff.txt`.

### Readable window bank at 3000 steps: +0.50 points, exactly at the bar, not yet proven (2026-11-19)

The first fair asking of the memory question, at a budget where the model is no longer far below trivial. Predictions and falsifier written before the run.

```
arm            params      HELD-OUT              train ema   gap
winoff (K=0)   1,684,224   14.75%  (loss 5.8357)   6.4652    -0.6294
winon  (K=8)   1,848,320   15.25%  (loss 5.8194)   6.4538    -0.6344
difference                                        +0.50 points

window at the end: ` | class | for | a | library | management | ` | book   (8 distinct real words)
```

**P1 MET barely** (both clear the 14.70% constant; the control by 0.05), **P2 MET exactly** (+0.50 against a +0.5 bar), **P3 MET**, and the **pre-registered falsifier did NOT fire - by 0.00 points**. **Why this is not a win yet:** a 0.50-point difference on **one seed** with a 104-window evaluation is inside the noise of the instrument (single-number SE ~1.5 points), so one seed cannot distinguish +0.50 from 0. The correct statement is that **the point estimate lands exactly on the pre-registered threshold**. It is the first time a memory mechanism has not been measurably negative - the slot bank gave +0.06 with collapsed slots, and this same window gave -0.03 at 1200 steps while the model was below trivial - but that is not evidence memory works. **The only thing that would settle it:** the same comparison at **3 seeds**, scoring the paired per-seed difference rather than arm means; if the mean paired delta stays near +0.50 with an interval excluding zero, the window earns its place, and if the interval straddles zero the honest conclusion is that a 1.68M model at 3000 steps cannot demonstrate a memory benefit and the line stops. Source: `docs/history/WINDOW-BANK-3K.md`, `lens/runs/WIN3K-winoff.txt`, `lens/runs/WIN3K-winon.txt`.

### The mean-singular-value repair failed badly: -11.28 accuracy points with a LOWER loss (2026-11-19)

The repair came from a real measurement - the memory operator `diag(1-tanh^2) @ Wh` had mean singular value **0.5613**, implying ~0.01% retention over 16 steps while the spectral radius was a satisfied 0.9. The repair raised the mean to 0.95 with the largest singular value capped at 2.0. The pre-registered falsifier was "P2 failing by more than 1.5 points means instability":

```
arm     what it does                        HELD-OUT      loss      train ema   gap
svoff   spectral-radius clamp (0.9)         17.70%       9.5525     7.0570    +2.4955
svon    mean singular value -> 0.95          6.42%       8.6336     7.4039    +1.2297

difference                                 -11.28 points  -0.92
```

**The falsifier fired on its most severe branch: -11.28 points.** The interesting part is the dissociation: `svon` has a **lower held-out loss** (8.6336 vs 9.5525) and a **far lower accuracy** (6.42% vs 17.70%) - probability mass spread more sensibly across plausible tokens while almost never putting the best token on top. Raising the mean singular value flattened the readout's confidence: better calibrated in aggregate, useless for greedy prediction. **This is also the third arm today whose generalisation gap shrank while the model got worse** (rarity +2.87 -> +0.77 as accuracy fell 2.91; the tied readout at 600 steps; now +2.4955 -> +1.2297 as accuracy fell 11.28). A smaller gap is a consequence of fitting less - any arm reporting it must report accuracy beside it. **What this does not refute:** the retention measurement stands; the operator's mean really is 0.5613 and really does imply ~0.01% retention over 16 steps. What failed is the assumption that forcing that number toward 1.0 every training step is free. **Next:** raise it only at inference, or with a gentler target, and measure the accuracy cost per step as a curve rather than one all-or-nothing point. Source: `docs/history/SVNORM-REFUTED.md`, `lens/runs/SVN-svoff.txt`, `lens/runs/SVN-svon.txt`.

### Depth at 1200 steps: falsifier fired, but the per-layer readout works (2026-11-19)

The audit said the core was 1.3% of the model and one tanh step cannot compose. `lens/model2.py` adds a second recurrent layer and gives layer 1 its own readout on the same objective (gate 4.026e-10).

```
arm                                  HELD-OUT            train ema   held-out loss
single layer (winoff, 1200 steps)     6.07%                7.0492       6.5013
two layers + per-layer readout        5.68% (layer 2)      9.1805       8.4180
                                       4.90% (layer 1)
difference                            -0.39 points
```

**P1 FAILED and the falsifier fired** (-0.39 against a +0.5 bar). **P2 MET and this is the transparency deliverable: layer 1's own readout reaches 4.90%**, above zero and 0.78 points from layer 2 - each layer produces its own next-token prediction and both are scored every run. **P3 MET** (5.68% vs 4.90% - the layers are not identical). **Why I am not yet accepting "depth does not help":** depth2 was **harder to optimise, not merely worse** - train EMA 9.18 against the baseline's 7.05, held-out loss 8.42 against 6.50 - and a second layer was added with the SAME learning rate, init scale and clip of 1.0 chosen for a one-layer model. A model that trains worse is evidence the optimiser settings suit a different model, not that the architecture is wrong. Honest status: **"depth with a one-layer learning rate does not help"**, which is weaker than "depth does not help". **One retry, pre-registered:** learning rate 0.03 -> 0.01 as the single change; if it clears 6.07% by +0.5 the earlier result was an optimisation artefact, and if it still fails the claim becomes "depth does not help at 1.8M parameters and 1200 steps" and the next repair is FLAW 4 (`<unk>` is 5.34% of the corpus). A single retry with a pre-registered outcome is not fishing; a second one after seeing the result would be. Source: `docs/history/DEPTH-REFUTED-AT-1K.md`, `lens/runs/DEPTH2.txt`, `lens/runs/WIN-winoff.txt`.

### CORRECTION: the window bank's +0.50 points does not survive an independent re-score (2026-11-19 13:20)

An hour ago I published the window bank beating its control by **+0.50 points**, exactly on the pre-registered bar, labelled "not yet proven". **That caution was correct and the number does not stand up.** The arms launched before the automatic self-check existed, so their numbers came from ONE instrument; re-scoring the same two checkpoints with the independent evaluator gives:

```
checkpoint                    trainer's own eval   independent re-score
win3kwinoff_K0_ep3000.npz     14.75%               12.19%   (top20 23.99%, r21-500 0.04%)
win3kwinon_K8_ep3000.npz      15.25%               12.03%   (top20 23.66%, r21-500 0.04%)

paired delta                  +0.50                -0.16   <- THE SIGN FLIPS
```

Three reasons the +0.50 cannot be quoted: **(1) it does not reproduce** - an independent instrument on the same files reads **-0.16**; both are 104-window evaluations with different draws, so the difference between draws exceeds the effect. **(2) It is smaller than seed noise** - seed 2's control finished at **10.50%** against seed 0's **14.75%**, **4.25 points of seed-to-seed variance** in the same arm and settings, so the claimed effect is **8.5x smaller** than the arm's own seed spread. **(3) The trainer reads ~2.6 points higher than the independent tool on both files** (14.75 vs 12.19, 15.25 vs 12.03) - a protocol difference that applies equally to both arms, which is why the delta matters and why a delta computed on one instrument must be confirmed on a second. **WITHDRAWN:** "the readable window bank gives +0.50 points at 3000 steps". **STILL TRUE:** the window is readable (8 distinct real tokens), its gradient is exact (4.334e-10), its batched path equals the certified one (1e-16), it does not collapse (slot cosine -0.081 vs the old 0.864), and it is not measurably negative at 3000 steps the way the slot bank was. **STILL OPEN:** whether any memory mechanism helps this model - three seeds are running, and the statistic is the mean **paired** delta with an interval. **Process lesson:** the self-check added two hours ago forces exactly this comparison at the end of every run; these two arms predate it and are the first case where its absence hid a withdrawn number. **A delta measured on one instrument is a hypothesis until a second instrument agrees on the same file.** Source: `docs/history/WINDOW-BANK-3K-CORRECTION.md`, `lens/runs/WIN3K-winoff.txt`, `lens/runs/WIN3K-winon.txt`, `lens/runs/WINS-s2off.txt`.

### The instrument is sharper than I said: 104-window draw noise is 0.41 points, not 1.5 (2026-11-19)

Twice today I wrote that a single 104-window number has a standard error "around 1.5 points" and used it to dismiss small effects. **I never measured it.** Measured now on one fixed checkpoint, five window draws, no training:

```
draw_seed 4242 -> 12.19%
draw_seed 11   -> 12.19%
draw_seed 22   -> 11.57%
draw_seed 33   -> 11.23%
draw_seed 44   -> 11.81%

mean 11.80%   sd 0.41 points   full range 0.96 points
```

**Draw noise is 0.41 points, about 3.6x smaller than the 1.5 I had been quoting** - wrong in the pessimistic direction, so I had been discarding effects as "inside the noise" that were larger than the instrument's real resolution. **The withdrawal of the +0.50 window-bank delta stands**, and not because of draw noise: the independent re-score put it at -0.16 (sign flipped) and the arm's own seed-to-seed spread is **4.25 points**. **The 2.6-point trainer-vs-evaluator gap is a protocol difference, not sampling** - at 0.41 points of draw noise it cannot be noise. **The measured noise hierarchy for this project:** seed variance **4.25** >> protocol difference **2.6** >> draw noise **0.41** - and every arm today was compared on one seed, so every arm was dominated by the largest source and the instrument was never the limiting factor. **Resolution floor for future single-draw comparisons: 0.41 points**, measured rather than guessed. Source: `docs/history/DRAW-NOISE-MEASURED.md`, `lens/runs/DRAW-NOISE.txt`, `lens/scripts/run_draw_noise.sh`.

### The 3000-step arms never cleared trivial: P1 was evaluated on the wrong denominator (2026-11-19 13:55)

Third correction today and the one that matters most, because it invalidates the gate I used to declare the memory test "fair". `lens/train_tokmem.py:119` computes accuracy with `<unk>` targets **excluded** (`wts = (tg != UNK)`), and `<unk>` is **5.19% of the validation region** (25,951 of 500,000 tokens). So the trainer's HELD-OUT number is a **non-`<unk>` accuracy**, while the constant from `tools/eval_next_bands.py` is over **all** positions - and the constant is even allowed to guess `<unk>` and score, which the model never can because `lg[UNK] = -1e9`. The P1 gate compared **14.75% over 94.81% of positions** against **14.70% over 100% of positions**: a ~0.77-point favourable denominator difference, larger than the entire effect the experiment was built to detect. Corrected, one protocol, one draw:

```
checkpoint                   model(<unk>-excl)   constant(<unk>-excl)   verdict
win3kwinoff_K0_ep3000         12.97%              13.14%                 BELOW constant by 0.17
win3kwinon_K8_ep3000          12.80%              13.14%                 BELOW constant by 0.34
    (constant over ALL positions on the same draw: 14.70%)
```

**P1 FALSE and P2 FALSE**: neither arm clears trivial once the denominators match, both are below it, and the window's paired delta on the independent instrument is **-0.17**. The budget ladder as previously published is wrong: **3000 steps does not clear trivial**, and the trainer's 14.75% was never comparable to the 14.70% constant. The 6000-step base (21.74%) has **not** been re-scored on the corrected protocol and must be before it is used as "clearly above trivial". The window bank returns to **untested at a qualifying budget**, because no arm so far has qualified. **The fix:** `tools/eval_any.py` now prints, per checkpoint and draw, the model's `<unk>`-excluded accuracy, the constant's `<unk>`-excluded accuracy, the constant over all positions, and a literal `BEATS constant` / `BELOW constant` verdict. **The rule: a gate that compares two numbers must prove both were computed on the same set of positions.** I wrote that gate, ran it, declared it met and published it, and found the defect only while auditing the instrument after a different discrepancy. The most dangerous measurement error is the one that makes your hypothesis look alive. Source: `docs/history/P1-WRONG-DENOMINATOR.md`, `lens/train_tokmem.py`, `tools/eval_any.py`, `lens/runs/DRAW-NOISE.txt`.

### Depth closed: the retry moved it further from the baseline (2026-11-19 13:51)

Pre-registered single change `LR0 0.03 -> 0.01`, and the stated failure outcome was "the claim becomes depth does not help at 1.8M parameters and 1200 steps, and the next repair is FLAW 4 (`<unk>`), not more architecture".

```
arm                         HELD-OUT (layer 2)   layer 1 own readout   train ema   held-out loss
single layer (baseline)      6.07%                -                     7.0492      6.5013
depth2, lr 0.03              5.68%                4.90%                 9.1805      8.4180
depth2, lr 0.01              5.08%                4.37%                 9.9654      9.4252
```

**The retry made it worse on every measure** (accuracy -0.60, train EMA +0.78, held-out loss +1.01), so the earlier failure was **not** a learning-rate artefact and the falsifier stands. **One retry was the budget and it is spent** - a second attempt after seeing this would be fishing for a setting, not testing a hypothesis. **Caveat that limits what "closed" means:** all three arms are **below the trivial constant** on the corrected protocol, so the comparison tests whether a second layer helps **a model that cannot yet beat a constant**. It is internally valid (same trainer, same instrument, fixed `rng(999)` so the runs are paired on windows) but narrow. Correct claim: **"a second recurrent layer does not help a model that is still below trivial at 1.8M parameters"** - not "depth does not help". **The transparency result survives:** layer 1's own readout scores **4.37-4.90%**, so every layer makes its own next-token prediction and both are scored on every run - that is what the two-layer model was built for and it is independent of whether depth improves accuracy. **Next, and it is a prerequisite rather than a hypothesis:** no model in this project has yet been shown to beat the trivial constant on a common protocol; the 6000-step base (21.74% on the trainer's line) is the only candidate and has never been re-scored with matching denominators. That re-score comes before any new architecture work. Source: `docs/history/DEPTH-CLOSED.md`, `lens/runs/DEPTH2-LR01.txt`, `lens/runs/DEPTH2.txt`, `lens/runs/WIN-winoff.txt`.

### The qualifying budget is 6000 steps: first model verified above trivial (2026-11-19 14:05)

After the denominator correction, the question that had to precede any new architecture work: does ANY model here beat the trivial constant once both numbers are computed on the same positions? Re-scored the 6000-step base with the corrected instrument:

```
scan12_r090_6k_ep6000.npz   table(Wa)  1-layer  no memory
  top20 36.68% | r21-500 12.45% | r501-2k 0.00% | tail 0.00% | overall 23.06%

  SAME-DRAW constant: all positions 14.70% | <unk>-excluded 13.14% | model <unk>-excluded 24.54%
  -> BEATS constant by +11.40 points
```

**First model in this project shown to beat trivial on a common protocol**, and the 23.06% overall figure reproduces the earlier evaluator number exactly - a consistency check that passed. **What this settles:** 6000 steps is the **qualifying budget**, the lowest measured where the model has headroom above the constant, so the lowest at which a mechanism test can mean anything. **Every memory and depth comparison run today (600-3000 steps) tested an under-trained model**, and that is now measured rather than suspected: at 3000 steps the arms sit at 12.80-12.97% against a 13.14% constant. The 3000-step "fair test" failed for two independent reasons, budget and denominator. **Launched `win6k`:** window vs control at 6000 steps on **both instruments** - P1 both arms beat the matched constant, P2 window beats control by >= 0.5 on the trainer line, **P3 the independent re-score reproduces the sign** (a delta that exists on only one instrument is not a delta), P4 the window still holds K distinct real words. **FALSIFIER:** both clear the constant AND the window is not >= 0.5 above control on both instruments -> the readable window buys nothing when the model is capable. Source: `docs/history/QUALIFYING-BUDGET-6K.md`, `lens/runs/chat/scan12_r090_6k_ep6000.npz`, `lens/scripts/run_winbank_6k.sh`.

### Seed noise is 2.89 points, so a 0.5-point memory effect is unmeasurable by seeds alone (2026-11-19 14:30)

Three controls now exist for the same configuration (K=0, 3000 steps, tied readout, identical settings, only the seed differs):

```
seed 0  win3kwinoff_K0_ep3000   14.75%
seed 1  s1off_K0_ep3000         16.03%
seed 2  s2off_K0_ep3000         10.50%

mean 13.76%   sd 2.89 points   range 5.53 points
```

**The arm's own seed spread is 10x the +0.50 effect the window bank was credited with this morning.** With the corrected protocol, where all three arms sit at 12.80-12.97% against a 13.14% constant, these arms **cannot resolve the effect they were built to find**. **What it would take, computed rather than guessed** (paired differences sd ~ 4.09): a **0.5-point** effect needs **258 seeds per arm**, 1.0 needs 65, 2.0 needs 17, 3.0 needs 8. So a sub-point memory effect **cannot be established by adding seeds** - two honest routes remain: **(1) only claim large effects** (3+ points, detectable with 8 seeds; anything smaller is unmeasurable here and must not be reported as a result in either direction), and **(2) use a sharper instrument** - the delayed-recall probe separates "memory works" from "memory does nothing" by **~97 points** (100.00% at delays 1/4/8 against 3.12% chance, 81.25% at delay 16) instead of 0.5. **Strategic conclusion: the memory bank should be judged by the delayed-recall probe, not by next-token accuracy.** Every arm today graded memory on a metric whose noise floor is 10x the effect size - partly the mechanism, partly the instrument. That does not excuse the mechanism: on the probe the bank holds a fact 8 steps perfectly and 16 steps at 81.25%, and whether it helps a 1.68M model generate is still open. Next-token accuracy should from now on only confirm a mechanism does not COST anything. **Correction:** the first version of this entry quoted sd 2.80 and 242/61/16/8 seeds, written from arithmetic not run; computed values are sd **2.89**, sd_diff **4.09**, **258/65/17/8**. Source: `docs/history/SEED-NOISE-FLOOR.md`, `lens/runs/WINS-s1off.txt`, `lens/runs/WINS-s2off.txt`, `lens/runs/WIN3K-winoff.txt`.

### The window bank's limit is addressing, not retention - measured, not argued (2026-11-19 15:10)

Before building a delayed-recall probe for the readable window bank, I read its read path: `q = h @ Wq`, `k = h @ Wk`, `s = (Km @ q)/sqrt(H)`, `a = softmax(s)`, `r = a @ C`, `out = h + tanh(r @ Wo)`. **There is no position or age term anywhere in the read** - a slot's index `j = t % K` exists in the implementation but is invisible to the attention, so the only way to select a specific slot is content similarity. **Retention is not the problem:** the ring buffer replaces a slot wholly and nothing decays, so a fact survives exactly K-1 steps. Measured on the **real trained embedding matrix** from `win3kwinoff_K0_ep3000.npz` (V=12002, D=128): the fraction of trials where the nearest vocabulary entry to the mean of K distinct token embeddings is one of those K:

```
K= 2   100.0%
K= 4    87.5%
K= 8    19.5%
K=16     5.0%   (typical token-to-token distance 1.60)
```

**With a full 16-slot window, a uniform read returns a vector whose nearest real token is one of the 16 only 5% of the time** - so unless attention is sharply peaked on exactly the right slot, the read is not noisy, it is **wrong**, and peakedness needs a discriminator the architecture does not provide. **Consequence for the probe I was about to run:** delayed recall asks the model to report the token from position 0 after D fillers; the query at step D is built from fillers and contains nothing that says "position 0", so **the probe would have failed by construction and I would have recorded it as "the window bank cannot hold 16 steps" - the wrong conclusion about the right mechanism.** **The repair this points to: give the read a position/age term** (`s = (Km + age_emb[j]) @ q / sqrt(H)` or concatenate age to the key) - small, and it makes arbitrary-delay addressing possible for the first time. **What the bank IS good for:** because the key comes from the state at write time and the query from the current state, it retrieves **recent or similar tokens**, which is what next-token prediction needs - a KV-cache-like recent buffer, not a long-delay store. Source: `docs/history/WINDOW-BANK-ADDRESSING.md`, `lens/winmem.py`, `lens/runs/chat/win3kwinoff_K0_ep3000.npz`.

### The age key: the window bank can now address slots by age, and its gradient is certified (2026-11-19 15:25)

The measured defect: the read had **no position or age information**, so the only way to select a slot was content similarity, and with a full 16-slot window a uniform read lands on a wrong token 95% of the time (nearest-token-is-one-of-the-16 = **5.0%** on a real V=12002 D=128 embedding matrix). Retention was never the problem - slots are replaced wholly and nothing decays - so addressing was the whole limitation. **The repair:** slot `j` at step `t` holds data written at step `t - age` with `age = (t - j) mod K`, identical for every sample, so the fix stays exactly batchable - `s = ((Km + age_emb[age]) @ q)/sqrt(H)`, with `np.add.at(g["age_emb"], age, ...)` in the backward. It is **opt-in** (activates only when `age_emb` is in the parameter dict), so no existing checkpoint, script or running job changes.

```
AGE-KEY FINITE-DIFFERENCE GATE  (H=4 D=3 V=5 K=3 T=5 B=3)
  parameters checked: 71  (includes age_emb (3,4) and every path the key changes)
  worst absolute error: 1.713e-09   at Wk[3,1]
  worst relative error: 4.157e-07
  GATE: PASS
  absence of age_emb gives identical forward: max |diff| 9.849e-02 (expected NONZERO: the key is active)
  zeroed age_emb gives identical forward:      max |diff| 0.000e+00 PASS (a zero key is a no-op)
```

Three things established, each measured: **the gradient is exact** (1.7e-09 over all 71 parameters including `age_emb` and the Wq/Wk/Wo/emb paths whose backward now differs); **the key is genuinely active** (removing it changes the forward by 9.8e-02, so the test does not pass on a no-op); **it is safe for existing artefacts** (zeroing `age_emb` reproduces the old forward bit-exactly at 0.000e+00). **What is NOT established: whether it helps.** The gate proves the mechanism is implemented correctly, not that it improves recall - and it must be judged on an instrument that can see the difference, not next-token accuracy whose seed noise (2.89) is 10x the effects this project has chased. The delayed-recall probe is now a **fair** test of the window bank for the first time, because the architecture can express positional addressing. **Next:** train with the key on delayed recall at delays 1/2/4/8/16 against a channel-removed control; predictions written before launch, falsifier = no better than the pre-age-key bank at every delay. Source: `docs/history/AGE-KEY-CERTIFIED.md`, `lens/winmem.py`, `tools/agekey_gradcheck.py`.

### The age key is decisive on the sharp instrument: 83.20% against 19.82% at delay 8 (2026-11-19 15:45)

The first measurement all day that is both large and on the right instrument. Delayed recall - remember one token from position 0 across D filler steps and output it; chance is **3.12%** (vocab 32). Both arms are the same readable window bank, same data stream, same frozen feature extractor, same parameters apart from `age_emb` (**K*H = 1024 params**):

```
delay   agekey      plain        diff        (chance 3.12%)
1       100.00%     100.00%      +0.00
2        99.90%      99.71%      +0.20
4        95.21%      56.93%     +38.28
8        83.20%      19.82%     +63.38
16       57.23%      14.06%     +43.16
```

**P2 NOT met and it was my prediction that was wrong** (I wrote the plain arm would sit near chance at delay 16 with an 8% threshold; it reads **14.06%**, 4.5x chance, so content similarity keeps real residual signal even with 17 slots filled - the measured decay 56.93 -> 19.82 -> 14.06 is the honest version), **P3 MET** (agekey clears 50% by a wide margin), **FALSIFIER DID NOT FIRE** (it required +5 points at delay 8; measured **+63.38**). **P1 was corrected by the smoke run before the full run:** both arms solve delay 1 at 100%, because the window holds two contents and the mean of two token embeddings still identifies one 100% of the time at K=2 (measured) - not a leak, since delay 4 shows the decay. **What this confirms:** the defect diagnosed at 15:10 was real and it **was** the limiter - the read had no position or age term, so slot selection depended entirely on content similarity, and a full window read as a blend whose nearest real token is one of the K only 5% of the time at K=16. A learned per-age key - **1024 parameters, 0.06% of a 1.68M model** - takes delay-8 recall from 19.82% to 83.20%. It also retro-explains the whole memory line: both earlier banks failed to show benefit at 16 steps because neither could address a slot by age. **What this does NOT establish:** it is a synthetic probe with a frozen feature extractor, a 32-token vocabulary and training that flows to the embeddings through the bank contents only - it says nothing yet about next-token accuracy on real text; it is **one seed**; and **no end-to-end arm has been trained with the age key**. **Next:** three seeds of this probe (minutes), then an end-to-end arm at the qualifying budget (6000 steps) scored on the trainer line and the independent evaluator. Source: `docs/history/AGE-KEY-DECISIVE.md`, `tools/delayed_recall_agekey.py`, `lens/runs/AGEKEY-PROBE.txt`.

### The age key at three seeds: +65.01 points at delay 8, every seed above +62 (2026-11-19 16:15)

The single-seed probe said 83.20% vs 19.82% at delay 8. One seed is one seed, so the probe was repeated at seeds 0, 1 and 2, reporting **each seed separately and the paired delta per seed** rather than arm means. Chance is **3.12%** (vocab 32):

```
delay  agekey            plain             delta     per-seed paired deltas
1      100.00% +/-0.00   100.00% +/-0.00   +0.00     +0.0   +0.0   +0.0
2       99.90% +/-0.10    99.71% +/-0.00   +0.20     +0.2   +0.3   +0.1
4       95.51% +/-2.11    59.44% +/-3.93  +36.07    +38.3  +40.3  +29.6
8       85.61% +/-4.09    20.61% +/-0.70  +65.01    +63.4  +69.5  +62.1
16      46.68% +/-14.78   13.44% +/-0.64  +33.24    +43.2  +16.3  +40.2
```

**At delay 16 the advantage survives on every seed but is variable** (agekey 46.68% +/-14.78 from seeds 57.2/29.8/53.0 against plain 13.44% +/-0.64; paired deltas +43.2/+16.3/+40.2) - and that is exactly the boundary of the buffer, since slot `j = t % K` is written after the read, so at `t = K = 16` the read still sees step 0's fact and the next step overwrites it. **The single-seed run OVERSTATED it:** it read 57.23% and the three-seed mean is 46.68%, because that seed was the best of the three - the clearest local example of why the three-seed rule exists, on my own result. **P1 MET** (paired delta at delay 8 above +30 on **every** seed; the smallest is **+62.1**, more than double the bar), **P2 MET** (same sign at every delay of 4 and above on all three seeds), **FALSIFIER DID NOT FIRE** (it needed any seed below +10; the minimum observed is +62.1). **The effect is not one seed's luck** - and the asymmetry is the honest shape of it: the plain arm is remarkably stable (20.61 +/- 0.70) while the age-key arm varies more (85.61 +/- 4.09), so the plain bank fails at the same place every time and the age key succeeds by a variable but always-large margin. **What this establishes:** the age key - **1,024 parameters, 0.06% of a 1.68M model** - is what the readable window bank was missing. Without a position/age term the read can only select a slot by content similarity and a full window reads as a blend; with it, delay-8 recall goes from **20.61%** (6.6x chance) to **85.61%** (27x chance), replicated across three seeds with per-seed deltas between +62.1 and +69.5. **What it still does not establish:** it is a synthetic probe with a frozen feature extractor, a 32-token vocabulary and embeddings trained through the bank contents only - nothing here is a claim about next-token accuracy on real text; and **no end-to-end arm has been trained with the age key**, which is the test that matters for the product. Source: `docs/history/AGE-KEY-THREE-SEEDS.md`, `tools/delayed_recall_agekey.py`, `lens/runs/AGEKEY-SEEDS.txt`.

### The window bank at three seeds: +0.87 points, all three seeds positive (2026-11-19 16:24)

The paired comparison that replaced this morning's withdrawn single-seed +0.50. Three seeds, both arms per seed, same data order, only the window differs, scored on the trainer's own 104-window line (`rng(999)` is fixed so the draws are identical across arms - a genuinely paired design):

```
seed   control (K=0)   window (K=8)   paired delta
0      14.75%          15.25%         +0.50
1      16.03%          17.11%         +1.08
2      10.50%          11.52%         +1.02

mean paired delta +0.87   sd 0.32   se 0.19   t = 4.73 (crit 4.303 at 95%, df=2)   CI [0.07, 1.66]
```

**P2 MET - all three seeds share the same sign**, and **the paired test is significant at 95%** (t 4.73 against 4.303) with an interval that excludes zero, just barely. **This is not the claim that was withdrawn this morning:** that one was a single seed compared as arm means and an independent re-score flipped its sign; this one is three paired seeds. **Two caveats keep it small:** (1) **it is measured on ONE instrument** - the independent evaluator gave 12.19% / 12.03% on the seed-0 pair, a NEGATIVE delta, so the sign has not yet been confirmed on a second instrument for seeds 1 and 2, which is the immediate next measurement; (2) **both arms are at or below the trivial constant** on the corrected protocol at 3000 steps, so a +0.87-point gain on a model still at the level of "guess the most frequent token in the window" is real but not useful yet. **Plain terms:** at a budget where the model cannot beat a constant, adding the readable window changes its score by **+0.87 consistently across three seeds** - small, but no longer explicable as one seed's luck. Source: `docs/history/WINDOW-BANK-THREE-SEEDS.md`, `lens/runs/WINSEEDS.txt`, `lens/runs/WINS-s1off.txt`, `lens/runs/WINS-s1on.txt`, `lens/runs/WINS-s2off.txt`, `lens/runs/WINS-s2on.txt`.

### The window bank effect shrinks 84% on a second instrument: +0.87 becomes +0.14 (2026-11-19 16:29)

The three-seed paired result said +0.87 points on the trainer's line. A delta that lives on only one instrument is not a delta, so all six checkpoints were re-scored with `tools/eval_any.py` on the same draw, using the `<unk>`-excluded model accuracy (the matched denominator):

```
              trainer line        independent instrument
seed 0        14.75 / 15.25       12.97 / 12.80        paired delta -0.17
seed 1        16.03 / 17.11       13.88 / 14.01        paired delta +0.13
seed 2        10.50 / 11.52        9.95 / 10.41        paired delta +0.46

mean paired delta   trainer +0.87        independent +0.14
independent sd 0.32, se 0.18, 95% CI [-0.65, +0.94], t = 0.74
```

**P1 MET** (sign holds on 2 of 3 seeds, not 3 of 3), **P2 MET barely** (mean +0.14), **falsifier did NOT fire** - **but the magnitude collapses by 84%** and the independent 95% interval **[-0.65, +0.94]** contains zero comfortably. **Honest conclusion: the +0.87 was mostly instrument effect.** Two facts force that reading: the trainer's line reads **~2 points higher than the independent tool on the same files** (14.75 vs 12.97, 16.03 vs 13.88, 10.50 vs 9.95) - and since the measured draw noise is **0.41 points**, a 2-point gap is not sampling - and whichever instrument is used, the effect is far below that instrument's resolution once protocols are matched. **The claim's whole history, recorded plainly: +0.50 (one seed, arm means) -> withdrawn -> +0.87 (three paired seeds, one instrument, t 4.71) -> +0.14 (three paired seeds, second instrument, t 0.74).** The direction survives; the size does not. What can be said without overreach: **adding the readable window does not measurably hurt** the language objective at 3000 steps, and **no positive claim is warranted**. Also learned: **one of three seeds clears the trivial constant** at 3000 steps (seed 1 at 13.88/14.01 against 13.14) while seeds 0 and 2 do not, so that budget is **seed-dependent** rather than uniformly below trivial. **The unexplained 2-point protocol gap is now the top instrument problem** - it is larger than every effect measured today, and until it is resolved every single-instrument delta carries a ~2-point unknown. Source: `docs/history/WINDOW-BANK-SECOND-INSTRUMENT.md`, `lens/runs/WINSEEDS-RESCORE.txt`, `lens/runs/WINSEEDS.txt`.

### At the qualifying budget the readable window buys nothing: +0.06 and +0.32 against a +0.5 bar (2026-11-19 16:38)

The test that was supposed to be the fair one: both arms at **6000 steps** - the budget established this afternoon as the lowest where a model clears the constant - on both instruments, falsifier written before the run ("both arms beat the constant AND the window is not >= 0.5 above its control on BOTH instruments -> the readable window buys nothing when the model is capable"):

```
                              trainer line          independent (matched protocol)
control (K=0)  params 1,684,224    22.82% (loss 5.3159)     20.30%   -> BEATS constant
window  (K=8)  params 1,848,320    22.88% (loss 5.3002)     20.62%   -> BEATS constant
paired delta                          +0.06                    +0.32

trivial constant on the same draw:    14.70% all positions | 13.14% <unk>-excluded
window at the end:  ` | class | for | a | library | management | ` | book   (8 distinct real words)
```

**P1 MET:** both arms beat the matched constant by **7.16 and 7.48 points** - the first properly qualified arms in this project. **P2 FAILED:** +0.06 against a +0.5 bar. **P3 MET:** the independent re-score gives the same sign (+0.32), so it is not an instrument artefact - both say "a little positive". **P4 MET:** eight distinct real tokens. **THE FALSIFIER FIRED.** **What this settles: the readable window bank, as built, buys nothing measurable on the language objective** - +0.06 and +0.32, both under 0.5, at the only budget where the model is capable enough to ask. It does not say the window is useless: it says the window is **neutral for next-token prediction in a model that already has a 2-token effective context**, which is what should be expected of a recent-token buffer when the objective rarely needs more than the preceding token. **What is still open, and it is one thing: the age-key arm (`glassnet-agekey6k`, in flight)** - the same bank with the addressing fix, paired against this very control. The probe says the key is worth **+65.01 at delay 8**; this run says the plain bank is worth **+0.06** on real text. If the key moves the language objective, addressing was the whole story; if it does not, real text at this budget does not demand positional recall and the honest next move is FLAW 4 (the 5.34% `<unk>` positions). Source: `docs/history/WINDOW-BANK-QUALIFIED-BUDGET-NEUTRAL.md`, `lens/runs/WIN6K.txt`, `lens/runs/WIN6K-off.txt`, `lens/runs/WIN6K-on.txt`.

### Correction: draw noise is 0.93 points, not 0.41 - and it explains the 2-point instrument gap (2026-11-19 17:04)

This morning I published a draw-noise figure of **0.41 points** from five draws and used it as the project's "resolution floor". Ten draws on a different checkpoint now give **0.93**, and the ten-draw test also prints each seed's first window index so an unapplied seed would be visible rather than inferred - all ten indices differ, so the seeds are being applied and this is real variance:

```
mean 22.67%   sd 0.93   range 2.77 points (21.47 to 24.24)   over 10 draws
```

**Why they differ:** the 0.41 came from a **12%-accuracy** checkpoint, the 0.93 from a **23%-accuracy** one. Draw noise is not a property of the tool, it is a property of **the checkpoint's position on the accuracy curve** - the same 104 windows sample a binomial, so variance scales with `p(1-p)`, and low-accuracy models are also more homogeneous in what they get right. Publishing one number as "the instrument's resolution floor" was a mistake; it is a function of what is measured. **What this explains: the 2-point trainer-vs-evaluator gap is fully accounted for** - `tools/instrument_agreement.py` showed the two forward passes agree to **+0.06 on identical windows**, so the gap is window selection, and the ten-draw range of **2.77 points** comfortably contains a 2-point difference between two different draws. **The distinction that matters: paired vs unpaired.** Unpaired single-draw comparisons carry the full 0.93 sd and 2.77-point range, so a claim built that way needs ~2 points to be safe (the model-vs-constant margins of 7.16 and 7.48 points clear it); **paired comparisons on identical windows cancel the draw entirely** - the per-window paired sd is **0.72**, so over 104 windows x 63 positions the standard error of a paired delta is **0.009 points**, which is why the window bank's +0.06/+0.32 are meaningfully measured even though they are far below the unpaired noise, and why the paired design was the right response to the earlier withdrawal. **Nothing measured today changes:** every paired result stands on its paired test, and the age-key probe's +65.01 is 70x the unpaired noise. **Correction:** the first version of this entry said the paired SE was 0.04; computed it is 0.72/sqrt(104*63) = **0.009**. Source: `docs/history/DRAW-NOISE-CORRECTED.md`, `lens/runs/DRAW-NOISE10.txt`, `tools/instrument_agreement.py`.

### Where the accuracy comes from, and why FLAW 4 is worth zero (2026-11-19 17:30)

Before spending a 7-hour arm on FLAW 4 (the `<unk>` positions), I measured what each frequency band is worth. Shares from the whole 500,000-token validation region, band accuracies from the 6000-step checkpoint on the independent instrument:

```
band        share of positions   measured acc    accuracy contributed
top20         45.31%            31.87%        14.44 pts
r21-500       36.08%             8.13%         2.93 pts
r501-2k        9.90%             0.00%         0.00 pts
r>2k           3.52%             0.00%         0.00 pts
<unk>          5.19%             0.00% (masked) 0.00 pts
```

**Reconstruction check:** these sum to **17.37 points** against a measured **23.06%** overall - the gap is because band accuracies come from one 104-window draw while band shares come from the whole region, and the draw is top-heavy. The ordering is the point, not a 0.1-point reconciliation. **Three conclusions:** (1) **83% of all accuracy comes from the top-20 tokens, which are 45% of positions** - the model is a frequent-token predictor and the rest is rounding; (2) **FLAW 4 is worth ~0 points** - a `<unk>` target is by construction a word outside the 12,002-token vocabulary, i.e. a rare word, and the model scores **0.00%** on both rare bands (13.42% of positions combined), so replacing `<unk>` with real tokens cannot turn a 0.00% band into accuracy; the ceiling of 5.19 points requires OOV words to become perfectly predictable, which no next-token model does - **the planned `<unk>` arm is cancelled on measurement, before 7 hours were spent on it**; (3) **the real headroom is the r21-500 band** - 36.08% of positions at only 8.13% accuracy, so moving it to 20% would add **+4.3 points**, more than every memory experiment today combined and 8x what a perfect `<unk>` fix could give. **What this changes:** the queue said FLAW 4 came next; measured, it is worth nothing and mid-frequency tokens are worth ~4 points. Those tokens have plenty of training signal at rank 21-500 in a 294M-token corpus, so this is a readout/capacity problem, the same family as the tied-readout question the parameter audit raised - and that is the next experiment to design, starting from a measurement rather than a hunch. Source: `docs/history/BAND-BUDGET-AND-FLAW4.md`, `lens/runs/WIN6K.txt`, `data/chat/prepared_mix_ctrl_honest/vocab_v12000.json`.

### The age-key end-to-end arm was a null experiment: the key was 1,117x too weak (2026-11-19 18:35)

The age-key arm was stopped at step 5025 by a forced shutdown. Before resuming it I compared its training line against its control (`win6kon` - same seed, same batch order, so the per-step lines are PAIRED):

```
matched steps: 201 (25..5025)
accuracy difference (agekey - control): mean -0.004 pts, sd 0.11; over steps >=4000 (n=42) mean -0.002, sd 0.09
loss difference mean -0.0018;  late steps where agekey acc > control: 10 of 42
step 500: 5.31 vs 5.31   step 2500: 8.51 vs 8.51   step 5000: 13.97 vs 13.97  (EXACTLY equal)
```

Two explanations survive: the key does nothing, or the key **cannot** do anything. So I measured the key in the checkpoint: it **is** saved (`bank_age_emb`) and it **is** training (|sum| 0.4544 -> 0.4617), but its **RMS is 0.000357 against a key magnitude RMS of 0.3987** - the age term is **0.1% of the key scale, a ratio of 1,117x**. Since the age vector is added to the key and then dot-producted with the query, a 0.1% perturbation cannot change the softmax ordering. **The arm could not test the hypothesis: it is a NULL EXPERIMENT, not a negative result, and it must not be quoted as evidence about the age key.** **Why the delayed-recall probe still stands:** the probe also started at zero, but there (H=64, 3000 steps, lr 0.05) the only useful signal is age, so the key grew to where it mattered; in the language model the content keys are informative and lr is sized for language modelling, so the key stayed negligible. **The probe proves the mechanism CAN express age addressing; the arm could not exercise it.** Those are consistent. **Repair, pre-registered:** `AGE_INIT` sets the initial RMS instead of zero and the new arm uses **AGE_INIT = 0.40** (the measured key RMS, 0.3987), so from step 0 the age term is comparable to the content term. **P1** the arm now differs from its control during training (paired |diff| > 0.5 somewhere) - if training lines stay identical the scale is still wrong and no endpoint number means anything; **P2** >= 0.5 points ahead of `win6kon` at 6000 steps on the trainer line; **P3** same sign on the independent evaluator; **FALSIFIER:** training lines diverge but neither instrument shows 0.5 points -> the key expresses age addressing and still does not help real text, and the memory line closes on measurement. The boot job now runs `win6kage2` (AGE_INIT=0.40) instead of resuming the null arm. Source: `docs/history/AGE-KEY-NULL-EXPERIMENT.md`, `lens/runs/AGEKEY-E2E-arm.txt`, `lens/runs/WIN6K-on.txt`, `lens/runs/chat/win6kage_K8_ep5000.npz`.

### The mid-band failure is frequency bias, not capacity: the correct token is usually at rank 34 (2026-11-19 19:15)

The band budget showed the biggest headroom here sits in the r21-500 band: **36.08% of scored positions at only 8.13% accuracy**, worth about **+4.3 points** overall if moved to 20%. Before designing a repair I measured WHICH failure it is:

```
MID-BAND DIAGNOSTIC  win6koff_K0_ep6000.npz
  windows 8, scored positions 483, of which r21-500: 192
  r21-500 accuracy 14.58%   top-20 accuracy 36.04%

  P1 prediction is a TOP-20 token when the target is r21-500 : 74.48%   -> MET (frequency bias)
  P2 of the wrong ones, prediction is also r21-500          : 12.80%   -> NOT met
  P3 median rank of the CORRECT token                       : 34      -> MET
     top-100 hit rate on r21-500 targets: 66.67%
  FALSIFIER: did not fire
```

**SMALL SAMPLE, stated up front:** 8 windows and 192 mid-band positions - treat the percentages as indicative and the direction as the finding. **What this establishes: (1) it is NOT a near-miss problem** - if the state could not separate mid-frequency tokens, wrong answers would be other mid-frequency tokens, and only **12.80%** are; **(2) it IS frequency bias** - on mid-band targets the model outputs a **top-20 token 74.48%** of the time; **(3) the information is present** - the correct token's **median rank is 34** and on **66.67%** of mid-band positions it is inside the model's own top-100, so the model is not ignorant of the answer, it is choosing not to say it. **Why this matters more than any architecture change:** a capacity repair would need to move the correct token from rank ~34 to rank 1 by representing more, which is expensive; a **calibration** repair only has to re-rank answers the state already contains, and can be tested at inference with no retraining. First candidate is **logit adjustment** (subtract `tau * log(prior)` before the argmax). **Pre-registered:** P1 some `tau > 0` raises OVERALL accuracy above the shipped baseline on 104 windows and both instruments; P2 the mid band gains more than the top-20 band loses; P3 the effect is monotone in `tau` up to a peak then reverses (a single lucky `tau` is a warning, not a result); **FALSIFIER** no `tau` beats the baseline by >= 0.5 points -> calibration is not the lever either. This is an inference-time experiment (minutes, not hours) aimed at a **+4.3-point** headroom, and it exists because the diagnostic ruled out the expensive explanation first. Source: `docs/history/MIDBAND-IS-FREQUENCY-BIAS.md`, `tools/midband_diagnostic.py`, `lens/runs/MIDBAND-DIAG.txt`.

### Calibration is not the lever: logit adjustment gains +0.84 on one checkpoint and nothing on another (2026-11-19 19:35)

The mid-band diagnostic said the failure is frequency bias (correct token at median rank 34, top-20 token emitted 74.48% of the time). The textbook correction is logit adjustment, `logits - tau*log(prior)`, inference-time only. Run on 104 windows it looked like a clean win:

```
LOGIT ADJUSTMENT  win6koff_K0_ep6000.npz
  tau    top20     r21-500   OVERALL
  0.00   36.15%    8.13%     20.30%
  0.20   36.90%    8.81%     20.92%
  0.35   36.70%    9.67%     21.14%   <- peak, +0.84 points
  0.50   33.97%    9.76%     19.88%
  1.00    1.74%    0.00%      0.83%

P1 overall >= 0.5 MET (+0.84)   P2 mid gains more than top-20 loses MET (+1.54 vs +0.55)
P3 monotone to a peak then reverses MET (peak 0.35)   FALSIFIER: did not fire
```

So two more checkpoints were tested before publishing: `win6kon_K8_ep6000` gives **+0.49** overall (P1 NOT met) and `scan12_r090_6k_ep6000` (untied table) gives **+0.00** with the optimum at `tau = 0` (P1 NOT met). **The falsifier fired on 2 of the 3 checkpoints.** The +0.84 does not replicate; on an untied model calibration buys exactly nothing, which is consistent with an untied answer table already absorbing frequency in its row norms. **What survives: (1) the DIAGNOSIS** - the mid band really is frequency-biased (median rank 34, inside the model's own top-100 on 66.67% of mid-band positions); **(2) the mid band does move on every checkpoint** (+1.54 and +1.67 points on r21-500) - the premise was right, the correction is just too blunt to convert it into overall gain without costing the top-20 band on two of three runs; **(3) it was cheap to be wrong** - minutes and no training. **The rule this re-confirms: a pre-registered falsifier is only worth writing if it is evaluated on more than the checkpoint that motivated it** - stopping at the first table would have published "+0.84 free points" from a single file, the same error shape as this morning's single-seed +0.50 that had to be withdrawn. **What is now established is that capacity is not the first thing to fix.** Source: `docs/history/CALIBRATION-NOT-THE-LEVER.md`, `tools/logit_adjust.py`, `lens/runs/chat/win6koff_K0_ep6000.npz`, `win6kon_K8_ep6000.npz`, `scan12_r090_6k_ep6000.npz`.

### Calibration line closed: gated adjustment does not replicate either, and the model has no confident set (2026-11-19 19:50)

Global logit adjustment failed to replicate (+0.84 / +0.49 / +0.00), so the one refinement allowed was to gate it on the model's own uncertainty - apply the correction only when the top1-top2 margin is small, protecting confident answers - with **+0.5 points on ALL THREE checkpoints** as the bar:

```
margin   tau   |  win6koff_K0    win6kon_K8    scan12 (untied)
0.5      0.35  |  ov +0.76       ov +0.41      ov -0.29
1.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
2.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
4.0      0.35  |  ov +0.84       ov +0.49      ov -0.37
ungated  0.35  |  ov +0.84       ov +0.49      ov -0.37

P2 settings clearing +0.5 on ALL THREE: 0
FALSIFIER: FIRED - calibration is not the lever and this line closes
```

**The finding inside the failure:** thresholds of 1.0, 2.0, 4.0 and ungated produce **identical numbers to two decimals**, which can only happen if **essentially every scored position has a top1-top2 margin below 1.0** - so there is **no confident set to protect**, and the model is **uniformly uncertain across positions**. Tightening to 0.5 does change the numbers and makes them worse (+0.84 -> +0.76, +0.49 -> +0.41), because the positions it protects were the ones the correction was helping least. **Closed by pre-registration, not opinion:** the plan allowed three refinements (global, cross-checkpoint, gated); all three are measured and none replicates. **Two facts now constrain the next attempt:** (1) the mid band **does** gain +0.47 to +1.67 points under a prior correction on every checkpoint, so the frequency bias is real and exploitable; (2) the **top-20 band pays more than the mid band gains** on the untied model, and there is no confident subset to spare. That points away from post-hoc re-ranking and toward changing the readout's behaviour **during** training - but the honest next step is a different diagnostic first: which specific mid-frequency tokens the model gets right when it does, since that determines whether the 480-token band should be split. Source: `docs/history/CALIBRATION-CLOSED.md`, `tools/gated_replication.py`.

### The cliff is at rank 50, not 500 - and the 8-window figures are corrected by the full sample (2026-11-19 20:00)

Running the mid-band diagnostic at the full **104 windows** and splitting the band produced two things, one of which corrects numbers I published an hour ago from 8 windows.

**1. The published 8-window figures are superseded:** r21-500 accuracy 14.58% -> **8.13%**; predicts a top-20 token 74.48% -> **84.12%**; median rank of the correct token 34 -> **63**; top-100 hit rate 66.67% -> **59.22%**. Every number moved the same way and the conclusion is unchanged and stronger - the failure is frequency bias, not ignorance - but the magnitudes were optimistic and are replaced here.

**2. The band is NOT homogeneous - the cliff is at rank 50:**

```
sub-band   n      accuracy   predicts top-20
20-50      786    24.05%     71.88%
50-100     451     0.22%     95.12%
100-250    690     0.00%     91.45%
250-500    410     0.00%     83.17%

weighted check: (786 x 24.05 + 451 x 0.22) / 2337 = 8.13%   <- reproduces the band figure exactly
```

**Ranks 21-50 sit at 24.05%** - nearly two-thirds of the top-20 rate, so the model is genuinely a **top-50 predictor**; **ranks 50-100 collapse to 0.22%** and 100-500 are **exactly zero** across 1,100 positions. That is a cliff, not a gradient. So the "36.08% of positions in r21-500 worth +4.3 points" framing was too coarse: the headroom is concentrated in ranks 21-50 where the model already works, and in the **cliff at 50+** (1,551 positions, 66% of the band) where it scores nothing. **The top-20 default rate is highest exactly where accuracy is lowest** (95.12% at ranks 50-100) - the signature of a model that answers "a frequent word" whenever it does not know. **Why this kills calibration more completely than the failed replication did:** the prior correction moved the mid band +1.5 points, but 66% of that band sits at **zero** accuracy, and a global re-ranking cannot lift a sub-population at 0.00% - it can only shuffle which frequent token wins. The earlier result is now explained rather than merely observed. **Next target: ranks 50+ specifically** - a per-frequency-band loss re-weighting that EXCLUDES the top-50 (unlike the rarity arms, which re-weighted the whole vocabulary and were refuted), or an untied readout with frequency-tiered capacity; predictions written before launch either way. Source: `docs/history/CLIFF-AT-RANK-50.md`, `tools/midband_split.py`, `lens/runs/MIDBAND-DIAG.txt`.

### The ranks-50+ arm: the loss now points at the measured cliff (2026-11-19 21:00)

The band analysis says the model is a **top-50 predictor** (ranks 21-50 **24.05%**, 50-100 **0.22%**, 100-500 **exactly 0.00%** over 1,100 positions). With calibration closed, this arm changes the **loss during training** instead, and only for the cliff: `forward_backward` takes an optional per-position weight `pos_w` that multiplies the `<unk>` mask before normalisation, so the gradient stays exactly consistent with the weighted loss - **certified by finite differences before launch: 84 params, worst abs error 3.366e-10, and uniform weights reproduce the unweighted loss to 0.000e+00 (verified no-op)**.

**How big the intervention is, measured over 193,725 scored positions using the trainer's own draws:** `rank >= 50` covers **50.06%** of positions and the 4x weight moves them from **50.1% to 80.0% of total loss weight**; `rank >= 20` covers 63.91%; `<unk>` covers 5.41% and is masked. Half of all training positions are up-weighted 4x, so this is a powerful test of whether the long tail responds to loss weight at all - and it puts P2 (overall falling no more than 2 points) genuinely at risk, because the top-20 band that supplies nearly all the measured accuracy keeps only 20% of the weight.

**Predictions, written before the run:** P1 ranks 50-500 rise above **0.22%**; P2 overall does not fall more than **2 points** against `win6kon`; P3 the arm still beats the matched constant. **Falsifier: ranks 50-500 stay at or below 0.22%** -> weighting the loss cannot teach the long tail, the limiter is the **readout's capacity**, and the next experiment must change the readout rather than any loss setting. **Status: 2125 of 6000 steps measured, then stopped for the night** with 8 checkpoints kept and no held-out line yet (correct for a mid-run stop). **Side observation:** `rank >= 20` is 63.91% of **training** positions but top-20 is 45.31% of **validation** positions - the validation region is more top-heavy than training, which inflates val accuracy relative to the training distribution. Recorded, not acted on. Source: `docs/history/RANK-BOOST-ARM.md`, `lens/runs/RANK50-arm.txt`, `lens/scripts/run_rank50_arm.sh`.

### Correction: an unguarded boot re-run overwrote the checkpoint behind the memory-line result, and thread count changes results (2026-11-20 09:20)

The 2026-11-19 memory-line closure (age key end-to-end: trainer **+0.14**, independent **-0.06**) cites `lens/runs/chat/win6kage2_K8_ep6000.npz`. **Those bytes no longer exist.** On the 08:53 boot `lens/scripts/run_after_boot.sh` line 38 re-ran the whole age-key arm **unconditionally** - no "already done" guard - and retrained the checkpoint over the top of it; mtime is now **2026-11-20 09:13**, size 7,404,494, sha256 `48abda39e04092d722ad1cbbe8469996dcb56b95531dec1d87215d8ec3972399` (recorded in `lens/runs/PROVENANCE-win6kage2-2026-11-20.txt`). **The guard** now requires **both** the checkpoint **and** `lens/runs/AGEKEY-E2E-DONE` to be absent, is pinned to two cores, and prints its skip decision - verified `SKIP-NOT-RE-RUN`.

**The unpinned duplicate also stole the live experiment's CPU:** per-25-step wall time on `boost50` went **17s -> 80s (~4.7x)** because the re-run used all 12 cores while the trainer is pinned to 6,7. Duplicate gone; only the trainer remains.

**The finding nobody ordered - the retrain did NOT reproduce the first run:** same seed, same config, same data order, and the trainer line reproduces to a hundredth (**23.02 -> 23.03**) while the independent evaluator moves a full point (**20.56 -> 19.55**). The only difference is thread count: multithreaded BLAS reduction order changes the last bits of every matmul, so an unpinned run is a **different model**, not the same model computed faster. **Consequence: a "same seed, same config" reproduction must also pin the same core count, or it is not a reproduction** - thread count joins draws, windows and denominators as something a paired comparison must hold fixed.

**The published conclusion stands and does not depend on the lost bytes:** trainer **23.02 vs 22.88 (+0.14)** then **23.03**; independent **20.56 vs 20.62 (-0.06)** then **19.55 vs 20.62 (-1.07)** - every version far below the +0.5 bar, the independent sign negative in both, the falsifier still fires. What is lost is exact byte-level reproducibility of the first run's independent figure, and this page now says so rather than citing a file that no longer holds those bytes. Source: `docs/history/PROVENANCE-OVERWRITE-CORRECTION.md`.

### Rank-boost verdict (falsifier fired on all three) and the untied readout is the first thing to ever move the tail (2026-11-20 09:52)

**Part 1 - `boost50` failed decisively** at its pre-registered 6000-step endpoint, against `win6kon`:

```
                          win6kon (control)   boost50      delta
ranks 20-50               25.19%              0.00%        -25.19
ranks 50-100               0.22%              1.33%        +1.11
ranks 100-250              0.00%              0.00%         0.00
ranks 250-500              0.00%              0.00%         0.00
r21-500 overall            8.52%              0.26%        -8.26
top-20 accuracy           35.81%             24.22%       -11.59
trainer line vs constant  22.88% (BEATS)     14.23% vs 14.47% = -0.24 -> BELOW
```

**P1 FAILED** (band fell 8.26 points; ranks 20-50, the only sub-band that ever worked, went to **exactly zero**), **P2 FAILED** (overall -11.59 against a 2-point allowance), **P3 FAILED** (on the trainer's own matched constant the arm scores **below a trivial predictor**). **The falsifier fired exactly as written:** weighting the loss cannot teach the long tail, so the limiter is the **readout**. 4x weight on half of all training positions starved the head so completely the model never beat a frequency table - it bought +1.11 in the 50-100 band for everything else.

**Part 2 - the untied readout moves the tail, which nothing else has.** Prediction written before the measurement: *if the limiter is readout capacity, the untied 6000-step checkpoint must show ranks 50-500 above the 0.22% tied baseline.* Measured on `scan12_r090_6k_ep6000` (untied table, 104 windows):

```
                          tied control   UNTIED      delta
ranks 20-50               25.19%         35.24%      +10.05
ranks 50-100               0.22%          2.66%      +2.44  (12x the tied rate)
ranks 100-250              0.00%          0.29%      +0.29  (from exactly zero)
ranks 250-500              0.00%          0.00%       0.00
r21-500 overall            8.52%         12.45%      +3.93
top-20 accuracy           35.81%         41.61%      +5.80
top-20 default rate        94.90%         70.73%     -24.17  (far less head-biased)
```

**Prediction MET.** First configuration in the entire memory/readout line where the mid band moves up **and** the head improves at the same time; the frequency bias drops 24 points. **The confound, stated before anyone quotes it:** `scan12_r090_6k` is **not** a clean one-change control - untied readout **and** ~3.07M answer-table parameters against the tied arm's 32,768 bottleneck (~2.8x params) **and** a rarity-weighted setting. So this shows capacity in the readout is **sufficient** to move the tail, not that tying alone was the cause. **Launched `untied6k`** - `win6kon`'s exact config with TIE=0 and nothing else changed: **P1** ranks 50-100 above 0.22%; **P2** top-20 no lower than 35.81%; **P3** overall at or above 21.88% on the trainer line; **falsifier:** ranks 50-100 at or below 0.22% -> the scan12 result was its config, not the readout, and capacity is not the lever either. Source: `docs/history/RANK-BOOST-VERDICT-AND-UNTIED-TAIL.md`.

### The untied-readout finding survives a different draw - and a silently ignored seed was caught by identical numbers (2026-11-20 10:17)

**The tool could not be re-drawn and said so silently:** `tools/midband_split.py` had `rr = np.random.default_rng(4242)` hardcoded, so `DRAW_SEED=777` was **ignored** and both checkpoints returned identical numbers to the last decimal - that identity is what exposed it, since a genuine re-draw cannot produce the same digits. (My first patch attempt also applied nothing at all: the regex looked for `rng` while the variable is `rr`.) Fixed properly - `DRAW_SEED` is read from the environment **and printed with every run**, so a reader can always see which draw produced a number. This is the fifth silently-ignored flag in this project, and the rule that catches all of them is already written: **a flag that does nothing must make the tool refuse to run, and every tool must print the settings that shaped its output.**

**The finding survives the re-draw** (different windows: sub-band counts 786/451/690/410 become 777/541/656/422):

```
draw 4242        tied     UNTIED      draw 777      tied     UNTIED
ranks 20-50      25.19%    35.24%      ranks 20-50  21.62%    31.02%
ranks 50-100      0.22%     2.66%      ranks 50-100  0.18%     2.77%
ranks 100-250     0.00%     0.29%      ranks 100-250 0.00%     0.76%
r21-500 overall   8.52%    12.45%      r21-500       7.05%    10.89%
top-20 accuracy  35.81%    41.61%      top-20       34.78%    43.07%
top-20 default   94.90%    70.73%      default      85.64%    71.83%
```

Both draws agree in sign and rough magnitude on every band, including the decisive one: the tail rate goes from **~1 position in ~500 to 12-15 positions in ~500** under an untied readout (1 of 451 -> 12 of 451; 1 of 541 -> 15 of 541). Absolute numbers move a point or two between draws - the measured draw noise (sd 0.93) - but **the sign never flips**. The claim stands with its confound still attached: capacity in the readout is **sufficient** to move the tail, on two independent draws; the clean attribution test `untied6k` is running. Source: `docs/history/UNTIED-TAIL-SURVIVES-REDRAW.md`.

### A guard against dead flags - and building it took two bugs in the guard itself (2026-11-20 10:32)

Three incidents today had one shape: a launcher passes a setting and the code never reads it (`RESUME=` ignored by the trainer so a "continuation" restarted from step 0; `DRAW_SEED=` ignored by midband_split so a "re-draw" measured the same windows, with identical digits the only clue; and my own patch matching `rng` while the variable is `rr`, applying nothing and reporting success). So `tools/check_env_wiring.py` takes a launcher, finds every UPPERCASE variable in the **env prefix of a python invocation**, and checks the target file (plus one or two hops of project-local imports) actually mentions the name - anything set but never read is DEAD, with a non-zero exit so a launcher can gate on it.

**It took two bugs of its own, caught by controls rather than by reading it:** (1) it first flagged shell bookkeeping (`RC=$?`, `CK=`, `FAIL=`) as dead flags - **12 false positives** across four launchers; (2) after restricting it to the env prefix, it reported **0 flags on everything including a planted dead flag**, because the preceding continuation lines were appended to the END of the joined string so `split(".py")[0]` cut off exactly the prefix being searched for. **A guard that silently never fires is worse than no guard, because it manufactures confidence.**

**Validated both ways before trusting it:**

```
planted dead flag  TOTALLY_DEAD_FLAG=1  -> DEAD FLAGS TOTAL: 1, exit 1   (caught)
live flag          RANK_BOOST=1.0       -> DEAD FLAGS TOTAL: 0, exit 0   (no false positive)
four real launchers (rank50, rank50resume, untied6k, agekey) -> 0 dead flags
```

The negative control matters as much as the positive one: a checker that flags everything is as useless as one that flags nothing. **Honest limit:** it is a name-based heuristic - it proves a name is never mentioned, not that a value is used correctly. That is exactly enough to catch all three incidents, because in each case the name appeared nowhere. Source: `docs/history/DEAD-FLAG-GUARD.md`, `tools/check_env_wiring.py`.

### Untied readout, clean test: the falsifier fired, and this corrects this morning's claim (2026-11-20 12:10)

This morning I published "the untied readout is the first thing to ever move the tail", with its confound stated. The clean one-change test has finished and **the confound was the cause**. `win6kon` (tied) vs `untied6k` (identical config, `TIE=0`, nothing else changed; same seed, batch order, 6000 steps, draw seed 4242, same 104 windows):

```
                          tied control    untied (only change)
ranks 20-50                25.19%          21.25%     -3.94
ranks 50-100                0.22%           0.00%     -0.22   <- the band the whole idea rested on
ranks 100-250               0.00%           0.00%      0.00
r21-500 overall             8.52%           7.15%     -1.37
top-20 accuracy            35.81%          34.79%     -1.02
trainer vs constant        22.88% BEATS    22.40% BEATS (+7.93)
```

**P1 FAILED** (ranks 50-100 at **0.00%**, at or below the 0.22% baseline), **P2 FAILED** (top-20 fell 1.02 points), **P3 PASSED** (22.40% is inside the 21.88% floor and beats the constant by 7.93, so the arm trained properly), **FALSIFIER FIRED exactly as pre-registered**. **The correction:** the tail movement I attributed to untying came from **the rarity-weighted setting and the extra parameters** - untying by itself makes every band slightly worse. **Where the lever is not:** loss weighting (14.23% vs a 14.47% trivial constant), calibration (+0.84/+0.49/+0.00, no replication), readout shape (today), depth (5.68/5.08 vs 6.07). All four changed the *strength* or *post-processing* of learning; none changed **what is supervised at each position**. The measured reason to expect something there is the delayed-recall probe: with a target that genuinely requires memory, the readable bank plus its age key lifted recall **20.61% -> 85.61%** over three seeds. So the next arm trains with the target at **offset 2**, with and without the readable bank. Source: `docs/history/UNTIED-READOUT-CLEAN-TEST.md`, `lens/runs/chat/untied6k_K8_ep6000.npz`, `lens/runs/chat/win6kon_K8_ep6000.npz`.

### Custom architecture, concepts load-bearing (scoped: 20k windows, 1 epoch) - 2026-11-20 12:33

`--direct` ON vs OFF, both arms identical otherwise, `--method ours`:

```
arm A  --direct ON    final VAL ppl 3841.51   acc 4.11%   concepts 13/36
arm B  --direct OFF   final VAL ppl 4520.79   acc 3.95%   concepts 14/36
```

**P2 PASSED** - removing the shortcut costs **0.16 points** (4.11 -> 3.95), far inside the 2.0 bar, with perplexity rising 3841.51 -> 4520.79 as the counterweight. **P3 PASSED** - passing concepts **13/36 -> 14/36** and seven of eight listed AUROCs improve (body 0.63923->0.65524, is_a_list 0.62948->0.63468, politics 0.55818->0.57263, a_size 0.57875->0.58962, law 0.54896->0.55316, colour 0.56389->0.56521, quantity 0.52239->0.52508). **P1 UNMEASURED and my pre-registration was flawed:** `train_chat.py` prints no concept-share metric, so the run could not answer its own primary question; and my script passed no `--out`, so both arms wrote `run1` and the second **overwrote** the first's checkpoint. Fixed with per-arm `--out`, clobbered file kept as `lens/runs/chat/ARCH-SMOKE-UNTAGGED-run1_ep1.npz` (sha256 171350909327a865), pair re-launched as `customarch2`. **The one thing this establishes:** removing the shortcut makes concept labels **more honest** while costing 0.16 points - the first time readability and accuracy moved the same way instead of trading off. Scoped to 20k windows; the full corpus under this recipe is 10.8-13.8 h per arm. Source: `docs/history/CUSTOM-ARCH-DIRECT-SCOPED.md`, `lens/runs/chain_arch_direct_on.out`, `lens/runs/chain_arch_direct_off.out`.

### P1 measured with the corrected instrument: the bottleneck moves share but not specificity - 2026-11-20 13:15

This SUPERSEDES the "P1 UNMEASURED" line in the scoped custom-arch entry above. The pair re-ran with per-arm `--out` tags and was
measured with the signed instrument, 96 usable windows per arm:

```
                          --direct ON        --direct OFF
signed concept share      88.76%             100.00%
direct term (max)         1.7263             0.0000
named / random ablation   5.39x              4.78x
concepts for 90% margin   22 of 36           22 of 36
reconstruct |error|       1.076              1.083
```

**P1 MET only in the trivial sense:** with the shortcut removed there is no other path, so 100% was guaranteed by construction.
**The informative measure moved the wrong way** - the named-vs-random ablation ratio FELL 5.39x -> 4.78x and the concepts needed
for 90% of the margin are identical at **22 of 36**. Raising share does not make concepts more *specifically* responsible. **Caveat:**
`C + D` misses the real predicted logit by up to **1.076/1.083**, so the split is approximate and no sharper claim is supported.
Source: `docs/history/CONCEPT-SHARE-MEASURED.md`, `lens/runs/chat/arch_direct_on_ep1.npz`, `lens/runs/chat/arch_direct_off_ep1.npz`.

### The concept layer answers by cancellation - 2026-11-20 13:35

Exact attribution (`contribution_i = h_i * (W[i,pred] - W[i,runner])`, runner-up held fixed so the split is additive and checkable;
residual 4.134e-02 OFF / 3.445e-01 ON, equal to the class-bias difference and named as such):

```
                                    --direct ON        --direct OFF
concepts supplying POSITIVE margin  17 of 36           17 of 36
positive supply vs the net margin   16.43x             9.45x
top-8 of the positive suppliers     68.7% of positive   67.4% of positive
strongest supplier                  is_a_request +223.8% is_a_request +123.9%
strongest opposer                  mentions_technology -262.0%  -153.7%
```

**P1 NOT MET both arms** (68.7%/67.4% against an 80% bar - attribution is spread over 17 concepts). **P2 MET**: the same concepts
lead in both arms, so the load-bearing set belongs to the architecture and data, not the arm. **The finding:** the concepts supply
**9.4x to 16.4x** the net margin - the answer is the small **residue of large opposing pushes**, with 19 of 36 concepts opposing and
`mentions_technology` alone at **-262%**. This is the structural explanation for the earlier 60x share overstatement and for
`explain()` being exact on 0 of 200 windows. **My first instrument was invalid, not the prediction:** ablating against a
re-computed runner-up gave -302%/-204% and is non-additive because ablation can change which class is second. Source:
`docs/history/CONCEPT-CANCELLATION.md`, `lens/runs/CONCEPT-EXACT-ON.json`, `lens/runs/CONCEPT-EXACT-OFF.json`.

### A bigger budget reverses the verdict, and cancellation gets worse - 2026-11-20 12:49

```
budget            --direct ON            --direct OFF            gap
20k windows    acc 4.11%  (13/36)     acc 3.95%  (14/36)      -0.16   <- P2 'passed'
100k windows   acc 12.18% (25/36)     acc 7.31%  (25/36)      -4.87   <- P2 FAILS
```

**Correction 1:** P2's bar is 2.0 points; at 100k the shortcut costs **4.87**, so "readability is nearly free" was a 20k-budget artefact. On this architecture removing the direct path costs real accuracy; both arms do reach **25/36** concepts with budget. **Correction 2:** cancellation gets worse, not better - positive supply **16.43x -> 8.476x** for the direct-path arm as it gains accuracy (4.11 -> 12.18%), but **9.45x -> 236.032x** for the bottlenecked arm as its accuracy falls to 7.31%, with `mentions_technology +3579.2%` and `mentions_sport +2599.5%`. Forcing the answer through the concepts makes them **fight harder**, which is the strongest evidence against a concept bottleneck and for the direct path. **Provenance damage, mine:** the arm tag carried no budget, so the 100k run **overwrote** the 20k checkpoints at 12:44/12:49; the 20k attribution survives only in `CONCEPT-EXACT-ON/OFF.json`. Tag now carries the budget (`arch_$1_m${MAXW}w`). Source: `docs/history/BUDGET-REVERSAL-AND-CANCELLATION.md`, `lens/runs/chain_arch_direct_on.out`, `chain_arch_direct_off.out`, `lens/runs/CONCEPT-EXACT-100K-ON.json`.

### Decorrelation removes 73.5% of the cancellation, for free - 2026-11-20 12:56

```
                        control (lam_decorr 0)   decorrelated (0.2)     delta
held-out accuracy       7.31%                    7.25%                  -0.06
perplexity              953.87                   955.09                 +1.22
concepts passing audit  25 of 36                 25 of 36               0
POSITIVE SUPPLY         236.039x margin          62.600x margin         -73.5%
top-8 share of positive 72.3%                   70.7%                  -1.6
reconstruct residual    2.556e-01                2.544e-01              closes
```

**The arm's P1 is MET:** cancellation falls **236.039x -> 62.600x (-73.5%)** against a 20% bar, for **-0.06 points** of accuracy (P2 PASSED). **My own script hid this** by labelling two different predictions "P1" - the arm's supply test and the tool's top-8 concentration test - and the log printed only the tool's, so a working result read as a failure. **Concentration is still NOT MET** (top-8 = 70.7% vs 80%). The control **reproduces the independent 100k arm exactly** (953.87 ppl / 7.31% / 236.03x), so the cancellation is a stable property of the architecture. **Residual resolved:** `sum(contribution)` can only reconstruct `(h@W)[pred]-(h@W)[runner]`, so the remainder is the class-dependent `(b+direct)` term; the 236x figure comes from the arm whose decomposition closes to **0.256**, so it is trustworthy while the `--direct` 8.476x carries that caveat. Source: `docs/history/DECORRELATION-REMOVES-CANCELLATION.md`, `lens/runs/DECORR-CANCEL.txt`, `lens/runs/DECORR-SUPPLY-ctrl.json`, `lens/runs/DECORR-SUPPLY-d02.json`.

### Decorrelation saturates at 0.5 and the falsifier fired - 2026-11-20 13:02

```
                                    accuracy   ppl      positive supply
control            lam_decorr 0     7.31%      953.87   236.039x
d02                lam_decorr 0.2   7.25%      955.09    62.600x
d05                lam_decorr 0.5   7.19%      956.79    38.051x
d10                lam_decorr 1.0   7.13%      959.33    38.356x
```

**P2 PASSED everywhere** (accuracy within 0.18 points). **P1 FAILED**: supply(1.0) is **+0.8% above** supply(0.5), not 20% below. **FALSIFIER FIRED as written** - decorrelation **saturates at 0.5** and the line **CLOSES**: 236.039x -> 62.600x -> 38.051x -> 38.356x, so the first two steps do all the work and the third does nothing. The remaining **38x** cancellation is **structural, not a tuning problem**, and the pre-registered remedy is **FEWER, WIDER concepts** - which needs a code change, because `--help` exposes no concept-count knob. **Instrument note:** for `d10` the log printed CONCENTRATION **MET** from the non-additive re-computed-runner path and **NOT MET (72.5%)** from `--fixed-runner`; only the fixed-runner number counts, and it says concentration is **not** met. Source: `docs/history/DECORRELATION-SATURATES.md`, `lens/runs/DECORR-SWEEP.txt`, `lens/runs/DECORR-SWEEP-d05.json`, `DECORR-SWEEP-d10.json`.

### Fewer concepts cuts cancellation ~48% and every concept passes audit - 2026-11-20 13:09

```
                              36 concepts (d05)     9 concepts (narrow)     delta
held-out accuracy             7.19%                 7.37%                   +0.18
perplexity                    956.79                1275.33                 +318.54   <- worse
concepts passing audit        25 of 36  (69%)       9 of 9  (100%)          +31 points
POSITIVE SUPPLY               38.051x margin        19.839x margin          -47.9%
```

**P1 MET** (bar was <=30.44x; actual **19.839x**), falsifier did **not** fire, line stays open. **P2 PASSED** (+0.18 points). First time three things moved the right way together: cancellation nearly halved, **every** concept passes audit (9/9 vs 25/36), and accuracy did not pay. **Three caveats:** perplexity is **33% worse** (956.79 -> 1275.33) while accuracy is flat-to-better, so the honest summary is accuracy flat / calibration worse; the concentration test is **vacuous at 9 concepts** (its top-8 bar is the whole layer when only 5 units are positive - the tool's "MET" means nothing and the bar needs restating as a fraction); and count and breadth are **confounded**, so this is "fewer, wider" as specified, not a pure count effect. **Verified not inferred:** the log prints `concepts 9/9`, so the arm really did load the 9-concept catalogue. Source: `docs/history/FEWER-CONCEPTS-HALVES-CANCELLATION.md`, `lens/runs/CONCEPT-COUNT.txt`, `lens/runs/chain_narrow.out`, `lens/runs/CONCEPT-COUNT.json`.

### One count-invariant scale for every concept arm - and it corrects the previous claim - 2026-11-20 13:15

The "top-8 >= 80%" concentration bar is **vacuous at 9 concepts** (top-8 of 5 positive units is the whole layer), so the instrument now reports **top-half share** and **supply entropy / log(n_pos)** (1.00 = perfectly spread). Every arm re-scored from saved JSON, no retraining:

```
result file                   n   n+    supply  top-half entropy   recon
CONCEPT-EXACT-100K-ON        36   20     8.48x    79.6%   0.903   4.63e+00
CONCEPT-EXACT-OFF            36   17     9.45x    74.2%   0.950   4.13e-02
CONCEPT-EXACT-ON             36   17    16.43x    74.6%   0.949   3.45e-01
CONCEPT-COUNT  (9 concepts)   9    5    19.84x    79.7%   0.927   7.10e-01
DECORR-SWEEP-d05             36   17    38.05x    76.6%   0.943   2.53e-01
DECORR-SWEEP-d10             36   19    38.36x    81.2%   0.906   2.50e-01
DECORR-SUPPLY-d02            36   17    62.60x    76.7%   0.943   2.54e-01
CONCEPT-EXACT-100K-OFF       36   19   236.03x    81.6%   0.906   2.56e-01
DECORR-SUPPLY-ctrl           36   19   236.04x    81.6%   0.906   2.56e-01
```

**Concentration is FLAT across every configuration: top-half 74.2-81.6%, entropy 0.906-0.950.** **Correction to the previous entry:** "fewer concepts cuts cancellation 48%" is true as a statement about **volume** (38.05x -> 19.84x), but the 9-concept arm's distribution shape (79.7% / 0.927) sits **inside the same band** as the 36-concept arms. Fewer concepts reduced **how much** the units push, not **how concentrated** the push is. Since the goal is "a human reads a few rules and knows the answer", **no configuration measured so far produces that**, at any count or strength. Two JSONs from the non-additive path are labelled **not quotable** and kept, not deleted. Source: `docs/history/CONCENTRATION-IS-INVARIANT.md`.

### The falsifier fired: the concept line closes at this scale - 2026-11-20 13:12

```
configuration                          acc      ppl      audit     supply
36 concepts + direct (decorr 0)     12.18%   712.92  25/36 (69%)    8.476x   <- most accurate
36 concepts, no direct (decorr 0)    7.31%   953.87  25/36 (69%)  236.039x
9 concepts, no direct (decorr 0.5)   7.37%  1275.33   9/9 (100%)   19.839x   <- most honest
9 concepts + direct (decorr 0.5)     9.35%  1269.90    5/9 (56%)   80.896x   <- neither
```

**narrowdir P1 FAILED** (9.35% is 2.83 below 12.18, audit 5/9 vs required 8/9), **P2 FAILED** (80.896x vs required <8.476x), **FALSIFIER FIRED** by its own condition (acc >2.0 below AND audit <6/9). The two good properties **refuse to combine**: narrowing buys honesty only in the bottlenecked shape (9/9) and destroys it with the direct path present (5/9, worse than 69%); the direct path buys accuracy and the least cancellation but leaves concepts a passenger. **Closes:** concept count is not the lever; **no configuration at 100k is both honest and accurate**; concentration never moved. **This is the honest answer to "is our readable architecture better than a regular model": at this scale it is not, and there is still **no plain matched control for the chat architecture** to measure that comparison - gap recorded in QUEUE.txt. One arm left open: the winner (36+direct) was only ever trained at `lam_decorr 0`, so it is tested at 0.5; if it loses more than 1.0 point, no further concept arm of any kind runs. Source: `docs/history/CONCEPT-LINE-CLOSED.md`, `lens/runs/NARROW-DIRECT.txt`, `lens/runs/NARROW-DIRECT.json`.

### Decorrelation is a no-op on the winner, and the plain control is blocked by design - 2026-11-20 13:15

```
                              reference (decorr 0)     windec (decorr 0.5)    delta
held-out accuracy             12.18%                   12.10%                 -0.08
perplexity                    712.92                   714.51                 +1.59
POSITIVE SUPPLY               8.476x                   10.107x                +19%
```

**P1 NOT MET** (accuracy within 0.5 is fine but supply went 19% the wrong way), **falsifier did NOT fire** (drop only 0.08). The reading is **"decorrelation does nothing here"**, not "it hurts": it is a lever only on the bottlenecked shape (236x -> 38x) and a no-op on the configuration that performs. Best setting stays `lam_decorr 0` by parsimony. **Concept line closed with full accounting** - count is not the lever, decorrelation is not the lever on the winner, concentration never moves (top-half 74-82%, entropy 0.906-0.950), and no configuration is both honest and accurate. **Plain control blocked by design, verified not assumed:** `lens.Net(vocab=100, concepts=[], ...)` raises `ValueError: LensNet needs at least one named concept; the concept layer IS the architecture.` So a regular-model control is a **new model class**, not a flag - queued with that spec and evidence rather than rushed. **Until it exists, no custom-vs-regular claim has been measured on the chat task**; the only matched figure is the image model's **-0.08 = parity**. Source: `docs/history/WINDEC-NOOP-AND-PLAIN-CONTROL-BLOCKER.md`, `lens/runs/WINNER-DECORR.txt`, `lens/runs/WINNER-DECORR.json`.

### Two closures: memory bank worth +0.49, and readable is 1.53 points worse at matched params - 2026-11-20 14:26

**1. Memory bank, decisive offset-2 test** (the target moves 2 ahead so the preceding token cannot answer it; arms differ only in K):

```
                          held-out acc   vs constant 14.47%
bank    K=8               16.57%         +2.10  BEATS
no bank K=0               16.08%         +1.60  BEATS
paired delta              +0.49
```

**P1 (+1.0 bar) NOT MET**, **FALSIFIER FIRED** (0.49 <= 0.5 by its own condition), **P2 PASSED** (both beat the constant, so the comparison is valid). By its pre-registered rule the **memory line is closed permanently** - no further seed or weight. It retains content in isolation (81.25% at delay 16 vs 3.22%) and does not convert it to next-token accuracy at any offset tested.

**2. Custom vs regular at matched parameters - readable LOSES.** The 3-seed run had already finished; its summary was **stale and wrong** ("NO RESULTS - every arm failed" from a pre-2026-11-18 glob). Recomputed from the per-seed JSONs and **deduplicated** (the `recovered/` copies are byte-identical, sha256 confirmed, so 3 seeds not 6):

```
seed   readable    plain     gap
 0     84.36%     86.42%   -2.06
 1     84.27%     85.81%   -1.54
 2     83.69%     84.67%   -0.98
mean: -1.53 points, sd 0.44   |  params 16,665 vs 16,705 (0.24% apart = matched)  |  honesty 97.4%
```

**P1/P2/P3 HOLD, falsifier not triggered.** So at matched parameters over 3 seeds the readable architecture is **1.53 points worse** than a plain network while being **97.4% honest**. **Honest is not better.** "10x better than a normal model" measures at **-15% relative** on the only matched task available. Chat has **no plain control** (refused by design, verified), so no chat claim exists. Source: `docs/history/TWO-CLOSURES-MEMORY-AND-MATCHED-GAP.md`, `lens/runs/OFF2-PAIR.txt`, `lens/runs/MATCHED-GAP-3SEED.txt`, `lens/runs/image/**/match3_s*_log.json`.

### Anatomy scan: what is actually inside, tensor by tensor - 2026-11-20 14:45

`tools/anatomy.py` inventories a checkpoint with no training and no data. Every number summed from the models own arrays; full inventory in `lens/runs/ANATOMY.json`.

```
architecture                          params      table%  compute%  concept%   state per token
LensNet 36 concepts + direct        2,225,284      89.0     11.0       1.7     none (window only)
LensNet  9 concepts + direct        1,873,609      88.4     11.6       0.5     none
tokmem + readable window bank       1,848,064      83.1     16.9       -       256 floats + bank
```

**The readable part is 1.7% of the model** (0.5% at 9 concepts). **69.0%** of `arch_direct_on` is the token lookup table (1,536,000 of `emb`), 20.0% the answer matrix, 9.4% the direct shortcut, and the entire named concept layer is **36,900 parameters**. The honest anatomy of the readable model is a word table with a small concept hat. **Two self-catches:** my role map was too thin and reported 208,384 params as OPAQUE (`dout`/`dproj` are the direct path under another name) - now **100.0% named**, because "transparent" must mean "we found everything"; and **state per token is none for LensNet** (no carried state at all, the 8-token window is the whole context) vs **256 floats + bank** for tokmem - so own-KV-cache work belongs to the recurrent family or a new attention block, not to LensNet. Source: `docs/history/ANATOMY-SCAN.md`, `lens/runs/ANATOMY.json`.

### Our own block, with our own cache, and three proofs it is right - 2026-11-20 15:08

`lens/glass_block.py`: LayerNorm, multi-head causal attention, MLP, residuals, and **our own KV cache** - **544 named parameters and nothing else to learn**; LayerNorm exposes the mean/variance it used; the causal mask is an explicit object; the cache prints its exact shape and byte cost.

```
TEST 1  attention + residual vs HAND reference (python loops, math.exp, no shared code)
  max |difference| = 4.441e-16   PASS
TEST 2  KV-cache equivalence: 4 tokens all-at-once vs one-at-a-time
  max |difference| over all 4 prefix positions = 1.192e-07   PASS (float32 rounding)
  KV cache: 1 layer x 2 heads x 4 tokens x 4 wide x 2 x 4 bytes = 256 bytes
TEST 3  LayerNorm vs hand computation (mean 3.000000, var 3.500000)
  max |difference| = 0.000e+00   PASS
```

**Two defects fixed in my own test first:** the hand reference omitted LayerNorm the block applies (TEST 1 would have failed on correct code) - fixed with an explicit `norm=False` switch so each test verifies ONE mechanism, and LayerNorm got its own hand test; and the cache byte report printed **0 bytes** because it read an outer cache never appended to - now reports the final stepped cache and **asserts** its token count. **Honest status: forward is proven, BACKWARD DOES NOT EXIST**, so the block cannot be trained yet - that is the next piece, gated on finite-difference checks with worst error < 1e-8. The "10x smarter per layer" gate was written first: **10x fewer parameters at equal accuracy, or 10x less compute at equal accuracy**. Source: `docs/history/GLASS-BLOCK-PROVEN.md`, `lens/glass_block.py`.

### Our own block learns real text: held-out 12.32% vs a 4.08% matched constant - 2026-11-20 16:05

600 steps, 286 s, 20,000 real-text windows. **836,352 parameters in 24 named tensors** (E 768,128 + P 2,048 + 2 blocks), tied output (`logits = h @ E.T`), our own momentum SGD, forward proven against a hand reference and backward certified at **8.815e-07**.

```
matched constant          4.08%   (most frequent TRAIN target after excluding <unk>: token 3)
HELD-OUT  loss 5.5956     acc 12.32%
delta                     +8.24 points -> BEATS the constant
```

Loss fell 9.4038 -> 5.4656 with no divergence; train ended 12.18% so held-out is not below train (no overfit signal). **P1 MET** (+8.24 against a +1.0 bar), **P2 MET** (largest upward step +0.60, under the 1.0 rule), **falsifier not triggered**. **Does not say:** it is competitive with anything - 600 steps at d=64 on 20k windows is not the same budget as the project's 6000-step arms, and no such comparison is claimed. **The flag that became the next job: E is 768,128 of 836,352 parameters = 92%** - the same lookup-table dominance the anatomy scan found (69% in the old shape). A 92%-table model can only memorise, so the next piece is a **factored embedding** (E = A @ B, rank r) with its own gradient check. Source: `docs/history/GLASS-LM-FIRST-RUN.md`, `lens/runs/GLASSLM.txt`.

### Factored embedding: 3.2x fewer parameters for 1.47 points - P1 missed, falsifier not fired, one retry owed - 2026-11-20 16:23

Both arms on the current code, same seed, same 20k windows, same 600 steps:

```
embedding              params     table share   held-out acc   delta vs 4.08% constant
full table E          836,352        91.8%         12.33%            +8.25
factored A @ B r=16   261,280        73.9%         10.86%            +6.78
```

**P2 MET** (params fall **3.20x**). **P1 NOT MET** (bar >= 11.32%, actual **10.86%** - missing by 0.46, i.e. 1.47 below the table). **FALSIFIER DID NOT FIRE** (needed below 10.32%), so the pre-registered action is **one retry at rank 32** - not a new idea, not a sweep. **The number that matters more:** even factored, the output table is still **73.9% of the model** (was 91.8%). Cutting rank moved the share the right way but nowhere near enough - the layers still hold a quarter of the capacity, and rank 32 would push the share back toward 85%, so it can rescue the accuracy question but **cannot** shift capacity into the layers. Both are recorded so the retry cannot be mistaken for progress on the real goal. Source: `docs/history/FACTORED-EMBED-R16.md`, `lens/runs/GLASSLM-EMBED.txt`, `lens/runs/chain_glasslm_emb0.out`, `chain_glasslm_emb16.out`.

### Factored rank 32: equal accuracy at 1.84x fewer parameters - 2026-11-20 16:32

```
embedding          params     table share   held-out acc   vs full table
full table E      836,352        91.8%         12.33%         -
factored r=16     261,280        73.9%         10.86%      -1.47
factored r=32     454,336        85.0%         12.51%      +0.18
```

**P1 MET** (12.51% vs a >= 11.33% bar, and 0.18 above the full table). **P2 MET** (454,336 params, **1.84x smaller**). **Falsifier not triggered** - factored embeddings work at rank 32. **May be said:** a factored output table reaches the same held-out accuracy with **1.84x fewer total parameters**. **May NOT be said:** that it is more accurate - **+0.18 on one seed at 600 steps is inside noise** (this project measured sd 0.93 draw noise on a neighbouring task, and no seed sweep has run here), so the supported claim is "equal within noise". **Capacity question still unanswered, as predicted before the run:** table share went **73.9% -> 85.0%**, the wrong direction - the model got *smaller*, not *more layer-heavy*. "Smaller at equal accuracy" achieved; "capacity in the layers" not advanced by this mechanism. Source: `docs/history/FACTORED-EMBED-R32.md`, `lens/runs/GLASSLM-R32.txt`, `lens/runs/chain_glasslm_r32.out`.

### Three seeds: equal within noise at 1.84x fewer params, and the +0.18 was a lucky draw - 2026-11-20 17:20

```
seed   full table   factored r=32   paired delta
 0       12.33%        12.51%          +0.18
 1       12.68%        12.05%          -0.63
 2       13.27%        12.78%          -0.49
mean     12.76%        12.45%          -0.31   (sd of deltas 0.35)
```

**P1 MET** (mean delta **-0.31**, inside -1.0..+1.0), **P2 MET** (every seed of both arms lands 11.3-13.5%, no collapse), falsifier not triggered. **Correction to the single-seed message:** rank 32 looked **+0.18** better; three seeds show that was **seed 0 being lucky** - the next two seeds are both negative. Defensible claim: a factored output table reaches the **same held-out accuracy within noise** (mean **-0.31 +- 0.35**) with **1.84x fewer total parameters** (454,336 vs 836,352). It is **not** more accurate. **Programme state:** scan / block / certificate / training / factored embedding all done; **capacity in the layers, automatic allocation, and the memorise-vs-generalise switch are not started** - and those three are what was actually asked for. Source: `docs/history/FACTORED-3SEED.md`, `lens/runs/GLASSLM-3SEED.txt`, `lens/runs/chain_gl3_s*_r*.out`.

### Per-tensor adaptive step sizes do NOT remove the learning-rate knob - 2026-11-20 17:56

```
arm                        lr      held-out acc   outcome
momentum (one lr)          0.05      12.51%       reference
momentum (one lr)          0.50       0.00%       DIVERGED, loss nan by step 300
adaptive (per-tensor)      0.50       0.03%       sank instead (loss 9.41 -> 27.57)
```

**P1 CONFIRMED** (a constant 10x too high destroys the model). **P2 FAILED** - the adaptive rule was meant to land within 1.0 of 12.51% and landed **12.48 below**. **FALSIFIER FIRED**, so the owed per-element retry is running. **Diagnosis that decides what's next:** normalising a gradient by its magnitude controls direction and relative scale, but not the **absolute** effect of a step - with gradients at order 1, `lr = 0.5` still means moving half a unit in every coordinate, which is why the arm sank instead of diverging. **Gradient-magnitude normalisation is the wrong family**; removing a learning-rate knob needs a rule that measures the loss's **response** to the step and grows/shrinks it (a gain rule), pre-registered separately. **Two runner defects, both mine, fixed:** a diverged arm aborted the run, and then a completed-but-bad arm aborted it too - one nonzero exit code covered both "bad result" and "crash". The trainer now prints an explicit **`OUTCOME:`** line and the runner aborts **only when it is missing**. Source: `docs/history/ADAPTIVE-STEPS-FAILED.md`, `lens/runs/chain_glopt_*_r2.out`.

### The owed retry failed too: gradient-normalisation closed, gain rule is next - 2026-11-20 18:20

```
optimiser                     lr / start    held-out acc   verdict
momentum (one lr)                0.05          12.51%      reference
momentum (one lr)                0.50           0.00%      DIVERGED, loss nan by step 300
adaptive per-tensor              0.50           0.03%      sank, 9.41 -> 27.57
adaptive per-ELEMENT             0.50           0.26%      sank, ~27.6
adaptive per-ELEMENT             0.05           0.00%      sank  <- broken at the lr that WORKS for plain momentum
adaptive per-ELEMENT             5.00           0.01%      sank
```

**Falsifier fired in its strongest form:** per-element is broken at **every** lr, including **0.05 where plain momentum reaches 12.51%** - not knob-insensitivity but an optimiser that cannot train this model at any setting. **Why:** per-element normalisation turns the update into sign-like steps of uniform magnitude, so unimportant coordinates move as far as critical ones; all three arms converge to the same loss ~27.6. **Gradient-magnitude normalisation is CLOSED** - no third variant, as pre-registered. **The mechanism the evidence demands:** choose the step by **measuring its effect on the loss** (one global direction `-g/||g||`, take the step, keep it if the loss fell else halve, up to 4 tries; grow 1.3x after a first-try accept), so the initial value is a guess, not a setting. Smoke: loss **9.38 -> 8.10**, step self-adjusted **0.5 -> 2.0 -> 1.25**, beat its constant. Real test running on the two initial values that defeated everything else (**0.50** and **0.005**) vs the 12.51% reference; both must land within 1.0 point, falsifier at 2.0 below (which would abandon step-size automation and move to capacity allocation). Source: `docs/history/GAIN-RULE-NEXT.md`, `lens/runs/GLASSLM-OPT-r3.txt`, `lens/runs/GLASSLM-GAIN-r4.txt`.

### Step-size automation is abandoned: it trains, but gives up 3.6-6.6 points to a hand-set constant - 2026-11-20 18:44

```
optimiser                  setting      held-out acc
momentum (hand-set)        lr 0.05        12.51%     <- shipping choice
gain (loss response)       start 0.50      8.90%     beat constant, 3.61 below reference
gain (loss response)       start 0.005     5.89%     beat constant, 6.62 below reference
adaptive per-element       lr 0.50         0.26%
adaptive per-tensor        lr 0.50         0.03%
adaptive per-element       lr 0.05         0.00%
momentum (hand-set)        lr 0.50       DIVERGED
```

**P1 FAILED** (gain@0.50 is **3.61** below 12.51), **P2 FAILED** (gain@0.005 is **6.62** below), **FALSIFIER FIRED on both** - so by pre-registration **step-size automation is abandoned**, hand-set lr 0.05 is the documented shipping choice, and work moves to capacity allocation. **Why:** the gain rule accepts a step if the loss on **that batch** fell - a greedy myopic test, which can accept steps that help the batch and not the objective; its size climbed to 1.11-1.58 and stayed. **The ordering is the lesson:** measuring a step's effect beats rescaling the gradient (the normalisation family cannot train at all) but loses to a constant chosen against the real objective - automation bought robustness at an accuracy cost too large to ship. **Untouched and next:** capacity allocation and the memorise-vs-generalise switch; neither is affected by any of this. Source: `docs/history/STEP-SIZE-AUTOMATION-CLOSED.md`, `lens/runs/GLASSLM-GAIN-r4.txt`.

### Capacity allocation: a 30%-table config beats the 85%-table default with 29% fewer parameters - 2026-11-20 19:12

**The first measurement in this programme that moves capacity out of the lookup table.** All four configs, 300 steps, seed 0, momentum lr 0.05:

```
config                     params     table share   held-out acc   vs default
A  default  rank32 d64  L2   454,336      85.0%         10.29%          -
B           rank8  d96  L3   322,384      30.0%         10.74%       +0.45
C           rank16 d96  L2   345,056      56.1%         10.87%       +0.58
D           rank8  d128 L2   364,560      26.6%          5.47%       -4.82
```

**P1 MET** (best beats default by **+0.58** vs a +0.5 bar). **P2 MET** (config **B** is **29% smaller** and still beats the default). **Why B is the interesting row:** it moved capacity from the table (**85.0% -> 30.0%**) into blocks and depth while using fewer parameters and scoring higher - the first evidence that the layers are where capacity belongs. **D shows the limit:** width 128 at 300 steps collapses to 5.47%. **What may NOT be claimed:** +0.58 and +0.45 are **one seed** and this project has measured paired delta sd **0.35**, so both are inside one sd - a direction, not a result; 3 paired seeds are the next run. **Is:** measurement-driven selection of where a fixed budget sits. **Is not:** growth or pruning during training (still unbuilt). Source: `docs/history/CAPACITY-ALLOCATION-SWEEP.md`, `lens/runs/ALLOC-SWEEP-r5.txt`, `lens/runs/chain_alloc_*_r5.out`.

### CORRECTION: three seeds refute the capacity signal I reported an hour ago - 2026-11-20 20:13

```
seed   A default   B (30% table)   C (accuracy winner)   C - A     B - A
 0      10.29%        10.74%           10.87%           +0.58     +0.45
 1      10.88%         7.59%           10.73%           -0.15     -3.29
 2      10.72%        10.73%           11.09%           +0.37     +0.01
mean                                                  +0.27     -0.94
sd                                                      0.31      1.67
```

**P1 NOT MET** (mean C-A **+0.27** vs a +0.5 bar, sd **0.31** - inside one sd). **P2 FAILED** (mean B-A **-0.94**; the "29% smaller and still better" was **seed 0 being lucky**, and on seed 1 that config is **3.29 below** the default). **P3 MET** (every arm beat the constant). **Falsifier did NOT fire, by 0.07.** **What is actually true:** capacity did **not** move; the 30%-table shape is **unstable** (paired-delta sd **1.67**, five times the default's spread) and the accuracy winner is **+0.27 +- 0.31 = indistinguishable from zero**. Not a training failure - B on seed 1 fell 9.41 -> 6.26 and still ended worse, so this is genuine seed sensitivity of the rank-8/3-layer shape. **My message an hour ago said "capacity finally moves"; that sentence was wrong, and three seeds caught it** - which is why the 3-seed rule exists even when one seed looks exciting. Next is the untouched third ask: **memorise versus generalise** with a trap dataset that separates them by construction. Source: `docs/history/CAPACITY-CORRECTION-3SEED.md`, `lens/runs/ALLOC-3SEED-r6.txt`, `lens/runs/chain_alloc3_*_r6.out`.

### Memorise vs generalise: the switch works on all 3 seeds, and the generalisation half is still weak - 2026-11-20 20:31

```
seed   rule acc (held-out)   with switch   fact acc (unique keys)   collisions
 0          23.33%             23.33%        13.50% -> 100.00%        0.00%
 1          26.33%             26.33%        11.75% -> 100.00%        0.00%
 2          25.33%             25.33%         8.00% -> 100.00%        0.00%
```

**P1 MET on all three seeds** (facts **8.00-13.50% -> 100.00%**; the model cannot generalise a unique key, as the trap intends). **P2 MET on all three** (held-out rule accuracy **+0.00** exactly). **P3 MET** (**0 of 600** held-out rule tokens matched a stored fact context). Falsifier not fired. **P2 is trivially true by construction** - an exact-match store cannot collide with held-out rule contexts, so it is a structural property of exact matching, not evidence the switch shares capacity cleverly; P3 checks it independently. **The weak half, stated before any win is claimed:** held-out rule accuracy is only **23-26%** while train accuracy is **49.7-52.5%**, so the model has **not learned two-digit addition** - the **memorise** half is demonstrated (**11.08% -> 100.00%** mean) but the **generalise** half is **not**. Honest sentence: the switch separates the two correctly; the model's ability to generalise is unproven at 1500 steps. **Scope:** inference-time decision rule with an exact store, not weight-integrated memory (already falsified at +0.49 vs a +1.0 bar). Source: `docs/history/MEMG-SWITCH-3SEED.md`, `lens/runs/MEMG-r7.txt`, `lens/runs/chain_memg_s{0,1,2}_r7.out`.

### The generalisation half failed: 4x training bought +11.7 points and the rule is still not learned - 2026-11-20 21:07

```
seed   held-out rule acc   train acc   gap      fact acc (with switch)   collisions
 0          35.00%           52.50%    17.5          100.00%              0.00%
 1          34.33%           53.44%    19.1          100.00%              0.00%
 2          40.83%           55.94%    15.1          100.00%              0.00%
mean        36.72%           53.96%    17.2
```

**P1 FAILED** (bar was above **60%** on all three seeds; actual **34.33-40.83%**). Against 1500 steps (mean 25.00%) four times the training bought **+11.72 points**. **Falsifier:** written as "below 40% even at 6000 steps" - two of three seeds are below 40 and the **mean 36.72% is below it**, so its substance is met but not its strictest reading (seed 2 reached 40.83%); both recorded. **The switch still behaves everywhere:** facts **10.00-12.75% -> 100.00%**, collisions **0 of 600**, rule accuracy **+0.00**. **Verdict without softening: the memorize half is demonstrated, the generalise half is NOT** - ~37% held-out against ~54% train is a 17-point gap, the signature of fitting 600 pairs rather than learning a rule. **Diagnosis setting the next single change:** 10,000 possible pairs, 600 given - data-starved, not obviously capacity-starved, and only raising the data with nothing else changed distinguishes them. **What this does NOT license:** widening the model to move the number; the pre-registration forbade that rescue. Source: `docs/history/MEMG-GENERALISATION-FAILED.md`, `lens/runs/MEMG-LONG-r8.txt`, `lens/runs/chain_memglong_s{0,1,2}_r8.out`.

### More data does not fix generalisation: 8.3x the pairs bought +6.1 points - 2026-11-21 11:50

```
seed   rule pairs   held-out rule acc   train acc   gap
 0       5000            38.17%          54.69%     16.52
 1       5000            44.83%          55.00%     10.17
 2       5000            45.50%          54.37%     8.87
mean     5000            42.83%          54.69%    11.85
      (600 pairs)        36.72%         53.96%    17.24
```

**P1 FAILED** (bar: above **60%** on all seeds; actual **38.17-45.50%**). **P2 FAILED but improved** (seed-matched gap fell **17.24 -> 11.85**, bar was under 10). **Falsifier:** written as "below 45% even with 5000 pairs" - mean **42.83%** is below and two of three seeds are, so its substance is met (one seed reached 45.50%, recorded not hidden). **The switch still works:** facts **9.25% / 11.50% -> 100.00%**. **Settled: data is NOT the binding constraint** - 8.3x the pairs bought **+6.11 points**, so the limit is architectural. Next change on this line must be structural (a loop over output digits). *CORRECTION 11:58: an earlier version reported a mean gap of 9.5; that mixed a 3-seed held-out mean with a 2-seed train mean. Seed 0's train figure (54.69%) was in `chain_memgdata_s0_r9.out` all along. Seed-matched the gap is **11.85**, so P2 is met on neither the mean nor two of three seeds. P1 and the falsifier are unchanged.* Source: `docs/history/DATA-SCALING-3SEED.md`, `lens/runs/MEMG-DATA-r9.txt`, `lens/runs/MEMG-DATA-r10.txt`.

### Self-assessment head fails, and the reason is the finding: the model already knows when it is right - 2026-11-21 12:18

```
seed  arm          head AUROC   free confidence   margin
 0    A_headoff      0.1049         0.9423       -0.8374
 0    B_headon       0.8517         0.9418       -0.0901
 1    A_headoff      0.7850         0.9384       -0.1534
 1    B_headon       0.7900         0.9140       -0.1239
 2    A_headoff      0.8173         0.9392       -0.1219
 2    B_headon       0.9122         0.9399       -0.0278
                    mean 0.8513     mean 0.9319   -0.0806
```

**P1 MET on all six arms** (head ranks its own correctness, AUROC 0.79-0.91). **P2 NOT MET on all three seeds** - the margin against free confidence is negative every time (mean **-0.0806**), against a +0.02 gate. Falsifier did not fire. **OUTCOME: SELF_THINKING_FAILED on all six arms.** **P4 is UNMEASURED**, not passed: the script never printed held-out language accuracy, so "thinking costs nothing" was never checked. **The finding that matters:** the model already knows when it is right - max-softmax confidence ranks correctness at **0.93** mean AUROC for zero extra parameters, and on seed 1 an **untrained random** linear projection of the hidden state scored **0.7850**, so the information sits openly in the final hidden state. **Closes** self-assessment as a new tensor; **opens** the common-sense loop on the free signal (6.2 already runs that way). Source: `docs/history/SELF-ASSESSMENT-3SEED.md`, `lens/runs/SELFTHINK-r1.txt`.

### The ladder: selection and comparison work, arithmetic does not - 2026-11-21 12:20

```
rung        what it requires                                          answer accuracy   verdict
copy1       find the ONE digit among 7 decoys and copy it                  98.0%        LEARNED
sum2fixed   two digits handed at positions 0 and 1, output their sum        2.0%        NOT LEARNED
max2        find TWO digits among 8 decoys, output the larger              92.0%        LEARNED
```

**The pipeline works** (copy1 98.0% as positive control). **Content-based selection works** (max2 finds two digits among eight decoys with no positional cue, 92.0%). **Comparison works** (max2 emits the larger of the two). **Arithmetic does not, and both arithmetic rungs fail BELOW chance** (sum2fixed **2.0%**, sum2 **8.0%**, against 10%): sum2fixed puts the digits at positions **0 and 1** - no selection needed at all - and scores **2.0%**, below the 10% chance line. **This overturns the week's attributions:** data (+6.11 from 8.3x pairs), capacity (+0.27+-0.31, unstable) and probe difficulty were all wrong; the model can select, can compare, and cannot add even when handed the operands. The missing piece is a **structure for computation**, and this is the first diagnosis here supported by a controlled contrast rather than a correlation. **Applied immediately:** L3's certificate moves to max2 (running, 3 seeds); the self-cleaning run on sum2fixed was **stopped** because its three arms sat at 9.0/10.0/7.5% - all at chance, so the comparison was meaningless - and will be re-run on max2. **My prediction R3 (a monotonic ladder) was WRONG**: sum2fixed at 2.0% sits far below max2 at 92.0% even though it removes a requirement. The inversion is the finding. Source: `docs/history/LEARNABILITY-LADDER.md`, `lens/runs/SALIENCE-LADDER-r2.txt`.

### Two gaps, not one: the model cannot read by position and cannot compute - 2026-11-21 13:35

**1. The memory precondition is blocked by something more basic.** Distance-recall seed 0: near (q=0..2) **8.3%**, far (q=16..19) **8.1%**, Spearman **-0.102** - **flat at chance for every distance including distance 0**, where it only had to repeat the most recent digit. This is not a long-range memory problem, and L1's precondition cannot be measured until reading-by-position works. (`lens/runs/RECALL-r1.txt`)

**2. The salience certificate still cannot issue a verdict, but its margin is consistently positive** - 6 arms across two attempts: margin over the free norm baseline **positive in 5 of 6** (+0.0460, +0.0126, +0.0216, +0.0126, +0.0476), AUROC **0.72-0.94** (crosses the 0.90 bar once), Spearman **0.39-0.43** against a 0.50 bar. The instrument is the problem: `max2` probe accuracy swings **60-92%** and the script refuses verdicts below 90%; the bar was NOT lowered - the certificate moved to `copy1` (98.0%).

**3. My bug, caught by disagreement:** P3 reported Spearman **0.0079** while AUROC was **0.9357** and the measured separation was **0.3773 vs 0.0107** (35x). Cause: predictions on the EVAL sequences were correlated with ground truth from DIFFERENT sequences - both arrays had the right shape so nothing complained. After the fix: **0.43 / 0.43 / 0.39**.

**The coarse story "cannot do arithmetic" is wrong - there are two gaps:** reading **by position** (fails even one position back) and **arithmetic** (fails even when selection works). `copy1` (98.0%) and `max2` (92.0%) work because they select by **content** - digits vs decoys - so no positional addressing is needed; `sum2fixed` puts operands at **fixed positions** and fails at **2.0%**, exactly when position must be used. Running: `copylast` vs `copyfirst`, identical job differing only in distance. Source: `docs/history/TWO-GAPS.md`.

### CORRECTION 13:55: the model CAN read its most recent position perfectly - `copylast` scores **100.0% on both seeds**, so my earlier claim that it "cannot read by position at all, even one position back" was wrong. The recall failure is not about reading: it is about turning a **symbolic query token into an offset** (chance at every distance), which is a computation. Corrected statement: positional reading is not broken; **addressing arithmetic** is. `copyfirst` (same digit at position 0) is the comparison that isolates distance and was still running. **Also: L3's certificate is not certifiable yet** - on copy1, seed 1 clears every bar (AUROC **0.9542**, margin **+0.3052**, Spearman **0.5080**, accuracy 98.0%) while seed 0 misses all three (0.7130 / +0.0142 / 0.3955). A layer that works on one seed and not another fails the "100% on every layer" rule, so L3 stays UNCERTIFIED. Source: `docs/history/CORRECTION-POSITIONAL-READING.md`.

### L1 memory closes as "not needed at this size", and the one real gap is computation - 2026-11-21 14:05

```
rung        where the informative digit sits          answer accuracy
copylast    the LAST position, 7 decoys before it      100.0% (seed 0)  100.0% (seed 1)
copyfirst   position 0, 7 decoys after it               80.0% (seed 0)  100.0% (seed 1)
```

**Reading by position works at both ends of an 8-token context.** Therefore **L1 CLOSES as "not needed at this scale"** - the readable memory bank has been HANGING on +0.49 against a +1.0 bar, and the promise of memory was that it would extend reach; at 8 tokens attention already reads every position, so a bank that re-stores the same tokens has nothing to add. L1 reopens only when context length exceeds what attention reaches - a condition that can be stated and tested. **The corrected capability map:** read recent (100%), read distant (80%), select by content (98%), select+compare (92%) all **WORK**; **add two digits fails (2.0% / 8.0%, below chance)** and **decode a query symbol into an offset fails** (flat at chance, 3 seeds). Both failures need **computation over values**; the missing layer is a **carrier for computation**, not memory and not attention. Source: `docs/history/L1-CLOSED-COMPUTATION-IS-THE-GAP.md`, `lens/runs/POSITION-LADDER-r1.txt`.

### Instrument defect found and fixed: the model was reading across its own batch - 2026-11-21 14:25

`glass_lm.forward` flattened `(B,T,d)` to `(B*T,d)` and `forward_record` read that as ONE sequence, so the causal mask spanned the whole batch: sequence k could read sequences 0..k-1. **Measured** (`lens/diag_batch_leak.py`, change only sequence 0): its own logits move **1.183650e-01**, and the **LAST** sequence's move **4.599481e-02**. Effective context was **B*T = 256 tokens**, not T, and batch size silently changed the model. **My first leak test was wrong** - it changed the last sequence and checked the first, which a causal mask blocks for free, so it printed 0.000000e+00; a control that cannot fail is worse than none. **Fixed** in `lens/glass_batch.py` (per-sequence attention) and certified three ways: B=1 equivalence forward **0.000e+00**, gradients **8.882e-16**; finite differences at B=3 worst **4.573e-07**; isolation **exactly 0.000e+00**. A second bug was caught by the equivalence check first (shared `ln_backward` sums axis 0 only, returning `(T,D)` scales instead of `(D,)` - forward matched at 0.000e+00 while gradients differed by **5.99**). `glass_lm` patched; leak now **0.000000e+00** and its own gradcheck still **PASS** (6.635e-10). **Effect on today's claims:** the arithmetic/selection contrast survives in one direction (the leak ADDED context, so a task failed while leaking more is still failed) but every number is provisional; the distance claims are **weakened**; L1's closure is **on hold**. Six-rung replication running with the falsifier written first: if either arithmetic rung now reaches >= 80%, "computation is the wall" is withdrawn in public. Source: `docs/history/INSTRUMENT-DEFECT-BATCH-LEAK.md`.

### The diagnosis survives the instrument fix: everything selects at 100%, arithmetic still fails at 2% - 2026-11-21 14:30

```
rung        what it requires                                  leaky instrument   FIXED instrument
copy1       find the ONE digit among 7 decoys, copy it              98.0%            100.0%
copylast    digit at the LAST position, copy it                    100.0%            100.0%
copyfirst   the same digit at position 0                         80.0/100.0%         100.0%
max2        find TWO digits among 8, output the larger              92.0%            100.0%
sum2fixed   two digits handed at positions 0 and 1, ADD              2.0%              2.0%
sum2        find two digits, ADD mod 10                              8.0%           (running)
```

**P1-P4 all MET: on a clean instrument every selection and comparison task is 100.0%**, and **arithmetic is unchanged at 2.0%** - below the 10% chance line, on a model that scores 100% everywhere else. The falsifier (either arithmetic rung >= 80%) did not fire. **This is stronger than before the fix:** the leak could only have HELPED the model (up to 256 tokens of context instead of 8), so doing better on four rungs and identically on arithmetic makes the contrast clean in both directions - *this model can select, copy and compare perfectly, and cannot add.* **L3 salience is FAILED as specified, on nine arms:** with the better-trained head (1500 steps on 3,200 points) AUROC was **0.7599 / 0.8058 / 0.7755** - P1 (0.90) missed on all three while P2 is met consistently. A better-fit head made the margin bigger (+0.17/+0.17) and the ranking worse, and the free baseline's own AUROC fell to 0.61-0.70, so part of the bigger margin is a weaker baseline. Caveat: all nine arms ran on the leaky instrument; a fixed-instrument re-run is in progress, but the leak could only have helped, so it is unlikely to rescue the bar. Source: `docs/history/DIAGNOSIS-SURVIVES-INSTRUMENT-FIX.md`.

### L3 salience closes as FAILED on a clean instrument - 2026-11-21 14:40

```
seed   AUROC    free norm   margin    Spearman   probe acc   verdict
 0     0.7409    0.4682    +0.2727    0.2996     100.0%      SALIENCE_FAILED
 1     0.7922    0.6351    +0.1570    0.3276     100.0%      SALIENCE_FAILED
 2     0.7655    0.6062    +0.1593    0.3989     100.0%      SALIENCE_FAILED
mean   0.7662    0.5698    +0.1963    0.3420
```

**P1 (AUROC >= 0.90) missed on all three seeds** - across **twelve arms and two instrument versions L3 cleared 0.90 exactly once** (0.9542, leaky instrument). **P2 met on all three decisively** (+0.16 to +0.27, the layer does carry information the norm baseline does not). **P3 missed** (0.30-0.40). The probe was solved **100.0%** on every seed, so this is not a broken instrument: the model answers right and still cannot expose through a linear read which position decided its answer. **The reason, named: importance is present in the hidden state but not linearly readable from it.** The earlier confusion where "a bigger margin" looked like progress is explained too - the free baseline fell to 0.47-0.64, so part of that margin was the baseline getting worse. **Verdict:** L3 as specified is FAILED and is **not being re-tuned**. The honest redesign is architectural - salience must be **inside the forward path and the loss**, jointly trained and used by the model rather than inspected afterwards, because a probe can only report what the network happened to encode. **A layer that is a probe is not a layer.** Source: `docs/history/L3-SALIENCE-CLOSED.md`.

### L7 loop fails: repeated passes change nothing (7.0% at R=1, R=4, R=8) - 2026-11-21 14:45

```
task       R=1     R=4     R=8
sum2fixed  7.0%    7.0%    7.0%      <- identical to the last decimal
sum2         -     15.0%    -
copy1        -    100.0%  100.0%     <- the loop runs correctly and costs nothing that works
```

**P1 control MET** (R=1 = 7.0% in-configuration). **P2 NOT MET** (expected >= 50%; the loop moved arithmetic by **0.0 points**). **P3 MET** (copy1 stays 100.0% at R=4 and R=8). **FALSIFIER FIRED.** Certified before use: FD worst **2.141e-07**, isolation **0.000e+00**, **zero extra parameters** - so this was genuinely compute passes, not capacity. **Eliminations now:** memory not needed (L1); selection/routing already perfect (copy1/max2 100.0%); capacity no reliable gain (+0.27+-0.31); data +6.11 bar missed; **compute passes 0.0 points**. **Honest caveat:** at this budget R=1 never learns the task either (7.0% is chance), so the loop was asked to improve something never learned - recorded as a limitation. But the pre-registered rule was explicit, so the next attempt is the **scratchpad** (`lens/scratchpad.py`: addition becomes counting - write (d1+d2) ONE tokens then STOP). Re-running the loop with longer training would be a rescue that needs its own pre-registration. Source: `docs/history/L7-LOOP-FAILED.md`.

### The scratchpad works: addition becomes counting and reaches 100.0% - with a memorisation test now running - 2026-11-21 14:25

```
task                                   exact-count accuracy   within +-1
single-token answer (sum2fixed)               2.0%                -
single-token answer, wider model (sum2)        6.0%                -
LOOP R=1 / R=4 / R=8                          7.0% each           -
SCRATCHPAD (write the count, then STOP)     100.0% / 100.0%     100.0% / 100.0%
```

**P1 MET** (>= 50% exact) and **P2 MET** (>= 80% within +-1) on both seeds; falsifier not fired. **Why this differs from the loop:** the loop transformed the same hidden state more times; the scratchpad changes **what the model writes** - to answer 3+4 it emits seven ONE tokens then STOP, one step per token, with the count carried in the open sequence where a human can check it. This explains the day's whole pattern: selection, copying and comparison are **one-step** operations and attention does them perfectly (100.0% on four rungs); addition is the first task needing **more than one step**, and the only thing that helped was more steps **in the output**, not in the depth. **The objection that must be answered first:** there are only **100 possible digit pairs**, and 80,000 training examples means every pair was seen hundreds of times, so a lookup table keyed on (d1,d2) would give this same 100% without computing anything. **The 100% is therefore not yet evidence of computation.** Two generalisation tests now running with predictions written first: **held-out pairs** (20 of 100 never in training) and **unseen sum range** (sums 13-18 never in training). P4 held-out >= 80%; P5 unseen range >= 60%. **MEMORISATION FALSIFIER: held-out <= 30% -> the 100% was a lookup table in the output format, not computation, and the claim is withdrawn in public.** Source: `docs/history/SCRATCHPAD-ADDTION-WORKS.md`.

### It is a procedure, not a lookup table - and it is bounded by the taught range - 2026-11-21 14:35

```
seed   TRAIN pairs        HELD-OUT pairs      unseen SUM RANGE (13-18)
 0     100.0% exact       98.0% exact          0.0% exact   (26.0% within +-1)
 1     100.0% exact      100.0% exact         14.0% exact   (62.0% within +-1)
```

**P4 MET on both seeds:** held-out digit pairs, never in training, answered at **98.0% and 100.0%** - a lookup table keyed on (d1,d2) cannot do that, and the **memorisation falsifier did not fire**. **P5 FAILED on both:** sums 13-18, never seen, answered at **0.0% and 14.0%**. **Honest reading: it learned a procedure bounded by the range it was taught** - not a table of pairs, but closer to *"count up to twelve"* than to *"add"*; seed 1's 62.0% within-+-1 shows it lands in the right neighbourhood above 12 without counting precisely, which is what a bounded counter looks like past its limit. **Why this matters:** every one-step attempt failed at 2-7%, serialising the steps reached **100%**, so the missing piece was never capacity, data, memory or depth - it was **a place for intermediate state to exist**. Tokens are that place and they are fully readable, which is what this project wants: **the computation is the trace.** **What this does NOT license:** "the model can add" - it adds **within the taught range**; whether the range widens without breaking generalisation, and whether it transfers to two-digit addition with carry, are new pre-registrations. Source: `docs/history/SCRATCHPAD-PROCEDURE-BOUNDED.md`.

### CARRY SOLVED: 100.0% on 2,000 held-out pairs, including every carry case - 2026-11-21 14:33

```
seed   TRAIN pairs      HELD-OUT pairs    no-carry cases    WITH-carry cases
 0     100.0% exact        100.0% exact      100.0% (n=124)     100.0% (n=76)
 1     100.0% exact        100.0% exact      100.0% (n=91)      100.0% (n=109)
```

**38,204 parameters, 4,000 steps**, two-digit addition **with carry**, on **2,000 pairs never trained on** - memorisation cannot explain it (only 10,000 pairs exist and every evaluated one was withheld). **P1, P2 and P3 all MET**, falsifier not fired, and the carry gap is **0 points** (100.0% vs 100.0%). **Genuine generation, not teacher forcing:** evaluation decodes greedily from the model's own outputs - it writes `units`, feeds that back to decide `carry`, then both to decide `tens`, with nothing supplied from the answer key. **And a human can check every step** (`73 + 4`: units 7, carry 0, tens 7 -> 77), so the explanation is not a post-hoc attribution: **the trace IS the computation.**

**The finding of the project:** one-token answer **2.0%/6.0%**; loop R=1/4/8 **7.0% each**; serialised counting **100.0%**; serialised with carry **100.0% on 2,000 held-out pairs**. The architecture never lacked capacity (no gain, 3 seeds), data (+6.11, bar missed), memory (not needed at this size) or depth (0.0 points) - it lacked **a place for intermediate state to exist**. Tokens are that place, and being tokens, the whole computation is human-readable. **What this does NOT license:** not "the model can do arithmetic" in general - it solves two-digit addition with **one** carry and a written intermediate; three-digit with two carries is the next pre-registration, and nothing here shows the trace can be compressed back into a single token. The bounded-range finding still stands. Source: `docs/history/CARRY-SOLVED.md`, `lens/runs/CARRY-r1.txt`.

### Serialisation scales: three-digit addition with a TWO-carry chain, 100.0% on 2,000 held-out pairs - 2026-11-21 14:39

```
seed   TRAIN pairs      HELD-OUT pairs    no-carry cases     WITH-carry cases
 0     100.0% exact        100.0% exact      100.0% (n=48)      100.0% (n=152)
 1     100.0% exact        100.0% exact      100.0% (n=57)      100.0% (n=143)
```

**One carry was not the limit.** Six written tokens (`sum0 carry0 sum1 carry1 sum2 carry2`), each depending on the carry before it, across **1,000,000 possible pairs** with every evaluated pair withheld. **P1/P2/P3 all MET**, falsifier not fired. Worked example it wrote itself: `928 + 367` -> units 5, carry 1, tens 9, carry 0, hundreds 2, final carry 1 -> **1295**. **The full arithmetic picture measured today:** one-token answer **2.0%/6.0%**; loop R=1/4/8 **7.0% each**; serialised counting **100.0%**; 2-digit one-carry **100.0% on held-out**; 3-digit two-carry **100.0% on held-out**. **The single change that made this work is where the intermediate state lives** - not capacity (no reliable gain, 3 seeds), data (+6.11, bar missed), depth (0.0 at any loop count) or memory (not needed here). In this architecture that place is a token, which is why every step of the computation is readable. **Licensed:** a **38,444-parameter** model learns multi-digit addition as a serial procedure that generalises to unseen operands, fully readable. **Not licensed:** that it can add in one step (2.0%); that it generalises beyond the trained range (0.0%/14.0% on unseen sums); or that subtraction with borrowing behaves the same - untested. Source: `docs/history/CARRY-CHAIN-SCALED.md`, `lens/runs/CARRY3-r2.txt`.

### Serialised arithmetic: 100.0% on held-out operands from 2 to 5 digits, no boundary found - 2026-11-21 15:01

```
width   chain length   possible pairs   TRAIN     HELD-OUT (2000 unseen)   with-carry cases
2       1 carry          10,000        100.0%          100.0%             100.0% (n=76 / n=109)
3       2 carries      1,000,000        100.0%          100.0%             100.0% (n=152 / n=143)
4       3 carries    100,000,000        100.0%          100.0%             100.0% (n=169)
5       4 carries 10,000,000,000        100.0%          100.0%             100.0% (n=179)
```

**Every result decoded greedily from the model's own outputs**, every evaluated pair withheld from training. At 5 digits a lookup table would need **ten billion entries**, so memorisation is inconceivable by the fourth width. **The project now has one sentence:** the architecture could not add because an intermediate value had nowhere to live; give it a token to live in and it adds five-digit numbers it has never seen, with every step readable. **Two defects found on the way, both mine:** (1) the 4/5-digit runs first reported 0.0% because I widened the input but left `OUT_START` at its 2-digit value - the printed example showed `got=(11, 20, 5, 1, 7, 0, 3, 0)` (11=NOISE, 20=DELIM) followed by the correct answer shifted by two, so **the model was right and the ruler was wrong**; fixed, gated, re-run to 100.0%. (2) `carryN.py` breaks at 2 digits with a broadcast error (my generalisation is off-by-one at the original width); the 2-digit result stands on the unchanged, gated `carry.py`. **Unproven and stated as such:** no boundary searched past 5 digits; **subtraction with borrowing untested** (running now - success there, not addition, is what would make the mechanism general); the bounded-range finding still stands. Source: `docs/history/ARITHMETIC-SCALED-TO-FIVE-DIGITS.md`.

### Subtraction with borrow transfers: 100.0% on held-out operands at 3 and 5 digits - 2026-11-21 19:30

```
arm                    TRAIN pairs    HELD-OUT pairs     no-borrow        WITH-borrow
3-digit subtraction       100.0%         100.0%         100.0% (n=63)    100.0% (n=137)
5-digit subtraction       100.0%         100.0%         100.0% (n=16)    100.0% (n=184)
```

**The mechanism is not tuned to addition.** Borrowing is a **different intermediate** (`d = da - db - borrow`, and a negative result must tell the next column). The model writes `[diff0][borrow0][diff1][borrow1]...`, decoded `741 - 269` -> 2,1,7,1,4,0 -> **472**. **The finding:** give an intermediate value a token to live in and this architecture performs multi-digit addition **and subtraction** on unseen operands, every step readable. **The 8-digit arm failed and it is MY mistake, not a boundary:** training token accuracy peaked at **67.5%** (every successful arm hit **100.0%**), I gave it **4,000 steps** against **6,000** for the arms it is compared with, on a task with 16 written tokens and a 36-token context, and the output is structurally right with wrong digits - **not** the decode-offset bug from earlier. **Recorded as INVALID-AS-A-BOUNDARY.** The fair re-test (10,000 steps, nothing else changed) is pre-registered: P1 converge (train >= 80%), P2 held-out >= 50%; **honest falsifier:** if training plateaus below 90% at 10,000 steps the limit is capacity per written token at this size, and that is stated as the boundary with the training curve as evidence. Source: `docs/history/SUBTRACTION-TRANSFERS.md`, `lens/runs/BOUNDARY-TRANSFER-r2.txt`.

### 8-digit addition solved at a fair budget: the "boundary" was my budget, confirmed - 2026-11-21 19:52

```
8-digit addition, 10,000 steps, 16 written tokens, 7-intermediate chain, 36-token context
training token accuracy:   100.0% at step 8,800   (vs a 67.5% plateau at 4,000 steps)
TRAIN pairs     100.0% exact
HELD-OUT pairs  100.0% exact   (2,000 unseen, 195 requiring carries)
```

**10^16 possible pairs** - a lookup table would need ten quadrillion entries, so this is a procedure by any measure. **The 4,000-step run had already been recorded as INVALID-AS-A-BOUNDARY before this existed**, on the strength of its own training curve, and that call was right. It establishes the rule worth keeping: **training-set accuracy is the tell** - if the model cannot reproduce what it was trained on, the run says nothing about generalisation or architecture, only that it has not converged, and the training curve is what separates the two. Same distinction as today's other defects (the padding mask that faked a 0.387 loss, the decode offset that faked 0.0%): **in every case the number was testable against a second measurement, and the second measurement won.** **The complete arithmetic line, all on 2,000 held-out pairs:** addition 2/3/4/5/8 digits **100.0%**, subtraction with borrow 3 and 5 digits **100.0%**, the same tasks asked in ONE token **2.0-7.0%**, the same tasks with a LOOP R=1/4/8 **7.0% at any depth**. Source: `docs/history/EIGHT-DIGITS-SOLVED.md`, `lens/runs/D8-FAIR-r1.txt`.
