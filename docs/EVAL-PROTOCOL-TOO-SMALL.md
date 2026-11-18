# The held-out evaluation was too small: every absolute accuracy was ~3.2 points too low (2026-11-18)

While checking why the 6000-step arm's held-out accuracy moved by less than its error bar, the error bar itself became the question. The evaluation protocol scored **12 chunks =
768 positions**, giving a standard error of **1.47 points** - it could not resolve anything smaller than ~2.9 points. The power calculation says what that costs:

```
scored positions     SE      can resolve (2 SE)   chunks of 64
     768          1.47 pts        2.94 pts            12   <- what every number used
   1,659          1.00 pts        2.00 pts            26
   6,636          0.50 pts        1.00 pts           104   <- now the default
  26,544          0.25 pts        0.50 pts           414
```

## What the tighter measurement found

The same three checkpoints, scored with **104 chunks = 6,552 positions** instead of 12:

```
checkpoint                   12 chunks (768)   104 chunks (6552)   difference
scan9_ep1500   (r=0.9)           16.27%            19.67%           +3.40
scan11_r100_ep3000 (r=1.0)       16.80%            20.83%           +4.03
scan11_r090_ep3000 (r=0.9)       21.03%            23.95%           +2.92
```

**All three were understated by 2.9 to 4.0 points**, because all three were scored on the *same* first twelve chunks - and that particular subsample is harder than the corpus. The
estimator was not biased; the **shared sample** was unrepresentative, which is why the error was the same sign and similar size for every model.

## What survives, and what is corrected

**Survives:** the **ordering** and the **paired comparisons**, because every model was scored on the same chunks. Radius 0.9 at 3000 steps beats radius 1.0 at 3000 steps either way.

**Corrected:** every absolute accuracy this project has published for these arms:
- radius 0.9 at 3000 steps is **23.95%**, not 21.03%
- radius 1.0 at 3000 steps is **20.83%**, not 16.80%
- `scan9_ep1500` is **19.67%**, not 16.27%

**Sharper as a result:** the radius-0.9 advantage over radius 1.0 at matched training is **3.12 points**, and with SE ~0.71 on the difference that is **~4.4 standard errors** - resolvable,
where before it was a 2-SE call. The earlier conclusion was right; it is now properly supported.

## Two consequences for work in flight

1. **The 6000-step arm's P1 needs restating.** It was "held-out >= 24% at 6000 steps" against a baseline believed to be 21.03%. The baseline is really **23.95%**, so 24% is nearly the
   starting point. The honest prediction for the extended arm is **>= 26%**, and the original 24% should not be counted as a success if it lands at 25%.
2. **The learning-curve shape stands** (all points shared the same biased sample, so the shape is unaffected), but its vertical axis should be read as ~3 points low at every step.

## The fix, in the tool

`tools/eval_radius_arm.py` now takes `EVAL_CHUNKS` with a default of **104** (SE 0.50, resolves 1-point differences), instead of the hardcoded 12. Final numbers for any arm should be
reported at that setting, and the 12-chunk figures kept only as the shared-sample comparisons they are.

## The lesson

The project has a rule that an effect needs three seeds. It needed a matching rule for the **measurement**, and this is it: **check the standard error before believing a difference,
and make the eval big enough for the smallest difference the experiment is supposed to detect.** A protocol that cannot resolve 1 point was being used to argue about 1-point effects
for several hours.
