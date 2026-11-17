
## The exposure-bias lever failed, and it made free-running accuracy worse (2026-11-17)

Pre-registered in `lens/runs/ROLLOUT-TRAINING-PREDICTIONS.md`. Arm: 300 steps, batch 8, lr 3e-4, half of each window rolled out from the model's own argmax, targets always the
untouched corpus (`tgt_chunks`), starting from `scan9_ep1500`. Judged on the same probe that measured the baseline.

```
                        baseline (scan9_ep1500)   rollout arm (300 steps)
teacher-forced accuracy        22.19%                   21.25%
free-running accuracy           4.06%                    2.19%
free-running repetition         0.93                     0.93
```

**The gate did not just fail; the intervention moved the wrong way.** Free-running accuracy **halved**, 4.06% -> 2.19%. Teacher-forced accuracy also fell slightly, 22.19% -> 21.25%. Repetition
was unchanged at **0.93**.

**The pre-registered falsifier fired exactly as written:** *if free-running accuracy stays under 6% after treatment, exposure bias was not the binding constraint.* It stays at 2.19%.

## The detectors were clean, so this is a real result rather than a broken arm

- **D1** constructed-vs-corpus difference over rolled positions: **0.9743 mean** over the run - the model's own tokens really did replace the corpus tokens in the input.
- **D2** max |constructed - corpus|: **11,999** - non-zero.
- **D3** fake-fix gate: **never fired**. Final teacher-forced accuracy on rolled contexts 4.29%, nowhere near the 90% contamination signature. The targets were the corpus throughout.
- Bit-identical check on the code change: loss 9.9041865856 / gnorm 1.1704698742 before and after, so the `tgt_chunks` refactor disturbed nothing.

**Nothing about this arm is void.** The mechanism was implemented correctly and it did not work.

## What the numbers say about why

Three things are visible together: free-running accuracy halved, **teacher-forced accuracy also dropped**, and **repetition did not move at all**.

If exposure bias were the binding constraint, training in the generation condition should have improved the generation condition and left the training condition roughly intact. Instead both fell and the
loop signature was untouched. The most consistent reading is that **300 steps of fine-tuning on self-generated context mostly moved the model's parameters away from its trained optimum without
teaching the state anything new** - the recurrent state still collapses to nearly the same prediction under self-generated input, and repetition 0.93 says that directly.

## The honest limit of this result

This is **one arm at one setting** - 300 steps, lr 3e-4, frac 0.5. It is not a sweep, and short fine-tunes can under-train. But the falsifier was fixed before the run and it fired, so **the claim
that exposure bias is the binding constraint is not supported** and will not be replaced by "it would work with more steps" without a pre-registered test of that specific claim.

## What the falsifier redirects to

Its own text: *the loop must then be sought in generation-context construction versus the training window.* That is now the leading question and it is testable on the same checkpoint without
retraining:

- **Context length at generation** - the trainer scores positions inside a 128-token window; the probe generates from a 64-token context. Does free-running accuracy depend on the context length used?
- **Where generation starts** - mid-conversation vs mid-document changes what the state has seen.
- **The state's own scale** - the readout normalises by RMS, so a state that drifts in magnitude is re-normalised into looking familiar. Measure state drift under free-running against the
  training distribution.

The first of those is a re-measurement with an existing instrument and should be done before any further training.
