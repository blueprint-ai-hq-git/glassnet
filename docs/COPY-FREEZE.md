# Freezing the copy path halves the copy decay (2026-11-13)

The ablation and drift measurements (`docs/COPY-WEIGHTS.md`) said copying lives in `ptrW`, `Wq`, `Wk` and `concept.W`, and that
the language phase rewrites exactly those tensors while barely touching the memorised-answer path. This tests whether protecting
them actually saves the ability.

## Design
All three arms continue from the same checkpoint - `early_s0_ep1000`, held-out copy **71.0%** - and train on
**language-only** batches (corpus fraction 1.0) for 4,000 steps, batch 128, identical in every other respect.

```
arm            frozen tensors                              reason
freeze_copy    ptrW, Wq, Wk, concept.W                     the ones the ablation says carry copying
freeze_control answer.W, dmix                              the ones it says do NOT (drift 0.0166 and 0.0000)
freeze_none    nothing                                    the reference decay
```

**Prediction, written before the run:** `freeze_copy` holds held-out copy within ~3 points of 71% while language improves;
`freeze_control` decays like `freeze_none`; `freeze_none` reproduces a large decay.

## Result
```
arm             late held-out copy (mean of last 3 evals)   drop from 71.0   language accuracy
freeze_copy     68.7%  (69.5, 68.5, 68.2)                    -2.3 points      16.15%
freeze_control  60.5%  (61.3, 63.5, 56.8)                   -10.5 points      17.70%
freeze_none     60.5%  (61.5, 63.3, 56.7)                   -10.5 points      17.70%
```

**Freezing the copy path is worth +8.2 points of held-out copy** (68.7% against 60.5%), with taught-word copy flat at 83% in
every arm. Prediction passed on all three counts.

**The control is what makes it mean something.** Freezing `answer.W` and `dmix` produced numbers identical to no freezing at all
(60.5% both; per-eval values within 0.1). So it is the *choice of what to freeze* that holds copy, not the act of freezing
parameters - which was the exact way this experiment could have fooled itself.

**The cost:** language accuracy in the frozen arm is **16.15%** against **17.70%** in the other two, i.e. 1.55 points of language
for 8.2 points of copy. That is a real trade, stated rather than hidden.

## Two honest limits
- **The decay here is 10.5 points, not the 25.2 points of the original curve.** This run continues from ep1000 with
  language-only batches at batch 128; the original `early_s0` run mixed its corpus differently. So these arms are internally
  comparable (identical recipe, only the freeze differs) but the absolute decay is not the same measurement as
  `docs/COPY-DECAY-CURVE.md`. Comparing them would be comparing two different recipes.
- **The freeze is not a fix for the language side.** It protects a capability by refusing to update weights; the frozen tensors
  are no longer learning. Whether a copy path can be made *non-interfering while still learning* - the separated-store answer
  from `docs/MEM-SPLIT.md` - is untested here.

## Where this leaves the memory work
- **Facts forgetting (the controlled 128-fact test): fixed by the separated store** - 96.1 -> 0.0 points of forgetting, because
  the store's parameters are disjoint from the language network (`docs/MEM-SPLIT.md`).
- **Mechanism decay (the real copy task): halved by freezing the copy path** - 10.5 -> 2.3 points, at a cost of 1.55 language
  points (`docs/COPY-FREEZE.md`).
- **Untested next step:** give the copy path its own parameters that keep learning from copy batches only while the language
  phase runs - the store idea applied to a mechanism instead of a table. That is the version that would keep both capabilities
  moving, and it is the honest successor to freezing.

## Reproduction
`lens/scripts/run_freeze_arms.sh` (three arms sequentially, nice 19, GPU stayed at ~425 MiB against a 5 GB budget); logs in
`logs/freeze/`; checkpoints `lens/runs/chat/freeze_{copy,control,none}*.npz`. The trainer gained `--freeze-extra`, which matches
parameters by their own `.name` (a tensor called `concept.W` is nested on the model, so `getattr` cannot find it - a first
attempt crashed on exactly that).
