# "Stop and start without losing anything" now exists - and is tested (2026-11-18)

A standing requirement on this project is **one switch to stop and start that loses nothing**. Until now it was not met: no trainer accepted a resume, so every invocation
re-initialised from scratch and stopping a 3000-step arm discarded everything it had learned. `grep -rn 'RESUME' lens/scripts/` returned nothing.

## What was added

`lens/scripts/train_scan9.py` accepts `RESUME=<checkpoint.npz>`. It loads the weights, reads the step from the checkpoint's `__meta__`, and continues the loop from **step + 1**, keeping
the same learning-rate decay curve (the schedule is a function of the absolute step, so a resumed run lands on the same curve rather than restarting it).

The existing guarantees still hold: `CK_PREFIX` keeps runs separate, and a checkpoint that already exists is **refused, never overwritten**.

## Tested, three ways

1. **It runs and logs the resume:**
   ```
   2026-11-18 11:16:07 | RESUMED from lens/runs/chat/scan11_r100_ep250.npz at step 250 (weights loaded; data order NOT restored ...)
   2026-11-18 11:16:10 | resumeSMOKE3 COMPLETE (radius 1.0)
   ```
2. **The source checkpoint is untouched** - `scan11_r100_ep250.npz` mtime stayed `09:04:47` before and after every resume test.
3. **The weights genuinely loaded, not silently re-initialised.** A run resumed from ep250 for 26 steps reports at step 275:
   ```
   step 275/275 loss 7.1772 ema 7.5038 acc 7.45% gnorm 2.99
   ```
   against a **fresh** run of the same batch size at step 25:
   ```
   step 25/3000 loss 9.5173 ema 9.7334 acc 0.42% gnorm 2.78
   ```
   0.42% accuracy is what a model that knows nothing gets; 7.45% after 26 steps at learning rate 0.003 is not reachable from scratch. The weights came from the checkpoint.

## My own bug, caught by the test rather than by reading

The first version called `log(...)` inside the resume block, but `log()` is **defined further down the file**, so every resumed run died instantly with
`NameError: name 'log' is not defined`. The patch parsed fine and looked correct. It was the smoke test that found it, which is the reason to run one before believing a patch.

## The honest limitation, written into the file

**The data order is not restored.** The sampler's rng state is not stored in the checkpoint, so a resumed run continues the weights exactly but sees a different (equally valid)
shuffle of the same corpus. Saving the rng state would make the resume bit-exact and is a small change to the saver; it is not done yet, and the note in the source says so rather
than leaving a reader to assume the resume is perfect.

## Why this matters for the work in flight

Two arms are running at 3000 steps and would take over an hour each to redo from zero. Before this change a reboot, a stop request, or a crash during one of them cost the whole run -
which is exactly what happened to a 3-seed run on 2026-11-17 at 14:23. Now a stop costs one checkpoint interval, 250 steps.
