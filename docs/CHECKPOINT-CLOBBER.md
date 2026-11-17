
## A resumed run destroyed a checkpoint, and the smoke test is what caught it (2026-11-17)

**The damage:** a smoke test of the new resume path overwrote the original `lens/runs/chat/warmstart_pass1.npz` - the pass-1 checkpoint from the epoch-mode arm.

**The cause:** `--passes` numbers its per-pass checkpoints from 1 on every invocation. A run that *resumes* from pass 2 therefore writes `warmstart_pass1.npz`
again, into the same path, and `np.savez` overwrites without a word. The filename carried no run identity, so two different runs shared a name.

**What was and was not lost:** the pass-1 *result* is safe - `lens/runs/WARMSTART-PASSES.txt` records `PASS 1/2 train 5.8526 VAL 6.5653 gap -0.0041`, and
`tools/eval_resolution.py` had already scored it for the published paired number. What was destroyed is the **weights**. The file is **quarantined, not deleted**, as
`warmstart_pass1_CLOBBERED_by_smoke_1115.npz` so nobody mistakes the 40-window-trained artifact inside it for the original. `warmstart_pass2.npz` (10:41) was
untouched and is what the extend arm resumes from.

## The two fixes

1. **`--tag` is required in practice**: checkpoints are named `warmstart_<tag>_pass<k>.npz`, so two runs cannot collide by construction.
2. **A run refuses to overwrite**: if the target exists, it prints `REFUSING to overwrite existing checkpoint ... results are never destroyed` and moves on, rather
   than replacing a checkpoint.

## Why the smoke test is the hero here

The whole point of a smoke test is to break things cheaply before they matter. This one ran **1 pass of 40 windows** and cost 38 seconds. Had it not run, the extend
arm would have resumed from pass 2, written `warmstart_pass1.npz` with its own weights, and I would have *found* a file at that path afterwards and had no reason to
suspect it was a different model's. The published pass-1 paired number would then have pointed at weights that no longer existed.

That is the fifth instrument-class defect this week and the most dangerous kind: **not a wrong number, but a right number attached to an artifact that had been
quietly replaced underneath it.**

## The extend arm, launched and verified

`glassnet-wsextend`, detached, resumed from `warmstart_pass2.npz`, **4 passes x 2000 distinct windows**, batch 8, lr 1e-4, clipped, **paired 200-window evaluation
every pass** so the answer is a curve rather than an endpoint.

Its own resume measurement, independent of the previous wake:

```
PAIRED BASELINE over 200 held-out windows: mean 6.0379 nats
  at resume paired -0.02555 nats  CI [-0.04864, -0.00219]  -> RESOLVABLE
```

That reproduces the published **-0.02623** from a separately computed baseline (6.0379 here against 6.0386 there). Two independent measurements of the same
checkpoints agree to 0.0007 nats, which is the first cross-check on this number that came from a different code path than the one that produced it.

**Falsifier, fixed before launch:** if the paired gain does not exceed the pass-2 value of **-0.02623** by the end of four more passes, the gain is saturated and the
block is not earning its 15.2M parameters.
