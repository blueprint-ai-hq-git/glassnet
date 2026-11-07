# 8M windows, gates off: the data axis did not flatten - it paid more than the doubling below it

`DATASCALE-8M-NOGATE.txt:236` (final), `:235` (epoch 3). Finished 19:57:44, exit 0, after resuming from the epoch-1
checkpoint that survived the 18:52 reboot.

    tag      datascale_8m_nogate_s0      d=256, window 32, batch 256, 3 epochs, gates off, seed 0
    VAL      24.51%      ASST 26.46%      ppl 87.55      train 34.31%      concepts 36/36
    path     ep1 23.46 (95.71) -> ep2 24.03 (91.04) -> ep3 24.51 (87.55)
    cost     1,021 s / 1,523 s / 1,508 s     updates 93,750     params 3,811,534
    provenance  1 arm verified, 0 problems; best = ep3; RESUME: loaded line present
    mechanical  0 agreement-step lines, 0 gate-step lines, METHOD = OURS STEP RULE, GATES OFF

## The pre-registered verdicts, all five

| # | prediction (written before the launch) | result | verdict |
|---|---|---|---|
| X1 | VAL > 23.74%, strictly above the 3-epoch 4M arm | 24.51% | **PASS**, +0.77 pp |
| X2 | lands 24.0 - 24.6% | 24.51% | **INSIDE** |
| X3 | ppl < 95.99 | 87.55 | **PASS** |
| X4 | 0 gate lines, METHOD reads GATES OFF | 0 / 0 | **PASS** |
| X5 | concepts >= 36 | 36/36 | **PASS** |

## What the number means, and the one thing it overturns

The rung below was 4M at 23.74%. Doubling the data bought **+0.77 pp**, which is *more*, not less, than the +0.62 pp
the 2M -> 4M doubling bought. My own X2 was written expecting the gain to decay below 0.62 ("every rung so far has
eventually decayed, so the band is centred below a repeat of 0.62"); it did not decay. Up to 8M windows, at this
width, more data keeps paying slightly better per doubling.

That is the finding, and it is a falsification of my prior, not a confirmation of it: the ladder is still climbing at
the last rung measured, so any sentence of the form "the data axis is flattening" is now unsupported at d=256.

Two limits that must travel with it:

* **The corpus ends here.** The prepared mix holds ~18.4M windows at stride 16; 8M is 43% of it. One more doubling
  (16M) is available before the axis runs out of data, and it has never been run at this width.
* **One seed.** The 200k rung's three-seed spread was 3.07 pp, far wider than 0.77. So this arm says "the axis has
  not turned down", NOT "the axis paid 0.77 pp with confidence". The three-seed block for the 4M rung is queued
  precisely because of this.

## The reboot, and what it cost

The machine rebooted at 18:52 with the arm mid-epoch-2. Epoch 1 (23.46%) was on disk with its rng sidecar, so the arm
resumed (`RESUME: loaded 5 parameter tensors ... continuing at epoch 2 of 3`) rather than restarting. Wall time lost:
about 11 minutes, not 51.

## What is not claimed

* Not the project's d256 record: full-corpus d256 reaches 35.72% by position (`docs/history/STATE.md:146`).
* Not a claim about the gates: this arm runs with them off, as the whole gates-off ladder does.
* Not a significance test: no p-value exists here, and none is invented.
* Headline unchanged: 37.37%, d=512, full corpus, 6 epochs.
