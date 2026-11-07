# Cost probe: d=512, full corpus, one epoch, our step rule, gates off - in the HEADLINE's own configuration

2026-11-07 14:3x IST, written before `lens/scripts/run_probe_d512.sh` is launched.

## A correction to what I said an hour ago

After the three-way (`docs/history/SGD-500K-RESULT.md`) I wrote that "the headline configuration has never been run
in the configuration that now measures best". True, but I made it sound like a one-flag change. It is not. The
headline run (`lens/runs/TIED-D512-EXT6.txt`) differs from the d=256 ladder in **four** ways at once:

| | d256 ladder arms | 37.37% headline |
|---|---|---|
| width | `--d 256` | `--d 512` (15,567,566 params) |
| head | no direct head (`direct: false` in the checkpoint meta) | `--direct --direct-mode tied` |
| data | 20k-2M windows | **16,900,000** windows, 6 epochs |
| lr | 0.4 | **0.3** decaying to 0.15 |

So "relgrad would beat 37.37%" is a hypothesis about the step rule and the gates **on top of** three other
differences, and the honest way to test it is to reproduce the headline's configuration exactly and change only
the step rule and the gates. That is what this probe starts to do.

## What the probe is

    --d 512 --window 32 --epochs 1 --batch 1024 --lr 0.3 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0
    --direct --direct-mode tied --method ours --no-gates      (full 16.9M-window corpus, no --max-windows)

One epoch. Its purpose is **cost**, not accuracy: the 6-epoch attempt is only worth scheduling if an epoch is
affordable, and relgrad's per-epoch cost at d=512 has never been measured. At d=256/500k the ratio was measured
at 72.5 s (relgrad, ungated) against 31.5 s (plain SGD, ungated) = 2.3x, and plain SGD at this scale took
1,720 s/epoch (`docs/history/STATE.md:2771`).

## Predictions

P1  COST. One epoch takes **2,500 - 6,000 s** (42-100 min), i.e. 1,720 x 1.5-3.5. Below 2,500 s means relgrad is
    cheaper at d=512 than at d=256 relative to SGD and a 6-epoch headline attempt is ~4 h, affordable today.
    Above 6,000 s means it is 10 h+ and needs a different plan.
    FALSIFIER: outside the band. Either way the measured number replaces the extrapolation, which is the point.

P2  FIRST-EPOCH ACCURACY, reported not predicted precisely. The headline's plain-SGD epoch 1 was VAL 31.52% /
    ASST 20.32% at 200k windows; the full-corpus epoch 1 in the ext6 run was VAL 31.52% (line 83). I expect the
    relgrad ungated epoch 1 in **28-36% VAL**. It is one epoch of six, so it is NOT a headline number.

P3  MECHANICAL. The log prints `STEP RULE:` relgrad, `METHOD: OURS STEP RULE, GATES OFF`, 0 agreement lines,
    `direct: true` in the checkpoint meta, and 15,567,566 params.

P4  NO REGRESSION IN THE OLD NUMBER. Nothing here may touch `TIED-D512-EXT6.txt` or its checkpoints; the probe
    writes a new tag (`probe_d512_relgrad`) and a new log.

## What will not be claimed

* One epoch is not a result. Even if epoch 1 looks strong, the headline is a 6-epoch number and this arm will not
  be quoted against 37.37%.
* VRAM guard: abort if free VRAM is below 3 GB at launch; the d512/batch-1024 family measured ~2,011 MiB used.
* Checkpointing: one epoch writes one checkpoint, well inside the 2 h rule; if the epoch exceeds 2 h the run is a
  cost answer anyway and will be reported as such, not as a training result.

## CORRECTION, added after the first probe ran (14:31-14:37)

The first probe did **not** test this configuration. Its log header reads `windows 900,000 train / 87,035 val`:
the trainer's `--max-windows` **defaults to 900,000** (`train_chat.py:345`), and I passed no cap, assuming "no
--max-windows" meant the whole corpus. It does not.

That arm is kept, not deleted, as a **cost calibration**: 878 steps in **164 s** = **0.187 s/step** at
d=512 / batch 1024 / relgrad / gates off. Its accuracy (VAL 14.46%, ppl 325.81) is **not comparable** to the
headline's epoch 1 (31.52%) because that epoch ran 16,504 steps and this one ran 878.

Scaled from the measured step cost - clearly labelled as scaling, not measurement:

    16,900,000 / 1024 = 16,504 steps per epoch  x  0.187 s = ~3,083 s = ~51 min per epoch
    six epochs = ~5.1 h

That lands inside P1's band of 2,500-6,000 s per epoch. The corrected probe was then launched with
`--max-windows 16900000` and a distinct tag (`probe_d512_relgrad_full`, unit `glassnet-probe-d512-full`), so the
calibration arm and the real probe cannot overwrite each other.

Two further corrections to what the earlier report implied:

* The headline differs from the d=256 ladder in **four** ways (d=512, `--direct --direct-mode tied`, 16.9M
  windows, lr 0.3), not one. The probe matches all four so that only the step rule and the gates change.
* The headline's own header says `tokens per parameter: 1.09 (UNDER-TRAINED)`. That is the 32x mislabel fixed
  earlier today: the true figure is **34.9 tokens/param**, i.e. the flagship run was *well trained*, not
  under-trained (`docs/history/TOKENS-PER-PARAM-MISLABEL.md`).
