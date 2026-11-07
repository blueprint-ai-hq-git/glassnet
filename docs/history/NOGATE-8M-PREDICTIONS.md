# 8M windows, gates off: does the data axis keep paying past 4M? (predictions, before the run)

2026-11-07 18:1x IST, written before `lens/scripts/run_datascale_8m_nogate.sh` is launched.

## Why 8M, and why not more epochs

Measured today at 4M, seed 0, gates off, 3 epochs (`docs/history/NOGATE-4M-EP6-RESULT.md`):

    doubling the DATA   2M -> 4M          23.12 -> 23.74   +0.62 pp
    doubling the STEPS  46,875 -> 93,750  23.74 -> 23.92   +0.18 pp

Data is worth ~3.4x steps per doubling at this margin, so the next GPU-minute belongs to data. Per epoch cost is
linear in data (4M = 485 s), so 8M is ~970 s/epoch, ~49 min for three epochs plus corpus assembly.

The corpus holds ~18.4M windows at stride 16, so 8M is 43% of it - a real increase, not a re-sample of the same
windows.

## Predictions

X1  STILL PAYING. VAL > 23.74% (strictly above the 3-epoch 4M arm - the matched comparison, same epochs).
    FALSIFIER: <= 23.74% means the data axis has flattened between 4M and 8M at this width.

X2  HOW MUCH. Land in **24.0 - 24.6%**, i.e. +0.3 to +0.9 pp for one doubling. The last doubling bought 0.62, and
    every rung so far has eventually decayed, so the band is centred below a repeat of 0.62.
    FALSIFIER: above 24.6% means the axis is accelerating; below 24.0% means it decayed harder than one step.

X3  PERPLEXITY below 95.99 (the 3-epoch 4M value; the 6-epoch 4M arm reached 93.72, so this is a loose bound).

X4  MECHANICAL. 0 `agreement step` / 0 `gate step` lines, METHOD reads GATES OFF, 3,811,534 params.

X5  CONCEPTS. Audit >= 36 of the kept set.

## A launch that failed before this one, and what it tells me

`lens/runs/chain_-c.out` holds exactly one line: `/bin/bash: /bin/bash: cannot execute binary file`. Someone
launched an 8M arm as an UNQUOTED inline command through `glassnet_chain.sh`, so the chain took `/bin/bash` as the
work script and died in 0 s with rc 126. Nothing ran; no checkpoint, no result, no claim was left behind.

Two differences in that attempt are worth recording rather than silently dropping: it pointed at
`data/chat/prepared_dialogue_only_honest` (dialogue-only corpus) and pinned `taskset -c 2,3`, while this whole
ladder uses `prepared_mix_ctrl_honest` and cores 0,1. **This arm uses the ladder's corpus and cores**, so its number
is comparable to the rest of the ladder. The failed attempt is not evidence of anything about 8M.

The chain now refuses a file that `bash -n` cannot parse, with a message naming the right usage; verified by passing
`/bin/bash` deliberately (rc 2, "REFUSED ... Nothing was launched").

## What will not be claimed

* One seed; the 3.07 pp seed spread at 200k dwarfs every delta here.
* Not the project's d256 record: full-corpus d256 reaches 35.72% by position (`docs/history/STATE.md:146`).
* The gates are not tested at this rung.
* Cost ~49 min + assembly; checkpoints every epoch (~16 min), inside the 2 h rule; VRAM ~490 MiB.

## ADDENDUM 2026-11-07 18:52 - the machine rebooted after epoch 1; the arm resumes rather than restarting

`uptime` reads 0 minutes at 18:52:56 and `who -b` shows a fresh boot: the 8M run was killed mid-epoch-2 with no exit
line (a hard stop, not a SIGTERM). What survived, measured on disk:

    datascale_8m_nogate_s0_ep1.npz          15,262,368 B   18:37
    datascale_8m_nogate_s0_ep1_rng.json       175 B        18:37
    datascale_8m_nogate_s0_best.npz        15,262,368 B   18:37
    epoch 1: VAL 23.46%  ASST 25.29%  ppl 95.71  (NEW BEST held-out 25.29% at epoch 1)

So 17 minutes of work is intact. `train_chat.py` has a documented resume path (`--resume <npz> --resume-ep N`, which
sets `start_ep = N+1` and restores the shuffle RNG from the sidecar, refusing on a corpus mismatch), so the arm
continues from epoch 2 instead of restarting. The resumed runner records the `RESUME: loaded ...` line and requires
it - a resume that did not actually load would be reported INCOMPLETE, not as a result.

Predictions X1-X5 are unchanged; epoch 1 is not a verdict (X1 needs the finished arm above 23.74%).
