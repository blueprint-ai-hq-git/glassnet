# The 19:05 "datascale" run: real numbers, not comparable, and not pre-registered

2026-11-06 19:0x IST, written by the chain turn woken by `label=datascale rc=0`. This is a provenance audit,
not a result. Nothing here re-labels that run as a test of the plan's data-scale branch.

## What actually ran

`lens/runs/chain_datascale.out` (the whole child output), started 19:05:43 and finished 19:06:34 - **51 s**:

    METHOD: standard (the old loop: plain SGD, our gates OFF)
    cached: 13,318,447 tokens, vocab 12,000, 86,360 conversations
    window cap: 200,000 windows ... ; windows 200,000 train / 13,651 val
    concept module: chat_concepts_wide (43 defined); concepts: 40 kept, 3 dropped
    model: 3,891,720 params (d=256, K=40)
    ep 3/3 ... VAL ppl 369.85 acc 13.12% ... concepts 23/40
    NEW BEST 13.93% at epoch 3 -> saved datascale_s0_best.npz

## Why its 13.12% cannot be compared to the wall A/B's 13.40% / 13.26%

Four axes differ, and each one alone is enough to void the comparison:

| axis | this run | every wall arm (WALL-AB and WALL-REP) |
|---|---|---|
| corpus | 13,318,447 cached tokens, vocab 12,000, 86,360 conversations | 294,225,090 cached tokens, vocab 12,002, 740,666 conversations |
| windows | 200,000 train / **13,651 val** | 20,000 train / **87,035 val** |
| trainer method | `standard` - plain SGD, gates OFF | `ours` - relgrad step rule + gap gate + agreement gate, `--opt-lr 1.0` |
| concept set | 40 kept of 43 (K=40, 3,891,720 params) | 36 kept of 43 (K=37 off / 36 on) |

A 13.12% on a 13.6k-window validation set drawn from a corpus 22x smaller, with a different optimiser and a
different concept set, says nothing about the 13.40% that was measured on 87k held-out windows from the
294M-token mix. They are different experiments that happen to share "VAL acc" as a column name.

## Provenance problems, stated plainly

1. **No queue entry.** `grep -i datascale lens/runs/QUEUE.txt` returns nothing. The standing rule is that the
   queue is the record of what is allowed to run; this run bypassed it.
2. **No pre-registered prediction.** There is no `docs/history/DATASCALE-PREDICTIONS.md`. A number produced
   without a prediction written first cannot pass or fail anything, and by this project's own rules cannot
   be called a result.
3. **Launched inline, against the standing rule.** The chain log records the work item as a command string,
   not a script file:
   `chain start label=datascale work=CHAT_DATA_DIR="$PWD/data/chat/prepared" taskset -c 2,3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 ...`
   `glassnet_chain.sh` falls back to `setsid --wait bash -c "$WORK"` when the argument is not a file, so any
   multi-flag trainer command can be fired without a script, a lock, a GPU claim or a skip-if-done test -
   exactly the failure modes that cost this project three arms today.

## What was NOT done about it

Nothing was deleted, nothing was killed, and the checkpoints (`datascale_s0_*`, best.npz included) are where
the trainer put them. The run is left as it is, on disk, with its number recorded here so that it is neither
lost nor mistaken for evidence.

## What a comparable data-scale test needs

Same corpus (`data/chat/prepared_mix_ctrl_honest`), same method (`ours --opt-lr 1.0`), same d/window/epochs,
the same `--assistant-only --train-turn-end`, and only `--max-windows` moved from 20,000 to 200,000, with
predictions written before the launch and a script file rather than an inline command. That is what
`docs/history/DATASCALE-PREDICTIONS.md` + `lens/scripts/run_datascale_honest.sh` were written for; the run
itself is a separate decision, because one seed and a 1.03 pp measured run-to-run spread (REF-REPRO.md)
means any gain under ~1 pp will not be separable from noise.
