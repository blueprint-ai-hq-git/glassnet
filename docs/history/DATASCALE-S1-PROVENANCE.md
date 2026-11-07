# The seed-1 tag was taken by a different job overnight; what that means for the data-scale result

2026-11-07 08:0x IST. Written after reading, not editing, `lens/runs/DATASCALE-HONEST.txt`.

## What happened, in order

1. **20:45:28, Nov 6** - `glassnet-datascale-honest-s1` launched the pre-registered seed-1 replication:
   METHOD `ours` (relgrad + gates), corpus `prepared_mix_ctrl_honest` (294,225,090 cached tokens),
   `--max-windows 200000`, seed 1. Predictions were already on disk
   (`docs/history/DATASCALE-SEED1-PREDICTIONS.md`).
2. **20:47:21, Nov 6** - it was stopped at epoch 0: the log says `SIGTERM: saving an interrupt checkpoint
   for epoch 0`, `exit=143`. `journalctl` names the cause exactly: *"Stopped glassnet-wake.timer"* at
   20:47:20, and the transient unit was stopped with it, after 1 min 52 s of CPU time. That is a deliberate
   shut-down, not a crash and not a defect in the arm. It produced **no result**.
3. **21:09:30, Nov 6** - the machine went down (boot log: that boot ended 21:09:30).
4. **08:01:27, Nov 7** - after the 07:50 boot, a different job ran: `/tmp/run_seed1.sh` (a script outside the
   repository, created 08:01) through the chain as label `datascale_s1`. It used **plain SGD with the gates
   off**, corpus `prepared_dialogue_only_honest` (38,996,999 cached tokens, 156,521 conversations),
   20,946 validation windows, **41 kept concepts**, and - the problem - `--out datascale_honest_s1`, i.e. the
   seed-1 replication's own tag.
5. Its result, from the log's last epoch: **VAL 11.04% / ASST 10.63% / ppl 693.45 / concepts 23/41**.

## What that number is, and is not

It is a real run with real numbers, and it is kept. It is **not** the seed-1 replication, and it is not
comparable to seed 0's 17.43%: different corpus (38.99M vs 294.23M cached tokens), different trainer method
(plain SGD, gates OFF vs `ours`), different validation set (20,946 vs 87,035 windows), different concept set
(41 vs 36 kept), different parameter count (the checkpoints are 15,667,640 bytes vs seed 0's 15,261,944).

## What was done about it

* Its checkpoints and JSONs were **moved, not deleted**, to
  `lens/runs/archive/foreign_datascale_s1_20261107/` - 11 files plus a `SHA256SUMS` receipt and a copy of
  the log segment (`run.segment.log`, 147 lines) - so the record survives with its hashes.
* A provenance note was appended to `lens/runs/DATASCALE-HONEST.txt` naming both runs and their line ranges.
* The tag `datascale_honest_s1` is now empty, so the honest-mix arm can run without its files being mistaken
  for this one.

## Why it happened (and the rule it broke)

`/tmp/run_seed1.sh` hard-codes `--out datascale_honest_s1` and appends to the same log. Nothing in the
project reserves a tag: the runner's skip-if-done test only checks *its own* tag's npz files, so a foreign
job can take a tag and the next honest run will look "already done". Two fixes belong here and are cheap:
one log file per tag, and a tag that names the corpus and method (`datascale_honest_ours_s1`), not just the
seed. The runner has been changed to write a per-tag log; a tag-reservation file is the next step if more
agents keep sharing this machine.

## Status of the replication

**Seed 1 of the 200k honest-mix arm has NOT run.** Seed 0 (VAL 17.43%, ASST 18.61%, ppl 207.29,
concepts 36/36) stands, and nothing in this note changes it. The seed-1 predictions remain the ones to test
when it runs.
