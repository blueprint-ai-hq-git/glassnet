# The seed-2 20k baseline existed already - and today's arm reproduced it exactly

2026-11-07 10:05:41 IST. Arm `datascale_base20k_s2` (20k windows, seed 2), predictions written before it ran
in `docs/history/DATASCALE-20K-S2-PREDICTIONS.md`. Log `lens/runs/DATASCALE-20K-S2.txt`,
runner marker `DATASCALE-20K-S2 DONE 2026-11-07 10:05`, `exit=0`.

## What was measured

    VAL 13.40%   ASST 13.83%   ppl 384.39   concepts 22/37   (ep 3/3, loss 5.4227)

## The two corrections this arm forces

**(1) I was wrong that seed 2 had no 20k arm.** `wall_rep_off_s2` - the `off` arm of the WALL-REP replication
block, run 2026-11-06 - is exactly this experiment: 20k windows, seed 2, same corpus, same command. Its log
line (`lens/runs/WALL-REP.txt:329`) reads `VAL ppl 384.39 acc 13.40%`, with the same `ep 3/3 ... loss 5.4227`.
The seed-2 predictions and the seed-2 result doc both said "there is no matched 20k arm at seed 2"; that was
wrong, and both now carry a correction.

**(2) Today's run reproduced that arm to the last printed digit.** Same ppl (384.39), same VAL (13.40%), same
loss (5.4227), same concept count (22/37), in a different transient unit, hours later, on a rebooted machine.
That is the strongest determinism evidence this project has produced: not "the spread is small", but "two
separate runs printed the same numbers".

So the arm was unnecessary. It cost 7 minutes of an idle card and bought an exact reproducibility check plus a
self-documented 20k baseline, and nothing was lost or overwritten.

## The verdicts

    B1  seed-2 20k VAL inside 12.10-12.90%     got 13.40%  -> FAIL, 0.50 pp above the band
    B2  within-seed gain inside +5.0..+9.0 pp   got +7.10  -> PASS
    B3  no re-grade of the 200k numbers         honoured: 20.50% / S1 band failure unchanged

B1's band was built from the two older baselines (12.18, 12.79). Seed 2's baseline is genuinely higher at
13.40%, so the miss is a fact about the baseline distribution, not about the 200k arm.

## The table, now fully paired

| seed | 20k VAL | 200k VAL | paired gain | sources |
|---|---:|---:|---:|---|
| 0 | 12.18% | 17.43% | **+5.25 pp** | `WALL-AB.txt:116`, `DATASCALE-HONEST.txt:230` |
| 1 | 12.79% | 19.71% | **+6.92 pp** | `WALL-AB.txt:850`, `DATASCALE-HONEST-s1.txt:262` |
| 2 | 13.40% | 20.50% | **+7.10 pp** | `WALL-REP.txt:329` = `DATASCALE-20K-S2.txt`, `DATASCALE-HONEST-s2.txt:256` |
| mean | 12.79% | 19.21% | **+6.42 pp** | |

    spreads across the three seeds:  20k 1.22 pp   200k 3.07 pp   paired gain 1.85 pp

## What the pairing shows that the absolute numbers did not

The three baselines and the three gains rise together (12.18->+5.25, 12.79->+6.92, 13.40->+7.10): a seed that
gives a stronger 20k baseline also gives a stronger 200k arm. So most of the 3.07 pp spread in the 200k values
is a **shared seed effect**, not noise in the effect itself - which is why the paired gain spans only 1.85 pp
and is the number to quote: **+6.42 pp on three seeds**, with mean VAL 19.21%.

## One more correction, about the concept set

The docs have been saying every arm used "the same 36 kept concepts". That is not exact: the keep/drop test
runs on the sampled training windows, so the 20k arms keep **37** concepts (`concepts: 37 kept, 6 dropped`) and
the 200k arms keep **36** (`36 kept, 7 dropped`). One concept crosses the support floor differently at the two
rungs. The comparison is still command-for-command and parameter-for-parameter identical, but the phrase "same
36-concept set" is replaced by "the same concept module, with 37 kept at 20k and 36 at 200k" in the documents
that make that claim.
