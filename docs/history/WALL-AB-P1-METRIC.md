# P1's control band was built from one column and applied to another

2026-11-06 18:0x IST. This is a measurement bug in how the wall A/B's control gate was evaluated, not a
training result. Every number below is copied from a log line; the arithmetic is the only thing computed,
and both inputs are printed.

## The claim the gate makes

`docs/history/WALL-AB-PREDICTIONS.md`, P1:

    CONTROL SANITY. wall_on s0/s1 land within 10.0-12.5% final val acc, i.e. reproduce the stored
    reference band.

and above it, the design section names that reference:

    This is field-for-field the config of the stored 20k reference in lens/runs/CTX-ATTN.txt
    (final val 11.96/10.08, mean 11.02%).

## Two things are wrong with the citation, and one with the arithmetic

**(1) Wrong file.** `lens/runs/CTX-ATTN.txt` does not contain 11.96 or 10.08. Its arms are `attn_w32` and
`w64`; their finals are 9.06 / 6.41 / 8.75 / 8.23 / 8.17 / 9.62 / 8.90 / 6.77 ... (grep for `11.9[0-9]`
in that file returns one line, an unrelated epoch-1 ppl of 711.97). The numbers are in
`lens/runs/LEARNCURVE.txt`:

    :4773  lc_w20000_s0  ep 3/3 ... VAL ppl 552.99 acc 11.32% ... ASST(by-position) ppl 484.65 acc 11.96%
    :4960  lc_w20000_s1  ep 3/3 ... VAL ppl 683.12 acc  9.53% ... ASST(by-position) ppl 600.84 acc 10.08%

**(2) They are the ASST column, not the VAL column.** 11.96 and 10.08 are `ASST(by-position)`; the same
two runs scored `VAL acc` 11.32 and 9.53. The grader (`wall_ab_grade.py`) reads the `VAL ppl ... acc`
field, i.e. VAL acc, and tests it against a band derived from ASST values.

**(3) The result of that mix.** On the graded column, `wall_on` s1 is 9.68, which is 0.32 pp under the
10.0 floor, so P1 prints FAIL and the pre-registered rule then forbids quoting any delta.

## Compared on the same column, the control reproduces

| seed | VAL ref (LC :4773/:4960) | VAL on (WALL-AB.txt:392/:952) | delta | ASST ref | ASST on | delta |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 11.32 | 10.79 | -0.53 | 11.96 | 11.42 | -0.54 |
| 1 |  9.53 |  9.68 | +0.15 | 10.08 | 10.30 | +0.22 |

Both columns, both seeds, land within 0.55 pp of the reference. Read the ASST column (the one the band was
built from) and both `wall_on` values, 11.42 and 10.30, sit inside 10.0-12.5 -> the band PASSES.
Read the VAL column (the one the grader uses) and 9.68 falls 0.32 pp below the floor -> the band FAILS.

## The two commands really are the same command

`lens/scripts/run_learncurve.sh:19-22` and `lens/scripts/run_wall_ab.sh:113-117` invoke the trainer with
the same flags for the 20k arm; `run_wall_ab.sh` receives `--direct --direct-mode tied` for `on`, which is
`run_learncurve.sh:21`'s own `--direct --direct-mode tied`. The config headers the trainer prints agree
field for field on all five lines that matter:

    assistant-only (BY POSITION): 213,046/300,000 (71.0%)      both
    concept module: chat_concepts_wide (43 defined)            both
    concepts: 36 kept, 7 dropped (floor 0.005)                 both
    train-turn-end: 173 targets (0.058%)                       both
    windows 20,000 train / 87,035 val  vocab 12,002  window 32  both

So the reference is a matched control, and the ~0.5 pp difference at seed 0 is not explained by the
config. That is a separate, smaller question (the project's measured run-to-run spread is 0.01-0.02 pp,
`docs/history/D512-REPRODUCIBILITY.md`), and one re-run of `lc_w20000` seed 0 would settle it.

## What is NOT changed here

The verdict in `lens/runs/WALL-AB-GRADE.txt` still says P1 FAIL, because that is what the code does with
the numbers it reads. This note does not re-label the gate: a gate that is re-read after seeing the result
is not a gate. The finding is recorded so the FAIL is not mistaken for evidence that the rung drifted,
which is what a reader would otherwise conclude.

The decision this leaves is a rule decision, not a measurement one: either
  (a) keep P1 as coded (VAL acc vs a band built from ASST) and keep "no delta is quotable", or
  (b) re-derive the band from the column it names, which makes the control pass and makes the A/B's
      off-vs-on deltas quotable (+1.39 / +3.11 pp on VAL acc, +1.58 / +2.91 pp on ASST),
and it is Yahya's to make, not this worker's.
