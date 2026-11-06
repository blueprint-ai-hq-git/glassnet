# DATASCALE-HONEST result: 10x the windows lifts the 20k rung by 5.25 points

2026-11-06 19:29 IST. One arm, seed 0, launched 19:08:57 and finished 19:29:28 (20.5 min). Predictions were
written before the launch: `docs/history/DATASCALE-PREDICTIONS.md`. Log: `lens/runs/DATASCALE-HONEST.txt`,
runner marker `DATASCALE-HONEST DONE 2026-11-06 19:29` at :323, `exit=0` at :317.

## The comparison, one flag apart

Both arms: same corpus (`data/chat/prepared_mix_ctrl_honest`, 294,225,090 cached tokens, 740,666
conversations), same command (`--d 256 --window 32 --epochs 3 --batch 256 --lr 0.4 --lr-decay 0.5
--lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only --train-turn-end --gate-tol 0.05
--method ours --opt-lr 1.0`, no bypass), same 36-kept concept set, **same parameter count 3,811,534**
(counted off the checkpoint, matching the log's `model: 3,811,534 params`).

    ONLY DIFFERENCE: --max-windows 20000 -> 200000

| arm | windows | VAL acc | ASST acc | final ppl | source |
|---|---:|---:|---:|---:|---|
| `wall_off` seed 0 (baseline) | 20,000 | 12.18% | 13.00% | 399.50 | `WALL-AB.txt:116` |
| `datascale_honest_s0` | 200,000 | **17.43%** | **18.61%** | **207.29** | `DATASCALE-HONEST.txt:230` |

    gain  VAL  +5.25 pp   (12.18 -> 17.43)
    gain  ASST +5.61 pp   (13.00 -> 18.61)
    ppl        399.50 -> 207.29   = 48.1% lower

Epoch path of the new arm (all three logged): VAL 17.48 → 17.90 → 17.43, ASST 18.19 → **18.87** → 18.61,
ppl 232.63 → 214.13 → 207.29. The best held-out checkpoint is epoch 2 at ASST 18.87%
(`datascale_honest_s0_best.npz`, 15,261,944 bytes), which is what the trainer saved.

## Against the predictions, as written

    D1 MAIN      predicted VAL 15.18 - 20.18% (+3 to +8 pp)     got 17.43%   -> PASS, inside the band
    D2 PPL       predicted final VAL ppl < 300 (baseline 399.5) got 207.29   -> PASS
    D3           one seed, no significance claim; the 20k rung's measured run-to-run spread is 1.03 pp
                 (REF-REPRO.md), and the gain is 5.1x that spread

Both falsifiers were registered in advance and neither fired: the gain is not <= +1.0 pp (so "data scale is
not the lever" is not what happened), and it is not > +8.0 pp (so the 20k rung was not so starved that every
earlier comparison is void - it was merely tight).

## Provenance check, done this run

`verify_wall_ab_ckpt.py` cannot read this log: the runner writes no `----- <arm> seed <N>` marker (a gap in
`run_datascale_honest.sh`, fixed for future runs below). The same checks were therefore run directly on the
files, and are recorded here with their output:

    log says model: 3,811,534        params counted off ep3: 3,811,534      -> match
    best sha == ep2 sha: True        best arrays == ep2 arrays: True
    best != ep3 (so best is the epoch the log named, not the last one): True
    meta seed/direct/mode: 0 False off                                      -> the no-bypass arm, seed 0

## What this means, and what it does not

* It is the first lever on this project that has moved the 20k rung by more than the noise: +5.25 pp,
  against a measured 1.03 pp spread and against every architectural lever tried so far (attention -2.81,
  window 64 -1.57, the bypass -1.4 to -3.4).
* It does **not** change the headline: best held-out next-token accuracy is still **37.37%** (d=512, full
  corpus, 6 epochs). 17.43% at d=256 / 200k windows is a different, smaller rung.
* It is **one seed**. The prediction said the signal would be 3-8x the spread and it is 5.1x, but a second
  seed is the honest way to publish it as a number rather than a lead.
* It says nothing about talk quality, "smart", or shipping. Next-token accuracy at 200k windows is all it
  measures.

## Hardening done in the same run

`run_datascale_honest.sh` now writes a standard arm marker (`----- off seed 0 : ... --max-windows 200000`)
before the trainer starts, so the checkpoint verifier and any future results parser can find an arm in this
log. The completed run's log is left exactly as it was written; its provenance is proved above instead.
