# Is the flat ceiling the data, or the gates refusing two-thirds of the steps? 500k, gates OFF

2026-11-07 13:5x IST, written before `lens/scripts/run_datascale_500k_nogate_s0.sh` is launched.

## The question

The seed-0 ladder is 12.18 -> 17.43 -> 20.81 -> 21.29% (20k -> 200k -> 500k -> 2M), i.e. 1.58, 2.56 then **0.24 pp
per doubling** (`docs/history/DATASCALE-2M-RESULT.md`). Meanwhile the arms increasingly refuse their own steps:
the 500k arm ended at **3,907 of 5,850 steps refused (66.8%)** and the 2M arm at 3,633/5,600 (64.9%) when sampled
at step 5600 (`docs/history/GATE-REFUSALS-VS-LENGTH.md`). If the gates spend most of the late run rejecting the
model's own updates, then "more data stopped helping" and "the optimiser stopped accepting" are the same
observation with two explanations, and no result so far separates them.

This arm separates them: the **same 500k rung, same step rule (relgrad), gates OFF**.

## Why this is now possible, and what had to change

`--method ours` hard-sets `a.gates = "gap+agreement"` (`train_chat.py:509`), so `--gates off` was unreachable
while using our step rule. A new flag, `--no-gates`, applies after that block and sets `a.gates = "off"` while
leaving `--opt relgrad` and `--opt-lr` untouched, and it prints a distinct METHOD line.

**Validated before use** (`lens/runs/GATE-PROBE.txt`): two 1-epoch 20k probes,
  * gates ON: `METHOD: OURS (relgrad step rule + gap gate + agreement gate ...)`, **3 agreement-step lines,
    2 gate-step lines**;
  * `--no-gates`: `METHOD: OURS STEP RULE, GATES OFF ...`, **0 and 0**;
  * identical param count (3,811,534) and identical inputs, so the flag isolates the gates and nothing else.
The probe script's first verdict line read FAIL because a `grep -c || echo 0` idiom emitted "0\n0" and the
comparison broke; that was a bug in the checker, not in the flag, and it is fixed and recomputed in the same file.

## Predictions

G1  THE GATES ARE THE CEILING. 500k gates-off VAL exceeds the gated 500k arm (20.81%) by >= 1.0 pp, i.e.
    **VAL >= 21.81%**.
    FALSIFIER: outside 19.81-21.81 means the gates are not what stopped the curve; below 19.81 means the gates
    were HELPING (they revert steps that worsen the held-out probe, which is regularisation), and the ceiling is
    then the data and the model, not the optimiser.

G2  OVERFITTING SIGNATURE. If the gates were helping rather than blocking, gates-off should show a **larger
    train-minus-validation gap** than the gated arm at the same rung. I will compute both from the epoch lines and
    report the pair; no threshold, because the gated arm's gap is what it is.

G3  MECHANICAL. The log contains no `agreement step` and no `gate step` lines, and its METHOD line reads
    "OURS STEP RULE, GATES OFF". If this fails, the arm did not test what it claims and the result is void.

G4  CONCEPTS. Audit >= 36 of the kept set, as in every arm of this family.

## What will not be claimed

* One seed. No interval and no significance test; a +-1 pp move is inside the seed spread at this rung.
* No claim that "gates off is better" in general - a single rung, one seed, and the gates' purpose is to protect
  held-out quality, not to raise this number.
* Not comparable to the 2M arm (different data), and not a headline: 37.37% (d=512, full corpus, 6 epochs)
  stands.
* Cost: ~28 min (measured 554-558 s/epoch at 500k), ~500 MiB VRAM, checkpoints every epoch.
