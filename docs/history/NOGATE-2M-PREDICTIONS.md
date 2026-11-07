# The 2M rung without the gates: was the flat ceiling the data, or the gates? (predictions, before the run)

2026-11-07 16:0x IST, written before `lens/scripts/run_datascale_2m_nogate.sh` is launched.

## Why this arm

Two facts from today sit next to each other and have not been reconciled:

* the data ladder **flattened** between 500k and 2M: 20.81% -> 21.29%, i.e. 0.24 pp per doubling
  (`docs/history/DATASCALE-2M-RESULT.md`);
* at the 500k rung, **removing the gates bought +1.24 pp and ran 7.7x faster** (20.81 -> 22.05,
  `docs/history/NOGATE-500K-RESULT.md`).

Every rung of the ladder was gated. So "more data stopped helping" may be a property of the gated method rather
than of the data. This arm re-runs the top rung **ungated**: same 2M windows, same seed 0, same batch 256, same
step rule, gates off. It is cheap because the ungated 500k arm ran 72 s/epoch, so ~3 epochs here is ~15 minutes.

It is also a second, independent test of the gates-off result at a different rung - 500k was one rung and one seed.

## Predictions

U1  LEVEL. 2M ungated beats the gated 2M arm (21.29%) by >= 0.5 pp, i.e. **VAL >= 21.79%**.
    FALSIFIER: below 21.29% would mean the gates HELPED at this rung, contradicting the 500k result; 21.29-21.79
    would mean the gates-off gain shrinks with data and is not a general property.

U2  FLATTENING, RE-TESTED WITHOUT GATES. Paired against the ungated 500k arm (22.05%), the ungated 2M arm lands
    in **21.5-23.0%**, i.e. no more than +1.0 pp for 4x data. Stated as a band because this is the number the
    whole "is data still the lever" question now hinges on.
    FALSIFIER: above 23.0% means the ungated curve is still climbing steeply and today's "flattened" conclusion was
    an artefact of the gates - which would reopen the ladder.

U3  PERPLEXITY. ppl < 149.14 (the gated 2M value).

U4  MECHANICAL. 0 `agreement step` and 0 `gate step` lines, METHOD reads `GATES OFF`, 3,811,534 params.

U5  CONCEPTS. Audit >= 36 of the kept set.

## What will not be claimed

* One seed per configuration. Both ungated arms are single draws; the seed spread at 200k was 3.07 pp, which is
  larger than every delta in this note. The ordering is a direction, not an effect size.
* No headline change (37.37%, d=512, full corpus, 6 epochs).
* This does not fix the d512 step-size confound from `D512-PROBE-RESULT.md`; it is a separate, cheaper question.
* Cost ~15 min, ~490 MiB VRAM, checkpoints every epoch, CPU pinned to two cores.
