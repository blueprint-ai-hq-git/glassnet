# Six epochs at 4M, gates off: is the plateau steps or data? (predictions, before the run)

2026-11-07 17:1x IST, written before `lens/scripts/run_datascale_4m_nogate_ep6.sh` is launched.

## The observation this tests

The 4M ungated arm's last two epochs were the same number to two decimals:

    ep1 23.11%  (ppl 106.06)
    ep2 23.75%  (ppl  97.08)
    ep3 23.74%  (ppl  95.99)   <- accuracy flat, perplexity still improving

So the arm stopped gaining accuracy at ~46,875 updates while its loss kept falling. That is a signature of a
**step-limited** arm, not a data-limited one: with 4x more windows the same run would have the same 3 epochs and the
same plateau. This arm holds the data fixed at 4,000,000 windows and doubles the epochs to 6 - same seed, batch,
step rule, gates off - so the only thing that changes is the number of updates (46,875 -> 93,750).

## Predictions

W1  STEPS ARE THE CONSTRAINT. The best VAL in six epochs exceeds 23.74% by at least 0.3 pp, i.e. **>= 24.04%**.
    FALSIFIER: below that means the arm saturated on data-independent terms - more steps at the same data buy
    nothing - and the next lever is not "more epochs" at all.

W2  WHERE IT LANDS. Best VAL in **24.0 - 25.5%**. The 4M three-epoch run gained +0.64 pp per epoch early
    (23.11 -> 23.75) and then nothing, so six epochs should recover part of that early rate, not all of it.
    FALSIFIER: above 25.5% would mean the plateau was a scheduling artefact; below 24.04 repeats W1's falsifier.

W3  PERPLEXITY below 95.99, since loss kept falling even while accuracy did not.

W4  MECHANICAL. 0 `agreement step` / 0 `gate step` lines, METHOD reads GATES OFF, 3,811,534 params.

W5  CONCEPTS >= 36 of the kept set.

## What will not be claimed

* One seed. A 0.3 pp threshold is well inside the 3.07 pp seed spread measured at 200k; the *shape* (does epoch 4-6
  move at all) is the finding, not the delta's size.
* This arm does not test more data (that would be 8M) and does not test bigger batches. If it comes back flat, both
  of those become live candidates and neither is answered here.
* Cost: 6 x ~487 s = ~49 min plus corpus assembly; checkpoints every epoch (~8 min each), VRAM ~490 MiB, CPU pinned
  to two cores.
* No headline change; 37.37% is 13.63 pp above the current best d256 arm.
