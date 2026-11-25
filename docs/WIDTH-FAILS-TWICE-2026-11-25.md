# 2026-11-25 - width, tested for sample efficiency: it fails for the SECOND time in one day

Source: lens/runs/COVERAGE-VS-SIZE.out (exit 0), COVERAGE-VS-SIZE.json. Same 24,000-step budget, same base set, same held-out
500 tens-carry pairs; only hidden width and the number of tens-carry examples changed.

    coverage   narrow (H=128, 30,570 params)   wide (H=512, 120,810 params)   gap
         100              25.6%                          23.6%                -2.0
         400              56.8%                          48.0%                -8.8
        1600              88.0%                          91.0%                +3.0

P1 MET (coverage is monotone at both widths). P2 NOT MET (wide does not win at every level). P3 NOT MET (the gap is not largest
at low coverage).

## The honest conclusion

A **4x wider model is not more sample-efficient at learning a mechanism.** It is behind at 2 of the 3 coverage levels, by as
much as 8.8 points, and its single win is 3.0 points at high coverage. Combined with this morning's capacity result (-0.017 on
language loss from 3.5x the parameters), width has now failed twice on two different questions.

So "small model that is smart per parameter" cannot be bought with parameters. It has to come from coverage plus replay plus
verification - the parts that have actually moved numbers today.

## A verdict-logic error, caught and fixed (fourth of the day)

The script printed **"CAPACITY BUYS SAMPLE EFFICIENCY"** on these numbers. It keyed off `max(gaps)`, which selects the single
positive value (+3.0) and ignores both that the pre-registered conditions had failed and that the largest absolute gap (-8.8)
favoured the narrow model. Rewritten so the conclusion is derived from the pre-registered predictions, never the most
flattering number in the table.

That is the fourth verdict-logic bug today and the same shape as the other three: a conclusion that was generated from
whatever number was convenient rather than from the test that was declared beforehand.
