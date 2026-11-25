# 2026-11-25 - the memorisation lock, measured: 100% where trained, 0.0% on the same mechanism unseen

Source: lens/runs/META-VS-MEMORISE.out (exit 0), META-VS-MEMORISE.json.
Trained ONLY on additions that need no carry (23+41=64). Tested on additions that need a carry (47+85=132) - the same
mechanism in a form never seen. 24,000 steps, identical net, identical held-out set.

    arm              no-carry held    CARRY held (unseen)
    plain                 100.0%              0.0%
    hint (flag given)     100.0%              0.0%
    plan_first            100.0%              0.0%
    few_shot (2% carry)   100.0%              5.0%

P1 MET, P2 MET, P3 NOT MET (+0.0), P4 NOT MET (+0.0), P5 NOT MET (+5.0).

## What this actually shows, and it is not "hints do not work"

The failure is not missing information. The **hint arm was handed the carry flag as a real input token and still scored 0.0%**,
because the blocker is the OUTPUT distribution, not the input: 100% of its training answers began with the digit 0 (every
no-carry sum is under 100), so the leading answer position was a constant carrying zero information. A model cannot learn to
write a 1 in a column that never varied. Knowing the mechanism was useless because it had nowhere to put the result.

The few-shot arm (80 carry examples = 2%) moved 0.0% to 5.0%. Slightly above zero, nowhere near transfer.

## The correction this experiment forced on my own design

My first version encoded the hint as `13 if hint else 13` - the same token whether the flag was set or not - so the hint arm
was never actually tested and P3/P4 were unevaluable. That is my error, found by the run, fixed in the same file. A second
tell I should have caught sooner: all three arms scoring identically meant the arms were probably not different.

## The design consequence

Any column that is constant in training carries no gradient and cannot be learned. So either the data must vary every output
position, or the model must WRITE the intermediate step explicitly (a trace), which makes the varying thing an output instead
of an input. The next experiment tests exactly that: give the model a dedicated carry slot in its OUTPUT and see whether it
then places a carry correctly in a position it never trained on.
