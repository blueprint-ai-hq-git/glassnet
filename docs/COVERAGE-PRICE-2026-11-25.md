# 2026-11-25 - the price of coverage, in examples: 1600 buys 91.4% transfer (my threshold prediction was WRONG)

Source: lens/runs/COVERAGE-CURVE.out (exit 0), COVERAGE-CURVE.json. Fixed 24,000-step budget, base set 2,000 no-carry + 2,000
units-carry pairs, varying ONLY how many tens-carry examples were added. Tested on 500 held-out tens-carry pairs never trained.

    tens-carry examples   trained classes   UNSEEN tens-carry position
            0                99.4 / 100.0%            0.0%
           25                97.0 /  99.6%            8.0%
          100                96.4 / 100.0%           24.8%
          400                97.0 /  99.0%           54.2%
         1600                95.4 /  99.8%           91.4%

## My prediction was falsified

I predicted a THRESHOLD - flat, then a jump of 50+ points between 100 and 400 examples. The largest single step is **+37.2
points**, so the pre-registered falsifier fired: coverage cost is a **slope, not a switch**. Mechanisms in this architecture
are bought gradually, per example, with no discrete "now it understands" moment.

That is the third falsified prediction of the day and it is recorded as such, not reframed as a partial success.

## What the curve is worth

It converts "we need coverage" into a PRICE: about **1,600 examples of a mechanism in the position where it occurs buys 91.4%**
of it in an unseen setting, at this model size and step budget. That is a number a data budget can be built on, and it is the
first quantitative answer today to "how much data per skill".

One secondary observation, recorded because it is a trade: the trained-class accuracies drift DOWN as tens-carry examples are
added (99.4% -> 95.4% on no-carry). A fixed-capacity model pays for new coverage with a little of the old, which is the same
interference that replay exists to counteract.
