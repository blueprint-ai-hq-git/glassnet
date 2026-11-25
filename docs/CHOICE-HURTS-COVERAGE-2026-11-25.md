# 2026-11-25 - curated examples HURT: boundary coverage is 46.8 points worse than random at equal count

Source: lens/runs/COVERAGE-BOUNDARY.out (exit 0), COVERAGE-BOUNDARY.json. Net identical in all arms, 24,000 steps, base set
2,000 no-carry + 2,000 units-carry pairs, only the KIND of tens-carry examples changed. Tested on 500 held-out tens-carry pairs
of both kinds, drawn naturally.

    arm                examples   trained classes   UNSEEN tens-carry
    random                 400      96.2 / 99.2%          58.4%
    boundary (causal)      400      98.2 / 99.6%          11.6%
    random                 100      96.0 / 98.8%          36.6%
    boundary (causal)      100      98.4 / 99.2%          10.2%

Boundary example: 87+19=106 - tens digits 8+1=9 and the units carry supplies the final unit, so the carry is visibly CAUSED by
the units column. That is the most instructive example of the mechanism available.

**It is 46.8 points WORSE than random at equal count.** P2 (boundary beats random by >=15) NOT MET. P3 NOT MET.

## Why, and this is the useful part

Notice the trained classes score HIGHER in the boundary arms (98.2/99.6 against 96.2/99.2) while the held-out score is
terrible. The boundary arm is not a worse learner - it is a better learner of a NARROW SLICE. The held-out set is drawn from the
NATURAL distribution, which is mostly case (ii) pairs where the tens digits carry on their own. Curating for the causal example
removed exactly the variation the test needs.

**The rule this gives us: coverage must MATCH THE TEST DISTRIBUTION.** An example that illustrates a mechanism perfectly is
still a bad training example if the test will not be drawn from its slice. Instructive is not the same as representative.

This is the THIRD independent failure of clever data selection in this project, and the first two were measured the same day:
  * model-chosen novelty/self-sense sampling: 0.004 spread = noise (lens/runs/SELF-SENSE.out)
  * boundary curation: -46.8 points (this run)
  * and earlier, distinct-data VOLUME worked while every selection scheme failed

## Fifth verdict-logic bug, same shape

The script printed "EXAMPLE CHOICE MATTERS ... (0.20x the transfer)" on a gap of -46.8 points, because the branch only tested
|gap| < 0.05 and never read the SIGN. Fixed: the conclusion now carries its direction, and a negative gap prints CHOICE HURTS
with the distribution explanation rather than a sentence that sounds like a win.
