# Rule ablation: the model's load-bearing rules are NOT the plausible ones (2026-11-12)

The cross-facet model has no contradictory rules and its heaviest rules read as sentences. That still says nothing about
whether the *plausible* rules are the ones doing the work. `tools/rule_ablation.py` answers it: it splits the 96 rules by
whether their two concepts actually co-occur in TRAINING data, deletes one group at a time from the answer layer, and
measures the accuracy lost on held-out windows.

Model: `rules_facetx.npz`, 12 facet concepts, 96 rules, 2 connections each. Held-out accuracy on this 4,000-window sample:
**16.07%** (the training-time evaluation reported 16.45% on its own sample; window sampling differs, so both are quoted).

```
DELETING A WHOLE GROUP                        rules   accuracy   loss     per rule
plausible   (pair lift >= 1, concepts co-occur)  44     8.92%    7.15 pts   0.163
implausible (pair lift <  1)                     52     8.90%    7.17 pts   0.138
random control, 44 rules                          -       -       8.10 pts   (spread 6.90 over 5 draws)
random control, 52 rules                          -       -       7.45 pts   (spread 1.88 over 5 draws)
```

**Deleting the plausible rules costs no more than deleting random rules.** With a 6.90-point spread across random draws,
7.15 is not distinguishable from 8.10.

## The clean statistic: every rule removed on its own
```
per-rule accuracy loss:  min +0.000   median +0.000   max +1.475 points
correlation (rule pair lift) vs (accuracy lost when the rule is removed):
    pearson -0.012        spearman -0.044
the 10 most load-bearing rules: 7 of them have pair lift < 1.0
their lifts: 0.86, 1.09, 2.12, 0.74, 0.69, 0.97, 0.86, 0.80, 1.56, 0.97
```

**Plausibility and usefulness are uncorrelated at the rule level (pearson -0.012).** The heaviest rules, in the sense of
"the model cannot lose them", are the ones whose concepts *do not* co-occur:

```
rule #21  lift 0.86  removing it costs 1.475 points   position: chat tag in last 8 AND earlier: token at -3 is punct
rule #46  lift 0.74  removing it costs 1.275 points   position: chat tag in last 8 AND earlier: token at -2 is punct
rule #15  lift 0.69  removing it costs 1.200 points   content: window has '$' AND position: chat tag in last 4
rule #11  lift 1.56  removing it costs 0.100 points   content: window has backslash AND earlier: token at -2 is punct
```
Rule #11 is the most *plausible* rule in that list and the least necessary of the four. The three most necessary rules are
all below lift 1.

## And most rules are individually redundant
**Median per-rule loss is 0.000 points.** Of 96 rules, roughly ten carry the model and a typical rule can be deleted
without changing a single answer. So "96 rules a human must read" is the wrong description of this model: a human must
read about **10** rules, and the other 86 are decoration that the L1 penalty did not remove.

## What this means for the readability claim, stated plainly
- The layer is **readable**: 12 named concepts, 2 connections per rule, no contradictions, every decision prints as
  "rule #k fired at value v and pushed word w by +x".
- The layer is **not yet informative** about why the model works: which rule carries the model is uncorrelated with whether
  the rule says something sensible. A plausible rule and an arbitrary rule are equally likely to matter.
- Two properties that were being treated as one are now separated: **readable** (can a person read it) and **faithful**
  (does the readable thing explain the behaviour). The first is achieved; the second is not, and the ablation above is the
  measurement that shows it.

## What would move faithfulness
1. **Cut the rule layer to the rules that matter.** About 10 rules carry the model; a sparse layer of 10-16 rules, each
   required to be plausible AND load-bearing, is a much stronger product claim than 96 mostly-inert ones.
2. **Tie each rule to a measurable outcome during training**, not just to a loss: a rule that never changes an answer should
   be removable by the optimiser, and the L1 penalty clearly is not doing that (median loss 0.000 while L1 is on).
3. Only then is it worth asking whether the plausible rules can be made the load-bearing ones - which is a training-objective
   question, not a concept-vocabulary one.
