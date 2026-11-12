# The sparse rule layer: 12 rules where every rule matters (2026-11-12)

The ablation of the 96-rule layer found that the median rule removed on its own costs **0.000** accuracy points - about ten
rules carried the model and 86 were inert. So the layer was cut to 12 rules, cross-facet wiring kept, everything else
identical. Then the same ablation was run on the result.

```
                              accuracy   honesty   ablation   price     median per-rule loss   lift-vs-usefulness
96 rules, cross-facet           16.45%     76.6%     18.00%    1.55 pp       0.000 points            -0.012
12 rules, cross-facet           13.90%     78.1%     17.55%    3.65 pp       1.613 points            +0.236
trivial baseline                 5.42%       -          -         -              -                      -
```

**Predictions: accuracy 11-15% passed (13.90%); median per-rule ablation loss above 0.05 points passed at 1.613; zero
contradictory rules passed.**

## What changed, and it is the thing that mattered
1. **Every rule is now load-bearing.** Median per-rule loss went from **0.000 to 1.613 points**. In the 96-rule model a
   typical rule could be deleted without changing a single answer; now the typical rule is worth 1.6 accuracy points, and
   the most necessary one (rule #1, `position: chat tag in last 4 AND earlier: token at -2 is punct`) is worth **4.725**.
2. **Plausibility finally correlates with necessity.** Pearson between a rule's concept-pair lift and its usefulness was
   **-0.012** in the 96-rule model (nothing) and is **+0.236** here (spearman +0.266). That is the first time in this
   project that "the rule says something sensible" and "the model needs the rule" have pointed the same way at all.
3. **A human has to read twelve rules, not ninety-six** - and all twelve change the answer, which is what a readable
   model is supposed to mean.

## What it cost
- **Accuracy 16.45% -> 13.90% (-2.55 points).** Twelve rules is a real bottleneck; the layer is now doing work it cannot
  spread around.
- **The readability price rose to 3.65 points** (17.55% ablation vs 13.90% with rules) from 1.55, because a bottleneck
  layer costs more than a wide one.
- **Half the load-bearing rules are still implausible**: 5 of the 10 most necessary rules have pair lift below 1.0. Rule
  #3 (`shape: last token is 6+ chars AND earlier: token at -3 is punct`, lift 0.70) is worth 3.750 points.

## The trade, stated plainly
Two configurations, two different claims, both true:

```
96 rules:  readable, cheap (1.55 pp), accurate (16.45%), but 86 of 96 rules are inert decoration
 12 rules: readable, faithful (every rule matters, plausibility correlates +0.236), but costs 2.55 points of accuracy
```

The 12-rule model is the one that can honestly be called **transparent**: every part of it changes its behaviour, and a
person can check all of it in a minute. The 96-rule model is legible but its own explanation does not identify what carries
it.

## What is next
The bottleneck is now the interesting question: 12 rules cost 2.55 points, so the curve between 12 and 96 should be swept to
find where "every rule matters" stops being true. Somewhere between 12 and 96 there is a rule count where the layer is both
cheap and fully load-bearing, and that number is a property of this task, not a matter of taste.
