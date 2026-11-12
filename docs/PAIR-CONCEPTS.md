# Token-pair concepts: the reason rules are meaningless is that the concepts are mutually exclusive (2026-11-12)

Hypothesis under test: rules join concepts that have nothing to do with each other because the concepts are single tokens.
Make the concepts positional and pair-level - "previous token is punctuation", "last two tokens both digits" - and rules
ought to become sayable. One change against the 12-honest-concept arm, identical recipe and step count.

```
                                accuracy   honesty   ablation   price     median lift   median |PMI|   independent rules
single-token concepts (12)        15.28%     84.0%     17.75%    2.47 pp      0.98          0.58 bits       23 of 96
pair/positional concepts (12)     15.40%     74.8%     17.65%    2.25 pp      0.72          2.18 bits       10 of 96
trivial baseline                   5.42%       -          -         -            -              -              -
```

Predictions: accuracy within 1 point of 15.28% **passed** (15.40%); concept honesty **>= 84%** **failed** (74.8%); median
rule pair lift **>= 1.5** **failed** (0.72, and worse than the single-token set); independent-concept rules **< 15 of 96**
**passed** (10). Two of four.

## What actually happened
The pairs are more strongly RELATED than tokens (median |PMI| rose 0.58 -> 2.18 bits) - but **negatively**. Median lift
fell to **0.72**, meaning the two concepts in a typical rule are less likely to be true together than chance. The top rules
make the reason obvious:

```
rule 16  influence 42.7  lift 0.00   previous token is punctuation AND last two tokens both digits
rule 25  influence 37.7  lift 1.09   last token is punctuation AND previous token is punctuation
rule 48  influence 36.4  lift 0.36   last token is punctuation AND tag in last four positions
rule 65  influence 35.0  lift 0.00   previous token is punctuation AND previous token is a digit
```

`previous token is punctuation AND previous token is a digit` has lift **0.00**: it is not rare, it is **impossible**, and it
is one of the model's four heaviest rules. This is not a wiring accident - the concepts are *mutually exclusive by
construction*. A token either is punctuation or is a digit. Nothing can satisfy both. So any conjunction of two
positional concepts about the same token is either a contradiction (lift 0) or a tautology (one implies the other).

**That is the structural reason 2-to-3-concept rules read as nonsense, and it is a design error in the concept set, not in
the training.** A rule that ANDs concepts can only mean something if its concepts describe DIFFERENT aspects of the window
(one about shape, one about content, one about position) and can therefore genuinely co-occur.

The influence-vs-lift correlation stayed negative (-0.104). The model's preference for contradictory rules is unchanged by
changing what the concepts are; with the vocabularies available it has almost no non-contradictory conjunctions to choose
from.

## What this rules out and what it points to
- Ruled out: "make the concepts richer and the rules will become meaningful". Bigger concepts alone do not help, because
  the meaningfulness of a conjunction depends on whether its parts CAN co-occur.
- Ruled out: single-token vocabulary was the bottleneck. It was not; the exclusivity structure was.
- Points to: concepts must be chosen as near-independent FACETS. Concretely, one concept per aspect - a shape concept
  ("last token is punctuation"), a content concept ("window contains the word 'is'"), a position concept ("the tag appears
  within the last four tokens") - so a rule can join shape AND content and mean something. That is testable on the same
  harness with the same measurement.

## Run hygiene
The first pair arm is **void**: its concept list contained a duplicate ("last two tokens both punctuation" and "punctuation
then punctuation" were the same test), so it had 11 distinct concepts named 12. The concept definitions now live in one
module-level function, `pair_concepts()`, which the training script and every analysis tool import, so a drifting copy
cannot happen again. The accepted run is `rules_pair2.npz` / `RULES-PAIR2.txt`.

The analysis tool also had to learn to read the checkpoint's own concept names and refuse to score a model against a
concept set it was not trained on - the first attempt crashed with an index error rather than silently mislabeling, which
is the behaviour I want from it.
