# Rule semantics: the model's biggest rules join concepts that avoid each other (2026-11-12)

The causal test showed the answer layer reads the concepts. It did not show the RULES mean anything, and the printed
explanations looked like nonsense ("window contains <unk> AND window contains '}'"). `tools/rule_semantics.py` puts a
number on that: for every rule it reports its two wired concepts, how hard it can push an answer, and how strongly those
two concepts co-occur in TRAINING data (lift and PMI in bits).

## The defect, measured on the unconstrained model
```
rules with 2 wires                                    96 of 96
rules joining statistically INDEPENDENT concepts      23 of 96      (lift 0.85-1.15)
of the 20 most influential rules, joining independent   5
median pair lift across all 96 rules                  0.98          <- the wiring was, on average, exactly independence
median |PMI|                                          0.584 bits
correlation rule influence vs pair lift               -0.100
```
And three rules were wired to **logically impossible** pairs: `last token is punctuation AND last token is a digit` has
lift **0.00** - a token cannot be both - and it still carried influence 50.7. One rule, `contains <|im_start|> AND
contains '{'`, is anti-correlated (lift 0.40) and pushed three different answers by +0.6 to +1.6 in the sample windows.

## The fix and what it bought
`COOC=1` builds the structural mask from training-window co-occurrence: a rule may only be wired to a pair with
**lift >= 1.5** and at least 20 co-occurrences. Everything else is unchanged, same 12 concepts, same 3000 steps.

```
                              accuracy   honesty   ablation   price    median lift   independent rules   impossible pairs
unconstrained (12 concepts)     15.28%     84.0%     17.75%    2.47 pp      0.98            23 of 96           3
semantic wiring (lift >= 1.5)   15.30%     79.0%     17.07%    1.77 pp      1.38            14 of 96           0
trivial baseline                 5.42%       -          -         -           -                -               -
```
Predictions: "every rule reads lift >= 1.5" **partly held** (median 1.38, impossible pairs gone, but rare-pair lift is too
noisy to guarantee per-rule); "accuracy within 1.5 points of 15.28%" **passed** (15.30%); "influence-vs-lift correlation
turns positive" **failed** (-0.127, slightly worse).

## What that means, honestly
- **Wiring is now plausible on average**: median pair lift 0.98 -> 1.38, median PMI 0.584 -> 1.279 bits, independent-pair
  rules 23 -> 14, logically impossible rules 3 -> 0. That is the "does the rule say something a person could use" check
  moving in the right direction.
- **But the model still prefers meaningless rules.** The correlation between how hard a rule pushes and whether its two
  concepts belong together stayed negative. Its single biggest rule in the constrained model (`contains <|im_start|> AND
  contains '}'`, influence 140) reads a pair with lift 0.41 - the concepts actively avoid each other. Constraining the
  vocabulary of rules changed which rules exist, not which ones the model chooses to rely on.
- **Honesty paid for it**: concept honesty fell 84.0% -> 79.0% while accuracy held. Two readability properties are not
  independent, so a change that improves one can quietly cost the other.
- **Only 16 of 66 concept pairs reached lift 1.5**, so this constraint is close to a bottleneck: there is not much room
  left to make rules both varied and meaningful with 12 concepts. More concepts, or concepts at the level of token PAIRS
  rather than single tokens, is the way out.

## Run hygiene, learned the hard way
Two void runs preceded this one. The first had a scope bug (`LIFT` assigned locally, so the mask silently stayed random -
telltale: identical 15.28%/84.0% to the previous arm), and the second overlapped the first, so both wrote the same log and
checkpoint and the saved weights came out identical to the unconstrained model. The accepted run uses its own unit name,
log and `rules_semantic3.npz`. The lesson is now enforced in code: the mask builder raises instead of falling back to
random wiring, and a config print states which pairs were used.
