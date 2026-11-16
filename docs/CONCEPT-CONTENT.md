# What the concept neurons actually promote: 51 of 53 are partly code detectors (2026-11-16)

Predictions in `lens/runs/CONCEPT-CONTENT-PREDICTIONS.txt`, written before the aggregate ran. Method: a concept's effect on the logits is
its own row of `answer.W`, so **no ablation and no reconstruction is involved** - the matrix is the model's own numbers. For each of the
53 concepts we take the 12 tokens it promotes most, and classify them with a fixed, stated heuristic (pure-symbol tokens, or a token in a
named python/latex/maths keyword list).

## Per-neuron examples, verbatim

```
is_a_request     promotes: 'sqrt' 'be' 'solve' 'frac' 'pi' 'get' '*' '^'          own token present: help
is_a_question    promotes: 'python' 'so' '00' 'import' 'in' '?' "let's"           own token present: ?
is_an_agreement  promotes: ']' 'right' ',' '^' '$' 'left' '\\' '!'               own token present: right
mentions_self    promotes: 'i' 'j' 'my' '#' 'me' '[' 'print' 'arr'               own tokens present: i, me, my
mentions_a_person promotes: 'she' 'he' 'her' 'frac' 'his' '<|im_end|>' 'people'   own tokens present: he, her, him, people, she
mentions_the_user promotes: 'your' 'you' 'answer' 'param' 'import' 'python'       own tokens present: you, your
```

## Aggregates across all 53 concepts

```
concepts promoting at least one of their OWN trained tokens in the top 12 : 23 of 53
concepts with at least one code/math token in the top 12                  : 51 of 53
median code/math share of a concept's top 12                              : 0.417
```

- **C1 FAILED**: predicted at least 25 concepts would promote one of their own tokens; got **23**.
- **C2 PASSED**: predicted at least 20 would carry a code/math token; got **51 of 53**.
- **C3 PASSED**: predicted a median code/math share between 0.08 and 0.50; got **0.417**, near the top of the band.
- **FALSIFIER NOT TRIGGERED**: 23 concepts do promote their own tokens, well above the 10 that would have called the concept layer
  decorative.

## What this explains, and why it is the most important honesty result of the project

The model's vocabulary mixes conversation, Python and LaTeX. A concept layer that is a **single linear map over a shared window
embedding** has no way to separate "this window is a request" from "this window is maths", so each neuron picks up a **code-or-math
co-factor alongside its name**. The measurements fit together exactly:

- **trained-label fidelity 0.999** (docs/history/SHIPPED-VERIFY.md): the listed words really are present when the neuron fires.
- **held-out synonym transfer only 0.59-0.64**: the neuron's promoted mass is largely on code and maths tokens, not on the concept's
  semantic neighbours.
- **`explain()` exact to 2.9e-06**: the arithmetic is right; what it is arithmetic ABOUT is partly something else.

**So `explain()` can truthfully say "I answered because `is_a_request` fired 0.8", while `is_a_request` is in large part pushing
`sqrt`, `frac` and `pi`.** That is precisely the failure mode this project exists to catch - a label that sounds meaningful and partly
measures something else - and it is now measured, named and public rather than hidden behind a passing audit.

## What this demands of the next arms

More data will not fix it: the co-factor is a property of how the concept layer is supervised against a mixed corpus. The fixes that now
have evidence behind them:

1. **Supervise concepts on paraphrases, not just listed tokens** - so the neuron's mass has to move onto meaning.
2. **Penalise the code/math co-factor** - a contrastive term that pushes a concept's row away from a nuisance token set.
3. **Supervise concepts on a corpus slice with code and maths removed**, then show the named neurons still work on the full corpus.

None of these is a data-volume experiment, which is why the 8M arm running now cannot fix it and is honest about that in its own
predictions.
