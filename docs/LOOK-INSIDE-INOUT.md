
## LOOK INSIDE: the name describes what the neuron DETECTS, not what it PUSHES - and the two are decoupled (2026-11-16)

Predictions in `lens/runs/INOUT-PREDICTIONS.txt`, written before this ran. Read-only; no training touched.

## The question

`docs/history/NUISANCE-DATA-DRIVEN.md` found a concept's promoted mass on its own trained words to be **0.002**, while its input-side
detection of those same words is **0.999**. If that were a flaw in a few neurons it would be a wiring bug. The alternative is a
**mechanism**, so it was tested as one: is a concept's input detector coupled to its output push?

## Measurement

For each concept: **input quality** = trained-label AUROC from the honesty audit; **output content** = the share of its absolute
`answer.W` row that lands on its own trained word ids.

```
concepts                                             : 53
M1  median output own-word mass                      : 0.0008
M2  concepts at or above 0.05 output own-word mass    : 0 of 53
M3  rank correlation (input quality, output content)  : -0.307   over 22 audited concepts
    input range 0.914-1.000   |   output range 0.0002-0.0024
```

```
mentions_law      input 0.9999   output 0.0010
mentions_weather  input 0.9998   output 0.0008
is_a_question     input 0.9997   output 0.0002
mentions_a_colour input 0.9996   output 0.0007
is_explanatory    input 0.9995   output 0.0007
```

**M1 PASSED, M2 PASSED, M3 PASSED; falsifier not triggered.**

## The mechanism, stated plainly

**A concept neuron is a near-perfect detector on the way in and a diffuse, almost contentless knob on the way out, and nothing in the
training ever required the two to agree.**

- `lam_concept` supervision acts on the **concept activations** - the input side. That is why AUROC is 0.9997-0.9999.
- The **answer matrix is never supervised against the concept's own words**; its rows are learned only by next-token prediction, so each
  row becomes whatever helps prediction - a broad, mildly code-flavoured direction containing 0.08% of its own words at the median.
- `explain()` then attributes the answer to those rows. So the explanation is **exact about the arithmetic and empty about the content**:
  it correctly reports which handle was turned, on a knob whose label was never attached to what the knob does.

This is the root cause, and it is structural. It explains the 0.999/0.59 honesty gap, the 0.002 own-word mass, the 0.204 code co-factor,
and the 2.9e-06 exactness all at once.

## The fix this points to, and it is not more data

**Supervise the output side.** Require that concept `c`'s row of the answer matrix promotes `c`'s own words and demotes the nuisance set -
a term on the trained parameter that has never been trained for meaning. Concretely the term is
`-log softmax(answer.W[c] . onehot(own words))` averaged over concepts, i.e. the concept's row must actually prefer its own vocabulary.

It must be built verify-first, like the decorrelation term: analytic gradient derived, finite-difference checked, registered through the
engine's `OPS` table, smoke-tested, and proven to enter the loss only at non-zero weight - before any arm launches.
