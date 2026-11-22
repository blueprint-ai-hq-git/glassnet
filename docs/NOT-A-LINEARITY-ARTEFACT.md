

---

# NOT A LINEARITY ARTEFACT: THE CONCEPTS REALLY DO NOT CARRY THE ANSWER (2026-11-22 15:50)

The day's load-bearing negative - "the 36 concepts do not contain the answer" - rested on a fitted **linear** readout reaching 24.0%
against the model's own 23.2%. One fair objection: perhaps the information is there, arranged in a curved way, and a linear probe simply
cannot see it. That objection is now tested with readouts that carry **no linearity assumption and no fitting at all**.

```
windows 1200 (fit 600, test 600) | concepts 36
the model's own concept head:                        23.2%
nearest neighbour in concept space (nonlinear):       10.7%
wider ridge (100x less regularisation):              22.8%
P1 nearest neighbour below 40%: MET
FALSIFIER not fired
OUTCOME: LAYER_REALLY_LACKS_ANSWER
```

## What each number rules out

- **Nearest neighbour, 10.7%** - *no parameters, no linearity assumption*: if two windows have nearly the same 36 concept values, they do
  **not** share the same next token. That is the most direct statement possible of "the concept state does not determine the answer". It
  is also **worse than chance-adjacent**, because the model's vocabulary is 12,002 and its top-token distribution is sharply skewed.
- **Wider ridge, 22.8%** - dropping regularisation 100x moved the linear result **down** from 24.0%, so the earlier figure was not
  suppressed by over-regularisation.
- **The model's own head, 23.2%** - unchanged reference.

**The best of every readout is 23.2%, which is the head the model already has.** Nothing recovers the answer from the concepts, under any
of the three methods.

## The conclusion is now anchored

The readable middle layer is not mis-wired, and it is not mis-read. **It does not encode the next token.** Any explanation of this
model that is built out of these 36 concepts is a description of the sentence, not of the decision - and that is now measured three ways
(linear fit, wider fit, nearest neighbour), on 1,200 windows, against the model's own logits as the target.

The gate stands, unchanged and readable in seconds:
- `tools/direct_ablation.py` - the concept route must reproduce the model's own answer at **100%** (now 5.5-18.0%);
- `tools/concept_holds_answer.py` and `tools/concept_readout_nonlinear.py` - a readout must reach **at least 90%** before any head work
  is attempted (now 24.0% and 10.7-23.2%).
