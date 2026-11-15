# The instrument was broken: my hand-built forward differs from the real model by 131 logits (2026-11-15)

Predictions were written in `lens/runs/INSTRUMENT-RECONCILE-PREDICTIONS.txt` before this ran, with an inverted falsifier: if the two paths
agreed and both reproduced the trainer's number, the earlier withdrawal would itself be withdrawn. That is not what happened.

## The comparison, on identical windows

```
path                       accuracy   mean NLL
hand-built numpy path       12.50%      35.433
REAL model forward          29.17%       5.033
max |logit difference| between the two paths: 131.8675
```

**My hand-built path - `emb[window] @ concept.W + concept.b -> answer.W` - is not the model.** It differs by up to 131 logits, halves the
accuracy, and inflates the negative log-likelihood seven-fold. The real model scores **29.17% / 5.03 nats**, which matches the trainer's
own **23.47% / 5.35 nats** on the same checkpoint.

- **I1 PASSED**: the paths differ by far more than 0.01.
- **I2 PASSED**: real-forward accuracy is 29.17%, well above 15%.
- **I3 not reached**: the difference is a reconstruction defect, not a population difference.
- **The inverted falsifier did NOT fire**: the paths do not agree, so nothing is restored.

## What this means, and it is the largest correction of the day

**Every number produced by my hand-built logit path today is withdrawn.** That covers:

- `tools/chat_generate_compare.py` - repetition, realistic-token coverage, KL and every generated sample in
  `docs/history/GENERATION-MEASURED.md` and `docs/history/ACCURACY-VS-GENERATION.md`
- `tools/exposure_bias_test.py` and `tools/exposure_bias_assistant_only.py` - already withdrawn an hour ago for exactly this reason,
  now explained
- `lens/scripts/measure_activation_spread.py` and every "distinct tokens" figure derived from it, including the 135/227/364/414/422
  breadth curve and its ceiling in `docs/history/BREADTH-IS-SUPERVISION.md`
- the "944 vs 364 tokens" argmax-versus-sampled correction in `docs/history/ARGMAX-IS-NOT-BREADTH.md`

**The conclusions those documents drew about generation quality - looping, junk tokens, a breadth ceiling - are not supported**, because
they were measured through a forward pass that is not the model. They are not merely imprecise; their subject was wrong.

## Why it went unnoticed for hours

The broken path is a *plausible* reconstruction: it uses the right parameter names and shapes, and it produces a probability
distribution on the right vocabulary, so every downstream metric "worked" and produced numbers of the right order. Nothing crashed. The
tell was that its absolute NLL disagreed with the trainer by 4x - and that only surfaced because a pre-registered falsifier forced a
reconciliation instead of allowing the number to stand.

The library had already met this exact defect class: `lens.load()`'s own docstring records that it once dropped the tied direct path and
produced "max |logit difference| 24.96 against the model the trainer had in memory". **The lesson the project had written down was to
load through the library's own loader. I built my own path anyway.**

## What stands, and what happens next

- **Stands**: everything the trainer measured - validation accuracy, perplexity, per-epoch curves, the three-seed concept-widening win
  (+0.57 +/- 0.04), the shipped lam 1.0 configuration, concept-name honesty (measured through the trainer's and `lens.concepts`' own
  paths), and explanation exactness.
- **Withdrawn**: the entire generation-quality and argmax-breadth line, pending re-measurement through `lens.load()` + `net.forward()`.
- **Next**: rebuild the generation comparison on the real forward and re-run it before any of those claims is repeated. That is the only
  legitimate next action; until it runs, the honest statement about this model's generation is "unmeasured".
