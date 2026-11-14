# Is chat explain() causal? Yes - but it accounts for only ~10% of the decision

Date: 2026-11-14. Tool `tools/chat_explain_causal.py`, runner `lens/scripts/run_chat_explain_causal.sh`, predictions
pre-registered in `lens/runs/CHAT-EXPLAIN-CAUSAL-PREDICTIONS.txt`, output `lens/runs/CHAT-EXPLAIN-CAUSAL.txt`.
Checkpoint `lens/runs/chat/exposure_d512_e9_s0_best.npz`, 200 held-out windows, 20 random draws, inference only.

## The question

`lens/concepts.py` explain() returns an algebraically exact decomposition of `logits[pred]` into a bias term plus per-concept
contributions. That is true by construction for any linear answer layer, including one whose named concepts carry almost none of
the decision. The image line answered the same question by ablation (removing named rules took 346.7% of the margin vs 10.8% random).
Chat has no rule layer, so the named stage is the concept layer itself.

## First run of my own tool was invalid, and the numbers said so

The first run reported a **negative** median margin removed (-56.55%) and explain() residuals up to 1.88e+01. A negative margin
removed is impossible, which is the signal that the instrument, not the model, was wrong.

Cause: `lens/model.py:427` adds a direct pathway, **`logits += relu(x @ dmix) @ E.T`**. My ablation recomputed logits as
`h @ W + b` and ignored that term entirely, so every "ablated" logit was a different model's output. Fixed by recovering the direct
term exactly (`direct_vec = logits - (h @ W + b)`) and holding it constant while concepts are ablated, which isolates the concept
path's causal role.

## Corrected result

```
median margin removed by ablating the NAMED top-3 concepts :  17.51%
median margin removed by ablating RANDOM 3 concepts        :   2.08%
median ratio (named / random)                              :   4.13x    mean 5.05x

prediction FLIPPED by named ablation  : 22.50% of windows
prediction FLIPPED by random ablation :  5.83% of windows

explain() residual: max 1.877e+01   (its docstring says this must be ~0)
concepts needed to remove 90% of the margin: median 21 of 36

HOW MUCH OF THE PREDICTED LOGIT COMES FROM THE CONCEPT PATH: median 9.74%
```

Predictions, scored:

- **P1 (named >= 3x random) HOLDS** at **4.13x**. **explain() is causal, not a story**: the concepts it names really do drive the
  prediction.
- **P2 (named flips >= 20%, random <= 5%) FAILS narrowly**: 22.50% passes on its own, but random ablation flipped **5.83%**, above
  the 5% ceiling I set. Random concepts are not inert, because a concept layer with 36 units and 21 of them needed for 90% of the
  margin has no truly idle units.
- **P3 (residual < 1e-4) FAILS, and this is a real defect, not a measurement error.** explain() reports residuals up to **18.7734**
  while its own docstring says the residual "must be ~0". On a checkpoint with the direct/tied path the decomposition is therefore
  **not** exact, and the honest reading is the one the tool already prints: the named concepts account for part of the logit.

## The finding that matters

**The concepts that explain() names account for a median of 9.74% of the predicted logit. Roughly 90% comes from the direct pathway,
which no concept explains and explain() does not decompose.**

So the readable story on the chat model is real but small: it is a true causal account of about a tenth of the answer. For the
project's first standing goal ("100% honest and understandable"), this is the sharpest defect found so far on the chat line, sharper
than the concept-name generalisation gap, because it is a **coverage** problem: the explanation is correct and incomplete, and the
incompleteness is invisible unless the residual is read.

Two possible repairs, to be tested rather than assumed:
1. **Decompose the direct term too**, so explain() reports "concepts said X (10%), direct path said Y (90%)" - honest, cheap, and it
   makes the gap visible in every explanation.
2. **Make the concept path carry the decision** (e.g. constrain or remove the direct path for a readable checkpoint, as the route work
   already began), then check the cost in accuracy and copy behaviour.

## Caveats

- One checkpoint, one seed, 200 windows, top-3 concepts.
- The ablation is a **counterfactual on the concept path** (concept activations zeroed, direct term held constant), not a re-run of the
  model with concepts forced off inside the forward pass - the model has no hook to force that, so this measures path attribution.
- `concept_share` is the concept path's share of the *predicted token's* logit against the absolute direct term, not a share of the
  whole logit vector.
