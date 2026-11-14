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

## Fixed the same wake: explain() now names the direct pathway instead of hiding it in a residual

explain() already reported `exact` and `residual`, so it was not lying - but a residual with no name makes an incomplete
explanation read like a complete one. `lens/concepts.py` now also returns:

- `direct_term` - exactly `logits[pred] - (b[pred] + sum of concept contributions)`, the undecomposed direct pathway
- `concept_share` and `direct_share` - the two shares of the predicted token's logit, which sum to 1

Verified on the d512 checkpoint, three held-out windows:

```
window pred    logit    residual  exact  concept_share  direct_share  direct_term
0      <unk>   10.011     5.8870  False       37.6%         62.4%        5.887
1      2       11.041    11.1718  False       14.1%         85.9%       11.172
2      int     10.920    10.8929  False        7.5%         92.5%       10.893
```

`residual` and `direct_term` agree, as they must (the residual IS the direct term), and the two shares sum to 1 on every window.
So a reader of a chat explanation now sees, in the same output, that the concepts account for 7.5-37.6% of the decision on these
windows and that the rest came from a pathway with no name in the concept layer. That is the honest version of the number.

The remaining open question is unchanged and is the important one: **should the concept path carry the decision** (removing or
constraining the direct path for a readable checkpoint), and what does that cost in accuracy and copy behaviour? That is a training
experiment, not a reporting fix, so it needs his call before it burns an hour of GPU.

## Verified the fix across consumers, and replaced the 3-window figure with a 200-window distribution

`tools/explain_share_distribution.py`, runner `lens/scripts/run_explain_share_distribution.sh`, output
`lens/runs/EXPLAIN-SHARE-DISTRIBUTION.txt` and `.json`. Inference only.

**Consumer regression check - the added fields broke nothing:**

```
api.explain returns all expected keys (incl. direct_term, concept_share, direct_share) : PASS
concept_share + direct_share == 1                                                        : PASS
explain_text() still produces output                                                     : PASS (884 chars)
```

**explain()'s own share, 200 held-out windows (d512, 9 epochs):**

```
concept_share   median 18.00%   q25 11.48%   q75 29.54%   min 3.90%   max 66.98%
direct_share    median 81.99%
windows where the concept layer is a MAJORITY contributor (>50%) :  10 of 200  (5.0%)
windows where it contributes less than 25%                        : 136 of 200 (68.0%)
explain() exact on this checkpoint                                :   0 of 200
residual  median 9.913   max 18.773
```

## Two share numbers, and why both are honest

- **9.74%** - the ablation tool's figure, the **net signed** concept contribution at the predicted token. This is the
  decision-relevant one: if two concepts push in opposite directions, the cancellation means neither moved the answer much.
- **18.00%** - explain()'s own figure, the sum of **absolute** per-concept attributions. Contributions that cancel still count
  here, so it is larger by construction.

Both say the same thing: the concept layer is a minority of the decision. Quoting either alone without saying which metric it is
would be exactly the kind of unattributed number this project keeps getting bitten by, so both are recorded with their definitions.

## The sharper statement for the goal

**On 95% of held-out windows the concept layer is not the majority reason for the prediction, and explain() is not exact on a single
one of 200 windows.** That is the honest state of chat readability today: causal (4.13x random) but covering a median 10-18% of the
decision, with 90%+ arriving through a pathway that has no name in the concept layer.

## Doubling the width HALVES the concept layer's share of the decision

Predictions pre-registered in `lens/runs/EXPLAIN-SHARE-SCALE-PREDICTIONS.txt`. Same instrument, same 200 held-out windows, only the
checkpoint changed (`exposure_d1024_e9_s0_best.npz`, 47M parameters). Output
`lens/runs/EXPLAIN-SHARE-DISTRIBUTION-d1024.json` / `.txt`, runner `lens/scripts/run_explain_share_d1024.sh`.

```
checkpoint   concept_share median   q25     q75     direct_share median   concepts majority   concepts <25%   exact
d512  e9 s0       18.00%          11.48   29.54         81.99%             10 of 200           136 of 200    0 of 200
d1024 e9 s0        8.89%           5.87   16.69         91.11%              1 of 200           174 of 200    0 of 200
```

- **P1 HOLDS**: the d1024 concept share (8.89%) is lower than d512's 18.00%.
- **P2 HOLDS**: explain() is exact on 0 of 200 windows at d1024 as well (residual median 11.389, max 20.295).
- **P3 HOLDS**: majority-contributor windows fall from 10 to **1** of 200.
- **Falsifier NOT TRIGGERED**: scale does not buy coverage.

## Why this matters more than the number

This is the **second independent instrument** pointing the same way, and they agree in direction:

```
metric                                              d512 -> d1024
name generalisation gap (own labels vs synonyms)   0.2117 -> 0.2576   WIDER
concept share of the decision                      18.00% -> 8.89%     SMALLER
windows where concepts are the majority reason      10/200 -> 1/200    FEWER
```

**The bigger model routes more of the decision around the named layer, while its own audit looks better.** On d1024 the concepts are
91.1% bypassed on a median window even though they score 0.9754 on the labels they were trained on. So "chat readability is not a
scaling problem" is no longer a single-metric claim: on both the meaning test and the coverage test, **width makes the readable story
weaker, not stronger.**

That directly contradicts the natural plan of "get a bigger model and the concepts will carry it". For the project's first standing
goal, the implication is that the direct pathway has to be constrained by design, and that any future width increase must carry the
share measurement with it or the honesty percentage will keep rising while the explanation keeps covering less.

## Caveats

- One seed per width (s0 for both), 200 windows, same instrument.
- The two checkpoints differ in more than width in principle (same recipe, same corpus, same epochs), but the trainer's own config
  records d512 vs d1024 as the only change; that is asserted from the reference invocation, not re-derived here.
- The share metric is explain()'s absolute-attribution share (18.00% / 8.89%); the net-signed ablation share was 9.74% at d512 and has
  not been re-run at d1024. Direction is consistent across both metrics where both exist.
