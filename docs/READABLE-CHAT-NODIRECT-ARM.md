# A fully readable chat checkpoint exists: forcing the concepts to carry the answer costs ~4 points, not 23

Date: 2026-11-14. Arms `nodirect_d512_s0` (direct pathway OFF) and its matched control `directctl_d512_s0` (direct pathway ON),
both trained for 3 epochs, d512, seed 0, 600k windows of the full 294,225,090-token corpus. Predictions
`lens/runs/NODIRECT-PREDICTIONS.txt` written BEFORE launch. Logs `logs/exposure/{nodirect,directctl}_d512_s0.log`,
outputs `lens/runs/NODIRECT-ARM.txt`, `NODIRECT-CONTROL.txt`, `NODIRECT-EXPLAIN-CHECK.txt`, `NODIRECT-VS-CONTROL.json`.

## Why this ran

The chat concept layer was measured decorative for prediction: the direct pathway carried 82-91% of the decision, the concept-only
answer was a constant `<unk>`, every post-hoc readout repair failed, and a probe on the 36 activations scored 18.32% train against
4.45% held-out (`CONCEPTS-CARRY-NOTHING.md`). The open question was whether a model **forced** to answer through the concept vector
can learn the task at all.

It needed no code change: the reference invocation passes `--direct --direct-mode tied` (verified by diffing the reference script),
so omitting those two flags leaves `logits = h @ W + b` with the concept vector as the only route to the output. Everything else is
the reference invocation verbatim, and a matched control was trained with the flags ON because the project's own rule forbids
comparing a readable arm against a number borrowed from another document.

## Result - matched, epoch by epoch

```
epoch      no-direct VAL   control VAL   direct ON   gap
1             9.94%          16.63%                  -6.69
2            13.53%          18.97%                  -5.44
3            14.84%          19.51%                  -4.67
held-out best 15.63%         21.53%                  -5.90
concepts pass  23 of 36      28 of 36                 -5
```

On the project's next-token metric, same 2,000 held-out positions, same instrument:

```
model                        accuracy   distinct tokens predicted   <unk> share
no-direct (3 ep)              17.25%              28                   0.0%
control, direct ON (3 ep)     21.20%             121                   0.0%
exposure d512, 9 ep           28.25%             132                  18.75%
```

## The headline

**A chat checkpoint whose every prediction is exactly decomposable into named concepts now exists.** On the no-direct checkpoint
`explain()` is **exact on 100 of 100 windows** (mean residual 0.000000, max 0.000003) with `concept_share` **100.00%** by
construction, against a direct-path checkpoint where it is exact on none of 200 and the concepts carry a median 8.9-18%.

**And the price is about 4 points of validation accuracy (3.95 on the next-token metric), not the 23-24 points the inference
ablation had suggested.** That earlier number was explicitly labelled an upper bound measured on a model *trained with* the path;
this matched pair shows a retrained model does recover most of it, exactly as that caveat said.

The `<unk>` degeneracy also disappears: **0.0% of predictions are `<unk>` in both 3-epoch arms**, against 18.75% in the 9-epoch
direct-path checkpoint. So `<unk>` dominance was a *product* of the direct pathway plus `--drop-unk`, not a property of the concept
layer.

## What it costs that accuracy does not show

**Output diversity collapses: 28 distinct tokens predicted versus 121 for the matched control.** The no-direct model leans on
`the`, `,`, backtick, quote and period. At 3 epochs it is a narrow talker, and the next-token accuracy hides that. Any claim about
this arm must carry the diversity number beside the accuracy.

## Predictions, scored honestly

- **P1 FAILED**: I predicted at most 12.0%; it reached **14.84%** VAL (17.25% on the next-token metric). The bottleneck carries more
  than I expected.
- **P2 FAILED**: I predicted >100 distinct tokens; it predicted **28**. My threshold was written for a 2,000-position sample and
  the prediction was wrong regardless - diversity is the real casualty of the bottleneck.
- **P3 HOLDS**: `explain()` is exact, residual ~0, `concept_share` 100% by construction.
- Decision rule from the predictions: 5-20% is the "graded cost" band, and 14.84% sits in it.

## What this changes

Readable chat is **not** hypothetical and does not need a new architecture. It needs a full-length arm of the arm that already
works, plus diversity work:

1. **Train `nodirect` at the shipping configuration** (9 epochs, and d1024 for the width comparison) with the same matched-control
   rule, and quote accuracy, diversity and share together.
2. **Fix the diversity collapse** - the 3-epoch bottleneck mode-collapses onto high-frequency tokens. Candidate causes: the 36-concept
   bottleneck is narrow, and `--drop-unk` plus a 12,002-token softmax over a 36-vector invites frequency-hugging.
3. Every readability claim for chat can now be stated in the strong form: **exact decomposition, and here is the accuracy it costs.**
