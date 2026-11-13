# RETRACTION: dmix is one of the most important components in the model, not dead weight (2026-11-13)

**This replaces the headline of `docs/PARAM-BUDGET-AUDIT.md`, which was measured on the wrong checkpoint. The numbers in it were
correct; the conclusion drawn from them was wrong, and the advice it gave for the 50M budget was backwards.**

## What was measured, and on which model
The audit ablated tensors on `early_s0_ep1000` - a COPY-finetuned checkpoint - and found `dmix` (8,388,608 parameters, 51.3% of that
model) worth **0.00** points of language accuracy. From that it advised spending the 50M budget on width and the pointer path rather
than on the direct pathway.

**The right model for a language-budget question is the pretrained language model.** Ablating `matched_d512_s0_best` (route off,
baseline 23.93% on the same held-out split), zeroing each tensor:

```
tensor        params      language after   damage     share of the model
emb           6,145,024       5.60%        +18.33        39.5%
dmix          8,388,608       5.60%        +18.33        53.9%
answer.W        432,072      20.33%         +3.60         2.8%
concept.W       589,824      22.00%         +1.93         3.8%
answer.b         12,002      23.47%         +0.47         0.1%
concept.b            36      24.00%         -0.07         0.0%
```

**`dmix` is tied with `emb` as the single most important component of the language model: removing either costs 18.33 points.** It
is not dead weight. My audit's advice to cut it would have removed the joint-largest contributor to the model.

## Why the two measurements disagree, and what the real finding is
Not a contradiction - a state difference, and the reason is a bug already described:

- **During pretraining** (`matched_d512`, route OFF) `dmix` learns normally. Drift between ep1 and ep3 is non-zero for **every**
  tensor including `dmix`; nothing is frozen.
- **In the copy arms** (`--route-gate`, route ON) `forward()` captures `_route_base = logits` **before** the direct path contributes
  and mixes that base with the pointer readout at the end, so the `dmix` term never reaches the output. No contribution -> no gradient
  -> drift **exactly 0.0000** between `early_s0_ep1000` and `ep4000` -> 0.00 in every ablation, on every task.

So the finding survives and is sharper than before:

```
model state                        dmix contribution to language     dmix drift
pretrained, route OFF              +18.33 points (joint largest)     normal
copy-finetuned, route ON            0.00 points                     exactly 0.0000
```

**Turning on `route` silently disables the direct pathway.** Every route-on arm in this project has been training with 8.4M parameters
computed and discarded - including all of today's copy arms, whose numbers remain valid for copy but say nothing about `dmix`'s worth.

## Corrected advice for the 50M budget
- **Keep the direct pathway.** It is measured at +18.33 points on the language model, tied with the embedding table.
- **Fix the ordering.** Either include the direct term in the route combination or do not enable `route` in arms that rely on `dmix`.
  Until that is fixed, `route` and `direct` are not independent switches.
- **The parameter-efficiency question is still open**, and the honest version of it is: which components earn budget *in the same
  model state*, since a component's measured worth can be zero simply because another switch is discarding it.

## What I got wrong, in one sentence
I measured a component in a model state where a routing bug had already disabled it, and published the result as a property of the
component - which is indistinguishable from the "identical output means the patch applied" errors this project has already been bitten
by, in a statistical costume.

## Reproduction
`tools/lang_ablation.py --ckpt lens/runs/chat/matched_d512_s0_best.npz --mode zero --windows 1500`;
drift comparison `early_s0_ep1000 -> early_s0_ep4000` and `matched_d512_s0_ep1 -> ep3`; the pre-route reasoning is in
`docs/PARAM-BUDGET-AUDIT.md`, now superseded on its headline.
