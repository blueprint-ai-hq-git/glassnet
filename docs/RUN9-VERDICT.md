# Run 9 verdict: the state finally reacts, the readout still does not (2026-11-10)

Run 9 (direct path + decorrelation + our own readout normalisation) completed at 19:19 after 1500 steps.
`tools/eval_run9.py` on `scan9_ep1500.npz`:

```
F1  KL between prompts                 1.1702   bar >0.05   PASS
F2  state cosine                       0.1736   bar <0.5    PASS
F4  trained readout vs noise readout   trained 1.1702 vs noise 1.1660, ratio 1.00x, needs >2x   FAIL
G1  gain_ro range                      0.978 .. 1.119                                  in bounds
G2  entropy                            8.835 = 94.1% of uniform                        no collapse
```

## What passed, and it is the biggest movement of the day
The state now reacts to its input. State cosine **0.1736** against **0.9716** at run 6's mid-checkpoint: the
representation went from nearly identical across prompts to clearly different. KL between prompts is **1.1702**,
well over the 0.05 bar. The guards also hold: the learned gain stayed between 0.978 and 1.119, so the KL was not
bought by saturation, and predictive entropy is 94.1% of uniform, so nothing collapsed.

## What failed, and it is the same failure as the readable model
F4 asks whether the TRAINED readout is better than a random-noise readout at the same scale. It is not:
**1.1702 against 1.1660, a ratio of 1.00x**. So the state carries prompt-specific variation that the answer layer
does not use. This is the same shape of finding as `READABLE-PATH-EXPLAINS-7PCT.md`, on a different architecture:
the representation is doing something, and the decision is not being made from it.

## Output, honestly
Run 9's own sample answers are still symbol soup:
```
H: what is the capital of france ?   A: - the : of the ` . ` , and , and to the ' of the in .
H: tell me about the sea .           A: the a of the and of the in . . # , we to the ` of this is
```
No answer, no capitals, no facts. Two of three gates passing does not make this a talking model, and it must not be
reported as one.

## Started immediately after
`lens/scripts/run_matched_selfctx_d512.sh` - exposure-bias training (`--self-context-frac 0.5`, `--self-context-rounds
8`), which trains half the steps on windows whose last 8 tokens are the model's own greedy predictions. It is the
arm the feedback finding points to (`FEEDBACK-LOOP-MECHANISM.md`), at 5x the forward passes per step. Prediction,
recorded before the result: the copy rate at generation steps 9-24 will be lower than the 90-97% measured today, runs
will be shorter, at a small accuracy cost.
