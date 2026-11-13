# Route and direct must stay mutually exclusive - both half-fixes measured, both rejected (2026-11-13)

The finding was real: with `route` on, `forward()` captures the route base **before** the direct path contributes, so `dmix` (8.4M
parameters) is computed every step and discarded - drift exactly 0.0000, zeroing it changes nothing
(`docs/PARAM-BUDGET-RETRACTION.md`). This tests the two ways to fix that, and both make things worse.

## The comparison: one script, one flag apart, paired seeds
Copy task from `matched_d512_s0_best`, `--pointer --pointer-qk --route-gate --unset-pointer-only` exactly as every earlier route arm,
mixed 50/50 stream, 4000 steps:

```
arm           held-out copy   taught copy   language     what it is
direct_out      83.1 / 85.1      83.4        12.62       current behaviour: the term is computed and discarded
direct_in        1.4 /  1.2      30.5         8.91       GLASSNET_ROUTE_DIRECT=1: the term is included in the route base
difference       -82.82          -52.9        -3.71
```

**Including the direct term inside the route destroys the copy pathway: -82.8 points of held-out copy.** The reported falsifier
("direct_in losing more than 3 points means the term damages the copy pathway") is triggered by a factor of twenty-seven.

Prediction `P1` ("the direct_in checkpoint shows dmix ablation damage > 1.0, proving the term now reaches the output") is also
**falsified**, and in the more informative direction: measured on the checkpoint that was actually trained with the term, the damage is
**-1.25 points** - removing `dmix` **improves** language accuracy by 1.25. So the term does reach the output, and it is actively
unhelpful there.

```
checkpoint          trained with   language   dmix zeroed   damage
rp_in_s0_final      term IN          9.75        11.00      -1.25   (valid: trained and evaluated the same way)
rp_out_s0_final     term OUT         0.25        14.25     -14.00   (invalid as a dmix measure: trained one way, evaluated the other)
```

## The mechanism, and why it is not surprising in hindsight
The direct term is enormous: a hand-computed pass put `relu(x @ dmix) @ E^T` at mean magnitude **2.63** against the concept/answer
logits' **0.093** - about 28x. Putting a term of that size inside the route base makes the base dominate whatever the pointer readout
offers, so the switch stops being a choice between two comparable readouts and becomes a choice that drowns the copy path. Copy
accuracy falls from 84% to 1.3% accordingly.

## What the correct fix actually is
**Do not include the term - refuse to pay for it.** When `route` is on, `dmix` cannot contribute, cannot learn, and costs 8.4M
parameters plus the model's largest matmul on every step. The repair is to stop building it in that configuration, which changes no
behaviour at all, instead of wiring a 28x-larger term into a switch that was tuned without it.

```
configuration                     dmix built   dmix used   measured
route OFF, direct ON                  yes         yes       +18.33 ablation, ~5 points of training deficit if removed
route ON,  direct ON (current)        yes          NO       current shippable behaviour
route ON,  direct ON (term wired in)  yes         yes       copy -82.8, language -3.71   REJECTED
route ON,  no dmix built              no           no        the proposed repair: no behaviour change, 8.4M params saved
```

The last row is the one to implement, and it is a configuration change rather than a model change: `route` and `direct` should be
mutually exclusive in the code that builds the model, with a clear error if both are requested.

## Two of my own mistakes on the way here, both caught by reading artifacts
1. An arm passed `--direct` to `train_chat.py`, which **has no route flag at all** - it silently became a reference replication
   (usefully: 19.88% against the reference's 19.69% at epoch 1).
2. An arm passed `--route-gate` but **not** `--pointer --pointer-qk`, so no pointer existed and copy read **0.0% taught and 0.0%
   held-out** - impossible for a copy task, which is how it was caught at seed 0.

Both were caught by printing the artifact's own state, which is why neither became a published claim.

## Reproduction
`lens/scripts/run_route_pair.sh` (predictions and falsifier in the header) - both orderings in one file, paired seeds, logs
`logs/routepair/{out,in}_s{0,1}.log`, checkpoints `rp_out_s*` / `rp_in_s*`. The switch: `GLASSNET_ROUTE_DIRECT`, default off.

## The repair implemented and verified: don't build it, and behaviour is identical

`GLASSNET_ROUTE_SKIP_DIRECT=1` makes the model builder leave `dmix` as `None` whenever `route` is on, so the pathway that would be
discarded every step is never created. Verified on a real route-on checkpoint (`rp_out_s0_final.npz`), same 200 held-out windows:

```
guard        dmix built   parameters   language accuracy
OFF (today)      yes      16,354,001        13.00
ON               no        7,965,393        13.00
```

**Half the parameters, identical output.** The guard is OFF by default so every existing checkpoint loads byte-identical, and it is a
pure configuration change: no weights move, no behaviour changes, 8,388,608 parameters and the model's largest matmul disappear from
every route-on arm.

Getting the guard in took three edits, each caught by running it rather than reading it: the first version skipped the tensor and
then fell through an `elif` chain that referenced the low-rank tensors that were never built (`AttributeError: dproj`); the second
still hit `UnboundLocalError: copy_out` in the return dict. Both branches now test `dmode`, and the guard-off path was re-verified
unchanged (16,354,001 params, 13.00) after every edit.

### Where this leaves the route/direct question
- Including the direct term in the route: **rejected**, copy -82.8 points.
- Not building it when route is on: **adopted**, identical behaviour and half the parameters.
- Using `direct` without `route`: **still the only configuration in which the direct pathway actually contributes** (+18.33
  ablation damage, ~5 points of training deficit if removed).

So the two switches stay mutually exclusive, but the cost of the exclusivity is now zero instead of 8.4M parameters.
