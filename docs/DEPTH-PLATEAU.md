
## Depth: the block gives a real 0.041-nat gain and it plateaus after two passes (2026-11-17)

Five checkpoints, paired over the same 200 held-out windows, bootstrap intervals computed per pass.

```
pass 2 (epoch arm, resume point)  -0.02623   CI [-0.04997, -0.00305]
extend pass 1                     -0.03502   CI [-0.06437, -0.00782]
extend pass 2                     -0.04021   CI [-0.07520, -0.00575]
extend pass 3                     -0.03989   CI [-0.07505, -0.00318]
extend pass 4                     -0.04127   CI [-0.07931, -0.00271]   <- endpoint
```

**Every interval excludes zero. The gain is real. And it stopped growing after extend pass 2** - passes 3 and 4 moved it by +0.0003 and -0.0014 nats, which is an order
of magnitude smaller than the 0.0046-0.0088 steps that preceded them.

## The falsifier, stated precisely because it did not fire in its literal form

Pre-registered before launch: *if the paired gain does not exceed the pass-2 value of -0.02623 by the end of four more passes, the gain is saturated and the block is
not earning its 15.2M parameters.*

**Final value -0.04127 exceeds -0.02623, so the literal threshold did NOT fire.** The substance of saturation did arrive - the curve flattened at pass 2 - but the
threshold as written was cleared, and that is the honest way to report it. The threshold is not restated after the fact to fit the outcome.

## What the endpoint supports, and what it does not

**Supports:** adding one certified 15.2M-parameter block to the shipped model and training with distinct windows per pass improves held-out loss by **0.041 nats**,
resolvable at 200 paired windows, plateaued after two to three passes.

**Does not support:** that this is a good trade, or that depth is a win. **0.041 nats is 0.67% of the 6.0379 baseline loss, bought with 51% of the parameter budget.**
On its own numbers the block is a poor purchase - real, resolvable, and small.

**Also does not support:** any claim the plateau is the architecture's ceiling rather than the data's. This arm saw ~10,000 distinct windows once each over five passes; the
shipped model trained on 4M windows for 12 epochs. A plateau at 10k windows says the *gain available from this data at this rate* is 0.041 nats, not that a block in
this position cannot do more with the corpus the other arms used.

## A labelling defect caught in the endpoint, same family as before

The run's own `VERDICT` line printed:

```
VERDICT: baseline 6.5648 -> final 6.5579 nats (-0.0069). held-out IMPROVED by 0.0069 nats, past the pre-registered threshold
```

That **-0.0069 is the unpaired 40-window reading**, printed directly beneath a paired 200-window line reporting **-0.04127**. Two numbers for one endpoint, six times
apart, in adjacent lines of the same log. The `VERDICT` line is computed from `base_val`, which is the 40-window path, and it was never updated when paired evaluation
was added.

The verdict now prints an explicit note that the unpaired figure must not be quoted as the arm's result. **This is the third variant of the same defect this week** -
after the 6.06-vs-6.57 mismatched comparison and the falling-number-called-did-not-improve label - and the pattern is now unmissable: *wherever two measurement paths
exist, the report will eventually mix them.* The structural fix is that the arm has **one** result path, and it is the paired one.
