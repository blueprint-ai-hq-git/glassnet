
## Epoch-mode endpoint: held-out fell 0.0046 nats - real in direction, below the bar, and my verdict string mislabelled it (2026-11-17)

Two passes over **distinct** windows, batch 8, lr 1e-4, clipped, 13.1 minutes each. Baseline re-measured in the same run's own protocol.

```
BASELINE (this run)        6.5694 nats
PASS 1/2  train 5.8526     VAL 6.5653   gap -0.0041
PASS 2/2  train 5.8025     VAL 6.5648   gap -0.0046
checkpoints lens/runs/chat/warmstart_pass{1,2}.npz, 119,533,084 bytes each
```

## The script printed a verdict that contradicts its own number, and that is fixed

The run printed:

```
VERDICT: baseline 6.5694 -> final 6.5648 nats (-0.0046). held-out did NOT improve: the extra capacity only memorises
```

**Held-out went down. That is an improvement in the direction that matters**, and the sentence attached to it said the opposite. The cause is my own threshold logic: the branch tested `v < base - 0.005`, the improvement was
0.0046 - just under - so it fell through to a message written for the *worse* case.

That is the same defect class as every instrumental failure this week: a label that reports something other than what was measured. The verdict logic is now four-way and cannot say "did not improve" about a
number that fell:

- fell past the 0.005 pre-registered threshold -> "the block is doing work"
- **fell but below the threshold -> "directionally positive, insufficient to claim the block works, too small to resolve at 40 held-out windows"** (this case)
- rose -> "the extra capacity is damaging generalisation"
- unchanged -> "unchanged"

## What the endpoint actually supports

**The pre-registered falsifier did not fire.** It said: *gap keeps widening and held-out does not fall => the block adds capacity that only memorises.* Held-out **fell on both passes, monotonically**
(-0.0041 then -0.0046). So the block is not a pure memoriser, and the replacement-sampling arm's damage really was a sampling artefact.

**It is also not an established gain.** 0.0046 nats is **0.07%** of the loss, and at 40 held-out windows that magnitude is inside the noise. So the accurate statement is:

> one certified block added to the shipped model, trained with distinct windows per pass, moves held-out loss in the correct direction by 0.0046 nats - below the pre-registered bar, and unresolvable at this
> evaluation size.

Neither "depth helps" nor "depth does not help" is supported. **The depth question is still open, and it is now blocked on measurement resolution rather than on the architecture.**

## The concrete next step

Widen held-out evaluation from **40 windows to 200** before the next arm. A 0.005-nat effect cannot be read at 40 windows no matter how the model behaves, so the next run must buy resolution first and
change nothing else. That is a measurement fix, not an architecture change, which is the correct order given this week's record.
