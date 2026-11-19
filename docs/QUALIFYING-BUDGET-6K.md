

---

# THE QUALIFYING BUDGET IS 6000 STEPS: FIRST MODEL IN THIS PROJECT VERIFIED ABOVE TRIVIAL (2026-11-19 14:05)

After the denominator correction, the question that had to come before any new architecture work: does ANY model here beat the trivial constant once both
numbers are computed on the same positions? Re-scored `scan12_r090_6k_ep6000.npz` (the 6000-step base) with the corrected instrument:

```
scan12_r090_6k_ep6000.npz   table(Wa)  1-layer  no memory
  top20 36.68% | r21-500 12.45% | r501-2k 0.00% | tail 0.00% | overall 23.06%

  SAME-DRAW constant: all positions 14.70% | <unk>-excluded 13.14% | model <unk>-excluded 24.54%
  -> BEATS constant by +11.40 points
```

**This is the first model in this project shown to beat trivial on a common protocol.** It also reproduces the earlier evaluator number (23.06% overall) exactly,
which is a consistency check that passed.

## What this settles

- **6000 steps is the qualifying budget.** It is the lowest budget measured where the model has headroom above the constant, so it is the lowest budget
  at which a mechanism test can mean anything.
- **Every memory and depth comparison run today (600-3000 steps) tested an under-trained model**, and that is now a measured statement rather than a
  suspicion: at 3000 steps the arms sit at 12.80-12.97% against a 13.14% constant.
- The 3000-step "fair test" I built at 12:45 was not fair. It failed on a denominator mismatch and on budget, for two independent reasons.

## Launched

`win6k`: window vs control at **6000 steps**, both instruments, predictions written before the run:
- **P1** both arms beat the matched constant (13.14%) - if not, the budget is still too low and I say so rather than interpreting the delta.
- **P2** the window beats control by >= 0.5 points on the trainer's line.
- **P3** the independent re-score reproduces the sign. **A delta that exists on only one instrument is not a delta** - that is the lesson of the
  withdrawal earlier today.
- **P4** the window still holds K distinct real words.
- **FALSIFIER:** both clear the constant AND the window is not >= 0.5 above control **on both instruments** -> the readable window buys nothing when the
  model is capable, and the memory line is dead on measurement rather than on another hypothesis.
