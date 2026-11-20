

---

# THREE SEEDS: EQUAL WITHIN NOISE AT 1.84x FEWER PARAMETERS, AND THE +0.18 WAS A LUCKY DRAW (2026-11-20 17:20)

The single-seed result said factored rank 32 was 0.18 points *better* than the full table. Three paired seeds say otherwise.

```
seed   full table   factored r=32   paired delta (factored - full)
 0       12.33%        12.51%              +0.18
 1       12.68%        12.05%              -0.63
 2       13.27%        12.78%              -0.49
mean     12.76%        12.45%              -0.31   (sd of the deltas 0.35)
```

- **P1 MET:** paired mean delta **-0.31**, inside the -1.0..+1.0 band. The two are **equal within noise**.
- **P2 MET:** every seed, both arms, lands between **11.3% and 13.5%** - no seed collapses, so this is not one bad run.
- **Falsifier not triggered** (it needed worse than -1.0).

## The correction this forces on my own last message

I reported rank 32 as "0.18 above the full table" and explicitly said it was not a claim. **Three seeds prove that caution was
correct**: the +0.18 was seed 0 being lucky, and the next two seeds were both negative. The defensible claim is:

> A factored output table reaches the **same held-out accuracy within noise** (mean delta **-0.31 +- 0.35** over 3 seeds) with
> **1.84x fewer total parameters** (454,336 vs 836,352).

It is **not** more accurate. Reporting the single seed as anything more would have been exactly the error this project keeps catching.

## Where the programme stands, measured

| piece | state | number |
|---|---|---|
| anatomy scan of our models | done | readable layer is 1.7% of params; 69-92% is a lookup table |
| our own attention block | done, proven | hand reference 4.441e-16, cache equivalence 1.192e-07 |
| backward pass | done, certified | worst relative 8.815e-07, input grad 2.650e-09 |
| trains on real text | done | 12.33% vs a 4.08% matched constant (+8.25) |
| factored embedding | done | equal within noise at **1.84x fewer params** |
| capacity IN the layers | **not started** | table share went 73.9% -> 85.0%, wrong direction |
| automatic allocation / alignment | **not started** | every knob is still hand-set |
| memorise vs generalise switch | **not started** | no mechanism exists |

The remaining three are the ones that were actually asked for. Next is automatic alignment: make the optimiser set its own per-tensor
step sizes from the data, so that the learning rate stops being a hand-tuned constant, and test it where a hand-set constant breaks.
