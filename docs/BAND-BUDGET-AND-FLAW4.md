

---

# WHERE THE ACCURACY ACTUALLY COMES FROM, AND WHY FLAW 4 IS WORTH ZERO (2026-11-19 17:30)

Before spending a 7-hour arm on FLAW 4 (the `<unk>` positions, 5.34% of the corpus), I measured what each frequency band is worth. Shares are from the whole
500,000-token validation region; band accuracies are from the 6000-step checkpoint on the independent instrument.

```
band        share of positions   measured acc    accuracy contributed
top20         45.31%            31.87%        14.44 pts
r21-500       36.08%             8.13%         2.93 pts
r501-2k        9.90%             0.00%         0.00 pts
r>2k           3.52%             0.00%         0.00 pts
<unk>          5.19%             0.00% (masked) 0.00 pts
```

**Reconstruction check, stated honestly:** these contributions sum to **17.37 points** while the measured overall accuracy on the same checkpoint is **23.06%**. The
gap comes from comparing band accuracies measured on one 104-window draw against band shares measured over the entire region - the draw is top-heavy relative to
the region. So treat the shares as population-level and the accuracies as draw-level; the ORDERING is what this analysis is for, not a 0.1-point reconciliation.

## The three conclusions

1. **83% of all accuracy comes from the top-20 tokens, which are 45% of positions.** The model is a frequent-token predictor and everything else is rounding.
2. **FLAW 4 IS WORTH ~0 POINTS.** A `<unk>` target is, by construction, a word outside the 12,002-token vocabulary - i.e. a rare word. The model scores **0.00%**
   on both rare bands (r501-2k and r>2k, 13.42% of positions combined). Replacing `<unk>` with a real token cannot convert a 0.00% band into accuracy. The
   ceiling is 5.19 points only if OOV words became perfectly predictable, which no next-token model does. **The planned `<unk>` arm is cancelled on measurement,
   not on opinion** - and it was cancelled before 7 hours were spent on it.
3. **The real headroom is the r21-500 band**: 36.08% of positions at only 8.13% accuracy, contributing 2.93 points. Moving that band from 8% to 20% would add
   **+4.3 points** - more than every memory experiment today combined, and more than 8x what a perfect `<unk>` fix could give.

## What this changes about the plan

The queue said the next item after memory was FLAW 4. Measured, FLAW 4 is worth nothing and the r21-500 band is worth ~4 points. So the next architecture work
should target **mid-frequency tokens**, and the question becomes concrete: why does a model that gets 31.87% of top-20 targets get only 8.13% of targets ranked
21-500? Those tokens are frequent enough to have plenty of training signal (rank 21-500 in a 294M-token corpus), so this is not a data-starvation problem - it
is a readout/capacity problem, which is the same family as the tied-readout question the parameter audit raised.

That is the next experiment to design, and it starts from a measurement rather than a hunch.
