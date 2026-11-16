
## The faster learner does not write better: next-token supervision gains +1.54 accuracy and LOSES generation quality, so it is not promotable (2026-11-16)

Predictions `lens/runs/NT-DECODE-PREDICTIONS.txt` written before the run. Numbers from `lens/runs/NT-DECODE.txt`, 12 prompts, 40 generated tokens,
checkpoint `nodirect1024_nexttarget_data4m_e12_s0_best.npz`.

```
decoder            repetition   coverage   KL
next-target greedy    0.075       0.190     18.44
next-target T=0.7     0.002       0.173     20.19
shipped    greedy     0.275       0.343     17.21
shipped    T=0.7      0.027       0.292     17.73
```

## The falsifier fired

I wrote, before running: *"if coverage drops below 0.20 or repetition exceeds 0.10 at T=0.7, the +1.54 accuracy gain came WITH a generation
regression, and the honest headline becomes: more accurate, worse text - which would mean the new config is NOT promotable and the shipped model
stays."*

Coverage at T=0.7 is **0.173**, under the 0.20 bar. So: **not promotable.** The prediction that failed is the one I most wanted to pass.

## What the accuracy gain actually bought

- **coverage 0.292 -> 0.173**, a 41% loss of vocabulary breadth
- **KL 17.73 -> 20.19**, a 2.46 increase in divergence from the target distribution
- **accuracy 24.58% -> 26.12%**, +1.54 paired on seed
- repetition improves dramatically, 0.275 -> 0.075 greedy and 0.027 -> 0.002 sampled

So sharper concept labels make the model **more accurate, less repetitive, and less varied** - it learns to say the likelier thing more often and the
range of what it says narrows. That is a coherent trade, not a defect, and it is the opposite of what the accuracy number alone suggested.

## Why this is the most useful negative of the day

It is the project's accuracy-versus-generation dissociation firing **in the direction that would have flattered me**: the arm that won on accuracy lost
on generation. Had I promoted the config on the accuracy number - which R1 of the decision rule explicitly permitted, and which would have been easy to
stop there - the shipped model would have become the narrower, worse-text model with a better benchmark score.

Two things saved it: the standing rule that VAL accuracy alone cannot promote a configuration, and a falsifier with a number in it, written before the
run. Neither is clever; both had to be there.

## What ships, and what stays open

**Shipped configuration is unchanged: 4M / 12 epochs, d1024, 53 concepts, lam_concept 1.0, next-token supervision OFF, decoding at T=0.7**, carrying
VAL 24.58% +/- 0.055, coverage 0.292, repetition 0.027, KL 17.73.

The +1.54 accuracy result stands as a measured fact and stays published (`docs/NEXT-TARGET-3SEED.md`); what does not follow from it is promotion. If a
future arm buys accuracy **without** losing coverage, that is a different and promotable result - and now there is a baseline curve to judge it against.
