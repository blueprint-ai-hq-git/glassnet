# 2026-11-26 - ablation: REPLAY alone adds 0.93 accuracy points, COVERAGE alone hurts, and no single mechanism explains the 1.46-nat loss

Source: lens/runs/BRAIN-ABLATION.json, journal of unit glassnet-ablate4. 8,000 vocabulary, 12,000 fresh-equivalent examples,
THREE seeds per arm, same architecture (named concept layer + direct route, tied embeddings), one mechanism on at a time.

    arm        held-out loss        accuracy         distinct predicted tokens
    plain      5.4407 +/- 0.0148    14.17 +/- 0.21%     23
    concept    5.4470 +/- 0.0136    14.43 +/- 0.45%     22      (+0.0063 nats)
    replay     5.4224 +/- 0.0324    15.10 +/- 0.25%     67      (-0.0182 nats)
    coverage   5.5511 +/- 0.0096    13.83 +/- 0.19%     37      (+0.1105 nats)

## My prediction was wrong

I predicted the CONCEPT loss was the expensive mechanism. It is not: it costs 0.0063 nats, which is nothing. The largest single
cost is COVERAGE at +0.1105 nats - stratified draws actively hurt. The falsifier condition ("no single mechanism costs more than
0.05") was almost met: two of three are inside that band.

## The one real win is replay, and it is on accuracy not loss

Replay alone: loss essentially unchanged (-0.0182 nats), but **accuracy 15.10% against 14.17% - a gain of 0.93 points with a
seed spread of 0.25**, roughly three and a half standard deviations, so it is not noise. And it predicts **67 distinct tokens
against 23**, so it is also three times less collapsed.

That is the first time in this project that a mechanism has moved ACCURACY, rather than loss or calibration, at a matched budget.

## And the 1.46-nat loss of the full method is an INTERACTION, not a part

Coverage alone explains 0.11 of the 1.46 - about seven per cent. Concept and replay explain nothing between them. So the three
mechanisms together are far worse than the sum of their parts, which means the combination must be tested, not the parts blamed.

## What this changes

* KEEP replay: +0.93 accuracy points, three seeds, tiny spread.
* DROP coverage as implemented: it costs 0.11 nats and 0.34 accuracy points on its own.
* Concept loss: neutral, keep it only if it buys readability, which is a separate question.
* Next: test the two-way combinations to find which pair is toxic, because the sum of the parts is not the whole.
