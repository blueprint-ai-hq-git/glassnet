# 2026-11-26 - the brain method, 3 seeds at 8,000 vocabulary: it COSTS loss and accuracy (my earlier positive result did not replicate)

Source: lens/runs/BRAIN-METHOD.json (written 15:09:45), journal of unit glassnet-brain3. Same architecture (named concept layer +
direct route, tied embeddings, 12 named concepts), 8,000-token vocabulary, 12,000 fresh-equivalent examples, THREE seeds per arm.

    arm     held-out loss        accuracy        distinct predicted tokens
    plain   5.4407 +/- 0.0149    14.22 +/- 0.23%     23
    brain   6.9035 +/- 0.0243    11.53 +/- 0.49%    153

    P1 lower loss by >=0.05 nats : NOT MET   (-1.4629 nats)
    P2 +1.5 points accuracy      : NOT MET   (-2.68 points)
    P3 wider variety of outputs  : MET       (153 vs 23)

## The honest reading

**At this scale the brain ideas make the model worse, not better.** It pays 1.46 nats and 2.68 accuracy points for six times the
output variety. The standard deviations are tiny (0.015 and 0.024 nats), so this is not seed noise - it is a real cost.

The earlier single-seed result at a 2,000-token vocabulary (+0.0837 nats, 22 vs 7 distinct) **did not replicate** at 8,000. A
0.08-nat gain from one seed, when the same method loses 1.46 nats at four times the vocabulary, was noise or a small-vocabulary
effect. I reported it as "the brain ideas pay". That claim is withdrawn.

## Why it plausibly costs

Three mechanisms are sharing capacity that next-token prediction needs: a concept loss at 0.3 on 12 targets, a replay stream that
revisits old windows, and a stratified draw. With more output classes (8,000 against 2,000) the prediction problem is harder, and
the auxiliary objectives take gradient that the harder problem needed. More variety in predictions is exactly what you would
expect from a model being pulled toward its concept targets: it spreads its guesses instead of sharpening them.

## Eighth verdict bug, caught on this very result

The script printed "FALSIFIER FIRED: ... the brain ideas are NEUTRAL" on a 1.46-nat DEFICIT, because the neutral branch tested
`dn < 0.02 and da < 0.005` and a large negative difference satisfies `< 0.02`. Magnitude tests now use abs(), and a negative
difference prints THE BRAIN IDEAS COST with both numbers in the sentence.

## What this changes

The brain-ideas-from-step-one line is now a measured failure at this vocabulary and budget, not a promising start. Before it is
retried it needs a reason to expect a different outcome: for instance the concept loss weighted down, replay kept but coverage
stratification dropped, or the vocabulary held at 2,000 where it did not hurt. Running the same configuration again would only
buy the same answer.
