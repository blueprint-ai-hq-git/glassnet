# 2026-11-25 - sparsity halves superposition but does not buy names

Source: lens/runs/EXCLUSIVITY-PRESSURE.json, journal of unit glassnet-excl5. Three arms, identical data and steps (24,000),
identical held-out set; only the pressure on the hidden layer differs.

    arm      held-out acc   honest names   mixed units
    none        91.0%        28 (10.9%)       117
    l1          88.2%        33 (12.9%)        63
    topk        76.2%         1 (0.4%)        129

## Predictions vs measured

    P1 hard top-k gives the most names      NOT MET  (top-k gave the FEWEST: 1)
    P2 some arm passes 20% honest naming    NOT MET  (best was 12.9%)
    P3 best-naming arm within 8 pts acc     MET      (l1 is 2.8 points behind)

The pre-registered falsifier fired: no pressure arm reaches 15% honest naming.

## What is real here

**L1 sparsity halves superposition** - mixed units fall from 117 to 63 - at a cost of 2.8 points of accuracy. That is a genuine
structural change: units really do respond to fewer concepts at once.

But the freed capacity does NOT become names. Honest naming moves 28 -> 33 units. So a unit can stop being a mixture and still
not be a single concept: it becomes a weak or partial detector instead. Naming needs a concept that matches, not merely absence
of competition.

Hard top-k is worse on both axes and shows why: forcing 8 of 256 units to carry everything loses 15 points of accuracy and
collapses naming to a single unit, because the surviving units must each cover many things.

## Consequence for the transparency claim

At this model size and task, roughly **11-13% of units can be honestly named**, and the rest are mixtures that the cards now
state plainly rather than mislabel. Sparsity pressure is not the lever that fixes this. Two options remain: make the concept
vocabulary richer still (measured: 24 concepts added no honest names over 6), or present explanations as COMBINATIONS of a few
named units rather than as single-neuron reasons. The second is the more honest direction, since it matches what the model
actually does.
