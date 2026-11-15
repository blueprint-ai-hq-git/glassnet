# Naming the collapse: which concepts carry it, and a redundancy I introduced myself (2026-11-15)

`docs/history/ACTIVATION-COLLAPSE.md` established that the readable model's 53 concept numbers move through about **10.4 effective
directions**, which is why vocabulary breadth is capped. This document names the concepts involved, using
`tools/chat_concept_entanglement.py` on 4,000 held-out windows of the 53-concept checkpoint.

## The dominant direction is conversational register, not topic

```
direction 1: 26.6% of variance | mentions_self 0.50, mentions_the_user 0.31, cp_thinking 0.26, is_a_question 0.24, mentions_a_number 0.20
direction 2: 10.1% of variance | mentions_a_number 0.40, is_a_question 0.36, mentions_self 0.27, cp_help_ask 0.25, cp_greeting 0.24
direction 3:  6.7% of variance | mentions_self 0.33, mentions_animals 0.30, mentions_a_group 0.29, mentions_travel 0.22, mentions_law 0.21
effective dimensionality: 10.03 of 53 (19%)
```

**A quarter of all concept variance is one axis: who is speaking and whether it is a question** (`mentions_self`, `mentions_the_user`,
`is_a_question`). Almost every other concept, and in particular every topic concept, moves very little by comparison. That is the
mechanism of the collapse in plain terms: the concept vector is mostly a **register detector**, so changing which words a concept set
contains does not change how many independent things the layer actually tracks.

## Most correlated pairs

```
cp_certainty        cp_emotion_pos       +0.671
mentions_a_number   cp_certainty         -0.643
mentions_a_number   cp_emotion_pos       -0.635
mentions_self       mentions_feeling     +0.617
mentions_self       cp_thinking          +0.616
mentions_the_user   cp_thinking          +0.615
mentions_the_user   mentions_feeling     +0.570
mentions_a_number   cp_help_ask          -0.570
pairs with |r| > 0.50: 20 of 1,378
```

## The correction: my own additions are partly redundant

`cp_certainty` (r = **+0.671** with `cp_emotion_pos`) and `cp_thinking` (r = **+0.61** with both `mentions_self` and
`mentions_the_user`) are concepts **I added in the widening arm**. The design rule I applied was that every added token must be
disjoint from all 283 tokens already in use - and both concepts pass that rule. **Token disjointness is not the same as
non-redundancy**: `sure/certain/probably` and `love/like/enjoy` co-occur in hedged, emotional chat, and `think/know/understand`
co-occurs with first- and second-person talk, so the neurons learned overlapping signals from disjoint words.

That is a real limitation of the widening that produced the +0.57 gain (`docs/history/WIDENING-3SEED.md`): **the accuracy win is
solid and seed-robust, but part of it was bought with redundant concepts**, and the shared-token self-check that I reported as a
clean pass was measuring the wrong property. The check that would have caught this is the one used here - activation correlation -
and it should have run before the arm launched.

## What this sets up

The two candidate fixes now separate cleanly:

1. **Supervision strength** (running): if a stronger concept loss spreads the activations, the collapse is a training-pressure problem.
2. **Decorrelation** (next, if (1) fails by its falsifier): penalise correlated activations directly, which targets the measured cause
   instead of hoping a weight change reaches it. This document is what makes that arm pre-registerable - the pairs to watch are the
   eight above, and the success criterion is effective dimensionality rising above 10.4 with the dominant register axis falling below
   its current 26.6% share.
