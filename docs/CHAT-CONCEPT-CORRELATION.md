# The chat concept labels are not redundant IN EFFECT either - and the one real correlation is a single ambiguous word

Date: 2026-11-14. Tool `tools/chat_concept_correlation.py` (instrument validated first: `--selftest` passes on synthetic labels -
identical +1.000, independent -0.003, opposite -1.000, 30%-flip +0.367). Output `lens/runs/CHAT-CONCEPT-CORRELATION.txt`.
Ran on **60,000 sampled training windows** of the prepared corpus (13,318,447 tokens, 86,359 conversations, vocab 12,000), one core
at nice 19 / ionice c3 because a game was running. The 200k default is available when the machine is free; 60k windows is ample for
phi estimates on concepts with supports from 0.0056 to 0.146.

## Why

`docs/history/CHAT-CONCEPT-OVERLAP.md` showed the chat concepts share almost no **vocabulary** (worst jaccard 0.08 across the 43
defined; the live set keeps 36). Disjoint words do not imply disjoint **events** - two concepts can fire together constantly with no
shared vocabulary. If a pair fires together almost always, no pair of neurons can separate them, and the labels are redundant in
effect even though they are disjoint in definition. This measured it.

## Result: no redundancy in effect

```
595 concept pairs
pairs above |phi| 0.50 : 0
pairs above |phi| 0.30 : 1
strongest pair        : is_an_agreement ~ mentions_law   +0.415
next                  : is_a_request ~ mentions_the_user +0.203
                        is_a_request ~ mentions_self     +0.143
commonest concept 'mentions_the_user': max |phi| with any other = 0.203
```

- **P1 (no pair above 0.50) HOLDS** - zero pairs.
- **P2 (at most two above 0.30) HOLDS** - exactly one, at 0.415.
- **P3 (commonest concept below 0.25 with all others) HOLDS** - 0.203. There is no general "something is happening" flag.
- Falsifier (any pair above 0.50 implies redundancy in effect) **not triggered**.

## The one real correlation is explained, and fixable

`is_an_agreement ~ mentions_law` at **+0.415** is a strange pairing semantically - agreement and law have nothing to do with each
other. The cause is already measured: those two concepts are the pair that **share a token** in the vocabulary check, the word
**"right"** ("you're right" versus "right to remain silent"). One ambiguous word in two lists manufactures the only meaningful
co-occurrence in the entire set.

**So the chat concept layer is not suffering from the image defect in either form** - not in vocabulary (max jaccard 0.08) and not
in effect (max phi 0.415, and that one is a word sense collision). The obvious tidy-up is to give "right" to only one of the two
concepts; after that the set is essentially orthogonal.

## What this closes, and what it does not

It closes the label-honesty question for the chat line: **redundant labels are not why chat concept neurons fail to mean their
names.** Any remaining dishonesty there must come from training pressure or the concept-to-rule wiring, not from the labels - which
is exactly the order the image line resolved it in (`alpha` sets honesty, `lam` sets rule structure, and the rules can even be
barred from concepts they should not read).

It does **not** show that the chat concept neurons are honest - that is the fidelity audit's job on a trained checkpoint, and the
last one reported 36/36 checks passing. What is now ruled out is a specific and plausible alternative explanation, with numbers.

## Follow-up the same day: the collision was removed, and removing it exposed something bigger

`"right"` sat in both `is_an_agreement` and `mentions_law`. Removing it from `mentions_law` (agreement-sense dominates in
conversation) was measured in three variants on the same 60,000 windows, each in its own process because the concept builder caches
a spec file and a single-process comparison gave contradictory numbers:

```
variant                                       concepts kept   mentions_law support   pairs above |phi| 0.30
A  "right"  (original, 7 tokens)                     35              0.0140                    1  (0.415)
B  no "right" (shipped, 6 tokens)                    35              0.0065                    0
C  "rights" instead of "right" (7 tokens)            35              0.0071                    0
```

- **The fix costs no concept** (35 kept in every variant) and **removes the only pair above 0.30**.
- **It halves `mentions_law`'s support**, from 0.0140 to 0.0065, against the 0.005 floor. A plural filler (`rights`) recovers only
  0.0071, so the missing support is not the plural form - **most of that concept's support WAS the ambiguous word in its agreement
  sense.**
- **That is the real finding: `mentions_law` was substantially measuring "agreement", not law**, which is exactly why it correlated
  at 0.415 with `is_an_agreement`. With the shared word gone it barely clears the support floor, and since C's filler token does not
  restore it honestly, **the shipped definition is B (no filler)**: if a run on the full corpus drops `mentions_law` for having
  support under 0.005, that is the correct outcome - a concept whose support depended on a word sense collision should not be kept
  by inventing a token to prop it up.

Also corrected in this follow-up: an earlier note in this document implied the concept count fell from 36 to 35 because of the fix.
It did not. The 36 figure comes from training runs on the **294M-token prepared corpus**, while every measurement here uses the
**13.3M-token cached corpus** that `chat_data.build()` loads by default, which keeps 35. Corpus difference, not causation.
