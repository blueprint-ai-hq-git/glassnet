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
