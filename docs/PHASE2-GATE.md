# PHASE 2 GATE: every item the brief asked for, measured (2026-11-11)

The brief's Phase 2 gate asks for four numbers plus an ablation. All five are below, from one run of the three-layer model
(36 named concepts -> 96 sparse rules, 3 wires each pruned to 2 -> answer), trained 3,000 steps on the matched split with
the concept layer fed both the window mean and the last token.

```
1. accuracy                         19.68% held-out next-token
   trivial baseline                  5.42%  (always output the most common token)
   difference                       +14.26 points
2. average connections per rule      3 before pruning, 2 after
3. rules a human must read           96, each printed as "rule #k fires when concepts A and B are on"
4. concept honesty                   79.2% mean, weakest 57.6%
5. ABLATION, rule layer removed      20.43%  ->  readability costs 0.75 points
```

**Predictions, both scored honestly:** "beats the trivial baseline by at least 1 point" PASSED at +14.26. "concept honesty
85% or better" **FAILED at 79.2%** - the concepts are still not honest enough to name in a sentence without a caveat.

## What the four input choices did, all on the same recipe
```
concept input        accuracy   concept honesty
window mean            9.20%        86.5%
last token only       15.38%        70.7%
mean + last token     19.68%        79.2%
```
Order is worth **+6.18 points** over the window mean. Adding both inputs is worth **+4.30 more** over last token alone.
Honesty moves the opposite way each time the input gains position information, which is a real tension and not a bug:
"this window contains 'the'" is a property of the whole window, while a next-token rule needs the last position.

## The honest price of readability
Removing the 96-rule layer entirely scores **20.43%** against **19.68%**: the middle layer costs **0.75 points**. That is
cheap for the ability to read every decision as two concepts and one rule, and it is measured rather than asserted. The
earlier mean-pool prototype looked like it cost 1.27 points, so the price also depends on the input, and at the best input
it is smaller.

## What this gate does NOT establish
- This is a **2,696,644 parameter** model on one harness, not the flagship. It is a mechanism test of the middle layer,
  and its absolute accuracy is not comparable to the 25.52% assistant-by-position number from the bigger models.
- Concept honesty **fails** the 85% bar. Every concept is still "window contains X", which is checkable but weak; naming
  the weakest one at 57.6% honesty is the honest way to report it.
- The rule layer's verbs are still shallow: a rule fires on two concepts, so it can say "contains the AND contains of",
  not "the word after a question mark". Richer concepts are what would make the rules worth reading.
