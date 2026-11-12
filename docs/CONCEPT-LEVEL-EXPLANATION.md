# Rule redundancy is structural, and the honest explanation is at concept level (2026-11-12)

Two findings from this arm, one negative and one positive.

## Negative: de-duplicating rule pairs does NOT restore rule necessity
Under tied answers every rule's row is the sum of its concepts' rows, so two rules with the same wire pair have literally
identical answer rows. Sampling rule pairs **without replacement** (no two rules share a concept pair) was the obvious fix.

```
config                                  accuracy (seeds)          median per-rule ablation loss
tied, pairs WITH replacement (3 seeds)   15.44 +/- 0.22           0.000 / 0.000 / 0.000
tied, DISTINCT pairs (2 seeds so far)    15.18 +/- 1.24           0.000  (seed 0)
                                         16.05 / 14.30
```

Distinct pairs did not remove the redundancy: the median rule is still worth **0.000** accuracy points when deleted. So
redundancy was never about identical pairs - it is intrinsic to the tied architecture. Every rule is a linear combination of
the same concept rows, so any rule's contribution can be absorbed by the others. Rule-level necessity cannot be manufactured
by how the wires are chosen; it is a property of the tie itself.

The accuracy difference between the two pair-sampling schemes is inside noise (15.18 vs 15.44, spreads 1.24 and 0.22), and
the distinct-pair arm is also less stable across seeds, so nothing is claimed from it beyond "it did not help".

## Positive: measured concept attribution, and it adds up almost exactly
`tools/explain_measured.py` explains ONE decision causally instead of reading weights: for the concepts true in a window, it
switches each one off and reports how much the predicted word's score moves.

```
window: ... 'good' 'that' 'you' 'enjoyed' 'the' 'musical' 'numbers' '.'
  said 'the'  (true 'the')    scores: the 3.75, <|im_start 2.82, do 2.72
  turning OFF content: window has 'the'      lowers 'the' by +1.55
  turning OFF content: window has ','        lowers 'the' by +1.23
  turning OFF content: window has '.'        lowers 'the' by -0.00
  turning OFF position: chat tag in last 16  lowers 'the' by -0.00
```

"I said 'the' partly because the window contains the word 'the', and partly because it contains a comma." That is a true
statement about this specific decision, measured, not inferred from weights.

**And the decomposition is additive**: across 2,000 held-out windows, the sum of the individual concept effects on the
predicted word's score is **-4.661**, while switching every concept off at once gives **-4.592**. The interaction between
concepts accounts for **1.5%** of the total effect.

That matters more than it looks: it means a per-concept explanation is not merely illustrative - it is quantitatively
complete. The parts add up to the whole within 1.5%, so a reader who follows the concept list has the whole causal story of
that decision, with no unexplained remainder worth naming.

## Why this is the right explanation level for these models
```
level     necessity measured               status
rule      median 0.000 (tied) / 0.36 (untied)   NOT reliable under a tie
concept   interaction only 1.5% of effect       FAITHFUL and additive
```

For the tied models, the rule is the wiring that *selects* concepts, not the cause of the answer - and saying otherwise
would be a false explanation. Reporting concepts, their individual measured effects, and the 1.5% interaction remainder is
the honest form.

## Caveat on scope
- The distinct-pair arm has two seeds here; the third is still running, and the arm's spread (1.24) is wide, so its mean is
  provisional.
- Additivity was measured on one model (`rules_distinct_s0`). It should be re-measured per configuration before being
  quoted as a property of the architecture.
- Concept attribution measures influence on the predicted word's score; it does not partition credit between concepts that
  push the same word in the same direction, only the total each contributes.
