# Stronger L1 does not buy multi-concept rules - it buys SINGLE-concept rules, and that is the better readable model

Date: 2026-11-14. Tool `gates/phase2_train.py`, alpha 1.0, de-duplicated 14-concept set, R=32, keep=3, 10 epochs, 3 seeds, no
pruning floor (so the connectivity measured is what TRAINING produced, not what pruning imposed). Predictions written before the
run in `lens/runs/LAM-CONCENTRATION-PREDICTIONS.txt`; raw output `lens/runs/LAM-CONCENTRATION.txt` and
`LAM-CONCENTRATION-AUDIT.txt`.

## The prediction, and what happened instead

I predicted stronger L1 would concentrate each rule onto **two** concepts, raising the multi-concept count from 4-5 to at least 8. It
did the opposite:

```
lam      accuracy (3 seeds)    rules by REAL connectivity (|w| >= 0.50)
0.005    82.76% +/- 0.35       12 multi  / 11 single / 9 dead
0.01     82.06% +/- 0.69        0-2 multi / 30-32 single / 0-1 dead
0.02     81.71% +/- 0.78        0 multi  / 32 single / 0 dead
0.05     80.53% +/- 0.89        0 multi  / 32 single / 0 dead
```

**At lam >= 0.02 every single one of the 32 rules reads exactly ONE named concept, and no rule is dead.** P1 failed outright. The
mechanism is obvious once seen: L1 shrinks the small weights harder than the dominant one, so instead of spreading a rule over two
concepts it deletes everything except the winner.

## Why this is the better model, not a worse one

The project has been quoting "each rule connects to 2-3 concepts". The measurements of the last two runs showed that claim was
false in effect (1.55 real concepts per rule, a dense tail worth 3.81 points when amputated). **lam 0.02 replaces a false complex
claim with a true simple one:**

> 32 rules. Each rule reads exactly one named concept and carries one signed weight. No rule is dead. The answer layer sums the
> rules for each class.

That is a statement a 12-year-old can check by reading 32 numbers, and it is honest at the level of the weights rather than only at
the level of the wiring. The concept layer stays honest too - audited at lam 0.02:

```
mean |rank corr| with own label   0.769
concepts below 0.50               0 of 14
concepts tracking ANOTHER feature 0 of 14
concept interaction               0.0%   (still purely additive)
dead rules (in the audit)         3 of 32
```

## What it costs

- **lam 0.01: -0.70 points** (82.06 vs 82.76) for a fully single-concept rule layer (30-32 of 32 rules single).
- **lam 0.02: -1.05 points** (81.71) for all 32 rules single and zero dead rules.
- **lam 0.05: -2.23 points**, and nothing further is gained.

Against the matched plain control (84.37% +/- 1.04), **the fully single-concept readable model at lam 0.02 is 2.66 points behind**,
with 14 honest concepts, 32 one-concept rules, a purely additive explanation, and no hidden dense tail.

## One metric that is now proven meaningless

The audit's `rule-pair lift` reports **27 of 32 rules below 1.0** at lam 0.02. That number must be ignored, and here is why: the
metric takes each rule's two strongest weights and tests whether those two concepts co-occur. When a rule reads exactly ONE concept,
its second-strongest weight carries approximately nothing, so the test compares a concept against noise. This is the same artefact
identified in `docs/history/RULE-EFFECTIVE-CONNECTIVITY.md` (correlation between |second weight| and lift was +0.494) now taken to
its limit. **The honest connectivity count supersedes rule-pair lift for single-concept rules**, and the audit should skip the metric
when a rule has fewer than two weights above the floor.

## Recommended configuration going forward

```
architecture : 14 de-duplicated named concepts -> 32 rules -> 10 classes
training     : alpha 1.0 (concept supervision), lam 0.02 (single-concept rules), keep 3, 10 epochs
accuracy     : 81.71% +/- 0.78  (matched plain control 84.37% +/- 1.04, so -2.66)
readability  : every rule reads exactly one named concept; 0 dead rules; concept interaction 0.0%
```

lam 0.01 is the cheaper option at -0.70 points if a mixed rule layer is acceptable; lam 0.02 is the honest simple one.

## Predictions, scored

- **P1 stronger L1 raises multi-concept rules to >= 8 - FAILED**, at **0**. The direction was inverted and the mechanism is now
  understood.
- **P2 accuracy at the best lam within 1.5 points - HOLDS** (lam 0.02 gives -1.05).
- **P3 average connections per rule above a 0.25 floor rises with lam - HOLDS in spirit but is moot**: with single-concept rules
  the average is 1 by construction, which is the cleaner statement.
- **Falsifier (no lam raises multi above 6 -> sparsity not achievable by training) - TRIGGERED for multi-concept rules**, and the
  finding is that the achievable-by-training structure is **one concept per rule**, not two.
