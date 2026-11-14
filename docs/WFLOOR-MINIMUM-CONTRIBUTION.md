# Enforcing a minimum contribution: the honest rule structure is ~1.5 concepts per rule, and forcing 2 costs 3.81 points

Date: 2026-11-14. Tool: `gates/phase2_train.py --wfloor` (new), `GlassNetPhase2.prune_rule_connections(keep, floor)` and
`.honest_connectivity(floor)` (new), alpha 1.0, the de-duplicated 14-concept set, R=32, keep=3, lam=0.005, 10 epochs, 3 seeds.
Predictions in `lens/runs/WFLOOR-PREDICTIONS.txt`, raw output `lens/runs/WFLOOR-RESULTS.txt`.

## What this run was for

`docs/history/RULE-EFFECTIVE-CONNECTIVITY.md` measured that every rule keeps 3 non-zero weights while at |w| > 0.50 the median rule
reads **one** concept, 27 of 32 rules reading one or fewer. So the advertised "each rule connects to 2-3 concepts" was true in span
and false in effect. The fix: a **minimum contribution floor** - after the top-`keep` cut, zero any weight with |w| < floor - so a
rule either genuinely reads two concepts or is honestly single-concept.

## Results

```
floor   accuracy (3 seeds)     pre-prune      avg conns/rule     rules reading >= 2 concepts (|w| >= 0.50)
none    82.76% +/- 0.35        82.49%            3.00 reported         12 / 11 / 9  (multi / single / dead)
0.25    82.22% +/- 0.35        82.49%            1.55                   5 / 24 / 3
0.50    78.95% +/- 0.88        82.49%            1.06                   5 / 24 / 3
```

**The real structure of the model is about 1.5 concepts per rule, not 3.** With a 0.25 minimum contribution the average kept
count is **1.55**, and only **4-5 of 32 rules genuinely read two or more concepts** - the rest are single-concept rules. That number
was previously hidden behind a reported "avg conns 3.00".

## And forcing two concepts is expensive

- A floor of 0.25 costs **-0.54 points** (82.76 -> 82.22), which is cheap and buys an honest connectivity number.
- A floor of 0.50 costs **-3.81 points** (82.76 -> 78.95) while the count of multi-concept rules stays at **5**.

**That triggers my own written falsifier**: "if the floor costs 2 or more points while reducing multi-concept rules by only a few,
then the near-zero weights are doing real work through many small contributions". Floor 0.50 costs 3.81 points and does not reduce
the genuine multi-concept count at all, so the weights in the 0.25-0.50 band are **collectively load-bearing**. The rules are
**sparse in span and dense in effect**: one concept carries the rule, but a tail of small weights across several concepts moves
accuracy by nearly 4 points.

## Predictions, scored

- **P1 floor 0.25 costs under 1.0 point - HOLDS** at **-0.54**.
- **P2 at most 20 of 32 rules keep two weights above the floor - HOLDS** (5), though I expected about 18; the true figure is far
  lower, which strengthens the core finding.
- **P3 at floor 0.50 accuracy stays above 80% - FAILED** at **78.95**.
- **Falsifier TRIGGERED** at floor 0.50, as described above.

## What the readable claim must now say

Not "each rule connects to 2-3 concepts". The defensible statement, measured on this checkpoint, is:

> 32 rules; average **1.55** concepts per rule above a 0.25 contribution floor; **4-5 rules read two or more concepts**; the rest
> read one. A rule's explanation is therefore usually a single concept with a signed weight, plus a small tail of minor
> contributions that are disclosed but immaterial individually.

**Recommended operating point: floor 0.25** - it costs half a point, and it makes the connectivity number honest. `explain()` should
report effective connectivity at that floor and should also state the tail, because the tail is worth ~3.8 points of accuracy and
silently dropping it would misreport the model's behaviour.

## What is still unknown

Whether the tail can be made to disappear with **training** rather than pruning: the L1 penalty (`lam 0.005`) is what allows a rule
to spread tiny weights across concepts. A larger `lam` might concentrate each rule on two concepts *during* training, which would
make the sparse claim true by construction instead of true-by-amputation. That is the next run, and the prediction should be that
a higher `lam` raises `multi` while keeping accuracy within 1.5 points - or the sparsity is cosmetic and this document says so.
