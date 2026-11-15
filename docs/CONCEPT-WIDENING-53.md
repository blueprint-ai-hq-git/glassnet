# Widening the concept set to 53: best readable accuracy yet, but vocabulary barely moved (2026-11-15)

Predictions were written before the run in `lens/runs/WIDENING-ARM-PREDICTIONS.txt`. This arm changes ONE thing: the concept table
grows from 36 kept to 53 kept (17 measured additions, every added token disjoint from the 283 tokens the existing set already used,
and from each other). No existing concept definition was altered. 18 epochs because the 27-epoch repeat showed this recipe peaks at
15-19 and decays after it.

## Result

```
endpoint                     36 concepts    53 concepts    change
final VAL accuracy              21.48%        22.04%       +0.56
held-out best                   23.67%        24.22%       +0.55
final VAL perplexity            253.02        240.55       better
concept path carries answer      83.4%         86.9%       +3.5
median concepts firing           13            19          +6
distinct tokens                    78            86          +8 (+10%)
concepts passing the audit     36/36         52/53        -
```

**Every accuracy endpoint improved, and the best checkpoint again came at the LAST epoch (24.22% at epoch 18, still improving).** The
concept path also carries more of the decision (83.4% -> 86.9%).

## Predictions, scored honestly

- **W1 FAILED**: predicted more than 110 distinct tokens; got **86** (+10%, not the +41% predicted).
- **W2 FAILED, and in the wrong direction**: predicted median concepts firing would fall below 13; it rose to **19**, which is what
  more concepts should do to a firing count. The prediction was written without thinking that through.
- **W3 FAILED**: predicted held-out concept honesty above 0.60; got **0.5884** (0.5676 before, so +0.021 in the right direction).
  11 of 22 concepts remain below 0.60 on synonyms they were never trained on.
- **W4 PASSED**: final VAL 22.04% against the 20.98% floor.
- **FALSIFIER did not fire**: it required diversity below 90 AND honesty below 0.58; diversity **86** (below) but honesty **0.5884**
  (above). It missed by 0.0084, so I am reporting this as a near miss rather than a clean survival.

## The two things this establishes

1. **Named concepts have sharply diminishing returns for vocabulary.** +47% concepts bought **+10%** vocabulary (+8 tokens). Extrapolating
   that rate, reaching the direct-path control's 226 tokens would need a concept count in the hundreds, which is no longer a set a
   human can read. So "more named concepts" is not the route to breadth, even though it is a reliable route to **accuracy**.
2. **The widening is worth keeping for accuracy and explanation quality**: +0.56 VAL, +0.55 held-out, better perplexity, and a decision
   that flows through named concepts 86.9% of the time. But it is **one seed**, and this project has twice seen a width-style verdict
   reverse on seed 2, so a paired 3-seed check is running before any of that is treated as established
   (`lens/runs/WIDENING-3SEED-PREDICTIONS.txt`).

## Also found: four pre-existing token-sharing pairs

The no-shared-token self-check written for this arm found that the **existing** set has four concept pairs sharing a token:
`later` (is_a_farewell / mentions_time), `study` (mentions_education / mentions_science), `law` (mentions_law / mentions_politics),
`country` (mentions_place / mentions_politics). This is the same defect class as the `right` collision fixed earlier, and the phi audit
never caught it because these pairs do not co-fire above 0.30. They are inherited unchanged by this arm, so they cannot explain any
difference here, and they should be fixed in a separate arm rather than mid-comparison.
