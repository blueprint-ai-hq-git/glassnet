

---

# A BIGGER BUDGET REVERSES THE VERDICT, AND THE CANCELLATION GETS WORSE, NOT BETTER (2026-11-20 12:49)

Two corrections land together. Both are here rather than edited into the entries above.

## 1. "The bottleneck costs 0.16 points" was a budget artefact

```
budget            --direct ON            --direct OFF            gap
20k windows    acc 4.11%  (13/36)     acc 3.95%  (14/36)      -0.16   <- P2 "passed"
100k windows   acc 12.18% (25/36)     acc 7.31%  (25/36)      -4.87   <- P2 FAILS
```

P2's bar was 2.0 points. At 20,000 windows the shortcut looked nearly free; at 100,000 it costs **4.87 points**, which is more than
double the bar. The 20k figure was not wrong, it was **too small a budget to show the cost** - and I reported it as evidence that
readability was nearly free. Corrected here: on this architecture, **removing the direct path costs real accuracy** once there is
enough data to train on. Both arms do gain concept honesty with budget (13-14/36 -> 25/36), so the structure is not lost - it is paid for.

## 2. Cancellation gets WORSE with budget, and explodes in the bottlenecked arm

Exact attribution, runner-up held fixed, 96 usable windows per checkpoint:

```
                                    20k                  100k
--direct ON    positive supply      16.43x               8.476x
               positive concepts    17 of 36             20 of 36
--direct OFF   positive supply      9.45x                236.032x
               positive concepts    17 of 36             19 of 36
               worst single concept                    mentions_technology +3579.2%
                                                          mentions_sport +2599.5%
```

The direct-path arm **halves** its cancellation as it trains (16.43x -> 8.48x) while gaining accuracy (4.11% -> 12.18%). The
bottlenecked arm does the opposite: its concepts end up pushing **236x** the net margin against each other while its accuracy falls
to 7.31%. Forcing the answer through 36 named concepts does not make the layer summarise the decision - it makes the concepts
**fight harder** as training proceeds.

That is the strongest evidence yet against "a concept bottleneck makes the model readable", and the strongest evidence FOR the
direct path: the trained model uses it to stop the concept layer cancelling itself to death.

## 3. Provenance damage, mine, recorded

`run_custom_arch_direct.sh` tagged its checkpoints by ARM NAME only (`arch_direct_on`), so the 100k run **overwrote** the 20k
checkpoints at 12:44 and 12:49. The attribution numbers published for the 20k budget (16.43x / 9.45x, `is_a_request +223.8%`,
`mentions_technology -262.0%`) can no longer be re-derived from weights - only from the JSON records
`lens/runs/CONCEPT-EXACT-ON.json` and `CONCEPT-EXACT-OFF.json` written at 12:41. Nothing was deleted; the newer, larger-budget
checkpoints are the ones that survive.

Fix: the tag now carries the budget (`arch_$1_m${MAXW}w`), because an arm name is not a run identity. This is the third clobber of
the same family in this project, and the rule it violates was already written down: a checkpoint filename must carry its run tag.
