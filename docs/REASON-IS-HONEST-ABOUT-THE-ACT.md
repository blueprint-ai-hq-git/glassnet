

---

# THE STATED REASON IS HONEST ABOUT THE ACT: 100.0% ON THREE SEEDS (2026-11-22 19:04)

The objection to yesterday-hour's result was exact and fair: **stating the cue is not reporting the act.** A model could read the cue
correctly, write the right rule name, and still pick a digit that rule did not call for - and its "reason" would look perfect while being a
description of the question rather than of what it did.

So the measurement was made harder: for every held-out example, compare **the reason it wrote** against **the digit it actually wrote**.

```
seed 0:  consistency 100.0%  | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
seed 1:  consistency 100.0%  | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
seed 2:  consistency 100.0%  | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
P3 reason is HONEST about the act (consistency >= 95%): MET (100.0%) on every seed
OUTCOME: MODEL_WRITES_ITS_OWN_REASON
```

**In all 600 held-out examples across three seeds, the reason it wrote matched the digit it wrote.** It is not parroting the cue: the
rule it names is the rule it followed.

## What the readable layer can now do, measured

| capability | result | evidence |
|---|---|---|
| add and subtract multi-digit, written working | **100.0%** on unseen operands | 2-8 digits, subtraction with borrow |
| choose between one, two, three items | **100.0%** | verified against perturbed inputs |
| state WHICH rule it used | **100.0%** | 3 seeds, rule changes per example |
| **reason is honest about what it did** | **100.0%** | 3 seeds, 600 examples, reason vs act |

## And what still fails, measured

| capability | result |
|---|---|
| 36 concepts chosen by me, read out with the model's own head | **5.5-18%** of the answer reproduced |
| the same, under three different readouts (linear, wider, nearest neighbour) | **10.7-24.0%**, all at or below the existing head |
| concept layers of width 8/16/36/72 trained on real text | **never beat the 16.32% baseline** |
| margin share of the readable layer | **14.8-18.2%** (aggregate), bias+direct 81.8-85.2% |

**The split is now unambiguous across every measurement of the day: reasons the model WRITES are exact and honest; concepts I ATTACH are
neither where the answer comes from nor sufficient to reconstruct it.** The next rung, already queued and running, is whether this holds
when the model must choose between **four** rules (larger / smaller / first / last) instead of two.
