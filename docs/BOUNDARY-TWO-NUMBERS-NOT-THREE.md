

---

# THE BOUNDARY IS DRAWN: TWO NUMBERS, NOT THREE (2026-11-22 13:52)

With the carry target fixed, three-addend addition **fails**. This is now a valid verdict, not a budget artefact.

## Measured, 12,000 steps, corrected target, three seeds

```
seed   final training token acc   TRAIN exact   HELD-OUT exact
 0            61.6%                  0.0%            0.0%        (never converged - no verdict)
 1            59.8%                  0.0%            0.0%        (never converged - no verdict)
 2            94.2%                 13.5%           13.0%        (VALID)
```

**Only seed 2 counts, and it fails.** But what it fails at is worth more than the failure:

```
want (1,1,3,1,7,1)  got (1,1,4,0,7,1)     <- only the carry-2 column is wrong
want (0,2,2,2,2,1)  got (0,2,3,2,3,1)     <- the two columns with carry 2 are wrong
```

Seed 2 reaches **94.2% token accuracy** and only **13.0% exact**, which is the diagnostic: if its mistakes were spread evenly over six
tokens, 94.2% per token would give roughly 70% exact. It does not. **Its mistakes sit on exactly the columns where the carry reaches 2** -
the one thing a ternary column has that a binary column does not.

## What this settles

- **The written-out mechanism does NOT extend to three addends** at this size and budget. Two budget points (6,000 and 12,000 steps) and
  three seeds were tried.
- **The reason is specific:** a column whose carry can be **2** is not handled. The digits are read and written correctly; the *carry value
  itself* is the blocker.
- **The earlier "100.0% on triples" measured an easier target** (carry recorded as a yes/no flag), which is precisely why the bug mattered:
  the flag version of the task removes the only part that is hard.

## The boundary of what this project has actually built, in one table

| capability | measured result | valid? |
|---|---|---|
| read, select, compare | 100.0% (4 different tasks) | yes |
| add two numbers, 2/3/4/5/8 digits | 100.0% on 2,000 unseen pairs | yes |
| subtract two numbers with borrowing, 3/5 digits | 100.0% on 2,000 unseen pairs | yes |
| add three numbers | 13.0% exact | yes, and it fails |
| answer arithmetic without working out | 2-7% | yes |
| learn faster (10x goal) | not achieved, five levers eliminated | yes |

**The honest boundary sentence, which now has evidence behind every clause:** this architecture reads, selects, compares and performs
**binary** multi-digit arithmetic perfectly and readably, and does **not** yet extend to three addends, because a carry of 2 is not
represented.
