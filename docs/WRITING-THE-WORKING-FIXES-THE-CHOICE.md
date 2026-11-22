

---

# WRITING THE WORKING OUT FIXES THE CHOICE: 88-94% -> 100% ON THE FLIP CASE (2026-11-22 14:54)

This is the result the whole day was pointing at. Two hours ago the model could point at **one** thing perfectly and managed only
**88-94%** when it had to choose between **two**. More parameters (3.3x) changed nothing, which ruled out capacity and left how the
choice is **written down**.

## The one change

Instead of answering the comparison in one step, the model must **write the comparison out first**:

```
  [pos of digit 1][digit 1] [pos of digit 2][digit 2] [pos of the LARGER][the larger]
   \____________________ the working, graded ____________________/  \___ the answer ___/
```

Six tokens where two used to be. Same parameters (67,844), same steps (6,000), same seeds, same 95% bar set before any of this ran.

## Measured

```
                        clean sample (200 held-out)      flip case: stated position
before (2 tokens)       pos/val/both 100%                88.0%   94.0%     <- FAILED the 95% bar
after  (6 tokens)       pos/val/both 100%               100.0%  100.0%     <- MET
```

- **seed 0:** moved 100.0%, swapped value 100.0%, swapped position 100.0% -> "it really follows the input"
- **seed 1:** moved 99.0%, swapped value 100.0%, swapped position 100.0% -> "it really follows the input"
- **P1, P2, P3 all MET. Falsifier not fired. `OUTCOME: EXPLAINS_BY_EMITTING`.**

## Why this is more than a bug fix

It is the same law, found twice in two different places:

- the model could not **add** until the carry had somewhere to be **written** - 2% to 100%;
- the model could not **justify a choice** until the comparison had somewhere to be **written** - 88-94% to 100%.

**Computation and explanation both improve the same way: by making the intermediate state a visible token instead of a hidden
vector.** That is the entire thesis of this project, and it now has two independent pieces of evidence instead of one - and in both
cases the fix cost **zero extra parameters**, only tokens.

The boundary from two hours ago is therefore **withdrawn and replaced**:

- old: "reliable reason for ONE item, 88-94% for a choice between TWO"
- new: **"a readable reason, written by the model itself, at 100% - for a single item and for a choice between two."**
