

---

# FOUR RULES, NOT TWO: STILL 100.0% AND STILL HONEST (2026-11-22 19:21)

The two-rule result could have been a special case: with only "larger" or "smaller", a model might be reading one bit off the cue. So the
cue now selects one of **four** rules - *take the larger*, *take the smaller*, *take the first*, *take the last* - and the model still has
to write `[which rule it used] [position of the digit that rule picks] [that digit]`.

## Measured, three seeds, 200 held-out examples each

```
seed 0:  consistency 100.0% | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
seed 1:  consistency 100.0% | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
seed 2:  consistency 100.0% | reason 100.0% | position 100.0% | value 100.0% | all three 100.0%
P1 states WHICH RULE it used: MET   P3 reason is honest about the act: MET   OUTCOME: MODEL_WRITES_ITS_OWN_REASON
```

**The reason is still honest against the act in all 600 examples**, so it is not reading the cue and reciting it - it reports the rule it
actually followed, out of four.

## Why this matters for the project's main bet

The readable middle layer has now been measured on both sides, repeatedly, today:

| what is readable | result |
|---|---|
| reasons the model **writes** (carry chains, item choices, rule choice, rule honesty) | **100.0%** on every test, every seed |
| concepts **I attach** and read out of the model | **5.5-24.0%** of the answer reproduced; never beats the model's own head |

Two rules could have been a coincidence. **Four rules at 100% with verified honesty is a capability**, and it is the one the project was
actually chartered to build: a model whose reason for an answer is visible, exact, and checkable by a person - **because the model writes
it**.

## Still queued behind this

`cuewide` (the same four-rule task with 16 input positions instead of 12, to test whether more distractors cost the reason) and
`ablate500` (the readable-route ablation at 500 windows, to re-check the 5.5-18.0% finding at a larger sample).
