

---

# IT CAN STATE ITS OWN REASON: 100% ON ALL THREE SEEDS, AND IT PROVABLY FOLLOWS THE INPUT (2026-11-22 14:05)

The gap this closes: for a day the model **behaved** as if it knew what mattered (100% finding one digit among 64 decoys) and
**could not say** what it acted on (the read-out probe scored 77/100 against a bar of 90). So it acted on importance and could not
report it - useless for a project whose entire point is a readable reason.

The fix came from the project's own discovery. The machine could not add until it was allowed to **write the working out**, so the
same medicine was applied to explanation: **do not probe the model with a head bolted on the side - ask it to write down which
input it used, as part of its own answer.**

```
  input :  12 positions, exactly one holds a real digit, the rest are decoys
  output:  [the position of that digit] [the digit itself]
```

## Result, three seeds, 6,000 steps, 200 held-out samples each

```
seed 0:  position 100.0% | digit 100.0% | both 100.0%
seed 1:  position 100.0% | digit 100.0% | both 100.0%
seed 2:  position 100.0% | digit 100.0% | both 100.0%
P1 (position >= 90%) MET, P2 (digit >= 95%) MET, P3 (both >= 85%) MET, falsifier not fired
OUTCOME: EXPLAINS_BY_EMITTING
```

## And then the second measurement, because a clean 100% is what hid this morning's defect

Three seeds at exactly 100.0% on the first attempt is the same shape as the ternary result that turned out to be measuring a
broken target. So before believing it, the input was changed in ways whose right answer is known in advance:

```
digit MOVED to a different position  -> the model follows it:            100.0%
digit value SWAPPED                  -> it reports the new value:        100.0%
digit value SWAPPED                  -> it leaves the position alone:     100.0%
VERDICT: it really follows the input
```

**It is not reciting a memorised pattern: move the digit and the stated position moves with it; change the digit and the stated
value changes while the position stays put.**

## Why this matters more than the accuracy number

The readable middle layer's failure was that concepts **cancelled** - 17 of 36 supplied their own label, 19 opposed it - and the
probe that tried to read importance out of hidden state scored 77. Both were attempts to **read the model's mind from the outside**.
Writing the reason into the answer instead is the same idea that made arithmetic work, and here it is **perfect on every seed**,
with the explanation being **the model's own output**, checkable by a person, not a probe's guess.

That is the first time in this project that "why" is something the model itself says.
