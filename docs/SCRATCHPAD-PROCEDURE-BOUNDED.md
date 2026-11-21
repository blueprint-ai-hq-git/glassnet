

---

# IT IS A PROCEDURE, NOT A LOOKUP TABLE - AND IT IS BOUNDED BY THE RANGE IT WAS TAUGHT (2026-11-21 14:35)

The memorisation test answered the objection and then produced a sharper one of its own.

## Measured, 2 seeds, 59 training pairs, 20 held-out pairs, sums above 12 never seen

```
seed   TRAIN pairs        HELD-OUT pairs      unseen SUM RANGE (13-18)
 0     100.0% exact       98.0% exact          0.0% exact   (26.0% within +-1)
 1     100.0% exact      100.0% exact         14.0% exact   (62.0% within +-1)
```

- **P4 MET on both seeds:** held-out pairs, never in training, are answered at **98.0% and 100.0%**. A lookup table keyed on (d1,d2)
  cannot do that. **The memorisation falsifier did not fire.**
- **P5 FAILED on both seeds:** sums 13-18, never seen in any training example, are answered at **0.0% and 14.0%**.

## The honest reading

**It learned a procedure, and the procedure is bounded by the range it was taught.** Not a table of pairs - unseen pairs generalise
almost perfectly. But what it actually acquired is closer to *"count up to twelve"* than to *"add"*. On seed 1 the within-+-1 figure of
62.0% shows it lands in the right neighbourhood above 12 without counting precisely, which is what a bounded counter looks like when it is
pushed past its limit.

That is a much better outcome than a lookup table would have been, and a much more specific one than "addition works":

> this architecture can learn a **multi-step counting procedure that generalises to unseen inputs**, and it does **not extrapolate**
> beyond the range of inputs it was trained on.

## Why this matters for the architecture, and the whole day's shape

Every earlier attempt asked for the answer in **one step** and failed at 2-7%. Serialising the steps in the output reached **100%**. So the
missing piece was never capacity, data, memory or depth - it was **a place for intermediate state to exist**. Tokens are that place, and
they are fully readable, which is exactly what this project wants: the computation is the trace.

## What this does NOT license

It does not license "the model can add". It adds **within the taught range**. The next question is whether the range can be widened
without breaking generalisation, and whether the procedure transfers to a harder task (two-digit plus two-digit with a carry) - both of
which are new pre-registrations, and neither is assumed here.
