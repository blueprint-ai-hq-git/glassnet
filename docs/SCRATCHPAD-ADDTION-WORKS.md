

---

# THE SCRATCHPAD WORKS: ADDITION BECOMES COUNTING AND REACHES 100.0% (2026-11-21 14:25) - WITH A MEMORISATION TEST NOW RUNNING

The first thing all day that made arithmetic work, and the reason it worked is that the computation stopped being hidden.

## Measured, 2 seeds

```
task                                   exact-count accuracy   within +-1
single-token answer (sum2fixed)               2.0%                -
single-token answer, wider model (sum2)        6.0%                -
LOOP R=1 / R=4 / R=8                          7.0% each           -
SCRATCHPAD (write the count, then STOP)     100.0% / 100.0%     100.0% / 100.0%
```

- **P1 MET** on both seeds (bar was >= 50% exact).
- **P2 MET** on both seeds (bar was >= 80% within +-1).
- **Falsifier not fired.**

## Why this is different from the loop

The loop asked the same hidden state to be transformed more times. The scratchpad changes **what the model is asked to write**: to
answer 3+4 it emits seven ONE tokens and then STOP. Each emitted token is one step, and the count is carried in the open sequence instead
of inside a single hidden state - so the addition happens **where a human can check it**, token by token. Nothing is hidden: the decode
trace is the explanation.

This also explains the whole day's pattern. Selection, copying and comparison are all **one-step** operations, and attention does them
perfectly (100.0% on four rungs). Addition is the first thing asked of this architecture that needs **more than one step**, and the only
change that helped was giving it more steps **in the output**, not in the depth.

## The objection that has to be answered before this is called computation

**There are only 100 possible digit pairs. 100 is small enough to memorise.** The training run had 2,500 steps x 32 = 80,000 examples, so
every pair was seen hundreds of times. A lookup table keyed on (d1,d2) would produce exactly this 100% result **without computing
anything** - it would just be a table written out in unary.

So the 100% is **not yet evidence of computation**, and the honest status right now is: *the scratchpad format makes the task learnable, and
we do not yet know whether what was learned is a procedure or a table.*

## The test now running, predictions written first

Two independent generalisation tests in one run:

- **held-out pairs:** 20 of the 100 pairs never appear in training.
- **unseen sum range:** pairs whose sum exceeds 12 never appear, so sums 13-18 are unseen by **range**.

```
P4 a PROCEDURE generalises:      held-out pairs >= 80% exact
P5 range generalisation:         unseen sum range >= 60% exact
MEMORISATION FALSIFIER:          held-out pairs <= 30%  ->  the 100% was a lookup table in the output format, NOT computation,
                                 and the scratchpad claim is withdrawn in public
```

If P4 and P5 both pass, this is the first genuine computation result in this project and it becomes the basis for the architecture. If
the falsifier fires, the honest sentence is that the output format made a memorisable task easy and nothing more - which is worth knowing,
but it is not computation.
