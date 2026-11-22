

---

# IT IS NOT CAPACITY: THE 88-94% SURVIVES 3.3x MORE ROOM (2026-11-22 14:31)

The two-item explanation's flip case (swap the larger digit so the other becomes larger, forcing the pointer to move) sat at
**88.0% / 94.0%** against a 95% bar. One change was made to test whether that was a room problem: **d 64 -> 96, layers 2 -> 3**
(67,588 -> **224,484 parameters, 3.3x**), same task, same steps, same seeds, same bar.

## Measured

```
                       params     clean sample        flip-case position
seed 0 (before)        67,588     pos/val/both 100%   88.0%
seed 0 (after)        224,484     pos/val/both 100%   92.5%
seed 1 (before)        67,588     pos/val/both 100%   94.0%
seed 1 (after)        224,484     pos/val/both 100%   89.5%
```

**FALSIFIER FIRED:** more room did **not** bring the flip case to 95%. 92.5 and 89.5 against 88.0 and 94.0 is the same thing twice,
inside noise. **P1 NOT MET, P2 MET.** The clean-sample headline stayed perfect, so the extra room was not even needed for the easy
part.

## The boundary, stated and closed

- **One item: the reason is reliable.** [position][digit] at **100.0%** on three seeds, surviving every perturbation
  (moved 100.0%, swapped value 100.0%, position held 100.0%).
- **Two items: the reason is 88-94% when the answer flips**, at either size. Mostly right, **not trustworthy**, and it is **not a
  capacity limit** - so it must be a limit of how the choice is represented, not of how much room the model has.
- The honest one-line boundary: **this architecture can say why it picked ONE thing, perfectly; it cannot yet reliably say why it
  picked one of TWO.**

For a project whose readable middle layer is the main bet, that is the sharpest statement available today, and it is bounded by
measurement rather than by optimism. The next real attack on it is representation (how the choice between two items is written),
not more parameters and not more steps.
