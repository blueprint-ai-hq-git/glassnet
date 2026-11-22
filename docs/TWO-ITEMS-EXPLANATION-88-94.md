

---

# TWO ITEMS: THE EXPLANATION IS 88-94% WHEN THE ANSWER FLIPS, NOT 100% - AND MY TEST WAS WRONG FIRST (2026-11-22 14:18)

One-item selection is settled: [position][digit] is 100.0% on three seeds and survives every perturbation. The honest next rung was
**two** items - two digits at random positions, report **where the larger one is** and **what it is**.

## Headline looks perfect. It is not the whole story.

```
seed 0:  position 100.0% | value 100.0% | both 100.0%   (200 held-out samples)
seed 1:  position 100.0% | value 100.0% | both 100.0%
```

## Then the perturbation check, which is where it breaks

```
                                   seed 0    seed 1
digit MOVED to a new position      100.0%    100.0%     <- it follows the item around
value SWAPPED -> correct value      98.0%     98.0%
value SWAPPED -> correct POSITION   88.0%     94.0%     <- the flip case
```

**The hard case is the flip:** swap the larger digit's value so the *other* digit becomes the larger one, and the model must move its
pointer to a position it was previously ignoring. It gets the value right 98% of the time but the **position only 88-94%**.

**Verdict against the bar I set before running (95% on all three): NOT MET.** I am not moving that bar afterwards. So:

- **ONE item: the reason is reliable** - 100.0% on all three checks, three seeds.
- **TWO items: the reason is 88-94% when the answer flips** - better than chance and mostly right, **not yet trustworthy**. The
  headline 100.0% alone would have been a false claim.

## My test was wrong the first time, and the record should say so

The first run reported "value SWAPPED -> position unchanged 66.5%" and blamed the model. **That expectation was my error:** in the
two-digit task, swapping the larger digit can make the *other* one larger, so the correct position changes too. The test assumed the
answer stayed put. Expected answers are now **computed from the perturbed tokens by the task's own definition**, unit-checked on
three cases including `after 7 becomes 4, the larger is 5 at position 1`. Re-running with the right expectations moved the number
from 66.5% to 88-94%.

That is the third time today the second measurement caught a defect, and the first time the defect was in the **test** rather than in
the model or the target. The pattern is consistent: **the claim is the least reliable part of any experiment.**
