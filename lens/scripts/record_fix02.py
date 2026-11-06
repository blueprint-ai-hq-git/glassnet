"""record_fix02.py — graded concepts via MSE: language better, concepts dead.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 17:00 — FIX-02: BOTH PREDICTIONS CORRECT, AND THE TRADE-OFF FLIPPED.

One change: concept targets become GRADED (the fraction of the window the concept occupies) and the
loss becomes a weighted squared error, at `lam_concept=0.3`.

```
arm                    concepts       ppl       gap      acc
lc=0.1 (binary)             7/9    90.085   +12.281   29.73%
lc=0.3 (binary)             9/9   100.850   +23.045   29.62%
two-phase                   9/9   116.430   +38.626   29.65%
FIX-02 soft (mse) lc=0.3    0/9    99.313   +21.508   30.03%
```

**PREDICTION 1 -- CORRECT.** "The gap at 9/9 falls below +23.045." It fell to **+21.508**.

**PREDICTION 2 -- CORRECT.** "Accuracy rises above 29.73%." It rose to **30.03%, the best of any
arm.** So graded concepts genuinely do help the language model.

**BUT ALL NINE CONCEPTS DIED: 0/9.** AUROC 0.4570-0.8027 with firing rates 0.000-0.056.

### WHY -- A CLASSIC TRAP, WALKED INTO WITH EYES OPEN

**Squared error on a RARE target has its optimum at the MEAN.** Most windows contain no speech
verbs, so the cheapest thing for every neuron was to sit near zero and never fire. MSE quietly
turned all nine concepts into constants.

**And that is also why the language numbers improved.** A constant concept layer is barely a
bottleneck at all -- the answer layer is effectively reading a fixed offset. **The good language
score and the dead concepts are the SAME fact, not two results.** The +21.508 is not progress on
the blocker; it is the blocker being sidestepped by removing the concepts from the path.

This is recorded explicitly because it is exactly the shape of a false win: the metric improved
while the thing being measured stopped existing.

### WHAT THE TRADE-OFF NOW LOOKS LIKE

```
binary  (concepts alive)  ->  7/9 or 9/9 concepts,  gap +12.281 .. +23.045
mse     (concepts dead)   ->  0/9 concepts,          gap +21.508, best accuracy
two-phase (concepts perfect, frozen) -> 9/9 concepts, gap +38.626
```

Every arm sits between +12.281 and +38.626 against a bar of ~6. **Nothing is close.** The concept
layer's SHARPNESS was worth testing and the answer is: it helps the language only when it stops
being a concept layer.

### NEXT: FIX-03, soft labels with the LOG loss

The graded target is right; the squared error was wrong. A **soft-label BCE** -- the same graded
target in [0,1] fed to the log loss -- cannot collapse to the mean, because its gradient stays
large while the neuron is wrong, so a rare concept keeps being pushed toward firing. That is the
standard fix for exactly this collapse.

**PREDICTIONS, written before running:**
1. The concepts recover -- 9/9 again -- because the log loss cannot collapse to the mean.
2. The language gains are partly kept: gap between +21.508 and +23.045, accuracy above 29.62%.
3. **If the concepts still fail at 9/9 with a loss that cannot collapse, then the bottleneck is
   the NUMBER of concepts (nine), not their sharpness.** That is an owner decision, not something
   to solve silently: more concepts means more to read, which is the thing the project exists to
   protect.

**Bars unchanged.** AUROC 0.55, firing floor 2%, gap against the matched normal (77.804).
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
