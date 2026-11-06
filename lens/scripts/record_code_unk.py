"""record_code_unk.py — the two real causes, separated by measurement.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "code")


def stats(f):
    p = os.path.join(P, f)
    return json.load(open(p))["stats"] if os.path.exists(p) else None


allp, clean = stats("code1_reexec.json"), stats("code1_reexec_clean.json")

LOG = f"""
---

## 2026-10-08 — CODE: A BIGGER VOCABULARY DID NOT HELP, AND THE MEASUREMENT SAYS WHY

### CHANGE 2 FAILED ITS PREDICTION

```
  run            corpus   vocab      ppl      acc     params      executable
  code1          both     6,000    44.26   36.44%   1,502,774      4/25
  codedoc_full   code     6,000    39.05   38.21%   1,502,774      6/25
  covocab        code    32,000    54.30   36.93%   7,508,774      6/25
```

Predicted: *"syntax rate rises WELL above 6/25."* **Measured: 6/25. Unchanged.** Five times the
parameters, better vocabulary coverage, and no improvement in executable code. (ppl getting worse was
predicted and did happen.)

### WHY, MEASURED — THE MODEL IS TRAINED TO EMIT `<unk>`

```
  vocab 6,000:   <unk> = 6.98% of TRAINING TARGETS (62,937 of 902,310)  -- the #1 most frequent target
  vocab 32,000:  <unk> = 3.48% of TRAINING TARGETS (31,423)              -- drops out of the top five

  the five most frequent TARGETS at 6k:
    1. '<unk>'  62,937  (6.98%)
    2. '.'      46,245  (5.13%)
    3. '`'      39,860  (4.42%)
```

**A bigger vocabulary changes how often the INPUT is unknown. It does not change what the model is
REWARDED for producing.** `<unk>` was the single most frequent token the model was asked to predict,
so it learned to produce it -- which is exactly what the output shows. The reasoning that produced
change 2 ("`<unk>` sits where identifiers go, so a bigger vocabulary will let the model name things")
was wrong about the mechanism.

### AND A SECOND, INDEPENDENT CAUSE IN MY OWN HARNESS

```
  validation prompts containing `<unk>`  : 20/25
  prompts clean of `<unk>`               :  5/25

  ' <unk> ( <unk> , <unk> = false , resume = false , <unk> = 0 ) : '   <- THE PROMPT IS ALREADY BROKEN
```

**The test handed the model `def <unk> ( <unk> , <unk> ) :` and then marked it wrong for continuing
with `<unk>`.** That is the locally correct continuation. So a large share of the generated `<unk>` is
READ-BACK, not invention, and the "24%" number was partly measuring prompt quality.

This is the **third** time the harness has been the problem, after the token-join bug and the
under-budget comparison.

### SEPARATING THE TWO CAUSES

The clean-prompt test needed no retraining, so it was run on the checkpoint already on disk:

```
  clean prompts (5, signature contains no <unk>):   syntax 2/5    runs 2/5    (40%)
  all prompts   (20 of 25 broken):                  syntax 6/25   runs 6/25   (24%)
```

**Both causes are real, and the clean prompt proves it:**

```
  PROMPT '` ` ` python def process_migration_operations ( operations : '     <- a REAL identifier
  WROTE  '<unk> ) : <unk> = 0 . the ` <unk> ( <unk> , <unk> , <unk> , <unk> ) ...'
```

Handed a real function name, the model **still** emits `<unk>`. So:
* prompt contamination explains **part** of the gap (+16 points), and
* the model's `<unk>` habit is a **separate, genuine** defect.

Both numbers are now reported on every run. The broken-prompt number was **kept**, not replaced --
deleting the hard cases would be lowering the bar, and the project's rule is that a bar cannot move
after a result.

### CHANGE 3, NOW RUNNING

Drop the training windows whose target is `<unk>`. Those positions are unanswerable by construction --
the true token was discarded before training started -- so training on them can only teach `<unk>`.

**The alternative not taken:** `ignore_index` inside the loss, which is the standard framework
approach. `softmax_ce` exists in **three** implementations here (numpy, the C library `liblens.so`,
and the CUDA path), so that is three changes each needing its own gradient check. Dropping the windows
is ONE change, loses no context (windows overlap at stride W/2), and is trivially inspectable. The
choice is recorded rather than left implicit.

Predictions: the generated `<unk>` rate falls sharply (not to zero -- `<unk>` can still come from an
unknown input); executable rate rises above 6/25 overall and above 2/5 on clean prompts; **val ppl
gets WORSE**, because the model loses the easy 7% of targets -- and a ppl improvement would be the
suspicious outcome.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("CODE — three changes, measured one at a time")
print("=" * 100)
print("  change 1  corpus = solution only   CONFIRMED: ppl 44.26->39.05, executable 4/25->6/25")
print("  change 2  vocab 6k -> 32k          FAILED:     ppl 39.05->54.30, executable 6/25->6/25")
print("  change 3  mask <unk> targets       RUNNING")
print()
print("  cause A (harness): 20/25 prompts hand the model `def <unk> (`  -> +16 pts on clean prompts")
print("  cause B (model):   <unk> is 6.98% of training targets         -> needs change 3")
print("=" * 100)
