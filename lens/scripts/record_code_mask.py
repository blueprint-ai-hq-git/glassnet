"""record_code_mask.py — change 3 worked on its target, and barely moved executability.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
P = os.path.join(PROJ, "lens", "runs", "code")
cmp = json.load(open(os.path.join(P, "unk_rate_compare.json")))
k = list(cmp)
a, b = cmp[k[0]], cmp[k[1]]

LOG = f"""
---

## 2026-10-08 — CHANGE 3: THE `<unk>` HABIT IS GONE, AND IT BOUGHT ALMOST NOTHING

### FIRST, A BROKEN INSTRUMENT THAT RETURNED A PERFECT-LOOKING ZERO

The first measurement of the generated `<unk>` rate reported **0.00% for every model**, including the
one whose output is visibly full of `<unk>`. Cause: the count re-tokenised the generated text with

    TOK = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\\sA-Za-z_0-9]")

and `TOK.findall("<unk>")` returns **['<', 'unk', '>']** -- three pieces, never the literal token. So
the pattern could not match the token it was counting, and the answer was a silent zero.

**This is the same regex that caused the original token-join bug**, and the second time this class of
error has produced a clean-looking zero. The visible text is what exposed it. Fixed by counting the
token IDS the model actually chose (id 0 IS `<unk>`) instead of re-tokenising.

The project's own rule applies here and was followed: **before reporting a zero, prove the test can
produce a non-zero.** The instrument was re-run and the UNMASKED model measured **7.06%** -- non-zero,
so the 0.00% on the masked model is a real measurement and not a silent failure.

### THE RESULT — ALL THREE PREDICTIONS HELD

Identical prompts, identical vocabulary, identical budget (d=192, 16 epochs, 1.2M windows, seed 0):

```
  model                    <unk> of generated   syntax   runs   clean syntax   clean runs
  codedoc_full (no mask)              7.06%      6/25   6/25       1/3           1/3
  codropunk    (masked)               0.00%      8/25   7/25       1/3           1/3

  generated <unk>   7.06%  ->  0.00%     PREDICTED "falls sharply"   HELD (to zero)
  syntax (all)      6/25   ->  8/25      PREDICTED "rises above 6/25"  HELD
  runs   (all)      6/25   ->  7/25      PREDICTED "rises"            HELD
  val ppl          39.05   -> 67.02      PREDICTED "gets worse"       HELD
```

**The literal token the model was trained to emit is completely gone.** 7.06% of generated tokens were
`<unk>`; now none are. That is a complete fix of the thing the change targeted.

### AND YET IT BOUGHT ONE PROGRAM

**6 of 25 runnable became 7 of 25.** 24% -> 28%.

So `<unk>` was **a real defect but not the binding constraint**. Removing it entirely changed
executability by four percentage points. That is the useful part of this result: it rules out the
symbol-level explanation and says the remaining gap is somewhere else.

### WHAT THE REMAINING GAP ACTUALLY IS

With `<unk>` eliminated, the output is still not code:

```
  '# example : ` the message to be a specified of the input . it returns the input ` ` ` the ` class
   is a enter it then then then returns the dictionary of the specified methods to the ` and ` class'
```

Readable English-ish boilerplate ABOUT functions. With the corpus now code-only and `<unk>` gone, the
remaining causes are the ordinary ones this project has already measured elsewhere:
* the training text is `problem + solution`-derived solution code, much of which contains English
  docstrings and comments, so prose is still a legitimate continuation;
* the model is small and undertrained: 1.5M parameters over 1.2M windows for 16 epochs.

That is a capacity and data position, not a symbol bug, and no further symbol-level patch will move
it. The honest statement is: **24-28% of generated programs parse, the `<unk>` cause is fixed and
eliminated, and the rest is the model being small.**

### THE FOUR-STEP LEDGER FOR CODE

```
  change 1  corpus = solution only   ppl 44.26 -> 39.05   executable 4/25 -> 6/25   CONFIRMED
  change 2  vocab 6k -> 32k          ppl 39.05 -> 54.30   executable 6/25 -> 6/25   FAILED
  change 3  mask <unk> targets       ppl 39.05 -> 67.02   executable 6/25 -> 7/25   TARGET FIXED
  harness   prompts were 20/25 broken with <unk>          clean prompts 40% vs 24%  CAUSE FOUND
```

Nothing here is claimed as working code generation. 7 of 25 programs run.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
for tag, r in ((k[0], a), (k[1], b)):
    print(f"  {tag:<28} <unk> {r['unk_rate']*100:5.2f}%  syntax {r['syn']:>2}/{r['n']}  "
          f"runs {r['ran']:>2}/{r['n']}  clean {r['csyn']}/{r['nc']}")
print()
print(f"  the <unk> habit: {a['unk_rate']*100:.2f}% -> {b['unk_rate']*100:.2f}%   COMPLETELY REMOVED")
print(f"  executables    : {a['ran']} -> {b['ran']} of {a['n']}                    +1 program")
print("  => <unk> was a real defect but NOT the binding constraint on executability")
print("=" * 100)
