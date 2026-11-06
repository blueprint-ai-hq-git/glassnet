"""record_code_doctext.py — the two code fixes: what was measured before building.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — CODE, CONTINUED: THE CORPUS MEASURED, AND THE VOCABULARY SIZED BEFORE BUILDING

### WHAT THE FIRST DIAGNOSIS FOUND

On the corpus built as `problem + "\\n\\n" + solution` (19,505,365 tokens, 43,254 documents):

```
  '`'        1,185,164  (6.08%)   <- THE MOST COMMON TOKEN IS A MARKDOWN BACKTICK
  '<unk>'    1,119,649  (5.74%)   <- THE SECOND IS UNKNOWN
  'the'      1,082,598  (5.55%)   <- THE THIRD IS ENGLISH PROSE
  43.0% of documents open with the problem statement
```

### CHANGE 1: THE DOCUMENT BECOMES THE SOLUTION ONLY

`code_data.DOC_MODE` selects what a training document is. Measured on the corpus BEFORE training --
which is the point, because it saved a training run:

```
  measure                       BOTH      CODE     verdict
  most common token is backtick  6.08%     5.64%   improved
  <unk> share                    5.74%     7.21%   WORSE     <- worsened
  'the' share                    5.55%     4.18%   improved
  prose-doc openings             43.0%      1.0%   improved hugely
  prose-marker share             15.57%     9.38%
  code-marker share              28.56%    32.99%
  tokens                    19,505,365  7,885,017
```

**The prose problem is FIXED -- 43% of documents opening with prose became 1%.** And the corpus is now
code-shaped: `.` `(` `)` `,` `:` `=` fill the top ten instead of backticks and "the".

**But `<unk>` got WORSE, from 5.74% to 7.21%, and it is now the MOST COMMON token.** Removing the
prose did not create identifier tokens; it made the identifier shortfall more visible. The model still
cannot name a function.

### CHANGE 2, SIZED BEFORE BEING BUILT

`measure_vocab_need.py` on the code-only corpus (6,273,090 tokens, 136,030 distinct tokens):

```
  vocab   <unk> rate
   6,000      7.18%
  12,000      5.58%
  20,000      4.52%
  32,000      3.51%
  50,000      2.47%     <- even 50k leaves 2.47%

  the same measurement on a REAL 322-token solution:
    <unk> at 6k  = 26 tokens (8.1%)
    <unk> at 32k =  1 token  (0.3%)     <- 27x fewer
```

**Two things follow, and both are honest:**

* **32k is a real fix for real solutions** -- 8.1% of a solution's tokens being unknown drops to 0.3%,
  which is the difference between `def <unk> ( <unk> , <unk> )` and a function that can be called.
* **No vocabulary size fixes it completely.** Even 50,000 tokens leaves 2.47% unknown, because
  identifiers are effectively unbounded (136,030 distinct tokens in only 6.2M). A compositional
  scheme that builds identifiers from sub-tokens would be needed for that, and it is not built.

### THE DISCIPLINE USED

Three things were measured BEFORE any training run:
1. the corpus composition, which named three structural defects;
2. the effect of change 1 on the corpus, which showed it fixes one defect and leaves the other;
3. the vocabulary size required, which produced the number 32k and the honest caveat that it is not
   a complete fix.

Change 1 is now training at the small rung with the executable test attached. Change 2 follows only if
change 1 confirms the diagnosis. **Neither is claimed as a fix until the executable test says so.**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("CODE — changes queued, each measured first")
print("=" * 100)
print("  change 1  DOC_MODE=code        corpus: prose-doc openings 43.0% -> 1.0%")
print("                                 BUT <unk> 5.74% -> 7.21% (now the #1 token)")
print("  change 2  vocab 6k -> 32k      measured need: <unk> on a real solution 8.1% -> 0.3%")
print("                                 caveat: even 50k leaves 2.47%; not a complete fix")
print()
print("  status: 4/25 syntax (16%) before either change. Nothing claimed until retested.")
print("=" * 100)
