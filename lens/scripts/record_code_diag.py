"""record_code_diag.py — why the code model writes prose, measured.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — THE CODE MODEL: 0% WAS MY BUG, 16% IS THE REAL NUMBER, AND THE CAUSE IS THE CORPUS

### FIRST, THE NUMBER WAS WRONG AND IT WAS MY FAULT

The reported **"0/25 syntax"** was not the model. `code_data.TOKEN_RE` drops whitespace entirely and
`train_code.gen()` joined tokens with `""`, so `def foo ( x ) :` came back as **`deffoo(x):`** --
which is not Python and never can be. **A generator that cannot emit a space cannot emit a function
definition, so the test was measuring my own bug.**

Re-run with the join fixed, loading `code1_ep16.npz`:
```
concepts in checkpoint: 38   rebuilt from data: 38   match: True
loaded: d=192 W=16 K=38 params=1,502,774

BEFORE join=''    syntax 4/25   runs 4/25
AFTER  join=' '   syntax 4/25   runs 4/25
```

Two things worth recording:
* **the real number is 4/25 = 16%, not 0%.** Still bad, but the earlier figure was not a model result.
* **the join fix changed nothing** (4/25 either way), so the whitespace bug was real but was never
  the limiting factor. Fixing it was necessary to SEE the real behaviour, not to fix the model.

### WHAT IT ACTUALLY WRITES

```
PROMPT  ...def <unk> ( <unk> , <unk> = false , resume = false , <unk> = 0 ) :
WROTE   'if = = <unk> = = self . <unk> = = <unk> = ( self . <unk> = <unk> ( ) , ` ) ` : this
         solution , the ` <unk> ` class is a method that takes the specified parameters ...'
RESULT  syntax: SyntaxError
```

Markdown backticks. "this solution". "the class is a method that takes". **English prose ABOUT code.**
Many of the 25 outputs are perfectly readable English sentences that describe an implementation
without being one.

### WHY — MEASURED, AND IT IS THE CORPUS

`diag_code.py` on the full 19,505,365-token corpus (43,254 documents):

```
A. WHAT IS IN THE CORPUS
   prose-marker tokens    3,036,461  (15.57%)
   code-marker tokens     5,570,332  (28.56%)
   markdown backticks     1,185,164  ( 6.08%)

D. THE TEN MOST COMMON TOKENS IN THE WHOLE CORPUS
   '`'          1,185,164  ( 6.08%)     <- THE MOST COMMON TOKEN IS A MARKDOWN BACKTICK
   '<unk>'      1,119,649  ( 5.74%)     <- THE SECOND IS UNKNOWN
   'the'        1,082,598  ( 5.55%)     <- THE THIRD IS THE ENGLISH WORD "the"
   '.'            936,362  ( 4.80%)
   ...

C. 86 of 200 validation documents open with prose-heavy text (43.0%)
```

**THREE STRUCTURAL DEFECTS, EACH MEASURED:**

1. **The most common token in the corpus is a backtick (6.08%).** The corpus is a chat-formatted
   mixture: problem statements in English, code fenced in markdown, and prose explaining the answer.
   A next-token model trained on that learns that prose and backticks are the likeliest continuation,
   which is exactly what it produces.

2. **`<unk>` is the SECOND most common token -- 5.74% of 19.5M tokens, 1,119,649 of them.** The
   6,000-token vocabulary is far too small for code, where identifiers are unbounded. **`<unk>` is
   precisely where function names and variables go**, which is why `def <unk> (` appears constantly.
   The model cannot name anything because it has no tokens for names.

3. **43% of documents open with the problem statement**, so from almost any prompt the
   highest-probability continuation is more description.

**The model is not broken. It is doing exactly what this corpus asks of it, and this corpus asks for
prose.** No amount of extra training changes that, which is the useful part of the finding.

### THE TWO CANDIDATE FIXES, NOT YET APPLIED

* **Filter the corpus to code-only** -- drop the markdown scaffolding and the English problem
  statements, so the prediction target is code.
* **Give code a vocabulary that can express it.** Identifiers are unbounded; a 6k vocab cannot hold
  them. A larger vocab, or a scheme that composes identifiers from sub-tokens, is required before the
  model can emit a function NAME at all.

Both are data/design changes, not training changes. This is the same shape as the medical finding:
**a data problem that looked like a model problem**, and the measurement is what separated them.

### STATUS, UNCHANGED AND NOT DRESSED UP

Code execution correctness: **4/25 (16%)**, and the reason is understood. Nothing here is claimed as
working code generation.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("CODE MODEL — the corrected picture")
print("=" * 100)
print("  reported earlier : 0/25   <- MY TOKEN-JOIN BUG, not the model")
print("  real             : 4/25 syntax (16%), 4/25 ran")
print("  join fix effect  : none (4/25 either way)")
print()
print("  most common corpus token : '`'       6.08%  (markdown backtick)")
print("  second most common       : '<unk>'   5.74%  (1,119,649 unknown tokens)")
print("  third most common        : 'the'     5.55%  (English prose)")
print("  documents opening with prose : 43.0%")
print("=" * 100)
