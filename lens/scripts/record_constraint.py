"""record_constraint.py — SyntaxConstraint PASSES all 7 tests; safe to wire to the model.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-09 — SyntaxConstraint PASSES (7/7); the coding fix is now sound to apply

`constrain_code.py` — constrained decoding for code. Masks out tokens that would make the
program syntactically invalid at every generation step. Same class of change as top-k and the
repetition penalty: a DECODER change, model weights and perplexity untouched.

### PROOF IT WORKS (against CPython's own ast.parse as ground truth)
```
  real programs (want ACCEPT):     3/3 correct
    'def add ( a , b ) : return a + b\\n'                       ACCEPT
    'class BankAccount : \\n def __init__ ( self ) : ...'       ACCEPT
    'for i in range ( 10 ) : print ( i )\\n'                    ACCEPT

  broken programs (want REFUSED):  4/4 correct
    'def add ( a , b : return a + b\\n'       (unclosed paren)   REFUSED
    'return a + \\n'                          (dangling op)      REFUSED
    'def f ( ) : \\n return 1\\n  x = 2\\n'    (bad dedent)      REFUSED
    'x = [ 1 , 2\\n'                          (unclosed bracket) REFUSED

  token-level agrees() query: 7/7 cases, ZERO disagreements with CPython
```

### THE BUG THAT SURFACED AND HOW IT WAS FOUND (this part matters)
The first test run reported **0/3 real programs accepted**, with every case showing "WRONG". The
tempting conclusion was "the constraint is broken". It was not. Debugging the actual token stream
showed the tokens arriving as `['def','add','(', 'a' ...]` — **no spaces at all** — because the
TEST's tokenizer used `.split()`, which discards whitespace. Joined together that is
`defadd(a,b):returna+b`, and refusing it was exactly what a syntax checker should do.

**The constraint was right; my test was wrong.** Printed the token stream instead of guessing, found
the missing whitespace in one step. Lesson recorded: when a checker rejects valid input, verify what
the checker actually RECEIVED before concluding the checker is broken.

### HONEST LIMITS (unchanged, still true)
* Enforces SYNTAX not sense. `def f ( ) : pass` is valid and will be allowed.
* It is a heuristic incremental checker, not CPython's grammar. The gate measures against
  ast.parse, so any disagreement surfaces as a failure rather than being hidden.
* It does not fix the prose problem by itself. If the model wants English, constraining it to valid
  syntax may yield valid-but-pointless code. That outcome will be reported, not spun.

### NEXT
Wire it to the existing code checkpoint (code1_ep16) and measure syntax/ran rates against the
unconstrained 8/25 and 7/25 baseline. Pre-registered gates before that run.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended constraint PASS record")
