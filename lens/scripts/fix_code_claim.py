"""fix_code_claim.py — CORRECTION: the 0/25 exec rate I just logged was the STALE harness-bug
number. Fix the record immediately; never leave an error standing.

The real story: `train_code.gen()` joined tokens with "" so `def foo ( x ) :` became `deffoo(x):`,
which can never parse. That bug was already found and fixed in a previous session (reexec_code.py),
and the corrected run gives 8/25 syntax and 7/25 that actually RUN — not 0/25.

The honest verdict changes from "coding does not work at all" to "coding works occasionally (8/25
parse, 7/25 run) and the output is still mostly rambling prose rather than code". Both are bad; only
one of them is true. The 0/25 figure must never be quoted again.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(P, "lens", "runs", "code", "code1_reexec.json")))
st = d["stats"]

LOG = f"""
---

## 2026-10-09 — CORRECTION to the capability audit above: the 0/25 code figure was STALE

I wrote in the entry above that the code model produces "syntax_ok 0/25, ran 0/25" and concluded
"coding is NOT working". **That number came from the artifact `codedoc_full_replies.json`, which was
produced BEFORE a harness bug was found and fixed.** Quoting it as the current state was wrong, and
the rule in this project is that an error gets fixed the moment it is spotted.

### WHAT THE BUG WAS (already found and fixed in a previous session)
`code_data.TOKEN_RE` drops whitespace, and `train_code.gen()` joined generated tokens with `""`:
`def foo ( x ) :` came back as `deffoo(x):` — which is not Python and never can be. **The reported
0/25 syntax rate was measuring my own generator bug, not the model.**

### THE CORRECTED NUMBER (real, from code1_reexec.json)
```
  code1, 25 samples, whitespace-join fixed:
    syntax OK : {st['syntax']}/25
    ran OK    : {st['ran']}/25
```

### WHAT IS STILL TRUE, AND STILL NOT GOOD
The model parses **{st['syntax']}/25 = {st['syntax']/25*100:.0f}%** of the time and runs
**{st['ran']}/25 = {st['ran']/25*100:.0f}%**. Looking at the actual generated text, it is mostly
rambling English prose with code fragments glued on ("...the ` solution ` the ` ` ` function first
first takes in the ` ` ` to the specified and iterates through the specified..."). So:

* "coding does not work at all" -> **FALSE**, corrected.
* "coding works reliably" -> also **FALSE**; {st['syntax']}/25 is poor.
* The accurate statement: **the code model sometimes emits syntactically valid Python
  ({st['syntax']}/25) and rarely emits something that runs ({st['ran']}/25), and its output is
  dominated by prose rather than code.** The concept audit is genuinely strong (29/38 concepts,
  has_a_loop AUROC 0.998, has_return 0.9997, has_error_handling 1.000), so READABILITY holds where
  GENERATION does not.

### THE LESSON (worth more than the number)
A metric from an artifact is only valid if you know WHICH HARNESS produced it. Two runs of the same
checkpoint gave 0/25 and 8/25 purely because the join changed. Before quoting any rate, check that
the harness that produced the file is the harness that is current. This is the same class as the
ladder's wrong-subject bug: the number was real, the conclusion drawn from it was not.
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended CORRECTION")
print(f"  corrected: code1 syntax {st['syntax']}/25, ran {st['ran']}/25 (was quoted as 0/25)")
