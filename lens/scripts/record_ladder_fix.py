"""record_ladder_fix.py — the big rung died twice, both times from my own bugs.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 — THE "BIG" RUNG DIED TWICE, AND BOTH CAUSES WERE MINE

### DEATH 1 — THE RUNG NAMED "BIG" WAS SMALLER THAN "SMALL"

```
  RUNG SMALL   text_lm ppl 1162.0    code_lm ppl 439.27
  RUNG MEDIUM  text_lm ppl  484.95   code_lm ppl 185.51
  RUNG BIG     text_lm ppl 1722.39   code_lm ppl 533.63    <- WORSE than small
```

Worse-at-big is impossible unless big is not big. The code computed `--max-windows max(2000, n*2)` and
for the big rung `n=0` (meaning "full dataset"), so it evaluated to **2000 -- less than the small
rung's 20,000.** Two of seven capabilities trained on less data at "big" than at "small", **and the
weak ppl gate (bar 7000) passed them anyway.**

### DEATH 2 — THE FIX DID NOT FIT THE TIMEOUT, AND THE HANDLER CRASHED

The first fix set the big rung to 4,000,000 windows. `train_code.py` then exceeded the 1800-second
per-capability timeout, and the handler that catches a timeout **itself raised**:

```
  subprocess.TimeoutExpired: train_code.py --max-windows 4000000 ... timed out after 1800 seconds
    File "ladder.py", line 56, in run
      return -1, (e.stdout or "") + "\\nTIMEOUT"
  TypeError: can't concat str to bytes
```

Two separate defects in that one line:
* **4,000,000 windows cannot finish in 1800s.** A budget that cannot complete is not a budget.
* **the error path had never been executed.** On timeout `e.stdout` can be BYTES even when
  `text=True` was requested, so the handler that was supposed to record "this rung exceeded its
  budget" crashed instead -- turning a clean budget failure into a traceback with **no rung recorded
  at all**. The failure was therefore silent in the artifact: `ladder_big.json` still held the OLD
  numbers from the pre-fix run, and reading it would have shown 7/7 PASS for a rung that had just
  failed.

### THE FIX, AND THE SMOKE TEST THAT PROVES IT

1. The handler accepts bytes or str.
2. The big rung uses **600,000 windows** (10x the medium rung) with a **5400-second** timeout.
3. `run(cmd, timeout=timeout)` -- the per-capability timeout is actually passed. It was not before;
   the variable was computed and never used, so every capability ran under the 1800s default.
4. **The budget is recorded in the rung JSON** (`max_windows`, `epochs`, `d`, `timeout_s`). The rung
   that lied about its size did so because the artifact recorded only pass/fail, so a "big" rung that
   trained on 2,000 windows looked identical to a real one.

Smoke test before trusting a long run -- **the failure was in an error path that had never been run,
so the fix has to be executed, not assumed:**

```
  1. a command that TIMES OUT (sleep 5, timeout 1)   rc=-1  out is str, ends with TIMEOUT  OK
  2. a normal command (echo hi)                      rc=0                           OK
  3. a command that FAILS (exit 3)                   rc=3                           OK
  4. window budget 600000, timeout 5400s, passed to run(), recorded in JSON           OK
```

**A LESSON THAT KEEPS REPEATING IN THIS PROJECT:** every one of these was a defect in the measuring
apparatus, not in the model -- an under-sized budget presented as "big", a gate too weak to notice it,
a recording format that omitted the number that mattered, and an error path that had never been
executed. **The models have been more reliable than the harnesses built to judge them.**
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
print("=" * 100)
print("BIG RUNG — both deaths and the fix")
print("=" * 100)
print("  death 1 : 4,000,000 windows > 1800s timeout")
print("  death 2 : the timeout HANDLER itself raised (bytes vs str) -> no rung recorded at all")
print("  danger  : ladder_big.json still showed the OLD 7/7, so reading it hid the failure")
print("  fix     : bytes-safe handler, 600k windows, 5400s timeout, timeout actually passed,")
print("            and the budget is now recorded in the rung JSON")
print("  smoke   : 4/4 checks passed before the re-run")
print("=" * 100)
