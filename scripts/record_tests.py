"""record_tests.py — record both scenario suites and refresh STATUS.md.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-05 22:10 — READABLE-MODEL SUITE: 10/10 PASSED (17/17 across both suites)

`tests/test_readable.py` (new) covers today's modules — `glassnet.explain` and
`glassnet.selfgen` — which until now had only been exercised by ad-hoc scripts.

```
PASS  1. generate() shape and seed preservation   -> (6,20), 8 given + 12 generated, prefix intact
PASS  2. generation is reproducible from its rng seed   -> same seed identical, different differs
PASS  3. windows and targets are aligned          -> (50,8), 50 targets, alignment verified
PASS  4. the pre-registered selection rules behave -> all 320/320; conf50 160/320 (50%); conf>=0.5 0
PASS  5. corpus health measurable, generated vs real -> 661 distinct tokens vs 1501; entropy 6.008 vs 5.064
PASS  6. explain() is EXACT, not a story           -> worst residual 4.768e-07 over 400 windows
PASS  7. explain() renders as plain English        -> 15 lines
PASS  8. what each concept pushes up is readable   -> 9 rows
PASS  9. the checkpoint-key mismatch is pinned by a test
PASS 10. the audit K guard holds
READABLE-MODEL SUITE: 10/10 passed
```

**Two of these tests exist specifically to stop today's mistakes from returning:**

* **Test 9** pins the checkpoint-key mismatch: `f_train` writes `['Wa','Wc','ba','bc','emb']`,
  real checkpoints write `['answer.W','answer.b','concept.W','concept.b','emb']`, and loading an
  f_train file with `M3.load_ckpt` gives concept mean **0.5063** instead of **0.0454**. That is
  the defect that produced the withdrawn "all nine concepts died" claim.
* **Test 10** pins the audit's `T3M_K`: a default of 32 against a K=9 checkpoint reports coverage
  as 9/32 (it printed **35%** once, which is impossible when all nine columns are named).

**Test 4 also pins the pre-registered selection amendment:** `conf >= 0.5` keeps **0** windows —
the empty-arm condition that forced the amendment to the quantile rule before the arms ran.

**Totals:** library suite **7/7** (`tests/test_library.py`) + readable-model suite **10/10**
(`tests/test_readable.py`) = **17/17**. Results JSON: `runs/t3/test_readable_results.json`.

**Interpreter rule, recorded:** the project python
(`/home/yahya/.hermes/hermes-agent/venv/bin/python3`, matplotlib 3.11.1) for anything that draws;
`.venv-gpu/bin/python3` (cupy, no matplotlib) for GPU work only.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = ("8. **Library suite: 7/7 PASSED** with the project python (matplotlib 3.11.1). It scores "
       "6/7 under `.venv-gpu` and the bare `python3`, which lack matplotlib — the library is "
       "fine, the interpreter must have it for `explain(save=)`. Rule recorded: project python "
       "for pictures, `.venv-gpu` only for cupy. The new readable-model modules "
       "(`glassnet.explain`, `glassnet.selfgen`) still need their own suite. **Was:**")
new = ("8. **Both scenario suites pass: 17/17.** Library `tests/test_library.py` **7/7**; "
       "readable-model `tests/test_readable.py` **10/10**. Two tests exist purely to stop "
       "today's mistakes returning: the checkpoint-key mismatch (test 9) and the audit's "
       "`T3M_K` default (test 10). **Interpreter rule:** project python for pictures, "
       "`.venv-gpu` only for cupy. **Was:**")
assert old in txt, "STATUS anchor not found"
txt = txt.replace(old, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
