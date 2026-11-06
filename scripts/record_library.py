"""record_library.py — record the library test results and the interpreter trap.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-05 22:00 — LIBRARY SCENARIO SUITE: 7/7 PASSED (with the right interpreter)

The Phase-5 library (`GlassNet / train / explain / accuracy / judge / save / load`) was run
through its full scenario suite. **7/7 pass** with the project python
(`/home/yahya/.hermes/hermes-agent/venv/bin/python3`, matplotlib 3.11.1):

```
PASS  test_20_line_gate_script          (the gate script is <= 20 lines)
PASS  test_construct                    (unknown task raises ValueError)
PASS  test_train_mini_explain_and_report (train -> explain -> saved PNG)
PASS  test_evo_switch                   (experimental optimizer behind the switch)
PASS  test_gate1_still_passes
PASS  test_judge_integration            (data judge trains + scores)
PASS  test_save_load                    (save/load roundtrip, explain still works)
LIBRARY TESTS: 7/7 passed
```

**Interpreter trap found and recorded:** the same suite scores **6/7** under `.venv-gpu/bin/python3`
or the bare system `python3` — both lack matplotlib, so only `test_train_mini_explain_and_report`
fails (`ModuleNotFoundError: No module named 'matplotlib'`), and only on the picture-saving
feature. The library itself is fine; the **interpreter must have matplotlib** if `explain(save=)`
is used.

| interpreter | numpy | matplotlib | suite |
|---|---|---|---|
| `/home/yahya/.hermes/hermes-agent/venv/bin/python3` | 2.4.3 | 3.11.1 | **7/7** |
| `python3` (system) | - | missing | 6/7 |
| `.venv-gpu/bin/python3` | yes | missing | 6/7 |

**Rule for later:** use the project python for anything that draws a picture; `.venv-gpu` only
for cupy/GPU work.

**Owner's request being answered here:** "you have to shoot tested in every single scenario" —
the scenarios above are the library's, and they pass. The remaining untested surface is the
new readable-model line (`glassnet.explain`, `glassnet.selfgen`), which has been exercised only
by the ad-hoc scripts in `scripts/`, not by a suite. That is the next test target.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

# add a short section to STATUS.md (read it first so the edit is anchored, not blind)
p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
anchor = "8. **Library testing in progress:**"
new = ("8. **Library suite: 7/7 PASSED** with the project python (matplotlib 3.11.1). It scores "
       "6/7 under `.venv-gpu` and the bare `python3`, which lack matplotlib — the library is "
       "fine, the interpreter must have it for `explain(save=)`. Rule recorded: project python "
       "for pictures, `.venv-gpu` only for cupy. The new readable-model modules "
       "(`glassnet.explain`, `glassnet.selfgen`) still need their own suite. **Was:**")
assert anchor in txt, "STATUS anchor not found"
txt = txt.replace(anchor, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
