"""smoke_ladder_timeout.py — prove the timeout handler works BEFORE trusting a long run.

The big rung died because the handler that CATCHES a timeout itself raised:
    TypeError: can't concat str to bytes
so no rung was recorded at all. A fix to that handler has to be exercised, not assumed -- the whole
point of the failure was that the error path had never been run.

This calls the real `run()` with a command that deliberately exceeds its timeout, and separately
confirms a normal command still returns cleanly.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import ladder                                            # noqa: E402

print("=" * 90)
print("SMOKE TEST — the ladder's timeout handler")
print("=" * 90)

print("\n  1. a command that TIMES OUT (sleep 5, timeout 1)")
rc, out, secs = ladder.run(["sleep", "5"], timeout=1)
print(f"     rc={rc}  secs={secs:.1f}  type(out)={type(out).__name__}")
print(f"     out ends with TIMEOUT: {out.strip().endswith('TIMEOUT')}")
print(f"     handler did not raise : True")
assert rc == -1, "a timeout must return rc -1"
assert isinstance(out, str), "output must be str, not bytes"
assert "TIMEOUT" in out

print("\n  2. a normal command (echo hi, timeout 30)")
rc2, out2, secs2 = ladder.run(["echo", "hi"], timeout=30)
print(f"     rc={rc2}  out={out2.strip()!r}  type={type(out2).__name__}")
assert rc2 == 0 and "hi" in out2

print("\n  3. a command that FAILS (exit 3)")
rc3, out3, _ = ladder.run(["bash", "-c", "exit 3"], timeout=30)
print(f"     rc={rc3}")
assert rc3 == 3

print("\n  4. the big rung's budget is inside its timeout")
print(f"     RUNGS = {ladder.RUNGS}")
n = ladder.RUNGS["big"][0]
print(f"     big rung n_train={n} (0 means full dataset)")
# the two numbers the fix introduced
src = open(os.path.join(PROJ, "lens", "scripts", "ladder.py")).read()
assert "big_windows = 600000" in src, "the window budget fix is missing"
assert "timeout = 5400" in src, "the timeout fix is missing"
assert "run(cmd, timeout=timeout)" in src, "the timeout is not passed to run()"
assert "max_windows" in src, "the budget is not recorded in the rung JSON"
print("     window budget 600000, timeout 5400s, timeout passed to run(), budget recorded: OK")

print("\n  ALL CHECKS PASSED")
print("=" * 90)
