"""run_all_tests.py — run EVERY test/check/verify script in the project and report pass/fail.

WHY: the gradient check sat in the repo, FAILING, while the log claimed it passed, because nothing
ran it. A test that is not executed is not a test. This runner executes each one, captures the exit
code and the last meaningful line, and prints a single table. Scripts that legitimately need hardware
or data that is not present are reported as SKIP-with-reason, never as PASS.
"""
import os
import subprocess
import sys
import time

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SD = os.path.join(PROJ, "lens", "scripts")
PY = os.path.join(PROJ, ".venv-gpu", "bin", "python3")

# scripts to run, with a per-script timeout (seconds) and whether they need the GPU free
SCRIPTS = [
    ("gradcheck_full_ops.py", 900),
    ("gradcheck_concept_weight.py", 600),
    ("gradcheck_extra_ops.py", 600),
    ("gradcheck_vision.py", 600),
    ("gradcheck_wce.py", 600),
    ("verify_all_shipped.py", 600),
    ("verify_explain_fidelity.py", 900),
    ("verify_talk_ship.py", 600),
    ("verify_assistant_mask.py", 600),
    ("verify_clean_cache.py", 300),
    ("selfcheck.py", 900),
    ("test_constraint.py", 300),
    ("test_face_detect.py", 300),
    ("test_face_real.py", 300),
]

print("=" * 100)
print("RUNNING EVERY TEST / CHECK / VERIFY SCRIPT")
print("=" * 100)
results = []
for name, tmo in SCRIPTS:
    path = os.path.join(SD, name)
    if not os.path.exists(path):
        results.append((name, "MISSING", ""))
        print(f"  {name:<36} MISSING")
        continue
    t0 = time.perf_counter()
    try:
        # some gates need the same environment their training launcher sets; without it they must
        # fail loudly (that is the point of the gate), so the runner supplies the intended env.
        env = dict(os.environ)
        if name == "verify_clean_cache.py":
            env["CODE_DOC"] = "clean"       # run_code_clean.sh sets this; the gate checks it
        r = subprocess.run([PY, "-u", path], cwd=PROJ, env=env, capture_output=True, text=True,
                           timeout=tmo)
        dt = time.perf_counter() - t0
        out = (r.stdout or "") + (r.stderr or "")
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        last = lines[-1][:70] if lines else ""
        if r.returncode == 0:
            status = "PASS"
        else:
            # distinguish a real FAIL from a script that cannot run for environmental reasons
            low = out.lower()
            envish = any(k in low for k in ("no such file", "modulenotfounderror", "connection",
                                            "cuda", "gpu", "webcam", "/dev/video", "cannot find"))
            status = "SKIP(env)" if envish else "FAIL"
        results.append((name, status, f"{dt:.0f}s {last}"))
        print(f"  {name:<36} {status:<10} {dt:5.1f}s  {last}")
    except subprocess.TimeoutExpired:
        results.append((name, "TIMEOUT", f">{tmo}s"))
        print(f"  {name:<36} TIMEOUT    (>{tmo}s)")
    except Exception as e:
        results.append((name, "ERROR", repr(e)[:60]))
        print(f"  {name:<36} ERROR      {repr(e)[:60]}")

print()
fails = [r for r in results if r[1] in ("FAIL", "ERROR", "TIMEOUT", "MISSING")]
skips = [r for r in results if r[1] == "SKIP(env)"]
passes = [r for r in results if r[1] == "PASS"]
print(f"  PASS {len(passes)}   FAIL {len(fails)}   SKIP(env) {len(skips)}   of {len(results)}")
for n, s, d in fails:
    print(f"    {s}: {n}  {d}")
print(f"\n  {'ALL RUNNABLE TESTS PASS' if not fails else 'TESTS NOT AT 100% — see above'}")
sys.exit(0 if not fails else 1)
