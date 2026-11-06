#!/usr/bin/env python3
"""regress_after_opt_change.py - the shipped (d) harness re-run AFTER the lens/opt.py change.

The project rule: when the engine changes, every claim measured on it is suspect until re-measured. The
change here was lens/opt.py (a `decay` knob + the `scale` factor moved to rmsprop only), and
lens/scripts/numeric_path_test.py does not import lens/opt at all - so this should reproduce
lens/runs/NUMERIC-PATH.txt exactly. Checked, not assumed: the two outputs are normalised (numpy RuntimeWarning
lines dropped, they are stderr noise) and compared line by line. The result is appended to
lens/runs/NUMERIC-PATH-OPT-DECAY.txt by the caller.
"""
import os
import subprocess
import sys

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
STORED = f"{ROOT}/lens/runs/NUMERIC-PATH.txt"
WARN = ("RuntimeWarning", "warnings.warn", "np.nanmean", "return Tensor", "_accumulate")


def normalise(text):
    out = []
    for line in text.splitlines():
        s = line.strip()
        if any(w in s for w in WARN):
            continue
        if s.startswith(ROOT + "/lens/"):
            continue
        if s:
            out.append(line.rstrip())
    return out


def main():
    env = dict(os.environ, NUMERIC_LR="0.2")
    p = subprocess.run([sys.executable, "lens/scripts/numeric_path_test.py"], cwd=ROOT, env=env,
                       capture_output=True, text=True)
    fresh = normalise(p.stdout + p.stderr)
    stored = normalise(open(STORED).read())
    print(f"  re-ran lens/scripts/numeric_path_test.py at NUMERIC_LR=0.2 (the stored run's setting), "
          f"exit {p.returncode}")
    print(f"  stored file {STORED}: {len(stored)} comparable lines; fresh run: {len(fresh)}")
    diffs = [(i, a, b) for i, (a, b) in enumerate(zip(stored, fresh)) if a != b]
    same = (len(stored) == len(fresh)) and not diffs
    if diffs:
        for i, a, b in diffs[:5]:
            print(f"    line {i}: stored {a!r} != fresh {b!r}")
    print(f"  SHIPPED (d) HARNESS AFTER THE OPT.PY CHANGE: {'IDENTICAL' if same else 'DIFFERENT'}"
          f"   ({len(diffs)} differing lines)")
    tail = [l for l in fresh if "NUMERIC PATH GATE" in l]
    print(f"  the gate line in the fresh run: {tail[0].strip() if tail else 'NOT FOUND'}")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
