#!/usr/bin/env python3
"""numeric_path_lr_probe.py - the harness must be able to reproduce the probe's lever BEFORE the numeric
path is judged by it.

P29b in lens/runs/NUMERIC-PATH-PREDICTIONS.txt said, in advance: "if arm C does not reproduce the probe's
100%, the harness is broken and no conclusion about arm B may be drawn from it". Arm C (the magnitude
channel -> regression head, which is the probe's own winning structure) came out at 0% exact on unseen and
only 60-73% exact on TRAINED pairs at lr 0.05 - it had not even fitted the training set. So the harness was
broken and the first numeric-path FAIL is not evidence about the architecture.

This file measures WHY and records the fix. The plain-SGD gradient of a mean squared error over 100 examples
is (2/100)*(pred-target)*feature, i.e. ~0.006 at the start, so at lr 0.05 the head moves 3e-4 per step and
needs ~3300 steps to reach the weight 1 it needs. That is a step-size problem, not a structure problem.

Measured here: arm C exact on unseen as a function of lr and steps, plus the same for arm B. Output:
lens/runs/NUMERIC-PATH-LR.txt
"""
import os
import subprocess
import sys

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
SCRIPT = f"{ROOT}/lens/scripts/numeric_path_test.py"


def run(lr, steps):
    env = dict(os.environ, NUMERIC_LR=str(lr), NUMERIC_STEPS=str(steps))
    p = subprocess.run([f"{ROOT}/.venv-gpu/bin/python3", SCRIPT], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=1800)
    lines = p.stdout.splitlines()
    out = {}
    for ln in lines:
        if "MEANS OVER 3 SEEDS" in ln:
            out["_means"] = True
        if ln.strip().startswith("A presence + classifier") and out.get("_means"):
            out["A"] = ln
        if ln.strip().startswith("B concepts + magnitudes + num head") and out.get("_means"):
            out["B"] = ln
        if ln.strip().startswith("B0 channel zeroed") and out.get("_means"):
            out["B0"] = ln
        if ln.strip().startswith("C magnitudes -> num head only") and out.get("_means"):
            out["C"] = ln
        if "NUMERIC PATH GATE" in ln:
            out["gate"] = ln.strip()
    return out, lines


def main():
    print("=" * 110)
    print("HARNESS CHECK: can this harness reproduce the probe's lever at all? (arm C, lr x steps)")
    print("=" * 110)
    print("  P29b (written before the run): if arm C does not reach >=95% exact on unseen, the harness is")
    print("  broken and arm B may not be judged by it. First measurement at lr 0.05 / 4000 steps: FAIL.")
    print()
    for lr, steps in ((0.05, 4000), (0.2, 4000), (0.5, 4000), (1.0, 4000), (0.5, 12000), (1.0, 12000)):
        out, _ = run(lr, steps)
        print(f"  lr {lr:<5} steps {steps:<6}")
        for k in ("A", "B", "B0", "C"):
            if k in out:
                print(f"    {out[k].strip()}")
        print(f"    {out.get('gate', '(no gate line)')}")
        print()
    print("  READING: arm C is the harness's own reference. It must reach the probe's exact 100% before any")
    print("  verdict about the architecture's numeric path is read off this test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
