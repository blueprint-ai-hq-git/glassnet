"""verify_all.py — the WHOLE project, fresh FULL runs, one command.

Runs every gate and phase script at full settings (no tiny variants), checks
every exit code and key artifact, collects the headline numbers from the
fresh runs, and writes the PASS/FAIL table to runs/VERIFY-ALL.md.

Run:  python3 verify_all.py      (any python with numpy + matplotlib)
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, "runs", "verify")
os.makedirs(VERIFY, exist_ok=True)
PY = sys.executable      # subprocesses inherit THIS interpreter (matches caller)

STEPS = [
    ("gate1 gradcheck", [PY, "tests/test_gradcheck.py"], HERE),
    ("library tests  ", [PY, "tests/test_library.py"], HERE),
    ("P1 xor gate     ", [PY, "phase1_xor.py"], HERE),
    ("P1 baseline     ", [PY, "phase1_fmnist.py", "--seeds", "0", "1", "2"], HERE),
    ("P2 train (full) ", [PY, "phase2_train.py", "--seeds", "0", "1", "2"], HERE),
    ("P2 audit        ", [PY, "phase2_audit.py"], HERE),
    ("P2 explain      ", [PY, "phase2_explain.py"], HERE),
    ("P2 ablations    ", [PY, "phase2_ablations.py", "--seeds", "0", "1", "2"], HERE),
    ("P3 auto-pilot   ", [PY, "phase3_autopilot.py"], HERE),
    ("P3 controls     ", [PY, "phase3_controls.py"], HERE),
    ("P4 judge        ", [PY, "phase4_judge.py"], HERE),
    ("P5 20-line gate ", [PY, os.path.join(HERE, "examples", "example_20_lines.py")], VERIFY),
    ("P6 fair fight   ", [PY, "phase6_comparison.py"], HERE),
]

ARTIFACTS = [
    "LICENSE", "README.md", "PHASES.md", "PROBLEMS.md", "CONCEPTS.md",
    "DATA_LICENSES.md", "HIDDEN_DEFAULTS.md", "docs/BEGINNER.md",
    "pyproject.toml", "examples/example_20_lines.py",
    "examples/example_evo_switch.py",
    "runs/phase1/BASELINE.md", "runs/phase1/baseline_summary.json",
    "runs/phase2/PHASE2-BASELINE.md", "runs/phase2/audit.md",
    "runs/phase2/ablations.md",
    "runs/phase3/PHASE3-REPORT.md", "runs/phase3/CONTROLS.md",
    "runs/phase4/PHASE4-REPORT.md", "runs/phase4/mistakes.png",
    "runs/phase4/summary.json",
    "runs/phase5/PHASE5-REPORT.md", "runs/phase5/gate_20lines.log",
    "runs/phase6/PHASE6-REPORT.md", "runs/phase6/summary.json",
]
GLOBS = [
    ("phase2 explain pictures", "runs/phase2/explain/*.png"),
    ("phase5 report pictures", "runs/phase5/report_*.png"),
    ("phase6 evo models", "runs/phase6/evo_seed*.npz"),
]


def run_step(name, cmd, cwd):
    log = os.path.join(VERIFY, name.strip().replace(" ", "_") + ".log")
    t0 = time.time()
    with open(log, "wb") as f:
        p = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    secs = time.time() - t0
    return p.returncode, secs, log


def headline_numbers():
    """Collect the fresh-run headline numbers (the tables the docs quote)."""
    out = []
    try:
        with open(os.path.join(HERE, "runs", "phase1", "baseline_summary.json")) as f:
            b1 = json.load(f)
        out.append(f"P1 baseline: {b1['mean']*100:.2f}% ± {b1['std']*100:.2f}%")
    except Exception as e:
        out.append(f"P1 baseline: unreadable ({e})")
    try:
        vals = []
        for s in (0, 1, 2):
            with open(os.path.join(HERE, "runs", "phase2", f"final_seed{s}.json")) as f:
                vals.append(json.load(f)["post_prune_acc"])
        out.append(f"P2 readable: {float(np.mean(vals))*100:.2f}% ± {float(np.std(vals))*100:.2f}%")
    except Exception as e:
        out.append(f"P2 readable: unreadable ({e})")
    try:
        vals = []
        for s in (0, 1, 2):
            with open(os.path.join(HERE, "runs", "phase3", f"eval_seed{s}.json")) as f:
                vals.append(json.load(f)["post_prune_acc"])
        out.append(f"P3 auto-pilot: {float(np.mean(vals))*100:.2f}% ± {float(np.std(vals))*100:.2f}%")
    except Exception as e:
        out.append(f"P3 auto-pilot: unreadable ({e})")
    try:
        with open(os.path.join(HERE, "runs", "phase4", "summary.json")) as f:
            j = json.load(f)
        out.append(f"P4 judge held-out {np.mean(list(j['heldout'].values()))*100:.2f}% "
                   f"unseen-type {np.mean(list(j['unseen_type'].values()))*100:.2f}%")
    except Exception as e:
        out.append(f"P4 judge: unreadable ({e})")
    try:
        with open(os.path.join(HERE, "runs", "phase6", "summary.json")) as f:
            j = json.load(f)
        sgd = np.mean([j[str(s)]["sgd_post"] for s in (0, 1, 2)])
        evo = np.mean([j[str(s)]["evo_post"] for s in (0, 1, 2)])
        out.append(f"P6 SGD {sgd*100:.2f}% vs EVO {evo*100:.2f}%")
    except Exception as e:
        out.append(f"P6: unreadable ({e})")
    return out


def main():
    results = []
    t_all = time.time()
    for name, cmd, cwd in STEPS:
        rc, secs, log = run_step(name, cmd, cwd)
        results.append((name, rc, secs, log))
        status = "PASS ✔" if rc == 0 else f"FAIL ✘ (rc={rc})"
        print(f"  {name} {status}  {secs:.0f}s  -> {os.path.relpath(log, HERE)}",
              flush=True)

    missing = [a for a in ARTIFACTS if not os.path.exists(os.path.join(HERE, a))]
    for label, pat in GLOBS:
        if not glob.glob(os.path.join(HERE, pat)):
            missing.append(pat)
    steps_ok = all(rc == 0 for _, rc, _, _ in results)

    nums = headline_numbers()
    lines = [
        "# VERIFY-ALL — full-project fresh run (" +
        time.strftime("%Y-%m-%d %H:%M") + ")",
        "",
        "Every step below ran at FULL settings (same commands as the phase "
        "gates), every log is in runs/verify/. Headline numbers below are "
        "read back from the FRESH run's own artifacts.",
        "",
        "| step | result | seconds |",
        "|---|---|--:|",
    ]
    for name, rc, secs, _ in results:
        lines.append(f"| {name.strip()} | {'PASS ✔' if rc == 0 else 'FAIL ✘'} | {secs:.0f} |")
    lines += ["",
              f"steps: {sum(1 for _, r, _, _ in results if r == 0)}/{len(results)} passed · "
              f"total {time.time() - t_all:.0f}s",
              "",
              "## Artifacts",
              f"- missing: {missing if missing else 'none — every key artifact exists'}",
              "",
              "## Headline numbers (fresh run)",
              "```",
    ]
    lines += nums
    lines += ["```", ""]
    rep = os.path.join(VERIFY, "VERIFY-ALL.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[verify] steps {sum(1 for _, r, _, _ in results if r == 0)}/{len(results)}"
          f" passed; artifacts missing: {missing if missing else 'none'}")
    for n in nums:
        print(f"[verify] {n}")
    print(f"[verify] wrote {os.path.relpath(rep, HERE)}")
    sys.exit(0 if (steps_ok and not missing) else 1)


if __name__ == "__main__":
    main()