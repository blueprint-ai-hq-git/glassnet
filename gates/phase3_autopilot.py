"""PHASE 3 — the auto-pilot trainer, gate run.

Zero settings typed in:  python3 phase3_autopilot.py

What happens:
- Gate 1 first (no training on broken gradients).
- HIDDEN_DEFAULTS.md is verified to list every numeric default (nothing secret).
- The auto-pilot trains the Phase 2 architecture for seeds 0/1/2, logging
  every action in plain English (runs/phase3/actions_seed*.log).
- The gate: does it match or beat the manually tuned score
  (79.04% ± 1.97%, runs/phase2)?
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.gates import gate1  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts  # noqa: E402
from glassnet.autopilot import AUTO_PILOT_DEFAULTS, train_autopilot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase3")
DATA_DIR = os.path.join(HERE, "data")
HIDDEN = os.path.join(HERE, "HIDDEN_DEFAULTS.md")
SEEDS = [0, 1, 2]


def verify_hidden_defaults_doc():
    """Every key in AUTO_PILOT_DEFAULTS must appear in HIDDEN_DEFAULTS.md."""
    with open(HIDDEN) as f:
        doc = f.read()
    missing = [k for k in AUTO_PILOT_DEFAULTS if k not in doc]
    if missing:
        print(f"HIDDEN_DEFAULTS.md MISSING keys: {missing}")
        return False
    print(f"HIDDEN_DEFAULTS.md covers all {len(AUTO_PILOT_DEFAULTS)} defaults ✔ "
          f"(nothing secret)")
    return True


def main():
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first (no training on broken gradients)")
    print("=" * 66)
    gate1()
    if not verify_hidden_defaults_doc():
        sys.exit(1)

    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)

    info = {}
    for seed in SEEDS:
        print(f"\n=== auto-pilot, seed {seed} ===", flush=True)
        net, res = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=seed)
        with open(os.path.join(RUNS, f"actions_seed{seed}.log"), "w") as f:
            for step, msg in res["actions"]:
                f.write(f"[auto-pilot @step {step:6d}] {msg}\n")
        with open(os.path.join(RUNS, f"eval_seed{seed}.json"), "w") as f:
            json.dump({"best_test": res["best_test"],
                       "post_prune_acc": res["post_prune_acc"],
                       "pre_prune_acc": res["pre_prune_acc"],
                       "eval_log": res["eval_log"]}, f, indent=1)
        net.save(os.path.join(RUNS, f"final_seed{seed}.npz"),
                 post_prune_acc=np.float64(res["post_prune_acc"]))
        info[seed] = res
        print(f"seed {seed} DONE | best {res['best_test']*100:.2f}% | "
              f"post-prune {res['post_prune_acc']*100:.2f}% | "
              f"restarts {res['restarts']} | silent-rule prunes "
              f"{len(res['silent_rules_pruned'])}")

    # ---- gate comparison vs the manually tuned Phase 2 score ---------------
    auto_fin = [info[s]["post_prune_acc"] for s in SEEDS]
    auto_best = [info[s]["best_test"] for s in SEEDS]
    manual = {}
    for s in SEEDS:
        with open(os.path.join(HERE, "runs", "phase2", f"final_seed{s}.json")) as f:
            manual[s] = json.load(f)["post_prune_acc"]
    am = float(np.mean(auto_fin))
    asd = float(np.std(auto_fin))
    mm = float(np.mean(list(manual.values())))
    msd = float(np.std(list(manual.values())))
    diffs = [(info[s]["post_prune_acc"] - manual[s]) * 100 for s in SEEDS]
    noise = float(np.sqrt((asd ** 2 + msd ** 2) / 2))
    amp = abs(am - mm)
    verdict = ("MATCH/EXCEED ✔ within noise" if amp <= noise
               else ("BEATS MANUAL" if am >= mm else "BELOW MANUAL ✘"))

    counts = {"lr_flat_cuts": 0, "lr_spike_cuts": 0, "restarts": 0,
              "early_stops": 0, "silent_flags": 0}
    for s in SEEDS:
        for _, msg in info[s]["actions"]:
            if "lowered step size" in msg and "spiked" not in msg:
                counts["lr_flat_cuts"] += 1
            if "loss spiked" in msg:
                counts["lr_spike_cuts"] += 1
            if "restarting from scratch" in msg:
                counts["restarts"] += 1
            if "STOPPING EARLY" in msg:
                counts["early_stops"] += 1
            if "silent" in msg and "neurons" in msg and "pruned" not in msg:
                counts["silent_flags"] += 1

    lines = [
        "# PHASE 3 REPORT — the auto-pilot trainer (zero settings typed in)",
        "",
        "Command run: `python3 phase3_autopilot.py` — no settings supplied.",
        "",
        "## Gate: matches or beats the manually tuned Phase 2 score "
        "(79.04% ± 1.97%, post-prune, seeds 0/1/2)",
        "",
        "| seed | auto-pilot post-prune | auto-pilot best | manual post-prune | delta |",
        "|------|---------------------:|----------------:|------------------:|------:|",
    ]
    for s in SEEDS:
        lines.append(f"| {s} | {info[s]['post_prune_acc']*100:.2f}% | "
                     f"{info[s]['best_test']*100:.2f}% | {manual[s]*100:.2f}% | "
                     f"{(info[s]['post_prune_acc']-manual[s])*100:+.2f} pts |")
    lines += [
        "",
        f"auto-pilot: **{am*100:.2f}% ± {asd*100:.2f}%** (post-prune)",
        f"manual:     **{mm*100:.2f}% ± {msd*100:.2f}%**",
        f"difference: {amp*100:.2f} pts vs seed noise {noise*100:.2f} pts "
        f"-> {verdict}",
        "",
        "## Action log summary (every action is in runs/phase3/actions_seed*.log, plain English)",
        f"- step-size cuts for flat loss: {counts['lr_flat_cuts']}",
        f"- step-size cuts for loss spikes: {counts['lr_spike_cuts']}",
        f"- explosion restarts (smaller step): {counts['restarts']}",
        f"- early stops (train >> test): {counts['early_stops']}",
        f"- silent-neuron flags: {counts['silent_flags']}",
        f"- silent rules pruned at the end: "
        f"{[len(info[s]['silent_rules_pruned']) for s in SEEDS]}",
        "",
        "## Defaults honesty",
        "Every number the auto-pilot uses is listed in HIDDEN_DEFAULTS.md; "
        "the runner verifies 100% coverage and refuses to run otherwise.",
        "",
        "## Risks / honest notes",
        "- The auto-pilot's budget (12 epochs) is 20% longer than the manual "
        "run's 10 — that runway is a listed default, not a free lunch.",
        "- Action choices come from the SAME measured logs the manual tuning "
        "produced; the watcher simply makes them without a human.",
    ]
    rep = os.path.join(RUNS, "PHASE3-REPORT.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n[wrote] {rep}")
    print("GATE 3:", verdict)
    sys.exit(0 if (am >= mm - 1e-9) else 1)


if __name__ == "__main__":
    main()