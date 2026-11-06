"""PHASE 3 — watcher controls: prove all four reactions actually work.

In the real gate run the pilot found nothing wrong (healthy task), so the
watcher never had to act. These controls FORCE each sign artificially and
require the auto-pilot to respond correctly — the same idea as Phase 2's
lie-neuron control: a reflex you can't show acting is not a reflex.

  EXPLODE  : start step size 10.0  -> loss must explode -> restart smaller
  FLAT     : flat_improvement_min = 0.9 (can never "improve") -> lr cuts
  EARLY    : tiny data (2000 rows) -> train >> test -> stop early
  SILENT   : silent_activation_min = 0.45 -> flags + end-pruning of silent rules

Run:  python3 phase3_controls.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.gates import gate1  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts  # noqa: E402
from glassnet.autopilot import AUTO_PILOT_DEFAULTS, train_autopilot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase3")


def main():
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first")
    print("=" * 66)
    gate1()
    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR := os.path.join(HERE, "data"))
    Ctr, Cte, _, _ = load_concepts(DATA_DIR)

    controls = []

    # 1. EXPLODE — absurdly large step size must trigger a restart-smaller
    for name, overrides, expect in [
        ("EXPLODE", {"start_learning_rate": 1e8, "max_epochs": 1},
         "LOSS EXPLODED"),
        ("FLAT", {"flat_improvement_min": 0.9, "flat_patience_windows": 2,
                  "max_epochs": 1}, "lowered step size"),
        ("EARLY", {"batch_size": 64, "eval_every_batches": 4, "max_epochs": 150,
                   "start_learning_rate": 2.0, "concept_loss_alpha": 0.0,
                   "l1_lambda": 0.0, "num_rules": 60,
                   "test_gap_threshold": 0.05, "early_stop_no_improve_evals": 2},
         "STOPPING EARLY"),
        ("SILENT", {"silent_activation_min": 0.45, "max_epochs": 2},
         "silent rule neurons"),
    ]:
        if name == "EARLY":
            n = 300
            _Xtr, _ytr, _Ctr = Xtr[:n], ytr[:n], Ctr[:n]
            net, res = train_autopilot(_Xtr, _ytr, _Ctr, Xte, yte, Cte,
                                       seed=0, log_actions=False, defaults=overrides)
        else:
            net, res = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=0,
                                       log_actions=False, defaults=overrides)
        texts = "\n".join(msg for _, msg in res["actions"]).lower()
        ok = expect.lower() in texts
        extra = ""
        if name == "EXPLODE":
            extra = f" (restarts={res['restarts']})"
        if name == "EARLY":
            extra = f" (best {res['best_test']*100:.1f}%)"
        if name == "SILENT":
            extra = f" (pruned {len(res['silent_rules_pruned'])} rules)"
        controls.append((name, ok, extra))

    lines = ["# PHASE 3 — watcher controls (do the reflexes actually fire?)",
             "",
             "| control | forced sign | expected action | fired? |",
             "|---|---|---|--:|"]
    rows = [
        ("EXPLODE", "step size 1e8 (insane)", "restart smaller", None),
        ("FLAT", "loss can never improve", "lower step size", None),
        ("EARLY", "tiny data, train >> test", "stop early", None),
        ("SILENT", "silent threshold 0.45", "flag + prune silent rules", None),
    ]
    for (name, ok, extra), (c, forced, expected, _) in zip(controls, rows):
        lines.append(f"| {c} | {forced} | {expected} | {'✔' if ok else '✘'}{extra} |")
    all_ok = all(ok for _, ok, _ in controls)
    lines += ["", f"CONTROLS: {'ALL FOUR REFLEXES FIRE ✔' if all_ok else 'FAIL ✘'}", ""]
    rep = os.path.join(RUNS, "CONTROLS.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {rep}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()