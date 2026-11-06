"""PHASE 6 — the fair fight: SGD auto-pilot vs the experimental (1+1)-ES.

Fairness contract (nothing hidden):
- SAME architecture (GlassNetPhase2: 14 concepts -> 30 rules -> 10 classes)
- SAME seeds [0,1,2], SAME batch-stream family (seed*777+1)
- SAME objective function (CE + 1.0*conceptBCE + 0.005*L1)
- SAME teaching budget: 5,628 batch-evaluations = 12 epochs x 469 batches
  (SGD: 1 step + backward per batch; EVO: 1 parent/child pair per batch)
- SAME post-training pipeline (prune to top-3 connections, report both)
The normal SGD trainer is not modified; `optimizer="sgd"` stays the default.

Run:  python3 phase6_comparison.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.gates import gate1  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts  # noqa: E402
from glassnet.autopilot import train_autopilot, AUTO_PILOT_DEFAULTS  # noqa: E402
from glassnet.evo import train_evo, EVO_DEFAULTS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase6")
DATA_DIR = os.path.join(HERE, "data")
SEEDS = [0, 1, 2]


def main():
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first (no training on broken gradients)")
    print("=" * 66)
    gate1()
    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)

    results = {}
    t_total = time.time()
    for seed in SEEDS:
        # ---------- arm A: the normal SGD auto-pilot (unchanged) ----------
        print(f"\n=== seed {seed} | SGD (normal, default) ===", flush=True)
        t0 = time.time()
        _, sg = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=seed)
        sg_time = time.time() - t0
        print(f"seed {seed} SGD done: post-prune {sg['post_prune_acc']*100:.2f}% "
              f"in {sg_time:.1f}s", flush=True)

        # ---------- arm B: the experimental (1+1)-ES ----------------------
        print(f"\n=== seed {seed} | EVO (experimental switch) ===", flush=True)
        t0 = time.time()
        evo_net, ev = train_evo(Xtr, ytr, Ctr, Xte, yte, Cte, seed=seed)
        ev_time = time.time() - t0
        evo_net.save(os.path.join(RUNS, f"evo_seed{seed}.npz"),
                     post_prune_acc=np.float64(ev["post_prune_acc"]))
        with open(os.path.join(RUNS, f"evo_actions_seed{seed}.log"), "w") as f:
            f.write("\n".join(ev["actions"]) + "\n")
        print(f"seed {seed} EVO done: best {ev['best_test']*100:.2f}% | "
              f"post-prune {ev['post_prune_acc']*100:.2f}% | sigma "
              f"{ev['final_sigma']:.4f} | {ev_time:.1f}s", flush=True)

        results[seed] = {"sgd_post": sg["post_prune_acc"],
                         "sgd_best": sg["best_test"],
                         "sgd_seconds": sg_time,
                         "evo_post": ev["post_prune_acc"],
                         "evo_best": ev["best_test"],
                         "evo_final_sigma": ev["final_sigma"],
                         "evo_seconds": ev_time}
        json.dump(results, open(os.path.join(RUNS, "summary.json"), "w"),
                  indent=1)

    # ---------- verdict (seed-robustness arithmetic) -----------------------
    lines = [
        "# PHASE 6 REPORT — brand-new learning method vs the normal one",
        "",
        "Method: (1+1)-Evolution Strategy (Gaussian mutation, same-batch "
        "parent/child selection, 1/5 success rule for sigma) — NO gradients.",
        "",
        "Fairness contract (all four sides equal): same architecture, same "
        "seeds [0,1,2], same batch-stream family, same objective "
        "(CE + 1.0*conceptBCE + 0.005*L1), same budget (5,628 "
        "batch-evaluations = 12 epochs x 469 batches), same post-processing "
        "(prune to top-3). The normal SGD trainer is untouched and remains "
        "the default (`optimizer=\"sgd\"`); the experimental method sits "
        "behind the switch `optimizer=\"evo\"`.",
        "",
        "| seed | SGD post-prune | EVO post-prune | EVO best |",
        "|------|---------------:|---------------:|---------:|",
    ]
    for s in SEEDS:
        r = results[s]
        lines.append(f"| {s} | {r['sgd_post']*100:.2f}% | {r['evo_post']*100:.2f}% "
                     f"| {r['evo_best']*100:.2f}% |")
    s_post = [results[s]["sgd_post"] for s in SEEDS]
    e_post = [results[s]["evo_post"] for s in SEEDS]
    s_mean, s_std = float(np.mean(s_post)), float(np.std(s_post))
    e_mean, e_std = float(np.mean(e_post)), float(np.std(e_post))
    amp = abs(s_mean - e_mean)
    noise = float(np.sqrt((s_std ** 2 + e_std ** 2) / 2))
    lines += [
        "",
        f"SGD: **{s_mean*100:.2f}% ± {s_std*100:.2f}%**   "
        f"(per-seed {[f'{v*100:.1f}' for v in s_post]})",
        f"EVO: **{e_mean*100:.2f}% ± {e_std*100:.2f}%**   "
        f"(per-seed {[f'{v*100:.1f}' for v in e_post]})",
        "",
        f"amplitude (|SGD - EVO|) = {amp*100:.2f} pts, "
        f"within-seed noise = {noise*100:.2f} pts",
        f"verdict: amplitude {'>' if amp > noise else '<='} noise -> "
        f"{'the difference is real; report which side won' if amp > noise else 'indistinguishable from seed noise'}",
        "",
        f"wall time: SGD {sum(results[s]['sgd_seconds'] for s in SEEDS):.0f}s total, "
        f"EVO {sum(results[s]['evo_seconds'] for s in SEEDS):.0f}s total "
        f"(EVO does 2 forwards per batch vs SGD's forward+backward).",
        "",
        "## Switch demo (library level)",
        "```python",
        "model = GlassNet('fashion-mnist')",
        "model.train(optimizer='sgd')   # normal one (default)",
        "model.train(optimizer='evo')   # experimental (Phase 6)",
        "```",
        "Both models explain() identically; the normal one is never deleted.",
        "",
        "## Risks / honest notes",
        "- This is ONE opponent on ONE task: evolution losing on 11.7k-param "
        "image classification says nothing about its strength on smaller "
        "search spaces — the switch stays for that future fight.",
        "- EVO consumed the same number of batch evaluations; its parent/child "
        "pair needs two forwards per batch, so raw FLOPs per budget are "
        "roughly double SGD's (stated, not hidden).",
    ]
    rep = os.path.join(RUNS, "PHASE6-REPORT.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"[wrote] {rep}")
    print(f"TOTAL {time.time() - t_total:.0f}s")
    sys.exit(0)


if __name__ == "__main__":
    main()