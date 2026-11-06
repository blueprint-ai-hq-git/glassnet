"""PHASE 2 — fidelity audit + lie-neuron control (the honesty gate).

For every named concept neuron (held-out test split):
  - fidelity  = AUROC of the neuron vs its OWN concept target (>0.5)
  - cross-leak: AUROC of the neuron vs every OTHER concept target — if
    another concept beats the namesake, it secretly leans elsewhere
  - flags: BROKEN (fidelity < 0.55), DETECTS-ABSENCE (< 0.50), QUIET
    (mean activation < 0.05)

Lie-neuron control: a model is deliberately trained with the "symmetry"
neuron's target REPLACED by "ink_amount". The audit must catch it — that
is the sanity test proving a neuron secretly detecting something else
cannot hide.

Run:  python3 phase2_audit.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402
from glassnet.concepts import load as load_concepts, NAMES  # noqa: E402
from glassnet.eval import auroc  # noqa: E402
from glassnet.gates import gate1  # noqa: E402
from glassnet.phase2_model import GlassNetPhase2  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs", "phase2")
DATA_DIR = os.path.join(HERE, "data")


def audit_table(net, Xte, Cte):
    """Returns rows + the cross-leak matrix for one model."""
    acts = net.predict(Xte)["concepts"].data
    rows = []
    matrix = np.zeros((net.K, net.K))
    for k in range(net.K):
        ok = auroc(acts[:, k], Cte[:, k] > 0.5)
        corr = float(np.corrcoef(acts[:, k], Cte[:, k])[0, 1])
        mean_act = float(acts[:, k].mean())
        for j in range(net.K):
            matrix[k, j] = auroc(acts[:, k], Cte[:, j] > 0.5)
        best_j = int(np.argsort(matrix[k])[-2])  # best OTHER than namesake
        flags = []
        if ok < 0.55:
            flags.append("BROKEN")
        if ok < 0.50:
            flags.append("DETECTS-ABSENCE")
        if mean_act < 0.05:
            flags.append("QUIET")
        rows.append({"name": NAMES[k], "fidelity": ok, "corr": corr,
                     "mean_act": mean_act, "leans_to": NAMES[best_j],
                     "leans_auroc": float(matrix[k, best_j]),
                     "flags": ", ".join(flags) if flags else "-"})
    return rows, matrix


def render_table(rows, title):
    lines = [f"## {title}", "",
             "| neuron | fidelity AUROC | corr | mean act | leans toward | flags |",
             "|---|---:|---:|---:|---|---|"]
    for r in rows:
        lines.append(f"| {r['name']} | {r['fidelity']:.3f} | {r['corr']:+.3f} | "
                     f"{r['mean_act']:.3f} | {r['leans_to']} ({r['leans_auroc']:.2f}) | "
                     f"{r['flags']} |")
    return lines


def train_lie_net(Xtr, ytr, Ctr, Xte, yte, Cte, epochs=6, seed=0):
    """Train a model whose 'symmetry' neuron is secretly supervised by
    'ink_amount' targets. Everything else identical."""
    net = GlassNetPhase2(K=14, R=20, rng=np.random.default_rng(seed))
    opt = SGD(net.parameters(), lr=0.1)
    rng_data = np.random.default_rng(seed * 777 + 1)
    n = Xtr.shape[0]
    lie_col = NAMES.index("symmetry")      # 1
    ink_col = NAMES.index("ink_amount")    # 0
    for epoch in range(epochs):
        order = rng_data.permutation(n)
        for i in range(0, n, 128):
            idx = order[i:i + 128]
            cb = Ctr[idx].copy()          # copy: we corrupt one column
            cb[:, lie_col] = cb[:, ink_col]  # THE LIE
            opt.zero_grad()
            f = net.forward(Tensor(Xtr[idx]))
            total = net.loss(Tensor(Xtr[idx]), ytr[idx], cb, alpha=0.5,
                             lam=0.02, f=f)
            total.backward()
            opt.step()
    net.save(os.path.join(RUNS, "lie_net.npz"))
    return net


def main():
    os.makedirs(RUNS, exist_ok=True)
    print("=" * 66)
    print("GATE 1 first")
    print("=" * 66)
    gate1()

    Xtr, ytr, Xte, yte = load_pixels(DATA_DIR)
    Ctr, Cte, lo, hi = load_concepts(DATA_DIR)

    # ---- honest model (trained with true targets) -------------------------
    with open(os.path.join(RUNS, "config.json")) as f:
        cfg = json.load(f)
    seed = cfg["seeds"][0] if isinstance(cfg.get("seeds"), list) else 0
    model = GlassNetPhase2.load(os.path.join(RUNS, f"final_seed{seed}.npz"))
    rows, matrix = audit_table(model, Xte, Cte)
    lines = ["# PHASE 2 — fidelity audit (held-out test split)", ""]
    lines += render_table(rows, f"Honest model (seed {seed}, post-prune)")
    honest_fidelity = {r["name"]: r["fidelity"] for r in rows}
    ok_count = sum(1 for r in rows if r["fidelity"] >= 0.55)
    lines += ["",
              f"Named-fidelity pass rate: {ok_count}/{len(rows)} neurons "
              f"(AUROC >= 0.55)", ""]

    # ---- lie-neuron control -------------------------------------------------
    print("[audit] training the LIE model (symmetry secretly = ink_amount) ...")
    lie = train_lie_net(Xtr, ytr, Ctr, Xte, yte, Cte)
    lrows, lmatrix = audit_table(lie, Xte, Cte)
    lines += render_table(lrows, "LIE model (neuron 'symmetry' secretly trained on 'ink_amount')")
    sym = next(r for r in lrows if r["name"] == "symmetry")
    honest_sym = honest_fidelity.get("symmetry")
    sym_k, ink_j = NAMES.index("symmetry"), NAMES.index("ink_amount")
    sym_vs_ink = float(lmatrix[sym_k, ink_j])   # how strongly it tracks the LIE target
    caught = (sym["fidelity"] < 0.60 and sym_vs_ink > 0.85)
    lines += [
        "",
        f"LIE CONTROL verdict: honest model 'symmetry' fidelity = {honest_sym:.3f}; "
        f"lie model fidelity = {sym['fidelity']:.3f}, and it secretly tracks the "
        f"'ink_amount' target at AUROC {sym_vs_ink:.3f}.",
        f"=> AUDIT {'CAUGHT THE LIE ✔' if caught else 'DID NOT CATCH IT ✘'}",
        "",
    ]

    rep = os.path.join(RUNS, "audit.md")
    with open(rep, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n[wrote] {rep}")
    print("AUDIT:", "PASS ✔" if caught else "FAIL ✘")
    sys.exit(0 if caught else 1)


if __name__ == "__main__":
    main()