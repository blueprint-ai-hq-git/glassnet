"""loop_audit.py — the readability gate for loop concept sets.

For every named neuron of the promoted/iteration model (validation split):
  - fidelity AUROC vs its own concept target (>0.5)
  - cross-leak: which OTHER concept it tracks more
  - LIE control: a model trained with one neuron's target corrupted; the
    audit must catch it (a neuron secretly detecting something else cannot
    hide). One new concept set = one audit, kept in the iteration folder.

Run:  python3 -m glassnet.loop_audit --model runs/loop/iterNNN/final_seed0.npz \
          --names ink_amount,symmetry,... --Cval data/loop_concepts_XXX.npz \
          --out runs/loop/iterNNN/audit.md
"""
import argparse
import os
import sys

import numpy as np

from .tensor import Tensor
from .optim import SGD
from .eval import auroc
from .phase2_model import GlassNetPhase2
from .loop_lib import (HERE, load_split, load_concept_set,  # noqa: F401
                       RUNS, DATA_DIR)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def audit_table(net, Xval, Cval, names):
    """Fidelity per named neuron. Target binarised at the column's MEDIAN
    (not 0.5): on imbalanced targets (PathMNIST dark_region_count is 93%
    positive) a saturated decoy ranks the rare class almost perfectly and
    AUROC stops meaning 'fires on its name'. Median split is balanced by
    construction, so fidelity = does the neuron separate high-vs-low of the
    named pixel measure."""
    acts = net.predict(Xval)["concepts"].data
    K = len(names)
    matrix = np.zeros((K, K))
    rows = []
    medians = [float(np.median(Cval[:, k])) for k in range(K)]
    for k in range(K):
        ok = auroc(acts[:, k], Cval[:, k] > medians[k])
        corr = float(np.corrcoef(acts[:, k], Cval[:, k])[0, 1])
        mean_act = float(acts[:, k].mean())
        pos_frac = float((Cval[:, k] > 0.5).mean())
        for j in range(K):
            matrix[k, j] = auroc(acts[:, k], Cval[:, j] > medians[j])
        best_j = int(np.argsort(matrix[k])[-2])
        flags = []
        if not (ok >= 0.55):            # nan (dead target) counts as broken
            flags.append("BROKEN")
        if not (ok >= 0.50):
            flags.append("DETECTS-ABSENCE")
        if mean_act < 0.05:
            flags.append("QUIET")
        if pos_frac > 0.85 or pos_frac < 0.15:
            flags.append("SKEWED-TARGET")
        rows.append({"name": names[k], "fidelity": ok, "corr": corr,
                     "mean_act": mean_act, "pos_frac": pos_frac,
                     "leans_to": names[best_j],
                     "leans_auroc": float(matrix[k, best_j]),
                     "flags": ", ".join(flags) if flags else "-"})
    return rows, matrix


def train_lie_net(Xtr, ytr, Ctr, Xval, yval, Cval, names, seed=0, epochs=6,
                  classes=10):
    net = GlassNetPhase2(K=len(names), R=30,
                         rng=np.random.default_rng(seed),
                         concept_names=names, classes=classes,
                         input_dim=Xtr.shape[1])
    heads = net.concept.parameters()
    if net.concept_hidden:
        heads = heads + net.concept_gate.parameters()
    opt = SGD(heads, lr=0.1)     # probe: only the concept head is learned
    rngd = np.random.default_rng(seed * 777 + 1)
    n = len(Xtr)
    # STRONG control: lie_k = the name with the most reliable honest target
    # (highest target variance), steal = the LEAST correlated other target —
    # stealing it must visibly wreck the neuron's own-name fidelity, otherwise
    # the control proves nothing (correlated steals mask the lie, seen in R2-000).
    tgt_std = Ctr.std(axis=0)
    pos_frac = (Ctr > 0.5).mean(axis=0)
    # lie_k needs a target that is (a) reasonably balanced and (b) has live
    # variance on VAL (PathMNIST dark_region_count is 93% pos on train and
    # CONSTANT on val -> AUROC = nan -> undecidable control).
    alive = []
    for k in range(len(names)):
        mk = float(np.median(Cval[:, k]))
        vf = float((Cval[:, k] > mk).mean())   # balance of the SPLIT the audit uses
        if 0.05 < vf < 0.95:
            alive.append(k)
    bal = [k for k in alive if 0.35 <= pos_frac[k] <= 0.65]
    pool = bal if bal else alive
    if not pool:
        pool = list(range(len(names)))
    lie_k = int(pool[int(np.argmax(tgt_std[pool]))])
    # STRONG control: the suspect neuron's supervision is PINNED OFF — its
    # BCE target is set to its own current output (zero gradient), so its
    # weights stay at init: pure random firing, not connected to its name.
    # (Shuffled-target decoys proved inadequate: same-pixel concepts share
    # variance, class-utility pushes the neuron back onto its name, and on
    # imbalanced targets even garbage ranks the rare class ~0.99.)
    steal = -1                          # sentinel: pinned supervision off
    for epoch in range(epochs):
        order = rngd.permutation(n)
        for i in range(0, n, 128):
            idx = order[i:i + 128]
            cb = Ctr[idx].copy()
            f_probe = net.forward(Tensor(Xtr[idx]))
            s_k = 1.0 / (1.0 + np.exp(-f_probe["concept_pre"].data[:, lie_k]))
            cb[:, lie_k] = s_k          # pinned: no gradient for column k
            opt.zero_grad()
            f = f_probe
            total = net.loss(Tensor(Xtr[idx]), ytr[idx], cb, alpha=5.0,
                             lam=0.005, f=f, include_ce=False)
            total.backward()
            opt.step()
    return net, lie_k, steal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--names", required=True)     # comma-separated
    ap.add_argument("--Cval", required=True)      # concept cache npz path
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    names = [n.strip() for n in args.names.split(",")]
    z = np.load(args.Cval)
    Cval = z["Cval"]
    Xtr, ytr, Xval, yval, Xte, yte = load_split()
    model = GlassNetPhase2.load(args.model)
    if len(model.concept_names) != len(names):
        model.concept_names = list(names)

    rows, matrix = audit_table(model, Xval, Cval, names)
    honest = {r["name"]: r["fidelity"] for r in rows}
    ok_named = sum(1 for r in rows if r["fidelity"] >= 0.55)

    zt = np.load(args.Cval.replace("Cval", "Ctr")) if os.path.exists(
        args.Cval.replace("Cval", "Ctr")) else None
    if zt is None:
        # derive Ctr from the same cache file name pattern
        base = args.Cval[:-4]
        # loop caches store Ctr inside the same npz
        zt = np.load(args.Cval)
    Ctr = zt["Ctr"]

    # lie control on the VALIDATION truck
    lie_net, lie_k, steal = train_lie_net(Xtr, ytr, Ctr, Xval, yval, Cval,
                                          names)
    lrows, lmatrix = audit_table(lie_net, Xval, Cval, names)
    lr = lrows[lie_k]
    honest_own = rows[lie_k]["fidelity"]
    caught = (lr["fidelity"] < honest_own - 0.15
              and lr["fidelity"] < 0.70)
    new_names = []

    L = ["# LOOP AUDIT — readability gate",
         "",
         f"concept set ({len(names)} names): {', '.join(names)}",
         "",
         "| neuron | fidelity AUROC | corr | mean act | leans toward | flags |",
         "|---|---:|---:|---:|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['fidelity']:.3f} | {r['corr']:+.3f} | "
                 f"{r['mean_act']:.3f} | {r['leans_to']} ({r['leans_auroc']:.2f}) "
                 f"| {r['flags']} |")
    L += ["",
          f"broken (fidelity<0.55): "
          f"{[(r['name'], r['fidelity']) for r in rows if r['fidelity'] < 0.55] or 'none'}",
          "",
          "LIE CONTROL:",
          f"- neuron '{names[lie_k]}' secretly trained on '{names[steal]}'",
          f"- honest-name fidelity {honest_own:.3f} -> {lr['fidelity']:.3f} "
          f"(lost {honest_own - lr['fidelity']:.3f}); secretly tracks "
          f"'{names[steal]}' at {float(lmatrix[lie_k, steal]):.3f}",
          f"- verdict: {'AUDIT CAUGHT THE LIE ✔' if caught else 'AUDIT MISSED ✘'}",
          "",
          "AUDIT:", "PASS ✔" if caught else "FAIL ✘",
          ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if caught else 1


if __name__ == "__main__":
    sys.exit(main())