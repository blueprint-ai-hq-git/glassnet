"""Is the labelling or the architecture at fault? Train a matched pair and compare honest concept counts.

WHY THIS EXPERIMENT
    Four arms showed a monotone accuracy/readability trade-off: shipped 80.78% / 9 of 14 honest, exclusivity alone 79.68% / 8,
    hidden 64 alone 84.91% / 6, hidden 64 + exclusivity 85.75% / 5. I first blamed the answer layer for having a pixel shortcut,
    then verified that it does not - the shipped model is concept(784->14) -> rules(14->30) -> answer(30->10) with no direct path.
    So the suspect moves upstream to the LABELS: seven of them correlate above 0.80 with each other, and a neuron cannot be pure
    about a property that shares its pixels with another property.

WHAT IT DOES
    Two trainings through the project's own trainer, same seed, same budget, same code path:
        CONTROL    all 14 concepts, exactly today's arrangement.
        DECORR     drop the most redundant concept out of every pair above the threshold (0.80) until no pair exceeds it,
                   then train on the surviving columns and score honesty only on those.
    The control is run first so the comparison is matched - the difference cannot come from the code path, the seed or the budget.

    A smaller concept set is not automatically comparable to 14, so the honest quantity is the FRACTION of surviving names that are
    true, reported beside the accuracy. If decorrelation is the right lever, the fraction rises while accuracy holds.

NO MACHINE PATHS IN THIS FILE - it is published.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from glassnet.data import load as load_fmnist  # noqa: E402
from glassnet.concepts import load as load_concepts, NAMES  # noqa: E402
from glassnet.autopilot import train_autopilot  # noqa: E402
from check_concept_honesty import auroc  # noqa: E402

THRESH = 0.80
SEED = 0


def trim_to_decorrelated(C, thresh=THRESH):
    """Greedily drop the more redundant member of every label pair above `thresh`.

    Redundancy is measured as the mean absolute correlation with the surviving labels, so the concept that carries the least
    unique information goes first. Ties break on index, deterministically, so the run reproduces.
    """
    keep = list(range(C.shape[1]))
    ranks = []
    while True:
        R = np.corrcoef(C[:, keep].T)
        np.fill_diagonal(R, 0.0)
        A = np.abs(R)
        if A.max() <= thresh:
            break
        redundant = A.mean(axis=1)
        # only candidates that take part in an offending pair may be dropped
        offenders = np.where(A.max(axis=1) > thresh)[0]
        drop = int(offenders[np.argmax(redundant[offenders])])
        ranks.append((NAMES[keep[drop]], float(redundant[drop]), float(A[drop].max())))
        keep.pop(drop)
    return keep, ranks


def honest_count(net, X, C):
    """How many neurons answer to their own label better than any other label, using the same rule as the gate."""
    A = np.asarray(net.forward(X)["concepts"].data)
    K = A.shape[1]
    honest = 0
    for i in range(K):
        aucs = np.array([auroc(A[:, i], C[:, j] > 0.5) for j in range(K)])
        if int(np.argmax(aucs)) == i:
            honest += 1
    return honest, K, A


def main():
    Xtr, ytr, Xte, yte = load_fmnist(os.path.join(ROOT, "data"))
    Ctr, Cte, _lo, _hi = load_concepts(os.path.join(ROOT, "data"))
    print("data: train %s  test %s  concepts %d" % (Xtr.shape, Xte.shape, Ctr.shape[1]))

    # ---- CONTROL: all 14, today's arrangement -------------------------------------------------
    print("\n=== CONTROL: all %d concepts ===" % Ctr.shape[1], flush=True)
    net, info = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=SEED, log_actions=False)
    acc_c = float((np.argmax(net.forward(Xte)["logits"].data, axis=1) == yte).mean()) * 100.0
    h_c, k_c, _ = honest_count(net, Xte, Cte)
    print("CONTROL RESULT: accuracy %.2f%%   honest %d of %d   (%.0f%%)"
          % (acc_c, h_c, k_c, 100.0 * h_c / k_c), flush=True)

    # ---- DECORRELATED: drop redundant labels, retrain ----------------------------------------
    keep, ranks = trim_to_decorrelated(Ctr, THRESH)
    print("\n=== DECORR: dropping redundant labels (|r| > %.2f) ===" % THRESH)
    for name, mean_r, max_r in ranks:
        print("   drop %-20s mean|r| %.3f  max|r| %.3f" % (name, mean_r, max_r))
    kept_names = [NAMES[i] for i in keep]
    print("   kept %d: %s" % (len(keep), ", ".join(kept_names)), flush=True)
    Ctr_s, Cte_s = Ctr[:, keep], Cte[:, keep]
    net2, info2 = train_autopilot(Xtr, ytr, Ctr_s, Xte, yte, Cte_s, seed=SEED,
                                  defaults={"num_concepts": len(keep)}, log_actions=False)
    acc_d = float((np.argmax(net2.forward(Xte)["logits"].data, axis=1) == yte).mean()) * 100.0
    h_d, k_d, _ = honest_count(net2, Xte, Cte_s)
    print("DECORR RESULT: accuracy %.2f%%   honest %d of %d   (%.0f%%)"
          % (acc_d, h_d, k_d, 100.0 * h_d / k_d), flush=True)

    # ---- verdict ------------------------------------------------------------------------------
    print("\nVERDICT")
    print("  control  %d of %d honest (%.0f%%)  accuracy %.2f%%" % (h_c, k_c, 100.0 * h_c / k_c, acc_c))
    print("  decorr   %d of %d honest (%.0f%%)  accuracy %.2f%%" % (h_d, k_d, 100.0 * h_d / k_d, acc_d))
    print("  bar 12 of 14 (86%%); decorrelation is the right lever only if the honest FRACTION rises")
    os.makedirs(os.path.join(ROOT, "runs", "decorr"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "decorr", "result.json"), "w") as f:
        json.dump({"threshold": THRESH, "seed": SEED, "dropped": ranks, "kept": kept_names,
                   "control": {"acc": acc_c, "honest": h_c, "k": k_c},
                   "decorr": {"acc": acc_d, "honest": h_d, "k": k_d}}, f, indent=2)
    print("written runs/decorr/result.json")


if __name__ == "__main__":
    main()
