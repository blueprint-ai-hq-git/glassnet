"""Does each concept neuron mean what its label says, and does it secretly detect something else?

THE GATE (from the project brief, Phase 2): a concept layer is only honest if a neuron's activation predicts ITS OWN concept
better than any other concept's label. A neuron that tracks a different concept than its name claims is a secret detector, and a
readable model containing secret detectors is not readable.

HOW IT IS MEASURED
    For every neuron i and every concept label j, compute AUROC between activation_i and label_j over the held-out test set.
    That is a K x K matrix. For each neuron:
        honest       argmax_j AUROC(i, j) == i          its name is the best explanation of its own behaviour
        secret       argmax_j AUROC(i, j) != i          it tracks something else, and we say what
        margin       AUROC(i, i) - max_{j != i} AUROC(i, j)  how much its own label wins by
    A high AUROC alone is not enough - exclusivity is the requirement - because a neuron can predict many labels at once and
    still be misnamed.

AUROC WITH TIES
    Rank-based, computed directly (no library), so equal activations cannot flatter a score.
"""

import os
import sys

import numpy as np

PUB = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PUB)

from glassnet.api import GlassNet  # noqa: E402
from glassnet.concepts import NAMES  # noqa: E402


def auroc(scores, labels):
    """Probability a positive outranks a negative, ties counted half. Rank based, no sklearn."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    g = GlassNet.load(os.path.join(PUB, "runs", "library", "model.npz"))
    z = np.load(os.path.join(PUB, "data", "concepts.npz"))
    Cte = z["Cte"]
    X = g.X_test
    out = g.model.forward(X)
    A = np.asarray(out["concepts"].data if hasattr(out["concepts"], "data") else out["concepts"])
    if A.shape[1] != Cte.shape[1]:
        print("SHAPE MISMATCH: %d neuron activations vs %d concept labels" % (A.shape[1], Cte.shape[1]))
        sys.exit(1)

    K = A.shape[1]
    names = [NAMES[i] if i < len(NAMES) else "col%d" % i for i in range(K)]
    print("held-out images: %d   concept neurons: %d   concept labels: %d" % (len(X), K, Cte.shape[1]))
    print()
    print("%-22s %8s %8s %8s  %s" % ("neuron (its name)", "own AUC", "best j", "margin", "verdict"))
    honest = 0
    secrets = []
    for i in range(K):
        aucs = np.array([auroc(A[:, i], Cte[:, j] > 0.5) for j in range(K)])
        own = aucs[i]
        # best label that is NOT its own
        other = np.where(np.arange(K) == i, -np.inf, aucs)
        j_best = int(np.argmax(other))
        margin = own - float(other[j_best])
        is_honest = int(np.argmax(aucs)) == i
        if is_honest:
            honest += 1
        else:
            secrets.append((i, j_best, own, float(other[j_best])))
        print("%-22s %8.3f %8s %8.3f  %s" % (
            names[i][:22], own,
            names[j_best][:8] if not is_honest else "-",
            margin,
            "honest" if is_honest else "SECRET: tracks %s" % names[j_best]))

    print()
    print("RESULT: %d of %d concept neurons mean what their label says." % (honest, K))
    if secrets:
        print("Secret detectors (their own name is NOT the best explanation of their behaviour):")
        for i, j, own, best in secrets:
            print("   %-20s claims %-18s but tracks %-18s (AUC %.3f vs own %.3f)" %
                  (names[i], names[i], names[j], best, own))
    print()
    print("Prediction bar for a readable layer: at least 12 of 14 honest, every margin positive.")


if __name__ == "__main__":
    main()
