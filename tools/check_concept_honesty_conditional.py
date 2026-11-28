"""INSTRUMENT AUDIT: is the honesty test measuring dishonesty, or measuring correlation in the world?

THE SUSPICION
    Three interventions all failed to raise the honest count - capacity (6 of 14), exclusivity (8 of 14), decorrelated labels
    (4 of 9) - against 9 of 14 for the shipped model. When every fix fails, the measuring instrument is the next suspect.

    The existing gate asks: for neuron i, which label j has the highest AUROC against activation_i? That is a MARGINAL test: it
    compares activation_i with each label on its own. If concept B almost always occurs when concept A occurs, a neuron that
    faithfully detects A will also score well on B - not because it is confused, but because B is nearly implied by A. A marginal
    test cannot tell "this neuron tracks the wrong property" from "this property and the other one appear together".

THE SHARPER TEST
    Regress activation_i on ALL labels at once and read the coefficients. A coefficient says how much a label moves the neuron
    holding the other labels fixed, so genuine confusion shows up as a large coefficient on a label that is not the neuron's own
    name, and honest co-occurrence does not, because the co-occurring label's coefficient collapses once its partner is in the model.

    honest_conditional   largest standardised coefficient belongs to its own label
    honest_marginal      largest AUROC belongs to its own label (the existing gate)

    Two instruments, one checkpoint, same held-out images. If the conditional count is much higher, the old gate was over-strict and
    the reported 9 of 14 UNDERSTATED the model. If it stays near 9, the model really does track other properties.

NO MACHINE PATHS. Published.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from glassnet.api import GlassNet  # noqa: E402
from glassnet.concepts import NAMES  # noqa: E402
from check_concept_honesty import auroc  # noqa: E402


def standardised_betas(A, C):
    """OLS of every neuron on every label, returning standardised coefficients (K_neuron x K_label).

    Standardising both sides makes coefficients comparable across neurons and labels, so "largest" means largest in units of its
    own spread. A ridge term keeps the solve stable when labels are near-collinear - which is exactly our situation, and is also why
    the marginal test misbehaves.
    """
    def z(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)

    Az, Cz = z(A), z(C)
    K = Cz.shape[1]
    X = np.hstack([Cz, np.ones((Cz.shape[0], 1))])
    # ridge with a modest penalty: the labels correlate above 0.8 in places, so plain OLS would explode
    G = X.T @ X + 1e-3 * np.eye(X.shape[1])
    B = np.linalg.solve(G, X.T @ Az)          # (K_label+1, K_neuron)
    return B[:K, :]                            # drop intercept row


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "runs", "library", "model.npz")
    g = GlassNet.load(ckpt)
    z = np.load(os.path.join(ROOT, "data", "concepts.npz"))
    Cte = z["Cte"]
    X = g.X_test
    A = np.asarray(g.model.forward(X)["concepts"].data)
    K = A.shape[1]
    names = [NAMES[i] if i < len(NAMES) else "col%d" % i for i in range(K)]
    print("checkpoint: %s" % ckpt)
    print("held-out images: %d   neurons: %d   labels: %d\n" % (len(X), K, Cte.shape[1]))

    B = standardised_betas(A, Cte)

    print("%-20s %10s %10s   %-22s %-22s" % ("neuron", "marginal", "condit.", "marginal says", "conditional says"))
    marg_honest = cond_honest = 0
    for i in range(K):
        aucs = np.array([auroc(A[:, i], Cte[:, j] > 0.5) for j in range(K)])
        j_marg = int(np.argmax(aucs))
        j_cond = int(np.argmax(np.abs(B[:, i])))
        marg_honest += (j_marg == i)
        cond_honest += (j_cond == i)
        flag_m = "own" if j_marg == i else names[j_marg]
        flag_c = "own" if j_cond == i else names[j_cond]
        print("%-20s %10.3f %10.3f   %-22s %-22s%s" % (
            names[i][:20], aucs[i], abs(B[i, i]), flag_m, flag_c,
            "" if (j_marg == i) == (j_cond == i) else "   <-- instruments disagree"))

    print("\nRESULT")
    print("  marginal (existing gate):    %d of %d honest (%.0f%%)" % (marg_honest, K, 100.0 * marg_honest / K))
    print("  conditional (controls for co-occurrence): %d of %d honest (%.0f%%)"
          % (cond_honest, K, 100.0 * cond_honest / K))
    print("  bar for a readable layer: 12 of 14")
    print("\n  If conditional >> marginal, the published figure understated the model and the gate needs replacing.")


if __name__ == "__main__":
    main()
