"""glassnet.honesty - does each concept neuron mean what its label says?

WHY THIS IS PART OF THE LIBRARY AND NOT A SIDE SCRIPT
    A readable model that prints a confident explanation for a name it does not honour is worse than an opaque model, because it
    misleads the person reading it. This project measured that on its own shipped model: 9 of 14 names were true by the simple test
    and only 6 of 14 under a stricter one. So `honesty()` ships with the library: any user can ask the model which of its own names
    it actually answers to, with two instruments and a warning when they disagree.

THE TWO INSTRUMENTS, AND WHY BOTH
    marginal    For neuron i, find the concept label with the highest AUROC against activation_i. Honest means that label is its own.
                Cheap, and generous: a neuron faithfully detecting a property that usually co-occurs with another gets charged with
                tracking that other property.
    conditional Regress every neuron on ALL labels at once and read the coefficients. A coefficient says how much a label moves the
                neuron with the other labels held fixed, so genuine confusion shows up and innocent co-occurrence does not.
    When the two disagree the conditional answer is the honest one, and reporting only the marginal count would flatter the model.

No sklearn, no library models: rank-based AUROC with tie handling and a ridge solve, implemented here.
"""

import numpy as np


def auroc(scores, labels):
    """Probability a positive outranks a negative, ties counted half. Rank based."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def standardised_betas(A, C):
    """OLS of every neuron on every label, standardised, with a small ridge for the near-collinear case."""
    def z(x):
        x = np.asarray(x, dtype=np.float64)
        return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)

    Az, Cz = z(A), z(C)
    K = Cz.shape[1]
    X = np.hstack([Cz, np.ones((Cz.shape[0], 1))])
    G = X.T @ X + 1e-3 * np.eye(X.shape[1])
    B = np.linalg.solve(G, X.T @ Az)
    return B[:K, :]


def concept_honesty(model, X, C, names=None):
    """Score every concept neuron against every concept label on held-out data. Returns a dict, never prints a claim it cannot back."""
    A = np.asarray(model.forward(X)["concepts"].data, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    K = A.shape[1]
    names = list(names or [("concept_%d" % i) for i in range(K)])[:K]
    B = standardised_betas(A, C)
    rows = []
    marginal = conditional = 0
    for i in range(K):
        aucs = np.array([auroc(A[:, i], C[:, j] > 0.5) for j in range(K)])
        j_m = int(np.argmax(aucs))
        j_c = int(np.argmax(np.abs(B[:, i])))
        marginal += int(j_m == i)
        conditional += int(j_c == i)
        rows.append({
            "neuron": names[i], "own_auroc": round(float(aucs[i]), 4),
            "marginal_best": names[j_m], "marginal_honest": j_m == i,
            "conditional_best": names[j_c], "conditional_honest": j_c == i,
            "conditional_own_beta": round(float(abs(B[i, i])), 4),
        })
    return {"k": K, "marginal_honest": marginal, "conditional_honest": conditional,
            "fraction_marginal": round(marginal / K, 4), "fraction_conditional": round(conditional / K, 4),
            "rows": rows}


def format_report(rep):
    """Plain text a person can read: which names are true, which are not, and what they track instead."""
    lines = []
    lines.append("concept honesty on held-out data: %d of %d names true (marginal), %d of %d under the stricter test (conditional)"
                 % (rep["marginal_honest"], rep["k"], rep["conditional_honest"], rep["k"]))
    if rep["conditional_honest"] < rep["marginal_honest"]:
        lines.append("NOTE: the simple test flatters the model here. Where the two disagree, trust the conditional column - it holds "
                     "the other concepts fixed, so it cannot be fooled by concepts that merely co-occur.")
    for row in rep["rows"]:
        if row["conditional_honest"]:
            lines.append("  %-20s honest (own AUROC %.3f)" % (row["neuron"], row["own_auroc"]))
        else:
            lines.append("  %-20s NOT what its name says - it tracks %s (own AUROC %.3f)"
                         % (row["neuron"], row["conditional_best"], row["own_auroc"]))
    return "\n".join(lines)
