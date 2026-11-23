"""HOW FEW CONCEPTS MUST A PERSON ACTUALLY READ?

WHY THIS EXISTS (2026-11-22 14:32): the certified audit found 12 of 14 concepts honest, but it ALSO measured that concepts are
massively redundant with each other - ink_amount is 92% predictable from the other concepts, quad_UL_ink 87%, ink_cols_nonzero 87%.
If most concepts are near-copies of each other, then a "readable middle layer" of 14 names is asking a person to read 14 things
when 4 or 5 would carry the same information. The original brief asks for exactly this number: HOW MANY RULES MUST A HUMAN READ.

This measures it directly on the real concept labels - no model, no GPU, nothing to tune:

  1. For each concept, how well is it predicted from all the OTHERS (linear, exact least squares)? That is its redundancy.
  2. Greedy forward selection: add the concept that most improves prediction of ALL concepts, until the remaining variance
     explained reaches a bar. The count at that point is the number a person must read.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 a subset of at most 6 concepts predicts every one of the 14 concepts with an average R^2 of at least 0.90.
  P2 the subset is smaller than 14 (obviously), and the number is reported exactly.
FALSIFIER: if 7 or more are needed to reach 0.90, the concepts are genuinely independent, pruning would lose real information, and
  the honest answer to "how many must a human read" is "almost all of them".
"""
import argparse
import json
import os

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT", os.environ.get("GLASSNET_ROOT", "."))


def r2_predict(A, B):
    """R^2 for predicting each column of B from the columns of A, exact least squares."""
    n = len(A)
    Ac = np.column_stack([A, np.ones(n)])
    out = []
    for j in range(B.shape[1]):
        y = B[:, j]
        coef, *_ = np.linalg.lstsq(Ac, y, rcond=None)
        pred = Ac @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        out.append(1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bar", type=float, default=0.90)
    a = ap.parse_args()
    z = np.load(os.path.join(ROOT, "data", "concepts.npz"))
    Y = z["Cte"].astype(np.float64)
    k = Y.shape[1]
    names = list(json.load(open(os.path.join(ROOT, "data", "concepts_disp10.json")))["names"])[:k]
    while len(names) < k:
        names.append("unnamed_%d" % len(names))

    print("=== how few concepts must a person read? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("concepts %d, samples %d" % (k, len(Y)))
    print()
    print("--- how much of each concept is already explained by the other 13? ---")
    red = []
    for j in range(k):
        others = np.delete(Y, j, axis=1)
        r2 = r2_predict(others, Y[:, [j]])[0]
        red.append(r2)
        print("  %-22s R^2 from the others = %.3f" % (names[j], r2))
    print()
    chosen = []
    remaining = list(range(k))
    history = []
    while remaining:
        best, best_score = None, -1.0
        for j in remaining:
            trial = chosen + [j]
            score = r2_predict(Y[:, trial], Y).mean()
            if score > best_score:
                best, best_score = j, score
        chosen.append(best)
        remaining.remove(best)
        history.append((len(chosen), names[best], best_score))
        print("  after %2d concept(s): %-22s average R^2 over all %d concepts = %.4f"
              % (len(chosen), names[best], k, best_score))
        if best_score >= a.bar:
            break
    n_needed = len(chosen)
    print()
    print("RESULT %d of %d concepts are enough to predict every concept at average R^2 >= %.2f"
          % (n_needed, k, a.bar))
    print("       the set a person must read: %s" % ", ".join(names[j] for j in chosen))
    print("P1 at most 6 concepts reach the bar: %s  (%d)" % ("MET" if n_needed <= 6 else "NOT MET", n_needed))
    if n_needed >= 7:
        print("FALSIFIER FIRED: the concepts are genuinely independent; the honest answer is that almost all must be read.")
        print("OUTCOME: CONCEPTS_INDEPENDENT")
    else:
        print("OUTCOME: CONCEPTS_PRUNE_WELL")
    print("CONCEPT-PRUNE COMPLETE")


if __name__ == "__main__":
    main()
