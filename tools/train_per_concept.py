"""Last untried lever: ONE DETECTOR PER CONCEPT. Does giving every name its own filter make the names true?

WHY
    Capacity (6 of 14), exclusivity (8 of 14) and decorrelated labels (4 of 9) all failed to raise the honest count, and the sharper
    conditional instrument put the shipped model at 6 of 14 rather than 9. Every one of those arms shared ONE linear pixel-to-concept
    filter, which forces 14 correlated properties into a single basis. This arm removes the sharing: each concept gets its own
    relu(x @ W1_k + b1_k) -> w2_k.

DESIGN
    Matched pair in ONE run, same seed, same budget, same trainer, control first:
        LINEAR      all 14 concepts through the shipped single filter (today's arrangement)
        PER-CONCEPT each concept through its own small detector, same K, same rules, same answer layer
    Scored with BOTH instruments, because they disagree about the shipped model (9 marginal vs 6 conditional) and reporting only the
    flattering one would be the exact dishonesty this project exists to avoid.

NO MACHINE PATHS. Published.
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
from check_concept_honesty_conditional import standardised_betas  # noqa: E402

SEED = 0
EACH = 8


def score(net, X, C):
    A = np.asarray(net.forward(X)["concepts"].data)
    K = A.shape[1]
    B = standardised_betas(A, C)
    marg = cond = 0
    for i in range(K):
        aucs = np.array([auroc(A[:, i], C[:, j] > 0.5) for j in range(K)])
        marg += int(np.argmax(aucs)) == i
        cond += int(np.argmax(np.abs(B[:, i]))) == i
    acc = float((np.argmax(net.forward(X)["logits"].data, axis=1) == net._y_eval).mean()) * 100.0 \
        if hasattr(net, "_y_eval") else None
    return acc, marg, cond, K


def main():
    Xtr, ytr, Xte, yte = load_fmnist(os.path.join(ROOT, "data"))
    Ctr, Cte, _lo, _hi = load_concepts(os.path.join(ROOT, "data"))
    K = Ctr.shape[1]
    print("data: train %s test %s concepts %d" % (Xtr.shape, Xte.shape, K), flush=True)

    def run(tag, defaults):
        net, _info = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=SEED, defaults=defaults, log_actions=False)
        acc = float((np.argmax(net.forward(Xte)["logits"].data, axis=1) == yte).mean()) * 100.0
        A = np.asarray(net.forward(Xte)["concepts"].data)
        B = standardised_betas(A, Cte)
        marg = cond = 0
        for i in range(K):
            aucs = np.array([auroc(A[:, i], Cte[:, j] > 0.5) for j in range(K)])
            marg += int(np.argmax(aucs)) == i
            cond += int(np.argmax(np.abs(B[:, i]))) == i
        npar = sum(p.data.size for p in net.parameters())
        print("%-12s accuracy %.2f%%   marginal %d/%d (%.0f%%)   conditional %d/%d (%.0f%%)   params %d"
              % (tag, acc, marg, K, 100.0 * marg / K, cond, K, 100.0 * cond / K, npar), flush=True)
        return {"accuracy": acc, "marginal": marg, "conditional": cond, "k": K, "params": npar}

    print("\n=== CONTROL: shared linear pixel->concept filter (today's arrangement) ===", flush=True)
    ctrl = run("CONTROL", {})
    print("\n=== TREATMENT: one nonlinear detector per concept (each %d wide) ===" % EACH, flush=True)
    trt = run("PER-CONCEPT", {"concept_each": EACH})

    print("\nVERDICT")
    print("  control      marginal %d/%d  conditional %d/%d  accuracy %.2f%%" %
          (ctrl["marginal"], ctrl["k"], ctrl["conditional"], ctrl["k"], ctrl["accuracy"]))
    print("  per-concept  marginal %d/%d  conditional %d/%d  accuracy %.2f%%" %
          (trt["marginal"], trt["k"], trt["conditional"], trt["k"], trt["accuracy"]))
    print("  bar 12 of 14. Both instruments must move, and accuracy must not collapse, for this to be the lever.")
    os.makedirs(os.path.join(ROOT, "runs", "perconcept"), exist_ok=True)
    with open(os.path.join(ROOT, "runs", "perconcept", "result.json"), "w") as f:
        json.dump({"seed": SEED, "each": EACH, "control": ctrl, "treatment": trt}, f, indent=2)
    print("written runs/perconcept/result.json")


if __name__ == "__main__":
    main()
