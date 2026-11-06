"""probe_t2_naming.py — pre-run probe: would cosine + support-gated naming
fix the audit on the ALREADY-TRAINED t2-002 (96-epoch) nets? No training.
Naming change: score_k(w) = cosine(row_k, emb_w) with a window-support gate
(>=1% of the windows where the concept fires, on TRAIN) + stopword drop.
Audit = standing meter (resid AUROC >= 0.55 AND logit contrib >= 0.25,
word-presence targets, causal removal = per-word row.emb)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from glassnet import tiny_data
from glassnet.tensor import Tensor
from scripts.run_t2_iter001 import (K, D, WIN, ITERDIR, TinyLM, _auroc,
                                    audit_t2)

STOP = {"i", "my", "me", "we", "you", "your", "the", "a", "an", "to", "up",
        "can", "is", "are", "am", "was", "were", "for", "why", "how", "what",
        "where", "when", "which", "who", "do", "does", "did", "it", "on",
        "in", "at", "and", "or", "of", "with", "about", "want", "need",
        "get", "got", "please", "would", "could", "should", "this", "that",
        "have", "has", "had", "be", "been", "so", "but", "not", "no", "yes",
        "ok", "thanks", "as", "if", "by", "from", "will", "shall", "may",
        "must", "now", "then", "yet", "all", "thy", "thou", "thee", "your",
        "his", "her", "our", "their", "sir", "lord", "duke", "duchess",
        "king", "queen", "s", "t", "d", "ll", "re", "ve", "o", "a"}


def main():
    d = tiny_data.build(win=WIN)
    X = np.asarray(d["Xtr"])
    Xva = np.asarray(d["Xva"])
    net = TinyLM(d["V"], K, D)
    z = np.load(os.path.join(ITERDIR, "final_seed0.npz"))
    net.emb.data[:] = z["emb"]
    net.concept.W.data[:] = z["cW"]
    net.concept.b.data[:] = z["cb"]
    net.answer.W.data[:] = z["aW"]
    net.answer.b.data[:] = z["ab"]

    h_tr = net.forward(X)["concepts"].data          # (182k, 24)
    fired = h_tr > 0.5
    row = net.concept.W.data.T                      # (K, D)
    emb = net.emb.data                              # (V, D)
    en = np.linalg.norm(emb, axis=1) + 1e-9
    rn = np.linalg.norm(row, axis=1) + 1e-9
    cos = (row / rn[:, None]) @ (emb / en[:, None]).T   # (K, V)

    names = []
    for k in range(K):
        E = X[fired[:, k]]
        if len(E) == 0:
            names.append(f"neuron{k}")
            continue
        sup = np.zeros(d["V"])
        np.add.at(sup, E.ravel(), 1.0)
        sup /= len(E)
        cand = [(j, cos[k, j]) for j in range(len(d["vocab"]))
                if d["vocab"][j] not in STOP and sup[j] >= 0.01
                and cos[k, j] > 0]
        cand.sort(key=lambda t: -t[1])
        ws = [d["vocab"][j] for j, _ in cand[:8]]
        names.append("_".join(ws[:6]) if ws else f"neuron{k}")

    rows, n_pass = audit_t2(net, names, d, Xva)
    print(f"cosine+support naming on 96-epoch nets: audit pass {n_pass}/{K}")
    for r in rows:
        print(f"    {r['name'][:42]:42s} auroc={r['auroc']:.2f} "
              f"contrib={r['contrib']:.3f} {r['flag']}")


if __name__ == "__main__":
    main()