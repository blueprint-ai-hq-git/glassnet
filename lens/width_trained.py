"""HOW WIDE MUST THE READABLE LAYER BE?  Measured by TRAINING, not by probing.

WHY THIS EXISTS (2026-11-22 16:10): probing cannot answer the width question. A linear probe on the 32,768-dim state that feeds the
concepts scores 10.4-11.7%, BELOW the 24.0% reached by a probe on the 36 sigmoided concepts themselves - because the concepts are a
NON-LINEAR map (sigmoid) of that state. So width must be measured the only way it can be: TRAIN concept layers of different widths and
compare them.

THE ARCHITECTURE MATTERS HERE, and it is checked, not assumed: in `lens/api.py` the answer head is `Dense(K, V)` applied to `h`, and
`logits = self.answer(h)` with no bypass unless direct=True. So in this model **the concepts ARE the only route to the answer** - which is
exactly the shape the repair needs. Whatever accuracy is reached here, it is reached THROUGH the readable layer.

  task    : next-token prediction on real corpus windows (vocabulary capped at the 2,000 most frequent tokens, rarer tokens mapped to a
            single UNK id so adjacency inside windows is preserved)
  concepts: word-presence concepts, the project's own definition, taken from the K most frequent tokens
  widths  : K = 8, 16, 36, 72

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 accuracy rises with K: the 36-wide layer beats the 8-wide layer.
  P2 the gain shrinks between 36 and 72 (diminishing returns rather than linear improvement).
FALSIFIER: accuracy flat across all four widths -> the COUNT of concepts is not the limiting factor at all, and the bottleneck is which
  concepts exist (their definitions), not how many. That would redirect the repair from width to concept design, which is a different day
  of work.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))   # chat_data lives here
os.chdir(ROOT)
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest"))

from lens.api import Net, train
from lens.concepts import Concept


def build(n_windows, window, vocab_cap, seed=0):
    import chat_data
    d = chat_data.build()
    ids = np.asarray(d["ids"][: n_windows * window + window], dtype=np.int64)
    ids = np.where(ids < vocab_cap, ids, 0)
    n = (len(ids) - 1) // window
    X = ids[: n * window].reshape(n, window)
    y = ids[1: n * window + 1].reshape(n, window)[:, -1]
    return X, y


def make_concepts(X_train, K, window):
    freq = np.bincount(X_train.reshape(-1), minlength=int(X_train.max()) + 1)
    freq[0] = 0                                    # UNK is not a concept
    top = np.argsort(-freq)[:K]
    return [Concept("has_%d" % int(t), [int(t)]) for t in top if freq[t] > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=20000)
    ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--vocab-cap", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--d", type=int, default=64)
    a = ap.parse_args()

    X, y = build(a.windows + 2000, a.window, a.vocab_cap)
    nh = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:nh], y[:nh], X[nh:], y[nh:]
    print("=== how wide must the readable layer be? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("windows %d (train %d, test %d) | window %d | vocabulary capped at %d"
          % (len(X), nh, len(Xte), a.window, a.vocab_cap))
    base = 100.0 * np.bincount(yte, minlength=a.vocab_cap).max() / len(yte)
    print("most-frequent-answer baseline: %.2f%%" % base)
    print()
    print("  %5s  %9s  %10s  %s" % ("K", "params", "test acc", "vs baseline"))
    results = {}
    for K in (8, 16, 36, 72):
        concepts = make_concepts(Xtr, K, a.window)
        net = Net(int(a.vocab_cap), concepts, d=a.d, window=a.window, seed=0)
        train(net, Xtr, ytr, epochs=a.epochs, batch=64, lr=0.3)
        out = net.forward(Xte)
        lg = out["logits"] if isinstance(out, dict) else out
        arr = lg.data if hasattr(lg, "data") else lg
        pred = np.asarray(arr)
        if pred.ndim == 3:
            pred = pred[:, -1, :]
        acc = 100.0 * float((pred.argmax(axis=-1) == yte).mean())
        results[K] = acc
        print("  %5d  %9d  %9.2f%%  %+8.2f" % (len(concepts), net_params(net), acc, acc - base))
    print()
    accs = list(results.values())
    # NO-LEARNING GUARD (2026-11-22 16:14). The 300-window smoke run printed 16.30% for ALL FOUR widths - exactly the most-frequent
    # baseline. That is not "width does not matter", that is a model that has not learned anything, and issuing the width verdict
    # from it would be the same class of error as the four false results caught earlier today. So a run where nothing beats the
    # baseline hands out NO verdict at all.
    if max(accs) - base < 2.0:
        print("RESULT best width %.2f%% is within 2 points of the %.2f%% baseline - the model has NOT learned the task at this budget."
              % (max(accs), base))
        print("OUTCOME: INVALID_INSTRUMENT - nothing learned, so no width verdict can be issued")
        print("WIDTH-TRAINED COMPLETE")
        return
    p1 = results[36] > results[8]
    p2 = (results[72] - results[36]) < (results[36] - results[8])
    spread = max(accs) - min(accs)
    print("RESULT accuracy by width: " + ", ".join("K=%d %.2f%%" % (k, v) for k, v in results.items()))
    print("P1 K=36 beats K=8: %s (%+.2f points)" % ("MET" if p1 else "NOT MET", results[36] - results[8]))
    print("P2 gain shrinks from 36->72: %s (%+.2f vs %+.2f points)"
          % ("MET" if p2 else "NOT MET", results[72] - results[36], results[36] - results[8]))
    if spread < 1.0:
        print("FALSIFIER FIRED: accuracy is flat across widths - the COUNT is not the limit; which concepts exist is.")
        print("OUTCOME: WIDTH_DOES_NOT_MATTER")
    elif p1:
        print("FALSIFIER not fired: width matters, and this model reaches its answer THROUGH the concepts (no bypass).")
        print("OUTCOME: WIDTH_MATTERS_AND_CONCEPTS_CARRY_THE_ANSWER")
    else:
        print("FALSIFIER not fired but width does not help in the predicted direction.")
        print("OUTCOME: WIDTH_UNPREDICTABLE")
    print("WIDTH-TRAINED COMPLETE")


def net_params(net):
    try:
        return sum(int(np.asarray(p.data).size) for p in net.parameters())
    except Exception:
        return -1


if __name__ == "__main__":
    main()
