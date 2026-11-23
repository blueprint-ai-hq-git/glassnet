"""IS THE 24% A LIMITED LAYER, OR ONLY A LIMITED *LINEAR* READOUT?

WHY THIS EXISTS (2026-11-22 15:50): the load-bearing negative result of today says the 36 concepts do not contain the answer, because a
BEST-POSSIBLE LINEAR readout from concepts to the model's own logits reaches only 24.0% agreement while the model's own head reaches
23.2%. That conclusion rests on one assumption: that a linear map is a fair test. If the information is present but arranged in a
curved way, a linear probe would miss it and the layer would be blamed unfairly.

So this tests the same question with a readout that has NO linearity assumption and NO fitting at all:

  1. NEAREST NEIGHBOUR: for each held-out window, find the TRAINING window whose 36-concept vector is closest, and predict that
     training window's own top token. If the concepts encode the answer, similar concept states must belong to similar answers.
  2. A WIDER RIDGE (regularisation 100x smaller) as a second linear reference.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 nearest-neighbour agreement is below 40%, i.e. the 24% was not a linearity artefact.
FALSIFIER: nearest-neighbour agreement at or above 70% -> the information IS in the concepts and only the readout was too simple, which
  would overturn the day's negative result and make the repair a nonlinear head rather than a wider layer.
"""
import os
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest"))
from concept_rank import load_windows


def main():
    ck = "lens/runs/chat/d1024_s1_best.npz"
    net, vocab, X, forward_arrays, backend = load_windows(ck, 1200)
    W = backend.to_cpu(net.answer.W.data).astype(np.float64)
    b = backend.to_cpu(net.answer.b.data).astype(np.float64)
    H, L, C = [], [], []
    for i in range(len(X)):
        act = forward_arrays(net, np.asarray(X[i])[None, :])
        h = backend.to_cpu(act["concepts"][0]).astype(np.float64)
        lg = backend.to_cpu(act["logits"][0]).astype(np.float64)
        H.append(h); L.append(lg); C.append(h @ W + b)
    H = np.array(H); L = np.array(L); C = np.array(C)
    top = L.argmax(axis=1)
    n = len(H); nh = n // 2
    print("=== is the 24%% a linearity artefact? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("windows %d (fit %d, test %d) | concepts %d" % (n, nh, n - nh, H.shape[1]))

    # standardise per concept dimension using TRAIN statistics only
    mu, sd = H[:nh].mean(axis=0), H[:nh].std(axis=0) + 1e-9
    Hs = (H - mu) / sd

    own = 100.0 * (C[nh:].argmax(axis=1) == top[nh:]).mean()
    print("  the model's own concept head:                        %.1f%%" % own)

    # 1. nearest neighbour, no parameters, no linearity assumption
    D = ((Hs[nh:, None, :] - Hs[None, :nh, :]) ** 2).sum(axis=2) if n <= 400 else None
    if D is None:                      # keep memory sane for 1200 windows: chunked
        hits = 0
        for i in range(nh, n):
            d = ((Hs[:nh] - Hs[i]) ** 2).sum(axis=1)
            j = int(np.argmin(d))
            hits += int(top[j] == top[i])
        nn = 100.0 * hits / (n - nh)
    else:
        nn = 100.0 * (top[:nh][D.argmin(axis=1)] == top[nh:]).mean()
    print("  nearest neighbour in concept space (nonlinear):       %.1f%%" % nn)

    # 2. wider ridge, second linear reference
    Hf = np.column_stack([H[:nh], np.ones(nh)])
    A = Hf.T @ Hf + 1e-4 * nh * np.eye(Hf.shape[1])
    B = np.linalg.solve(A, Hf.T @ L[:nh])
    wide = 100.0 * ((np.column_stack([H[nh:], np.ones(n - nh)]) @ B).argmax(axis=1) == top[nh:]).mean()
    print("  wider ridge (100x less regularisation):              %.1f%%" % wide)
    print()
    best = max(nn, wide, own)
    print("RESULT best non-linear / wide readout agreement %.1f%% (own head %.1f%%)" % (best, own))
    print("P1 nearest neighbour below 40%%: %s" % ("MET" if nn < 40 else "NOT MET"))
    if nn >= 70:
        print("FALSIFIER FIRED: the information IS in the concepts; only the readout was too simple.")
        print("OUTCOME: NONLINEAR_READOUT_RECOVERS_ANSWER")
    elif nn < 40:
        print("FALSIFIER not fired: the concepts do not carry the answer even without a linearity assumption.")
        print("OUTCOME: LAYER_REALLY_LACKS_ANSWER")
    else:
        print("FALSIFIER not fired (40-70%%): partial information, the repair needs both a wider layer and a better head.")
        print("OUTCOME: PARTIAL")
    print("CONCEPT-READOUT-NONLINEAR COMPLETE")


if __name__ == "__main__":
    main()
