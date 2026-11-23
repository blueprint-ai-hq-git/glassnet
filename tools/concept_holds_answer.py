"""DOES THE CONCEPT LAYER CONTAIN THE ANSWER, EVEN IF THE CURRENT HEAD DOES NOT READ IT?

WHY THIS EXISTS (2026-11-22 15:40): measured on three checkpoints, the concept route with the model's own answer head reproduces the
model's answer only 5.5-18.0% of the time - the readable path is a side channel. Before rebuilding the answer head, one question decides
what the repair must be:

    is the information even THERE in the 36 concept activations, or does the layer itself lack it?

So a NEW linear readout is FITTED from the 36 concepts to the model's own logits, on one half of the windows, and agreement is measured on
the other half. This is deliberately generous: it is allowed to use the model's own logits as its target, and it has least squares to find
the best linear map. If even that cannot reproduce the model's answers, no head change can rescue the readable path.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 a fitted concept->logits readout reaches >= 70% agreement with the model's own top token on held-out windows. (The information is
     present; the repair is a head problem.)
FALSIFIER: agreement below 40% -> the 36 concepts do not contain the answer, and the LAYER must change, not the head. That would mean the
  readable middle layer needs to be wider or differently built, not merely re-wired.
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
    H, L, C = [], [], []
    for i in range(len(X)):
        act = forward_arrays(net, np.asarray(X[i])[None, :])
        H.append(backend.to_cpu(act["concepts"][0]).astype(np.float64))
        L.append(backend.to_cpu(act["logits"][0]).astype(np.float64))
        W = backend.to_cpu(net.answer.W.data).astype(np.float64)
        b = backend.to_cpu(net.answer.b.data).astype(np.float64)
        C.append((H[-1] @ W + b))
    H = np.array(H); L = np.array(L); C = np.array(C)
    n = len(H)
    nh = n // 2
    print("=== does the concept layer contain the answer? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("windows %d (fit %d, test %d) | concepts %d | vocabulary %d" % (n, nh, n - nh, H.shape[1], L.shape[1]))

    top_full = L.argmax(axis=1)
    # how does the model's OWN head do, for reference
    agree_own = 100.0 * (C[nh:].argmax(axis=1) == top_full[nh:]).mean()
    print("  the model's own concept head agrees with its own answer: %.1f%% (this is the 18%% from the ablation, re-measured)" % agree_own)

    # fit a fresh linear readout concepts -> the model's own logits
    Hf = np.column_stack([H[:nh], np.ones(nh)])
    ridge = 1e-2 * nh
    A = Hf.T @ Hf + ridge * np.eye(Hf.shape[1])
    B = np.linalg.solve(A, Hf.T @ L[:nh])
    pred = np.column_stack([H[nh:], np.ones(n - nh)]) @ B
    agree_fit = 100.0 * (pred.argmax(axis=1) == top_full[nh:]).mean()
    # graded: how often is the model's own top token in the fitted readout's top 10
    top10 = np.argsort(-pred, axis=1)[:, :10]
    in10 = 100.0 * np.mean([top_full[nh + i] in top10[i] for i in range(n - nh)])
    print("  a NEW FITTED linear readout from the 36 concepts reaches: %.1f%% agreement (top-10 hit %.1f%%)"
          % (agree_fit, in10))
    print()
    print("RESULT fitted concept readout agreement %.1f%% vs the model's own head %.1f%%" % (agree_fit, agree_own))
    print("P1 fitted readout >= 70%%: %s" % ("MET" if agree_fit >= 70 else "NOT MET"))
    if agree_fit < 40:
        print("FALSIFIER FIRED: the 36 concepts do not contain the answer - the LAYER must change, not the head.")
        print("OUTCOME: LAYER_LACKS_ANSWER")
    elif agree_fit >= 70:
        print("FALSIFIER not fired.")
        print("OUTCOME: INFORMATION_PRESENT_HEAD_IS_THE_PROBLEM")
    else:
        print("FALSIFIER not fired (40-70%%): the information is partly there; a head repair would recover part of the answer.")
        print("OUTCOME: PARTIAL_INFORMATION")
    print("CONCEPT-HOLDS-ANSWER COMPLETE")


if __name__ == "__main__":
    main()
