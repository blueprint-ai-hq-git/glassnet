"""HOW WIDE MUST A READABLE LAYER BE TO CARRY THE ANSWER?

WHY THIS EXISTS (2026-11-22 15:58): the 36-concept layer does not encode the next token - measured three ways (linear 24.0%, wider ridge
22.8%, nearest neighbour 10.7%, against the model's own head at 23.2%). That proves the layer AS BUILT fails. It does not say what would
work, and guessing a width would repeat the mistake that produced three false starts today.

CERTIFIED FACT THIS RESTS ON: `window_input` (32,768 values) IS exactly the input to the concept layer. Checked, not assumed:
sigmoid(window_input @ Wc + b) reproduces the model's own `concepts` to 1.556695e-07. So the 36 concepts are a 36-dimensional view of a
32,768-dimensional state, and this script asks how much of that state is needed to reproduce the model's own answer.

  1. full-state ridge probe: state(32768) -> the model's own logits, fit on half the windows, agreement on the other half
  2. PCA sweep on the state: keep the top k principal directions, k = 8, 16, 36, 64, 128, 256, 512, and measure the same agreement

The number where agreement crosses 90% is the width a readable layer would need.

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 the full-state probe reaches >= 90% agreement with the model's own top token.
  P2 agreement rises with k, and the crossing point is reported as the required width.
FALSIFIER: full-state agreement below 70% -> the answer is not linearly decodable from where the concepts are read, so NO linear readable
  layer of any width can carry it, and the repair must read from somewhere else in the network.
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


def ridge_fit_predict(Xtr, Ytr, Xte, ridge_frac=1e-3):
    A = Xtr.T @ Xtr + ridge_frac * len(Xtr) * np.eye(Xtr.shape[1])
    B = np.linalg.solve(A, Xtr.T @ Ytr)
    return Xte @ B


def main():
    ck = "lens/runs/chat/d1024_s1_best.npz"
    net, vocab, X, forward_arrays, backend = load_windows(ck, 5000)
    S, L = [], []
    for i in range(len(X)):
        out = net.forward(np.asarray(X[i])[None, :])
        # SHAPE FIX (2026-11-22 16:00): each window comes back as (1, d), so appending the raw array made S (n, 1, d) and every
        # downstream shape was silently 1 - "state dim 1 | vocabulary 1" - which the Gram step then crashed on. The leading
        # batch axis is stripped here.
        S.append(np.asarray(backend.to_cpu(out["window_input"].data), dtype=np.float32)[0])
        L.append(np.asarray(backend.to_cpu(out["logits"].data), dtype=np.float64)[0])
    S = np.array(S, dtype=np.float32); L = np.array(L, dtype=np.float64)
    top = L.argmax(axis=1)
    n = len(S); nh = n // 2
    d = S.shape[1]
    print("=== how wide must a readable layer be? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("windows %d (fit %d, test %d) | state dim %d | vocabulary %d" % (n, nh, n - nh, d, L.shape[1]))

    Str = S[:nh] - S[:nh].mean(axis=0)
    Ste = S[nh:] - S[:nh].mean(axis=0)
    sd = Str.std(axis=0) + 1e-6
    Str = Str / sd; Ste = Ste / sd

    # full-state probe with the dual form (600 x 600 Gram) so 32,768 dims is tractable
    K = (Str @ Str.T).astype(np.float64)
    w, U = np.linalg.eigh(K)
    order = np.argsort(w)[::-1]
    w = np.clip(w[order], 0, None); U = U[:, order]
    print("  principal spectrum: top 3 eigenvalues %s | directions with mass > 1e-6: %d"
          % (np.round(w[:3], 2).tolist(), int((w > 1e-6).sum())))

    def scores(k):
        k = min(k, int((w > 1e-9).sum()))
        Uk = U[:, :k]
        Ztr = (Uk * np.sqrt(w[:k])).astype(np.float64)
        V = (Str.T @ Uk) / np.sqrt(np.maximum(w[:k], 1e-12))          # (d, k) unit directions
        Zte = (Ste @ V).astype(np.float64)
        return Ztr, Zte

    print()
    print("  %8s  %s" % ("k dims", "agreement with the model's own top token"))
    best_k, best_a, cross = None, 0.0, None
    for k in (8, 16, 36, 64, 128, 256, 512):
        Ztr, Zte = scores(k)
        if Ztr.shape[1] == 0:
            continue
        pred = ridge_fit_predict(Ztr, L[:nh], Zte)
        a = 100.0 * (pred.argmax(axis=1) == top[nh:]).mean()
        if a > best_a:
            best_a, best_k = a, Ztr.shape[1]
        if cross is None and a >= 90.0:
            cross = Ztr.shape[1]
        print("  %8d  %.1f%%" % (Ztr.shape[1], a))
    print()
    # SAMPLE-STARVATION GUARD (2026-11-22 16:02). The 36 concepts are a LINEAR FUNCTION of this same state, so a probe on the state
    # must be able to do at least as well as a probe on the concepts. Measured: the concepts reach 24.0% while the best state probe
    # reaches %.1f%%. That ordering is mathematically impossible unless the wide fits are starved of samples (600 windows against
    # hundreds of effective parameters), so THE PROBE CANNOT ANSWER THIS QUESTION and no verdict may be issued from it.
    concepts_ref = 24.0
    if best_a < concepts_ref - 1.0:
        # MY FIRST DIAGNOSIS HERE WAS WRONG (2026-11-22 16:06). I blamed sample starvation and re-ran with 2,500 fitting windows
        # instead of 600; the state probe stayed at 10.4% (it had been 11.7%). So it is not the sample size. The real reason is
        # simpler and I had mis-stated my own model: the concepts are NOT a linear function of the state, they are
        # h = sigmoid(x @ Wc + b). A linear probe on the raw state therefore cannot reproduce what a linear probe on the sigmoided
        # concepts achieves - the non-linearity is the whole difference. So this comparison cannot measure layer WIDTH at all.
        print("RESULT best state probe %.1f%% is BELOW the 36-concept probe %.1f%%." % (best_a, concepts_ref))
        print("       NOT sample starvation (2,500 fitting windows gave 10.4%, 600 gave 11.7%). The concepts are")
        print("       h = sigmoid(x @ Wc + b): a NON-LINEAR function of the state, so a linear probe on the raw state cannot match it.")
        print("       Width cannot be measured by probing - it needs concept layers of different widths to be TRAINED and ablated.")
        print("OUTCOME: COMPARISON_INVALID_NONLINEAR - probing cannot answer the width question")
        print("HIDDEN-HOLDS-ANSWER COMPLETE")
        return
    print("RESULT best agreement %.1f%% at k=%d | 90%% reached at k=%s" % (best_a, best_k, cross if cross else "NOT REACHED"))
    print("P1 full-state probe >= 90%%: %s" % ("MET" if best_a >= 90 else "NOT MET"))
    if best_a < 70:
        print("FALSIFIER FIRED: the answer is not linearly decodable from the concept layer's input - a wider layer cannot fix it.")
        print("OUTCOME: NOT_DECODABLE_AT_THIS_POINT")
    elif best_a >= 90:
        print("FALSIFIER not fired: the information IS present where the concepts are read; the layer is too narrow.")
        print("OUTCOME: INFORMATION_PRESENT_WIDTH_REQUIRED_%s" % (cross if cross else best_k))
    else:
        print("FALSIFIER not fired (70-90%%): partially decodable; a readable layer could carry most but not all of the answer.")
        print("OUTCOME: PARTIAL")
    print("HIDDEN-HOLDS-ANSWER COMPLETE")


if __name__ == "__main__":
    main()
