"""experiments/bottleneck_proof2.py - CORRECTED: the direct path is POSITION-AWARE.

Task: predict the MIDDLE token of the window BY IDENTITY (chance = 1/vocab = 2.0%). No semantics can help:
the model must know WHICH token, not what kind of token it was. Smallest honest stand-in for "which word next".

  A "concepts only"   : logits = Wa @ concepts                        <- OUR MODEL TODAY (41 semantic values)
  B "position-aware"  : logits = Wa @ concepts + sum_i E[x_i] @ Wo_i  <- the fix, one path PER POSITION

The first version of this test was invalid: its direct path read only the LAST position while the answer was
the middle one, so it could not test anything (my error, recorded). This one gives every position its own path.

PREDICTION, written BEFORE running: A sits at chance, B learns well above chance. If B does not, the diagnosis
(that our answer layer cannot see token identity) is WRONG, and that result stands as the evidence against it.
"""
import numpy as np


def data(n, vocab, window, rng):
    X = rng.integers(0, vocab, size=(n, window))
    return X, X[:, window // 2]


def run(variant, X, y, Xv, yv, vocab, d, rng, steps=800, lr=0.5):
    E = rng.standard_normal((vocab, d)) * 0.1
    K = 8

    def concepts(Xm):
        H = np.zeros((len(Xm), K))
        for k in range(K):
            H[:, k] = (Xm % K == k).mean(axis=1)
        return H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)

    H, Hv = concepts(X), concepts(Xv)
    Wa = rng.standard_normal((K, vocab)) * 0.1
    Wo = rng.standard_normal((X.shape[1], d, vocab)) * 0.1 if variant == "pos" else None
    for _ in range(steps):
        logits = H @ Wa
        Emb = E[X]
        if variant == "pos":
            logits = logits + np.einsum("nwd,wdv->nv", Emb, Wo)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1.0; g /= len(y)
        Wa -= lr * (H.T @ g)
        if variant == "pos":
            Wo -= lr * np.einsum("nwd,nv->wdv", Emb, g)
    logits = Hv @ Wa
    if variant == "pos":
        logits = logits + np.einsum("nwd,wdv->nv", E[Xv], Wo)
    return float((logits.argmax(axis=1) == yv).mean())


def main():
    vocab, window, d = 50, 8, 16
    rng = np.random.default_rng(0)
    X, y = data(8000, vocab, window, rng)
    Xv, yv = data(2000, vocab, window, rng)
    print("=" * 92)
    print("BOTTLENECK PROOF (corrected) - predict the middle token BY IDENTITY, chance = %.1f%%" % (100 / vocab))
    print("=" * 92)
    for v in ("concepts_only", "pos"):
        acc = run(v, X, y, Xv, yv, vocab, d, np.random.default_rng(1))
        label = ("A concepts only (our model today)" if v == "concepts_only"
                 else "B concepts + position-aware direct path (the fix)")
        print(f"  {label:52s} held-out {acc*100:5.1f}%")
    print()
    print("  A at chance and B above it => the diagnosis holds: the answer layer cannot see WHICH token.")
    print("  A above chance would falsify the diagnosis.")


if __name__ == "__main__":
    main()
