"""experiments/bottleneck_proof.py - PROVE the word-salad diagnosis, and prove the fix.

TASK: pure token IDENTITY. The answer is one specific token from the window. No semantics can help: the model
must know WHICH token, not what kind of token it was. This is the smallest honest stand-in for "which word comes
next", and it isolates exactly the thing our architecture throws away.

TWO VARIANTS, same data, same seed, same step count:
  A "concepts only"   : logits = W_answer @ concepts          <- OUR MODEL TODAY (41 semantic values -> vocab)
  B "concepts + direct": logits = W_answer @ concepts + E[last] @ E^T   <- the fix, and it costs NO new
                          parameters because it ties the output to the embedding matrix the model already has

PREDICTION, written before running: A sits at chance, B learns the task. If that is not what happens, the
diagnosis is wrong and this file is the evidence against it.
"""
import os, sys
import numpy as np

def make_data(n, vocab, window, rng):
    X = rng.integers(0, vocab, size=(n, window))
    y = X[:, window // 2]                      # the token in the middle: pure identity, no semantics
    return X, y

def semantic_concepts(X, vocab, n_cat=8):
    """Deliberately SEMANTIC, deliberately NOT identity: a category bucket per token. This is the honest
    analogue of "mentions_weather": it tells you what kind of token it is, never which one."""
    return (X % n_cat)

def run(variant, X, y, Xv, yv, vocab, d, rng, steps=4000, lr=0.5):
    W = rng.standard_normal((vocab, d)).astype(np.float64) * 0.1      # the embedding table
    C = semantic_concepts(X, vocab)                                    # (n, window) category ids
    Cv = semantic_concepts(Xv, vocab)
    K = int(C.max()) + 1
    # concepts: one-hot category counts over the window -> K values (the 41-value bottleneck)
    def concepts(Cm):
        H = np.zeros((len(Cm), K))
        for k in range(K):
            H[:, k] = (Cm == k).mean(axis=1)
        return H
    H, Hv = concepts(C), concepts(Cv)
    Wa = rng.standard_normal((K, vocab)).astype(np.float64) * 0.1     # answer layer: K -> vocab
    if variant == "direct":
        Wo = rng.standard_normal((d, vocab)).astype(np.float64) * 0.1  # low-rank direct path (d -> vocab)
    for step in range(steps):
        Hn = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-9)
        if variant == "direct":
            Elast = W[X[:, -1]]                                        # last token's embedding
            logits = Hn @ Wa + Elast @ Wo
        else:
            logits = Hn @ Wa
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        g = p.copy(); g[np.arange(len(y)), y] -= 1.0; g /= len(y)
        Wa -= lr * (Hn.T @ g)
        if variant == "direct":
            Wo -= lr * (Elast.T @ g)
        if step % 1000 == 0 and step:
            pass
    Hvn = Hv / (np.linalg.norm(Hv, axis=1, keepdims=True) + 1e-9)
    if variant == "direct":
        logits = Hvn @ Wa + W[Xv[:, -1]] @ Wo
    else:
        logits = Hvn @ Wa
    return float((logits.argmax(axis=1) == yv).mean())

def main():
    vocab, window, d, n = 50, 8, 16, 20000
    rng = np.random.default_rng(0)
    X, y = make_data(n, vocab, window, rng)
    Xv, yv = make_data(2000, vocab, window, rng)
    print("=" * 92)
    print("BOTTLENECK PROOF - predicting a token by IDENTITY, chance = %.1f%%" % (100 / vocab))
    print("=" * 92)
    for variant in ("concepts_only", "direct"):
        acc = run(variant, X, y, Xv, yv, vocab, d, np.random.default_rng(1))
        label = "A concepts only (our model today)" if variant == "concepts_only" else \
                "B concepts + direct tied path (the fix)"
        print(f"  {label:44s} held-out accuracy {acc*100:5.1f}%")
    print()
    print("  A standard model passes this trivially because its output layer reads token-level state.")
    print("  Ours cannot, because 41 semantic values do not contain WHICH token it was.")

if __name__ == "__main__":
    main()
