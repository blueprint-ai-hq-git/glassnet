"""experiments/direct_path_real_model.py - the fix, tested in the REAL model with OUR engine.

Same task as bottleneck_proof2.py (predict the middle token BY IDENTITY, chance 2.0%), but now through
lens/model.py and lens/tensor.py - the actual architecture and the actual autograd, not a numpy stand-in.

  direct=False : the model as it is today (answer layer sees only the 41-style concept values)
  direct=True  : the same model plus the position-aware low-rank direct path

PREDICTION, written BEFORE running: direct=False sits near chance, direct=True learns well above it.
"""
import os, sys, time
import numpy as np
ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
import lens.api as api
from lens.concepts import Concept

def data(n, vocab, window, rng):
    X = rng.integers(0, vocab, size=(n, window))
    return X, X[:, window // 2]

def main():
    vocab, window, d, K = 50, 8, 16, 8
    rng = np.random.default_rng(0)
    X, y = data(3000, vocab, window, rng)
    Xv, yv = data(1000, vocab, window, rng)
    concepts = [Concept(f"cat{k}", [t for t in range(vocab) if t % K == k]) for k in range(K)]
    print("=" * 92)
    print("REAL MODEL, REAL ENGINE - predict the middle token BY IDENTITY, chance = %.1f%%" % (100/vocab))
    print("=" * 92)
    for direct in (False, True):
        net = api.Net(vocab, concepts, d=d, window=window, seed=1, device="cpu", lr=0.5,
                      direct=direct, direct_rank=8)
        npar = sum(int(np.prod(p.data.shape)) for p in net.parameters())
        t0 = time.time()
        api.train(net, X, y, epochs=8, batch=64, lr=0.5, seed=1)
        out = net.forward(Xv)
        logits = np.asarray(out["logits"].data)
        acc = float((logits.argmax(axis=1) == yv).mean())
        label = "direct=False (our model today)" if not direct else "direct=True  (position-aware path)"
        print(f"  {label:38s} held-out {acc*100:5.1f}%   params {npar:,}   {time.time()-t0:.0f}s")
    print()
    print("  Prediction was: direct=False near chance, direct=True well above it.")

if __name__ == "__main__":
    main()
