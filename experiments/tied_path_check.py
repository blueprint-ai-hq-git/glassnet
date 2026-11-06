"""experiments/tied_path_check.py - does the TIED direct path run, cost what it should, and have correct
gradients? The gradient check is the point: the readout is TIED to the embedding matrix, so if the gradient
does not reach that matrix the tied path trains nothing while looking healthy.

Two bugs in the FIRST version of this file are fixed here, and both are worth keeping in mind:
  1. it wrapped a raw array in a fresh Tensor, which detached it from the graph, so no gradient could
     propagate and every analytic gradient came back as nan - the MODEL was fine, the TEST was wrong;
  2. it printed PASS on nan, because max(0.0, nan) returns 0.0 in Python. A nan must never read as PASS.
"""
import sys, numpy as np
ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
import lens.api as api
import lens.tensor as T
from lens.concepts import Concept


def main():
    # GRADIENTS ARE CHECKED IN FLOAT64. In float32 the finite difference itself is noisy: with a loss near 4.3
    # the float32 resolution is about 4e-7, and at eps=1e-3 that noise shows up as a 7e-3 relative difference,
    # which says nothing about the gradient. Double precision is the standard way to check, and the model still
    # TRAINS in float32 afterwards.
    T.set_dtype(np.float64)
    vocab, window, d, K = 50, 8, 16, 8
    rng = np.random.default_rng(0)
    X = rng.integers(0, vocab, size=(64, window))
    y = X[:, window // 2]
    concepts = [Concept(f"cat{k}", [t for t in range(vocab) if t % K == k]) for k in range(K)]
    net = api.Net(vocab, concepts, d=d, window=window, seed=1, device="cpu", lr=0.5,
                  direct=True, direct_mode="tied")
    npar = sum(int(np.prod(p.data.shape)) for p in net.parameters())
    print(f"  tied mode: params {npar:,}   dmix {tuple(net.dmix.data.shape)}   readout TIED to emb {tuple(net.emb.data.shape)}")

    def fval():
        return float(np.asarray(T.softmax_ce(net.forward(X)["logits"], y).data))

    for p in net.parameters():
        p.grad = None
    ce = T.softmax_ce(net.forward(X)["logits"], y)   # GRAPH-CONNECTED loss, unlike the first version
    ce.backward()
    print(f"  loss {fval():.6f}")
    eps, worst, checked = 1e-5, 0.0, 0
    for name, p in (("dmix", net.dmix), ("emb (tied readout)", net.emb)):
        if p.grad is None:
            print(f"  grad check {name:20s} NO GRADIENT REACHED IT -> FAIL")
            worst = float("inf")
            continue
        idx = tuple(int(np.random.default_rng(7).integers(0, s)) for s in p.data.shape)
        orig = float(p.data[idx]); base = fval()
        p.data[idx] = orig + eps; hi = fval()
        p.data[idx] = orig - eps; lo = fval()
        p.data[idx] = orig
        num = (hi - lo) / (2 * eps)
        ana = float(p.grad[idx])
        rel = abs(num - ana) / (abs(num) + abs(ana) + 1e-12)
        if not np.isfinite(rel):
            worst = float("inf")
        else:
            worst = max(worst, rel)
        checked += 1
        print(f"  grad check {name:20s} numeric {num:+.6f}  analytic {ana:+.6f}  rel diff {rel:.2e}")
    ok = np.isfinite(worst) and worst < 1e-6 and checked == 2
    print(f"  WORST relative difference {worst:.2e} over {checked} parameters  -> {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
