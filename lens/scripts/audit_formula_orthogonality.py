"""audit_formula_orthogonality.py — THE PRE-GATE for every concept set, every line.

A neuron can only be label-honest (strict meaning audit) if its defining formula is the best
predictor among all formulas. If two formulas are collinear, a neuron can track "the other one"
and still pass its own AUROC. So BEFORE any training, the concept SET must be measured:
  gate: max pairwise |corr| between any two concept formulas on the line's validation data <= 0.2
        (strict margin rule: at 0.2 you cannot secretly track another formula without your own
        correlation beating it — the >0.10 honesty margin then has headroom).
USAGE: audit_formula_orthogonality.py [talk|code|medical|fashion|all]
Exit: 0 = set is shippable (all pairs <= 0.2); 2 = collinear pairs exist (named).
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                                # noqa: E402
CODE_DOC = "clean"


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def measure(tag, labels_matrix, names):
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = abs(corr(labels_matrix[:, i], labels_matrix[:, j]))
    np.fill_diagonal(M, 0.0)
    pairs = [(names[i], names[j], float(M[i, j]))
             for i in range(n) for j in range(i + 1, n) if M[i, j] > 0.2]
    pairs.sort(key=lambda t: -t[2])
    print(f"  [{tag}] {n} concepts: {len(pairs)} collinear pair(s) (>0.2; {'PASS set' if not pairs else 'REDESIGN REQUIRED'})")
    for a, b, c in pairs[:12]:
        print(f"      {a:<26} <-> {b:<26} |corr| {c:.2f}")
    return pairs


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fail = False
    if which in ("talk", "all"):
        import chat_data
        d = chat_data.build()
        net = lens.load(os.path.join(PROJ, "lens", "runs", "chat", "t5_ep192_ep192.npz"), device="cpu")
        X, _ = chat_data.make_windows(d["ids"], d["bounds"], d["split"], net.W, which=1)
        X = X[:4000]
        T = np.stack([c.labels(X) for c in net.concepts], axis=1).astype(np.float64)
        p = measure("talk (shipped set)", T, [c.name for c in net.concepts])
        fail |= bool(p)
    if which in ("code", "all"):
        os.environ["CODE_DOC"] = "clean"
        import code_data
        d = code_data.build()
        net = lens.load(os.path.join(PROJ, "lens", "runs", "code", "cclean32k_ep16.npz"), device="cpu")
        X, _ = code_data.make_windows(d["ids"], d["bounds"], d["split"], net.W, which=1)
        X = X[:4000]
        T = np.stack([c.labels(X) for c in net.concepts], axis=1).astype(np.float64)
        p = measure("code (cclean32k set)", T, [c.name for c in net.concepts])
        fail |= bool(p)
    if which in ("medical", "all"):
        import vl_data, vl_data3, image_concepts, image_concepts_adaptive as ic_mod
        try:
            import colour_concepts
        except ImportError:
            colour_concepts = None
        _base = vl_data.prepare_vqa("pathmnist", max_train=None, seed=0, color=True)
        _d = vl_data3.prepare(_base, "medical", seed=0)
        _Xtr = np.asarray(_d["Xtr"], dtype=np.float32)
        _kept, _spec = ic_mod.build(_Xtr, min_support=0.02, verbose=False)
        names = [c.name for c in _kept]
        Q = np.asarray(_d["Qte"][:4000], dtype=np.int64)
        X = np.asarray(_d["Xte"][:4000], dtype=np.float32)
        T = np.stack([ic_mod._target(c, X, Q) for c in _kept], axis=1).astype(np.float64) \
            if hasattr(ic_mod, "_target") else None
        if T is None:
            # formula targets via each concept's own machinery
            T = np.stack([c.fn(X, Q) if hasattr(c, "fn") else None for c in _kept], axis=1)
        if T is None or np.isnan(T).any() or T.shape[1] != len(names):
            print("  [medical] formula-target machinery mismatch — print specs instead")
            print("     concepts:", len(names), [n[:20] for n in names[:6]])
            fail |= True
        else:
            p = measure("medical (mlow set)", T, names)
            fail |= bool(p)
    print("\n" + ("OVERALL: FAIL — collinear pairs exist (named above); set must be redesigned before training"
                  if fail else "OVERALL: PASS — every pair <= 0.2; set is orthogonality-clean"))
    sys.exit(2 if fail else 0)


if __name__ == "__main__":
    main()