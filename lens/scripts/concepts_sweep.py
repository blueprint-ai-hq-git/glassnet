"""concepts_sweep.py — follow-up to concepts_more: where does WIDTH flatten?

concepts_more (2026-10-08) found the 12-concept bottleneck was UNLOCKED BY WIDTH (+4 pts), with
real-vs-dummy inside noise (+0.77). Product-critical question for GlassNet: how many concepts
should the shipped net have before adding more stops paying? This maps the curve at K=12/22/32/48
with REAL concepts (named, formula-stated, support-fit), and tests one more understandability claim
at the wide end.

PRE-REGISTERED PREDICTIONS (binary, fixed before running):
  s1  K=32 beats K=22 by >2 pts           (width still paying)
  s2  K=48 beats K=32 by >2 pts           (width still paying at 48)
  s3  the K-curve flattens by K=48 at this task's pixel budget (i.e. NOT both s1 and s2 pass)
      — a ceiling claim: 10-way class information in 784 px should saturate the bottleneck
  s4  (understanding probe) at K=48 REAL vs a K=48 DUMMY: |difference| < 2 (i.e. at the wide end,
      named concepts and random projections converge — the base-K gap was starvation, and the
      honest reading then is: concepts cost nothing and buy explainability, not accuracy)
GPU, n=10000, d=64, 12 epochs, seeds 0,1,2.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import image_concepts                                    # noqa: E402
import lens.tensor as T                                  # noqa: E402
import train_seetalk as ts                               # noqa: E402
from concepts_more import FormulaConcept, extended_specs, dummy_specs  # noqa: E402


def more_specs():
    """12 additional readable concepts (on top of base 12 + extended 10) to reach K=32/48."""
    A = lambda X: X.reshape(-1, 28, 28)

    def row_mass(X, r0, r1):
        return A(X)[:, r0:r1, :].mean(axis=(1, 2))

    def col_mass(X, c0, c1):
        return A(X)[:, :, c0:c1].mean(axis=(1, 2))

    return [
        ("top_eighth_mass", lambda X: row_mass(X, 0, 4) > 0.06, "mean(top 4 rows) > 0.06"),
        ("bottom_eighth_mass", lambda X: row_mass(X, 24, 28) > 0.06, "mean(bottom 4 rows) > 0.06"),
        ("left_eighth_mass", lambda X: col_mass(X, 0, 4) > 0.06, "mean(left 4 cols) > 0.06"),
        ("right_eighth_mass", lambda X: col_mass(X, 24, 28) > 0.06, "mean(right 4 cols) > 0.06"),
        ("diagonal_tl_br", lambda X: np.abs(
            (A(X) * np.eye(28)).sum(axis=(1, 2)) / 28 - 0.04) < 0.04, "main-diagonal mean ~ 0.04"),
        ("diagonal_tr_bl", lambda X: np.abs(
            (A(X) * np.fliplr(np.eye(28))).sum(axis=(1, 2)) / 28 - 0.04) < 0.04,
            "anti-diagonal mean ~ 0.04"),
        ("ring_shape", lambda X: (row_mass(X, 4, 8) > 0.12) & (row_mass(X, 20, 24) > 0.12)
         & (row_mass(X, 12, 16) < 0.05), "top+bottom bands inked, centre clear"),
        ("column_shape", lambda X: (col_mass(X, 12, 16) > 0.20) & (col_mass(X, 0, 8) < 0.05),
         "centre column band > 0.20 with clear left edge"),
        ("two_blobs_h", lambda X: (col_mass(X, 4, 12) > 0.08) & (col_mass(X, 16, 24) > 0.08)
         & (col_mass(X, 12, 16) < 0.04), "left+right bands inked, centre gap"),
        ("mass_upper_half", lambda X: row_mass(X, 0, 14) > row_mass(X, 14, 28) + 0.03,
         "upper mean exceeds lower by > 0.03"),
        ("edge_gradient_h", lambda X: np.abs(
            col_mass(X, 14, 28) - col_mass(X, 0, 14)) > 0.03, "|R-L half means| > 0.03"),
        ("sparse_centre_dense_rim", lambda X: (row_mass(X, 10, 18) < 0.05) & (
            ((row_mass(X, 0, 10) + row_mass(X, 18, 28)) / 2) > 0.06),
            "centre rows clear, outer rows inked"),
    ]


def dummy_specs_n(X_train, seed=123, count=10):
    """PARAMETRIC random pixel projections — dummy arms must reach the SAME K as real arms."""
    rng = np.random.default_rng(seed)
    Wd = (rng.standard_normal((784, count)) / 28.0).astype(np.float32)
    out = []
    for j in range(count):
        def fn(X, j=j):
            v = np.asarray(X, dtype=np.float32).reshape(len(X), -1) @ Wd[:, j]
            return (v > 0).astype(np.float32)
        out.append((f"dummy_{j + 1}", fn, f"(random pixel projection #{j + 1}, no meaning)"))
    return out


def band_specs():
    """14 more readable concepts: row/column bands and corner eighths. Needed because the K=48
    arm must be reachable with REAL specs — the first draft's pool topped out at 34 definitions,
    which would have made K=32 and K=48 secretly the same arm (caught by counting before running)."""
    A = lambda X: X.reshape(-1, 28, 28)
    out = []
    edges = [(0, 7), (7, 14), (14, 21), (21, 28)]
    for i, (r0, r1) in enumerate(edges):
        out.append((f"row_band_{i}_mass",
                    (lambda X, r0=r0, r1=r1: A(X)[:, r0:r1, :].mean(axis=(1, 2)) > 0.10),
                    f"mean(rows {r0}-{r1}) > 0.10"))
    for i, (c0, c1) in enumerate(edges):
        out.append((f"col_band_{i}_mass",
                    (lambda X, c0=c0, c1=c1: A(X)[:, :, c0:c1].mean(axis=(1, 2)) > 0.10),
                    f"mean(cols {c0}-{c1}) > 0.10"))
    corners = [(0, 7, 0, 7), (0, 7, 21, 28), (21, 28, 0, 7), (21, 28, 21, 28)]
    for i, (r0, r1, c0, c1) in enumerate(corners):
        out.append((f"corner_{i}_eighth",
                    (lambda X, r0=r0, r1=r1, c0=c0, c1=c1: A(X)[:, r0:r1, c0:c1].mean(axis=(1, 2))
                     > 0.04),
                    f"mean(corner {r0}:{r1}x{c0}:{c1}) > 0.04"))
    out.append(("top_edge_line", lambda X: A(X)[:, 0:2, :].mean(axis=(1, 2)) > 0.12,
                "mean(rows 0-1) > 0.12"))
    out.append(("bottom_edge_line", lambda X: A(X)[:, 26:28, :].mean(axis=(1, 2)) > 0.12,
                "mean(rows 26-27) > 0.12"))
    return out


def build_k(kind, K, Xtr):
    base, _ = image_concepts.build(Xtr, min_support=0.02, verbose=False)
    if K == 12 or kind == "base":
        return base
    if kind == "real":
        pool = extended_specs() + more_specs() + band_specs()
    else:
        pool = dummy_specs_n(Xtr, seed=123, count=max(0, K - 12))
    cons = list(base)
    for name, fn, f in pool:
        if len(cons) >= K:
            break
        lab = np.asarray(fn(Xtr), dtype=np.float32)
        s = float(lab.mean())
        if kind == "real" and (s < 0.02 or s > 0.98):
            continue
        cons.append(FormulaConcept(name, fn, f, support=max(s, 1e-6)))
    return cons


def noun_val(net, data):
    vi = {w: i for i, w in enumerate(data["vocab"])}
    prefix = [vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"], vi["is"], vi["a"]]
    Xv, Sv = data["Xva"], data["Sva"]
    lg = net.forward_arrays(Xv, np.tile(prefix, (len(Xv), 1)))["logits"]
    pred = np.array([data["vocab"][int(i)] for i in lg.argmax(axis=1)])
    tokens = {w for nm in data["names"] for w in nm.split("_")}
    words = np.array([data["vocab"][int(y)] for y in data["Yva"]])
    is_noun = np.isin(words, list(tokens))
    hit = (pred == words)[is_noun]
    return float(hit.mean()) * 100


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()
    T.set_device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]
    data = ts.build_caption_data(n=a.n, seed=0)
    Xu = data["Xtr"][::4].astype(np.float32)

    print("=" * 100)
    print("CONCEPT WIDTH SWEEP — K = 12 / 22 / 32 / 48 (real) + K=48 dummy probe")
    print("=" * 100)
    curves = {}
    for kind, K in (("real", 12), ("real", 22), ("real", 32), ("real", 48), ("dummy", 48)):
        accs = []
        for seed in seeds:
            cons = build_k(kind, K, Xu)
            net = ts.SeeTalkNet(cons, vocab_size=len(data["vocab"]), d=a.d, window=data["W"],
                                seed=seed, lam_concept=0.3)
            ts.train(net, data, epochs=a.epochs, seed=seed, tag=f"{kind}{K}_s{seed}")
            accs.append(noun_val(net, data))
            print(f"  {kind}{K:<3} seed {seed}: noun val {accs[-1]:.2f}% (K used {len(cons)})")
        v = np.asarray(accs)
        curves[f"{kind}{K}"] = {"mean": float(v.mean()), "std": float(v.std()), "accs": accs}

    def m(k):
        return curves[k]["mean"]

    s1 = m("real32") - m("real22") > 2.0
    s2 = m("real48") - m("real32") > 2.0
    s3 = not (s1 and s2)
    s4 = abs(m("real48") - m("dummy48")) < 2.0
    print()
    print(f"  curve: K12 {m('real12'):.2f}  K22 {m('real22'):.2f}  K32 {m('real32'):.2f}  "
          f"K48 {m('real48'):.2f}   dummy48 {m('dummy48'):.2f}")
    print(f"    s1 K32>K22 by 2: {m('real32')-m('real22'):+.2f} -> {'PASS' if s1 else 'FAIL'}")
    print(f"    s2 K48>K32 by 2: {m('real48')-m('real32'):+.2f} -> {'PASS' if s2 else 'FAIL'}")
    print(f"    s3 flattens by 48 (not both):        -> {'PASS' if s3 else 'FAIL'}")
    print(f"    s4 real48 vs dummy48 within 2: {abs(m('real48')-m('dummy48')):.2f} -> "
          f"{'PASS' if s4 else 'FAIL'}")
    OUT = os.path.join(PROJ, "lens", "runs", "concepts")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"args": vars(a), "curves": curves,
               "gates": {"s1": bool(s1), "s2": bool(s2), "s3": bool(s3), "s4": bool(s4)}},
              open(os.path.join(OUT, "concepts_sweep.json"), "w"), indent=1)
    print(f"  saved concepts_sweep.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
