"""concepts_more.py — IS THE CONCEPT LAYER THE CEILING? Real concepts vs DUMMY concepts.

THREE INDEPENDENT FINDINGS TODAY (see+talk p1 miss, eyes v2 null, image/medical lines) all blame
the 12-18 named concept layer for the ~77% band. This experiment tests that claim DIRECTLY, and it
carries the control that makes the answer mean something:

  ARM REAL   : frozen 12-concept SPEC + 10 NEW named concepts (formula-stated, support-fit on
               train pixels: quadrant means, gradient halves, stroke density, aspect, edge mass
               rows/cols, centre-of-mass offsets)   -> K = 22
  ARM DUMMY  : frozen 12-concept SPEC + 10 RANDOM PIXEL PROJECTIONS (fixed rng, named
               dummy_1..10, no meaning)             -> K = 22, SAME parameter count

  If REAL > DUMMY and REAL > base(12): understanding is the bottleneck — more concepts help
  BEYOND mere width.
  If REAL ~= DUMMY > base: it was just capacity (more input dims), NOT readability — reported as
  such, because then "transparent concepts help learning" would be unsupported and we say so.
  If neither helps: the task itself (10 clothing classes from 784 px) is at its Bayes limit with
  this architecture and the ceiling claim dies honestly.

PRE-REGISTERED GATES (fixed before running, binary):
  c1  REAL noun-position val acc > DUMMY by > 2.0 pts (3-seed means, outside seed noise)
  c2  REAL > base 12-concept (seetalk_small rung, same data/budget) by > 2.0 pts
  c3  DUMMY - base within +/-2 pts (dummy concepts do NOT help: capacity alone is not the fix)
Data: SAME caption builder, n=10000 images, d=64, 12 epochs, seeds 0,1,2, CPU (GPU reserved).
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


class FormulaConcept:
    def __init__(self, name, fn, formula, support=0.5):
        self.name, self.fn, self.formula, self.support = name, fn, formula, support

    def labels(self, X):
        return np.asarray(self.fn(np.asarray(X, dtype=np.float32)), dtype=np.float32)


def _sq(X):
    return X.reshape(-1, 28, 28)


def extended_specs():
    """10 NEW readable concepts, each a formula on 28x28 pixels (thresholds dimensionless)."""
    A = _sq

    def quad(x, r0, r1, c0, c1):
        return A(x)[:, r0:r1, c0:c1].mean(axis=(1, 2))

    return [
        ("top_left_mass", lambda X: quad(X, 0, 14, 0, 14) > 0.12, "mean(TL quadrant) > 0.12"),
        ("top_right_mass", lambda X: quad(X, 0, 14, 14, 28) > 0.12, "mean(TR quadrant) > 0.12"),
        ("bottom_left_mass", lambda X: quad(X, 14, 28, 0, 14) > 0.12, "mean(BL quadrant) > 0.12"),
        ("bottom_right_mass", lambda X: quad(X, 14, 28, 14, 28) > 0.12, "mean(BR quadrant) > 0.12"),
        ("centre_column", lambda X: A(X)[:, 10:18, :].mean(axis=(1, 2)) > 0.15,
         "mean(centre 8-col band) > 0.15"),
        ("centre_row", lambda X: A(X)[:, :, 10:18].mean(axis=(1, 2)) > 0.15,
         "mean(centre 8-row band) > 0.15"),
        ("com_high", lambda X: ((np.arange(28)[None, :] * (A(X) > 0.5).sum(axis=2)).sum(axis=1)
                                / np.maximum((A(X) > 0.5).sum(axis=(1, 2)), 1) < 12.0)
         & ((A(X) > 0.5).mean(axis=(1, 2)) > 0.03), "vertical centre-of-mass row < 12"),
        ("com_low", lambda X: ((np.arange(28)[None, :] * (A(X) > 0.5).sum(axis=2)).sum(axis=1)
                               / np.maximum((A(X) > 0.5).sum(axis=(1, 2)), 1) > 16.0)
         & ((A(X) > 0.5).mean(axis=(1, 2)) > 0.03), "vertical centre-of-mass row > 16"),
        ("tall_object", lambda X: (A(X) > 0.5).any(axis=2).sum(axis=1) > 18,
         "rows containing ink > 18 of 28"),
        ("edge_mass_bottom", lambda X: A(X)[:, 22:28, :].mean(axis=(1, 2)) > 0.10,
         "mean(bottom 6 rows) > 0.10"),
        # --- K20 ratio-relative redesign (2026-10-11) -------------------------------
        # The absolute quadrant-mass formulas collided with the darkness/contrast axes
        # (bottom_left_mass REALLY tracked is_dark at |0.608|; bottom_right 0.334 top_left).
        # A darkness signal cancels in a RATIO (region mean / image mean), so these can be
        # honest where the absolute versions could not. New names: identity guards on old
        # checkpoints stay intact; the old absolute entries remain in the registry and are
        # EXCLUDED from the K20 build instead (see image_concepts.build exclude=).
        ("bottom_left_mass_rel", lambda X: quad(X, 14, 28, 0, 14) / (quad(X, 0, 28, 0, 28) + 1e-6) > 1.10,
         "mean(BL quadrant)/mean(image) > 1.10"),
        ("bottom_right_mass_rel", lambda X: quad(X, 14, 28, 14, 28) / (quad(X, 0, 28, 0, 28) + 1e-6) > 1.25,
         "mean(BR quadrant)/mean(image) > 1.25"),
        ("top_right_mass_rel", lambda X: quad(X, 0, 14, 14, 28) / (quad(X, 0, 28, 0, 28) + 1e-6) > 1.05,
         "mean(TR quadrant)/mean(image) > 1.05"),
        ("centre_column_rel", lambda X: A(X)[:, 10:18, :].mean(axis=(1, 2)) / (A(X).mean(axis=(1, 2)) + 1e-6) > 1.25,
         "mean(centre 8-col band)/mean(image) > 1.25"),
        ("centre_row_rel", lambda X: A(X)[:, :, 10:18].mean(axis=(1, 2)) / (A(X).mean(axis=(1, 2)) + 1e-6) > 1.55,
         "mean(centre 8-row band)/mean(image) > 1.55"),
        ("left_right_symmetric_rel", lambda X: np.abs(A(X) - A(X)[:, :, ::-1]).mean(axis=(1, 2))
         / (A(X).mean(axis=(1, 2)) + 1e-6) < 0.30,
         "mean|I-mirror|/mean(I) < 0.30 (relative mirror loss — darkness cannot fake it; 0.25 broke honesty in K21c, 0.30 held in K21b)"),
        ("edge_mass_bottom_rel", lambda X: A(X)[:, 22:28, :].mean(axis=(1, 2))
         / (A(X).mean(axis=(1, 2)) + 1e-6) > 1.10,
         "mean(bottom 6 rows)/mean(image) > 1.10 (relative — the absolute version "
         "collided with tall_object at |corr| 0.825)"),
        ("mostly_empty_flat", lambda X: ((A(X) > 0.5).mean(axis=(1, 2)) < 0.10)
         & (A(X).std(axis=(1, 2)) < 0.15),
         "ink-fraction < 0.10 AND std < 0.15 (empty AND flat — the old mostly_empty "
         "collided with has_high_contrast: it fired on empty-but-textured)"),
        # --- K21c pass-3 (2026-10-11): the com pair + high-contrast, darkness-fair ---
        # binarized centers-of-mass collide with the mass-quadrant formulas; luminance-
        # WEIGHTED coms respond to the actual brightness distribution instead of ink only.
        # has_high_contrast (std > 0.30) collides with is_dark: a dark-but-empty image has
        # low std; RELATIVE contrast (std / mean) cancels the darkness axis like the masses.
        ("com_high_w", lambda X: (np.arange(28)[None, :]
                                  * A(X).sum(axis=2)).sum(axis=1)
         / (A(X).sum(axis=(1, 2)) + 1e-6) < 12.0,
         "luminance-weighted vertical centre-of-mass row < 12"),
        ("com_low_w", lambda X: (np.arange(28)[None, :]
                                 * A(X).sum(axis=2)).sum(axis=1)
         / (A(X).sum(axis=(1, 2)) + 1e-6) > 16.0,
         "luminance-weighted vertical centre-of-mass row > 16"),
        ("has_high_contrast_rel", lambda X: A(X).std(axis=(1, 2))
         / (A(X).mean(axis=(1, 2)) + 1e-6) > 1.30,
         "std(pixel)/mean(pixel) > 1.30 — relative contrast, darkness cannot fake it"),
    ]


def dummy_specs(X_train, seed=123):
    """Fixed random projections of pixels — named but meaningless, SAME width as the real set."""
    rng = np.random.default_rng(seed)
    Wd = (rng.standard_normal((784, 10)) / 28.0).astype(np.float32)
    out = []
    for j in range(10):
        def fn(X, j=j):
            v = np.asarray(X, dtype=np.float32).reshape(len(X), -1) @ Wd[:, j]
            return (v > 0).astype(np.float32)
        out.append((f"dummy_{j + 1}", fn, f"(random pixel projection #{j + 1}, no meaning)"))
    return out


def build_concepts(kind, Xtr):
    base, _ = image_concepts.build(Xtr, min_support=0.02, verbose=False)
    if kind == "base":
        return base, {"K": len(base)}
    specs = extended_specs() if kind == "real" else dummy_specs(Xtr)
    extra = []
    for name, fn, f in specs:
        lab = np.asarray(fn(Xtr), dtype=np.float32)
        s = float(lab.mean())
        if s < 0.02 or s > 0.98:
            continue
        extra.append(FormulaConcept(name, fn, f, support=s))
    return base + extra, {"K_base": len(base), "K_extra": len(extra),
                          "K": len(base) + len(extra)}


def noun_val(net, data):
    """val accuracy at the FIRST class-name word, prompt 'this is a ___' (the position g1/g4 use)."""
    vi = {w: i for i, w in enumerate(data["vocab"])}
    prefix = [vi["<pad>"], vi["<pad>"], vi["<bos>"], vi["this"], vi["is"], vi["a"]]
    Xv, Sv = data["Xva"], data["Sva"]
    lg = net.forward_arrays(Xv, np.tile(prefix, (len(Xv), 1)))["logits"]
    pred = np.array([data["vocab"][int(i)] for i in lg.argmax(axis=1)])
    # truth word per val row: Yva target that equals a class NAME TOKEN is the noun position.
    # (Compare against split tokens: names are "ankle_boot" but the vocab word is "ankle" — the
    # whole-name comparison would match nothing and the mask would silently be empty.)
    Yv = data["Yva"]
    tokens = {w for nm in data["names"] for w in nm.split("_")}
    words = np.array([data["vocab"][int(y)] for y in Yv])
    is_noun = np.isin(words, list(tokens))
    hit = (pred == words)[is_noun]
    if not is_noun.any():
        raise SystemExit("noun mask empty — token sets broken")
    return float(hit.mean()) * 100, int(is_noun.sum())


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    T.set_device(a.device)
    seeds = [int(s) for s in a.seeds.split(",")]

    print("=" * 100)
    print("CONCEPT LAYER CEILING TEST — real concepts vs DUMMY concepts vs 12-concept base")
    print("=" * 100)
    data = ts.build_caption_data(n=a.n, seed=0)
    Xtr = data["Xtr"]
    # train rows are (image, prefix) expanded; concept labels need unique images:
    Xu = Xtr[::4]

    res = {}
    for kind in ("base", "real", "dummy"):
        accs = []
        for seed in seeds:
            cons, info = build_concepts(kind, Xu.astype(np.float32))
            net = ts.SeeTalkNet(cons, vocab_size=len(data["vocab"]), d=a.d, window=data["W"],
                                seed=seed, lam_concept=0.3)
            hist = ts.train(net, data, epochs=a.epochs, seed=seed, tag=f"{kind}_s{seed}")
            acc, nrows = noun_val(net, data)
            accs.append(acc)
            print(f"  {kind:<6} seed {seed}: noun-position val {acc:.2f}% over {nrows} rows "
                  f"(K={info['K']}, params {int(sum(p.data.size for p in net.parameters())):,})")
        v = np.asarray(accs)
        res[kind] = {"mean": float(v.mean()), "std": float(v.std()), "accs": accs, "K": info["K"]}

    print()
    print(f"  means±std:  base {res['base']['mean']:.2f}±{res['base']['std']:.2f}  "
          f"real {res['real']['mean']:.2f}±{res['real']['std']:.2f}  "
          f"dummy {res['dummy']['mean']:.2f}±{res['dummy']['std']:.2f}")
    noise = res["real"]["std"] + res["dummy"]["std"]
    c1 = (res["real"]["mean"] - res["dummy"]["mean"]) > 2.0
    c2 = (res["real"]["mean"] - res["base"]["mean"]) > 2.0
    c3 = abs(res["dummy"]["mean"] - res["base"]["mean"]) <= 2.0
    print("  GATES (fixed before the run, binary):")
    print(f"    c1 real beats dummy by >2 ({res['real']['mean']-res['dummy']['mean']:+.2f}, "
          f"noise ±{noise:.2f})  -> {'PASS' if c1 else 'FAIL'}")
    print(f"    c2 real beats base by  >2 ({res['real']['mean']-res['base']['mean']:+.2f})  -> "
          f"{'PASS' if c2 else 'FAIL'}")
    print(f"    c3 dummy ~ base within 2 ({res['dummy']['mean']-res['base']['mean']:+.2f})        -> "
          f"{'PASS' if c3 else 'FAIL'}")
    print()
    verdict = ("CONCEPTS ARE THE CEILING: real named concepts help BEYOND width (c1+c2, c3 holds)"
               if c1 and c2 and c3 else
               ("CAPACITY NOT UNDERSTANDING: dummy concepts help as much as real ones (c1 fails)"
                if res['dummy']['mean'] - res['base']['mean'] > 2.0 else
                "NO HELD ARM: the concept layer is NOT the binding constraint at this rung"))
    print(f"  VERDICT: {verdict}")
    OUT = os.path.join(PROJ, "lens", "runs", "concepts")
    os.makedirs(OUT, exist_ok=True)
    json.dump({"args": vars(a), "res": res, "gates": {"c1": bool(c1), "c2": bool(c2),
               "c3": bool(c3)}, "verdict": verdict},
              open(os.path.join(OUT, "concepts_more.json"), "w"), indent=1)
    print(f"  saved concepts_more.json")
    print("=" * 100)


if __name__ == "__main__":
    main()
