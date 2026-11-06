"""glassnet.concepts — measurable pixel concepts for Fashion-MNIST.

Every concept is computed from the pixels by a fixed, documented formula
(see CONCEPTS.md). Each column is min-max normalised using TRAIN statistics
ONLY — the test set never leaks into the concept scale.

The Phase-2 set (NAMES / compute_raw / load) is unchanged so archived runs
stay reproducible; FEATURE_FUNCS extends it with named candidates so the
autonomous loop can build new concept SETS (compute_selected).
"""
import os

import numpy as np

from . import data as data_mod

# ------------------------------------------------------------------ helpers
K = 14
NAMES = [
    "ink_amount", "symmetry", "top_heavy", "horizontal_edges", "vertical_edges",
    "quad_UL_ink", "quad_UR_ink", "quad_LL_ink", "quad_LR_ink", "center_ink",
    "left_heavy", "border_ink", "tall_shape", "wide_shape",
]
DESC = {
    "ink_amount":       "how much dark ink (mean pixel intensity)",
    "symmetry":         "mirror-symmetric left/right (1 - 2*mean |X - fliplr(X)|)",
    "top_heavy":        "more ink in the top half than the bottom half",
    "horizontal_edges": "mean |horizontal pixel difference| — vertical stripes",
    "vertical_edges":   "mean |vertical pixel difference| — horizontal stripes",
    "quad_UL_ink":      "ink in the upper-left quarter",
    "quad_UR_ink":      "ink in the upper-right quarter",
    "quad_LL_ink":      "ink in the lower-left quarter",
    "quad_LR_ink":      "ink in the lower-right quarter",
    "center_ink":       "ink in the middle 14x14 block",
    "left_heavy":       "more ink on the left half than the right half",
    "border_ink":       "ink touching the outer frame of the image",
    "tall_shape":       "height of the silhouette (fraction of 28 rows with ink)",
    "wide_shape":       "width of the silhouette (fraction of 28 columns with ink)",
    # ---- loop candidates ----
    "diag_edges":       "mean |diagonal pixel differences| (both directions)",
    "edge_balance":     "horizontal_edges / (vertical_edges + 1e-6) — orientation ratio",
    "ink_contrast":     "p95 - p05 of pixel values — outline sharpness",
    "outline_ratio":    "boundary pixels / ink pixels — shape complexity",
    "top_band_ink":     "ink in the top 7 rows (neckline zone)",
    "bottom_ink":       "ink in the bottom 14 rows (hem/sole zone)",
    "row_profile_std":  "std of row ink sums — silhouette profile complexity",
    "quad_top_diff":    "(UL - UR + 1)/2 — upper-left vs upper-right balance",
    "quad_bottom_diff": "(LL - LR + 1)/2 — lower-left vs lower-right balance",
    "lowres_symmetry":  "symmetry measured on a 14x14 downsampled image",
    # ---- loop widening: region variants (same meaning, different field) ----
    "symmetry_top":     "left/right symmetry measured on rows 0-14 only",
    "symmetry_bottom":  "left/right symmetry measured on rows 14-28 only",
    "symmetry_center":  "left/right symmetry measured on the middle 14x14 block",
    "left_heavy_top":   "left/right ink balance on rows 0-14 (collar zone)",
    "left_heavy_bottom":"left/right ink balance on rows 14-28 (hem/sole zone)",
    "center_ink_top":   "ink in the central 7-wide column band, rows 0-14",
    "center_ink_bottom":"ink in the central 7-wide column band, rows 14-28",
    # ---- Rung-2 texture concepts (PathMNIST): documented formulas ----
    "mean_gradient":    "overall texture wiggle: mean(|dx|+|dy|)/2 across pixels",
    "dark_region_count":"fraction of 4x4 blocks whose mean ink > 0.5 — dark speckles/dots",
    "light_region_count":"fraction of 4x4 blocks whose mean ink < 0.2 — bright gaps",
    "corner_ink":       "mean ink of the four 5x5 image corners — frame/background presence",
    "streak_ink":       "longest horizontal run of ink>0.25 pixels over all rows, /28 — streaks",
}


def _f(X):
    """Feature functions. X: (N, 28, 28) floats in [0, 1]. Return (N,) arrays."""
    top = X[:, :14, :].mean(axis=(1, 2))
    bot = X[:, 14:, :].mean(axis=(1, 2))
    left = X[:, :, :14].mean(axis=(1, 2))
    right = X[:, :, 14:].mean(axis=(1, 2))
    mask = X > 0.25
    return {
        "ink_amount": X.mean(axis=(1, 2)),
        "symmetry": 1.0 - 2.0 * np.abs(X - X[:, :, ::-1]).mean(axis=(1, 2)),
        "top_heavy": (top - bot + 1.0) / 2.0,
        "horizontal_edges": np.abs(X[:, :, 1:] - X[:, :, :-1]).mean(axis=(1, 2)),
        "vertical_edges": np.abs(X[:, 1:, :] - X[:, :-1, :]).mean(axis=(1, 2)),
        "quad_UL_ink": X[:, :14, :14].mean(axis=(1, 2)),
        "quad_UR_ink": X[:, :14, 14:].mean(axis=(1, 2)),
        "quad_LL_ink": X[:, 14:, :14].mean(axis=(1, 2)),
        "quad_LR_ink": X[:, 14:, 14:].mean(axis=(1, 2)),
        "center_ink": X[:, 7:21, 7:21].mean(axis=(1, 2)),
        "left_heavy": (left - right + 1.0) / 2.0,
        "border_ink": (X[:, 0, :].mean(axis=1) + X[:, -1, :].mean(axis=1)
                       + X[:, :, 0].mean(axis=1) + X[:, :, -1].mean(axis=1)) / 4.0,
        "tall_shape": mask.any(axis=2).sum(axis=1) / 28.0,
        "wide_shape": mask.any(axis=1).sum(axis=1) / 28.0,
        # ---- loop candidates ----
        "diag_edges": 0.5 * (np.abs(X[:, 1:, 1:] - X[:, :-1, :-1]).mean(axis=(1, 2))
                             + np.abs(X[:, 1:, :-1] - X[:, :-1, 1:]).mean(axis=(1, 2))),
        "edge_balance": _f_edge_balance(X),
        "ink_contrast": np.quantile(X, 0.95, axis=(1, 2)) - np.quantile(X, 0.05, axis=(1, 2)),
        "outline_ratio": _f_outline_ratio(mask),
        "top_band_ink": X[:, :7, :].mean(axis=(1, 2)),
        "bottom_ink": bot,
        "row_profile_std": X.sum(axis=2).std(axis=1),
        "quad_top_diff": (X[:, :14, :14].mean(axis=(1, 2))
                          - X[:, :14, 14:].mean(axis=(1, 2)) + 1.0) / 2.0,
        "quad_bottom_diff": (X[:, 14:, :14].mean(axis=(1, 2))
                             - X[:, 14:, 14:].mean(axis=(1, 2)) + 1.0) / 2.0,
        "lowres_symmetry": 1.0 - 2.0 * np.abs(X[:, ::2, ::2] - X[:, ::2, ::-2]).mean(axis=(1, 2)),
        # ---- loop widening: region variants ----
        "symmetry_top": 1.0 - 2.0 * np.abs(X[:, :14, :] - X[:, :14, ::-1]).mean(axis=(1, 2)),
        "symmetry_bottom": 1.0 - 2.0 * np.abs(X[:, 14:, :] - X[:, 14:, ::-1]).mean(axis=(1, 2)),
        "symmetry_center": 1.0 - 2.0 * np.abs(X[:, 7:21, 7:21]
                                              - X[:, 7:21, 7:21][:, :, ::-1]).mean(axis=(1, 2)),
        "left_heavy_top": ((X[:, :14, :14].mean(axis=(1, 2))
                            - X[:, :14, 14:].mean(axis=(1, 2)) + 1.0) / 2.0),
        "left_heavy_bottom": ((X[:, 14:, :14].mean(axis=(1, 2))
                               - X[:, 14:, 14:].mean(axis=(1, 2)) + 1.0) / 2.0),
        "center_ink_top": X[:, 0:14, 7:21].mean(axis=(1, 2)),
        "center_ink_bottom": X[:, 14:28, 7:21].mean(axis=(1, 2)),
        # ---- Rung-2 texture formulas ----
        "mean_gradient": 0.5 * (np.abs(X[:, :, 1:] - X[:, :, :-1]).mean(axis=(1, 2))
                                + np.abs(X[:, 1:, :] - X[:, :-1, :]).mean(axis=(1, 2))),
        "dark_region_count": _f_block_fraction(X, 0.5, True),
        "light_region_count": _f_block_fraction(X, 0.2, False),
        "corner_ink": ((X[:, :5, :5].mean(axis=(1, 2))
                        + X[:, :5, -5:].mean(axis=(1, 2))
                        + X[:, -5:, :5].mean(axis=(1, 2))
                        + X[:, -5:, -5:].mean(axis=(1, 2))) / 4.0),
        "streak_ink": _f_streak(X, 0.25),
    }


def _f_block_fraction(X, thresh, above):
    """Fraction of 4x4 blocks whose mean ink is above/below `thresh`.
    Blocks: 7x7 grid of 4x4 cells over the 28x28 image (49 blocks)."""
    B = X.reshape(len(X), 7, 4, 7, 4).mean(axis=(2, 4))
    return ((B > thresh) if above else (B < thresh)).mean(axis=(1, 2))


def _f_streak(X, thresh):
    """Longest horizontal run of pixels > thresh, over all rows, /28 (0..1)."""
    r = (X > thresh)
    best = np.zeros(len(X))
    for row in range(28):
        for start in range(28):
            for end in range(start + 1, 29):
                seg = r[:, row, start:end].all(axis=1)
                best = np.maximum(best, seg.astype(np.float32) * (end - start))
    return best / 28.0


def _f_edge_balance(X):
    h = np.abs(X[:, :, 1:] - X[:, :, :-1]).mean(axis=(1, 2))
    v = np.abs(X[:, 1:, :] - X[:, :-1, :]).mean(axis=(1, 2))
    return h / (v + 1e-6)


def _f_outline_ratio(mask):
    m = mask.astype(np.uint8)
    # eroded by all 4 neighbours (interior pixels)
    ero = (m[:, 1:-1, 1:-1] * m[:, 1:-1, 2:] * m[:, 1:-1, :-2]
           * m[:, 2:, 1:-1] * m[:, :-2, 1:-1])
    boundary = m.sum(axis=(1, 2)) - ero.sum(axis=(1, 2))
    return boundary / (m.sum(axis=(1, 2)) + 1e-6)


def _stack(X, names):
    f = _f(X)
    return np.stack([f[n] for n in names], axis=1).astype(np.float32)


def compute_raw(X_flat):
    """Phase-2 set, unchanged: (N, 784) -> (N, 14) raw features."""
    return _stack(X_flat.reshape(-1, 28, 28), NAMES)


def normalize(raw, lo, hi):
    return np.clip((raw - lo) / (hi - lo + 1e-9), 0.0, 1.0).astype(np.float32)


def compute(X_flat, lo=None, hi=None):
    """Phase-2 set: concepts for a pixel batch, normalised [0,1]."""
    raw = compute_raw(X_flat)
    if lo is None or hi is None:
        lo, hi = raw.min(axis=0), raw.max(axis=0)
    return normalize(raw, lo, hi), np.asarray(lo, np.float32), np.asarray(hi, np.float32)


def compute_selected(X_flat, names, lo=None, hi=None):
    """Any named set (Phase-2 names + candidates). Returns (C, lo, hi)."""
    names = list(names)
    raw = _stack(X_flat.reshape(-1, 28, 28), names)
    if lo is None or hi is None:
        lo, hi = raw.min(axis=0), raw.max(axis=0)
    return normalize(raw, lo, hi), np.asarray(lo, np.float32), np.asarray(hi, np.float32)


def separation_score(C, y):
    """Between/within class sum-of-squares per column (decision data only!)."""
    cols = []
    for k in range(C.shape[1]):
        bss = wss = 0.0
        grand = C[:, k].mean()
        for c in np.unique(y):
            m = C[y == c, k]
            bss += len(m) * (m.mean() - grand) ** 2
            wss += ((m - m.mean()) ** 2).sum()
        cols.append(bss / (wss + 1e-9))
    return np.asarray(cols)


def load(data_dir, force=False):
    """Phase-2 set for both splits (cached; archive format untouched)."""
    cache = os.path.join(data_dir, "concepts.npz")
    if os.path.exists(cache) and not force:
        z = np.load(cache)
        return z["Ctr"], z["Cte"], z["lo"], z["hi"]
    Xtr, _, Xte, _ = data_mod.load(data_dir)
    Ctr, lo, hi = compute(Xtr)
    Cte, _, _ = compute(Xte, lo, hi)
    np.savez(cache, Ctr=Ctr, Cte=Cte, lo=lo, hi=hi)
    print(f"[concepts] cached -> {cache}", flush=True)
    return Ctr, Cte, lo, hi


def main():
    import os as _os
    here = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    data_dir = _os.path.join(here, "data")
    Xtr, ytr, Xte, yte = data_mod.load(data_dir)
    Ctr, Cte, lo, hi = load(data_dir, force=True)
    lines = [
        "# concept-class stats (generated by glassnet/concepts.py)",
        "",
        "Rows = concepts (normalised [0,1]), columns = Fashion-MNIST classes.",
        "Classes: 0 t-shirt/top 1 trouser 2 pullover 3 dress 4 coat 5 sandal "
        "6 shirt 7 sneaker 8 bag 9 ankle boot.",
        "",
        "| concept | " + " | ".join(str(c) for c in range(10)) + " |",
        "|---|---" + "---|" * 10,
    ]
    for k, name in enumerate(NAMES):
        means = [f"{Ctr[ytr == c, k].mean():.2f}" for c in range(10)]
        lines.append(f"| {name} | " + " | ".join(means) + " |")
    lines.append("")
    for k, name in enumerate(NAMES):
        lines.append(f"- {name}: {lo[k]:.4f} .. {hi[k]:.4f}")
    doc = _os.path.join(here, "docs", "concept-class-stats.md")
    _os.makedirs(_os.path.dirname(doc), exist_ok=True)
    with open(doc, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[concepts] wrote {doc}")


if __name__ == "__main__":
    main()