"""image_concepts_adaptive.py — concepts whose thresholds are FITTED to the dataset.

THE DIAGNOSED PROBLEM
---------------------
`image_concepts.SPEC` uses FIXED thresholds chosen for Fashion-MNIST: "is_bright = mean(pixel) > 0.50".
On PathMNIST (colon histology) the same test is true for **93.5%** of images, `is_dark` for **0.0%**,
and `has_low_contrast` for **82.8%**. Three of twelve concepts were effectively constants.

That breaks the layer twice over:
  1. a constant concept cannot be learned, so its AUROC is meaningless;
  2. a near-constant concept gets pos_weight (1-s)/s up to ~20, and its balanced BCE at
     initialisation is 1.31 x lam 0.3 = 0.39, against ~2.56 for the answer loss. **The concept term
     was shouting louder than the task**, which is a large part of why the answer collapsed.

THE FIX
-------
Same concepts, same meanings, but the THRESHOLD is a QUANTILE OF THE TRAINING DATA rather than a
magic number:

    is_bright            mean(pixel)    > q50 of train means      "brighter than typical"
    is_dark              mean(pixel)    < q15 of train means      "among the darkest"
    has_high_contrast    std(pixel)     > q75 of train stds
    top_half_heavier     (top - bottom) > q80 of train differences
    ...

Every threshold is still EXACTLY documentable -- the formula states the quantile and the value it
resolved to is stored in the spec, so a reader can recompute it. What changes is that the concept now
splits the actual data instead of assuming a Fashion-MNIST brightness range.

Pre-registered prediction: every concept lands in a usable support band (roughly 5-50%), pos_weight
stays modest, and the medical run stops collapsing onto a single answer.
"""
import numpy as np


def _img(X):
    """(B, 784) grayscale OR (B, 2352) channel-major colour -> (B, 28, 28) grayscale.

    The MODEL can be given colour while the CONCEPTS are still computed on brightness. That is a
    deliberate separation: the colour change is one experiment, and keeping the concept definitions
    identical means any difference in the result comes from the model seeing colour, not from the
    concepts changing underneath it. Colour-aware concepts are a separate, later change.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[-1] == 2352:
        X = X.reshape(-1, 3, 28, 28).mean(axis=1)
    return X.reshape(-1, 28, 28)


def _flat_gray(X):
    """The (B, 784) grayscale view, for measures that work on flattened pixels."""
    return _img(X).reshape(len(_img(X)), -1)


def _lap(I):
    return (I[:, 1:-1, 1:-1] * 4 - I[:, :-2, 1:-1] - I[:, 2:, 1:-1]
            - I[:, 1:-1, :-2] - I[:, 1:-1, 2:])


# Each entry: (name, raw_measure, quantile, direction)
#   direction "hi" -> concept fires when the measure is ABOVE the quantile
#   direction "lo" -> concept fires when the measure is BELOW the quantile
MEASURES = [
    ("is_bright", lambda X: _img(X).mean(axis=(1, 2)), 0.50, "hi",
     "mean(pixel) > q50 of TRAIN means"),
    ("is_dark", lambda X: _img(X).mean(axis=(1, 2)), 0.15, "lo",
     "mean(pixel) < q15 of TRAIN means"),
    ("has_high_contrast", lambda X: _img(X).std(axis=(1, 2)), 0.75, "hi",
     "std(pixel) > q75 of TRAIN stds"),
    ("has_low_contrast", lambda X: _img(X).std(axis=(1, 2)), 0.25, "lo",
     "std(pixel) < q25 of TRAIN stds"),
    ("top_half_heavier",
     lambda X: _img(X)[:, :14, :].mean(axis=(1, 2)) - _img(X)[:, 14:, :].mean(axis=(1, 2)),
     0.80, "hi", "(mean(top half) - mean(bottom half)) > q80 of TRAIN"),
    ("bottom_half_heavier",
     lambda X: _img(X)[:, 14:, :].mean(axis=(1, 2)) - _img(X)[:, :14, :].mean(axis=(1, 2)),
     0.80, "hi", "(mean(bottom half) - mean(top half)) > q80 of TRAIN"),
    ("left_half_heavier",
     lambda X: _img(X)[:, :, :14].mean(axis=(1, 2)) - _img(X)[:, :, 14:].mean(axis=(1, 2)),
     0.80, "hi", "(mean(left half) - mean(right half)) > q80 of TRAIN"),
    ("centre_heavier",
     lambda X: (_img(X)[:, 7:21, 7:21].mean(axis=(1, 2))
                - (_img(X).sum(axis=(1, 2)) - _img(X)[:, 7:21, 7:21].sum(axis=(1, 2))) / 588),
     0.80, "hi", "(mean(centre 14x14) - mean(border)) > q80 of TRAIN"),
    ("left_right_symmetric",
     lambda X: np.abs(_img(X) - _img(X)[:, :, ::-1]).mean(axis=(1, 2)),
     0.30, "lo", "mean|I - mirror(I)| < q30 of TRAIN"),
    ("top_bottom_symmetric",
     lambda X: np.abs(_img(X) - _img(X)[:, ::-1, :]).mean(axis=(1, 2)),
     0.30, "lo", "mean|I - flipud(I)| < q30 of TRAIN"),
    ("vertical_edges_dominate",
     lambda X: (np.abs(np.diff(_img(X), axis=2)).mean(axis=(1, 2))
                / (np.abs(np.diff(_img(X), axis=1)).mean(axis=(1, 2)) + 1e-6)),
     0.70, "hi", "mean|dI/dx| / mean|dI/dy| > q70 of TRAIN"),
    ("has_fine_texture", lambda X: np.abs(_lap(_img(X))).mean(axis=(1, 2)), 0.70, "hi",
     "mean|laplacian| > q70 of TRAIN"),
    ("has_flat_region", lambda X: np.abs(_lap(_img(X))).mean(axis=(1, 2)), 0.20, "lo",
     "mean|laplacian| < q20 of TRAIN"),
    ("mostly_ink", lambda X: (_flat_gray(X) > 0.5).mean(axis=1), 0.75, "hi",
     "fraction(pixel>0.5) > q75 of TRAIN"),
    ("mostly_empty", lambda X: (_flat_gray(X) > 0.5).mean(axis=1), 0.20, "lo",
     "fraction(pixel>0.5) < q20 of TRAIN"),
    ("is_thin_structure", lambda X: (_flat_gray(X) > 0.5).mean(axis=1), 0.30, "hi",
     "fraction(pixel>0.5) > q30 of TRAIN"),
]


def build(X_train, min_support=0.02, verbose=True):
    """Fit each threshold on TRAIN, then keep only concepts that split the data usefully."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lens.vision import ImageConcept

    X = np.asarray(X_train, dtype=np.float32)
    kept, dropped, spec_rows = [], [], []
    for name, fn, q, direction, formula in MEASURES:
        vals = np.asarray(fn(X), dtype=np.float64)
        thr = float(np.quantile(vals, q))
        if direction == "hi":
            lab = (vals > thr).astype(np.float32)
        else:
            lab = (vals < thr).astype(np.float32)
        s = float(lab.mean())
        full_formula = f"{formula}  (resolved threshold = {thr:.6f})"

        # reject degenerate concepts rather than storing a constant
        if s < min_support or s > (1.0 - min_support):
            dropped.append({"name": name, "support": s, "threshold": thr,
                            "reason": f"support {s:.4f} outside [{min_support}, "
                                      f"{1-min_support}] -- constant, cannot be learned"})
            if verbose:
                print(f"    DROPPED {name:<24} support {s:.4f}  thr {thr:.6f}")
            continue

        pw = (1.0 - s) / s
        kept.append(ImageConcept(name, _make_fn(fn, thr, direction), support=s,
                                 formula=full_formula))
        spec_rows.append({"name": name, "support": s, "threshold": thr, "pos_weight": pw,
                          "quantile": q, "direction": direction, "formula": full_formula})
        if verbose:
            print(f"    kept    {name:<24} support {s:.4f}  thr {thr:+.6f}  "
                  f"pos_weight {pw:5.2f}")

    out = {"kept": len(kept), "dropped": len(dropped), "n_defined": len(MEASURES),
           "min_support": min_support, "spec": spec_rows, "dropped_detail": dropped,
           "adaptive": True}
    return kept, out


def _make_fn(measure, thr, direction):
    """Bind the fitted threshold into a plain callable, so the concept stays self-contained."""
    def fn(X):
        v = np.asarray(measure(np.asarray(X, dtype=np.float32)), dtype=np.float64)
        return (v > thr).astype(np.float32) if direction == "hi" else (v < thr).astype(np.float32)
    return fn


if __name__ == "__main__":
    import os
    import sys
    PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
    sys.path.insert(0, PROJ)
    sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
    import vl_data

    for which, label in (("fashion", "Fashion-MNIST (MIT)"), ("pathmnist", "PathMNIST (CC BY 4.0)")):
        print("=" * 100)
        print(f"ADAPTIVE CONCEPTS — {label}")
        print("=" * 100)
        d = vl_data.prepare_vqa(which, max_train=20000)
        kept, spec = build(d["Xtr"], min_support=0.02, verbose=True)
        print()
        print(f"  {spec['kept']} kept, {spec['dropped']} dropped of {spec['n_defined']}")
        if spec["spec"]:
            pws = [r["pos_weight"] for r in spec["spec"]]
            sups = [r["support"] for r in spec["spec"]]
            print(f"  support range {min(sups):.4f} - {max(sups):.4f}")
            print(f"  pos_weight: median {np.median(pws):.2f}  max {max(pws):.2f}")
        print()
