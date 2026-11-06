"""colour_concepts.py — named concepts measured on the RGB CHANNELS.

THE GAP THIS CLOSES
-------------------
The colour change fed RGB into the model but the concept layer still saw grayscale, so the readable
middle could CONSUME colour without being able to NAME it. A reader of the model's explanation would
see `is_bright`, `has_fine_texture` and no idea that the model was actually using stain colour.

These concepts are measured on the channels directly. For H&E histology the stain is the point:
haematoxylin stains nuclei blue/purple, eosin stains cytoplasm pink, so "is pink-dominant" and
"blue exceeds green" are meaningful tissue properties, not decoration.

Every threshold is a QUANTILE OF TRAIN, resolved value stored in the formula, same rule as the
grayscale concepts. A concept whose support falls outside [min_support, 1-min_support] is REJECTED
rather than stored as a near-constant -- the exact failure that made three of the Fashion-MNIST
concepts useless on PathMNIST.
"""
import numpy as np


def _rgb(X):
    """(B, 2352) channel-major -> (B, 3, 28, 28). Only meaningful when the data kept colour."""
    X = np.asarray(X, dtype=np.float32)
    if X.shape[-1] != 2352:
        return None
    return X.reshape(-1, 3, 28, 28)


def _chan_mean(X, c):
    R = _rgb(X)
    if R is None:
        return None
    return R[:, c].mean(axis=(1, 2))


MEASURES = [
    # --- pure channel levels
    ("red_channel_high", lambda X: _chan_mean(X, 0), 0.65, "hi",
     "mean(R) > q65 of TRAIN mean(R)"),
    ("red_channel_low", lambda X: _chan_mean(X, 0), 0.35, "lo",
     "mean(R) < q35 of TRAIN mean(R)"),
    ("green_channel_high", lambda X: _chan_mean(X, 1), 0.65, "hi",
     "mean(G) > q65 of TRAIN mean(G)"),
    ("green_channel_low", lambda X: _chan_mean(X, 1), 0.35, "lo",
     "mean(G) < q35 of TRAIN mean(G)"),
    ("blue_channel_high", lambda X: _chan_mean(X, 2), 0.65, "hi",
     "mean(B) > q65 of TRAIN mean(B)"),
    ("blue_channel_low", lambda X: _chan_mean(X, 2), 0.35, "lo",
     "mean(B) < q35 of TRAIN mean(B)"),

    # --- STAIN HUE. For H&E histology these are the real signal.
    ("is_pink_dominant",
     lambda X: _chan_mean(X, 0) - _chan_mean(X, 1), 0.65, "hi",
     "mean(R) - mean(G) > q65 of TRAIN   (eosin: pink cytoplasm)"),
    ("is_purple_dominant",
     lambda X: (_chan_mean(X, 0) + _chan_mean(X, 2)) / 2 - _chan_mean(X, 1), 0.65, "hi",
     "((mean(R)+mean(B))/2 - mean(G)) > q65 of TRAIN   (haematoxylin: purple nuclei)"),
    ("is_green_dominant",
     lambda X: _chan_mean(X, 1) - (_chan_mean(X, 0) + _chan_mean(X, 2)) / 2, 0.65, "hi",
     "mean(G) - (mean(R)+mean(B))/2 > q65 of TRAIN"),
    ("blue_exceeds_green",
     lambda X: _chan_mean(X, 2) - _chan_mean(X, 1), 0.60, "hi",
     "mean(B) - mean(G) > q60 of TRAIN"),
    ("red_exceeds_blue",
     lambda X: _chan_mean(X, 0) - _chan_mean(X, 2), 0.60, "hi",
     "mean(R) - mean(B) > q60 of TRAIN"),

    # --- how much colour at all
    ("has_strong_colour",
     lambda X: (np.abs(_rgb(X)[:, 0] - _rgb(X)[:, 1]).mean(axis=(1, 2))
                + np.abs(_rgb(X)[:, 1] - _rgb(X)[:, 2]).mean(axis=(1, 2))), 0.65, "hi",
     "mean|R-G| + mean|G-B| > q65 of TRAIN   (strongly tinted)"),
    ("is_nearly_greyscale",
     lambda X: (np.abs(_rgb(X)[:, 0] - _rgb(X)[:, 1]).mean(axis=(1, 2))
                + np.abs(_rgb(X)[:, 1] - _rgb(X)[:, 2]).mean(axis=(1, 2))), 0.25, "lo",
     "mean|R-G| + mean|G-B| < q25 of TRAIN   (little hue variation)"),

    # --- spatial colour: is one channel unevenly spread?
    ("red_vs_blue_top_bottom",
     lambda X: (_rgb(X)[:, 0, :14, :].mean(axis=(1, 2)) - _rgb(X)[:, 2, 14:, :].mean(axis=(1, 2))),
     0.70, "hi", "mean(R, top half) - mean(B, bottom half) > q70 of TRAIN"),
    ("colour_contrast_high",
     lambda X: (np.maximum(np.maximum(_rgb(X)[:, 0], _rgb(X)[:, 1]), _rgb(X)[:, 2])
                - np.minimum(np.minimum(_rgb(X)[:, 0], _rgb(X)[:, 1]), _rgb(X)[:, 2])).mean(
                    axis=(1, 2)),
     0.60, "hi", "mean(max(R,G,B) - min(R,G,B)) > q60 of TRAIN"),
]


def build(X_train, min_support=0.02, verbose=True):
    """Fit colour-concept thresholds on TRAIN. Returns ([], spec) if the data is not colour."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lens.vision import ImageConcept

    X = np.asarray(X_train, dtype=np.float32)
    if X.shape[-1] != 2352:
        if verbose:
            print("    (no colour concepts: this dataset is grayscale, not 2352 values)")
        return [], {"kept": 0, "dropped": 0, "n_defined": 0, "colour": False,
                    "reason": "data is not channel-major colour"}

    kept, dropped, spec_rows = [], [], []
    for name, fn, q, direction, formula in MEASURES:
        vals = np.asarray(fn(X), dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            dropped.append({"name": name, "reason": "non-finite measure"})
            continue
        thr = float(np.quantile(vals, q))
        lab = (vals > thr).astype(np.float32) if direction == "hi" \
            else (vals < thr).astype(np.float32)
        s = float(lab.mean())
        full = f"{formula}  (resolved threshold = {thr:.6f})"
        if s < min_support or s > (1.0 - min_support):
            dropped.append({"name": name, "support": s, "threshold": thr,
                            "reason": f"support {s:.4f} outside [{min_support}, {1-min_support}] "
                                      f"-- constant, cannot be learned"})
            if verbose:
                print(f"    DROPPED {name:<26} support {s:.4f}")
            continue
        kept.append(ImageConcept(name, _bind(fn, thr, direction), support=s, formula=full))
        spec_rows.append({"name": name, "support": s, "threshold": thr, "quantile": q,
                          "direction": direction, "formula": full,
                          "pos_weight": (1.0 - s) / s})
        if verbose:
            print(f"    kept    {name:<26} support {s:.4f}  thr {thr:+.6f}  "
                  f"pos_weight {(1.0-s)/s:5.2f}")

    return kept, {"kept": len(kept), "dropped": len(dropped), "n_defined": len(MEASURES),
                  "colour": True, "min_support": min_support, "spec": spec_rows,
                  "dropped_detail": dropped}


def _bind(measure, thr, direction):
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

    print("=" * 100)
    print("COLOUR CONCEPTS — PathMNIST (CC BY 4.0), fitted on TRAIN")
    print("=" * 100)
    d = vl_data.prepare_vqa("pathmnist", max_train=20000, color=True)
    print(f"  input shape {d['Xtr'].shape}  (channel-major RGB)")
    kept, spec = build(d["Xtr"], min_support=0.02, verbose=True)
    print()
    print(f"  {spec['kept']} kept, {spec['dropped']} dropped of {spec['n_defined']}")
    if spec["spec"]:
        sups = [r["support"] for r in spec["spec"]]
        pws = [r["pos_weight"] for r in spec["spec"]]
        print(f"  support range {min(sups):.4f} - {max(sups):.4f}")
        print(f"  pos_weight median {np.median(pws):.2f}  max {max(pws):.2f}")
    print()
    print("  --- do these actually separate the tissues? support per class ---")
    names = vl_data.MEDMNIST_NAMES["pathmnist"]
    for r in spec["spec"][:4]:
        k = [x for x in kept if x.name == r["name"]][0]
        lab = k.labels(d["Xtr"])
        per = []
        for c in range(9):
            m = d["ytr"] == c
            per.append(float(lab[m].mean()) if m.sum() else float("nan"))
        spread = max(per) - min(per)
        print(f"    {r['name']:<26} min {min(per):.3f}  max {max(per):.3f}  spread {spread:.3f}"
              f"  {'SEPARATES' if spread > 0.25 else 'weak'}")
    print("=" * 100)
