"""image_concepts.py — named, MEASURABLE image concepts, each with its exact formula.

The brief's Phase 2 requirement, applied to images: "create simple measurable concepts from the
pixels (like symmetry, brightness regions, edge counts) and document exactly how each is computed."

Every concept here is a formula a person can recompute by hand from the pixels. None is learned,
none is a black box. The model is then trained to match each one, and audited on whether it does.
"""
import numpy as np


def _img(X):
    return X.reshape(-1, 28, 28)


def _mean(X):
    return _img(X).mean(axis=(1, 2))


def _std(X):
    return _img(X).std(axis=(1, 2))


# ------------------------------------------------------------------ brightness
def bright(X):
    return (_mean(X) > 0.50).astype(np.float32)


def dark(X):
    return (_mean(X) < 0.20).astype(np.float32)


def high_contrast(X):
    return (_std(X) > 0.30).astype(np.float32)


def low_contrast(X):
    return (_std(X) < 0.12).astype(np.float32)


# ------------------------------------------------------------------ regions
def top_heavier(X):
    I = _img(X)
    return ((I[:, :14, :].mean(axis=(1, 2)) - I[:, 14:, :].mean(axis=(1, 2))) > 0.05).astype(np.float32)


def bottom_heavier(X):
    I = _img(X)
    return ((I[:, 14:, :].mean(axis=(1, 2)) - I[:, :14, :].mean(axis=(1, 2))) > 0.05).astype(np.float32)


def left_heavier(X):
    I = _img(X)
    return ((I[:, :, :14].mean(axis=(1, 2)) - I[:, :, 14:].mean(axis=(1, 2))) > 0.05).astype(np.float32)


def centre_heavier(X):
    I = _img(X)
    c = I[:, 7:21, 7:21].mean(axis=(1, 2))
    border = (I.sum(axis=(1, 2)) - I[:, 7:21, 7:21].sum(axis=(1, 2))) / (784 - 196)
    return ((c - border) > 0.05).astype(np.float32)


# ------------------------------------------------------------------ symmetry
def left_right_symmetric(X):
    I = _img(X)
    return (np.abs(I - I[:, :, ::-1]).mean(axis=(1, 2)) < 0.05).astype(np.float32)


def top_bottom_symmetric(X):
    I = _img(X)
    return (np.abs(I - I[:, ::-1, :]).mean(axis=(1, 2)) < 0.05).astype(np.float32)


# ------------------------------------------------------------------ edges
def vertical_edges(X):
    """Mean absolute left-right pixel difference: high when there are vertical edges."""
    I = _img(X)
    return (np.abs(np.diff(I, axis=2)).mean(axis=(1, 2)) > 0.18).astype(np.float32)


def horizontal_edges(X):
    I = _img(X)
    return (np.abs(np.diff(I, axis=1)).mean(axis=(1, 2)) > 0.18).astype(np.float32)


def edge_dominant_vertical(X):
    """Vertical edges outweigh horizontal ones -- a striped/linear shape rather than a flat one."""
    I = _img(X)
    v = np.abs(np.diff(I, axis=2)).mean(axis=(1, 2))
    h = np.abs(np.diff(I, axis=1)).mean(axis=(1, 2))
    return (v > h * 1.15).astype(np.float32)


# ------------------------------------------------------------------ texture
def fine_texture(X):
    I = _img(X)
    lap = I[:, 1:-1, 1:-1] * 4 - I[:, :-2, 1:-1] - I[:, 2:, 1:-1] - I[:, 1:-1, :-2] - I[:, 1:-1, 2:]
    return (np.abs(lap).mean(axis=(1, 2)) > 0.30).astype(np.float32)


def flat_region(X):
    I = _img(X)
    lap = I[:, 1:-1, 1:-1] * 4 - I[:, :-2, 1:-1] - I[:, 2:, 1:-1] - I[:, 1:-1, :-2] - I[:, 1:-1, 2:]
    return (np.abs(lap).mean(axis=(1, 2)) < 0.12).astype(np.float32)


# ------------------------------------------------------------------ mass
def mostly_ink(X):
    return ((X > 0.5).mean(axis=1) > 0.55).astype(np.float32)


def mostly_empty(X):
    return ((X > 0.5).mean(axis=1) < 0.10).astype(np.float32)


def thin_structure(X):
    """Ink occupies a small share: a thin outline rather than a filled blob."""
    frac = (X > 0.5).mean(axis=1)
    return ((frac > 0.10) & (frac < 0.35)).astype(np.float32)


SPEC = [
    ("is_bright", bright, "mean(pixel) > 0.50"),
    ("is_dark", dark, "mean(pixel) < 0.20"),
    ("has_high_contrast", high_contrast, "std(pixel) > 0.30"),
    ("has_low_contrast", low_contrast, "std(pixel) < 0.12"),
    ("top_half_heavier", top_heavier, "mean(top 14 rows) - mean(bottom 14 rows) > 0.05"),
    ("bottom_half_heavier", bottom_heavier, "mean(bottom 14) - mean(top 14) > 0.05"),
    ("left_half_heavier", left_heavier, "mean(left 14 cols) - mean(right 14 cols) > 0.05"),
    ("centre_heavier", centre_heavier, "mean(centre 14x14) - mean(border) > 0.05"),
    ("left_right_symmetric", left_right_symmetric, "mean|I - mirror(I)| < 0.05"),
    ("top_bottom_symmetric", top_bottom_symmetric, "mean|I - flipud(I)| < 0.05"),
    ("has_vertical_edges", vertical_edges, "mean|dI/dx| > 0.18"),
    ("has_horizontal_edges", horizontal_edges, "mean|dI/dy| > 0.18"),
    ("vertical_edges_dominate", edge_dominant_vertical, "mean|dI/dx| > 1.15 * mean|dI/dy|"),
    ("has_fine_texture", fine_texture, "mean|laplacian| > 0.30"),
    ("has_flat_region", flat_region, "mean|laplacian| < 0.12"),
    ("mostly_ink", mostly_ink, "fraction(pixel>0.5) > 0.55"),
    ("mostly_empty", mostly_empty, "fraction(pixel>0.5) < 0.10"),
    ("is_thin_structure", thin_structure, "0.10 < fraction(pixel>0.5) < 0.35"),
]


def build(X_train, min_support=0.02, verbose=True):
    """Keep only concepts that fire on at least `min_support` of TRAIN images.

    The floor is 2%, pre-registered, and it does not move. A concept that fires on 0.3% of images
    is not something a human can use to understand the model, and the text side already learned
    that lesson: 16 of 17 failures there were perfect AUROC at 0.8-1.3% firing rate.
    """
    from lens.vision import ImageConcept
    kept, dropped = [], []
    for name, fn, formula in SPEC:
        lab = np.asarray(fn(np.asarray(X_train, dtype=np.float32)), dtype=np.float32)
        s = float(lab.mean())
        if s < min_support:
            dropped.append({"name": name, "support": s, "formula": formula})
            continue
        kept.append(ImageConcept(name, fn, support=s, formula=formula))
        if verbose:
            print(f"    kept    {name:<24} support {s:.4f}   {formula}")
    if verbose:
        for d in dropped:
            print(f"    DROPPED {d['name']:<24} support {d['support']:.4f} < {min_support}")
    return kept, {"kept": len(kept), "dropped": len(dropped), "n_defined": len(SPEC),
                  "min_support": min_support, "dropped_detail": dropped}
