"""face_concepts.py — FROM-SCRATCH face detection as NAMED, MEASURABLE concepts.

No pretrained weights. No Haar cascades. No OpenCV. Every "concept" here is a formula you can read
and re-compute by hand, exactly like the image concepts that already work on Fashion-MNIST.

A face is detected by a small set of measurable signals, each with a NAME and an EXACT formula:
  skin_fraction     fraction of pixels whose colour falls in the skin-tone band
  skin_cluster      largest connected skin region / total skin pixels (a face is ONE blob, not noise)
  skin_centred      how close the skin blob's centre is to the frame centre
  eye_dark_bands    two dark horizontal bands in the upper half of the skin blob (eye sockets)
  mouth_dark_band   one dark horizontal band in the lower third of the skin blob
  vertical_symmetry left/right symmetry INSIDE the skin blob (faces are near-symmetric)
  blob_aspect       height/width of the skin blob (a face is taller than wide, ~1.2-1.5)
  blob_solidity     blob area / bounding-box area (faces fill their box better than hands do)

The DETECTOR is a transparent weighted sum of these, with every weight printed. It is NOT trained
on any dataset — the thresholds are stated constants, so there is nothing hidden. This is
deliberately the simplest thing that can work, and the honest framing is:
  * DETECTION/TRACKING: can be done well from scratch (this file)
  * RECOGNITION (who is this person): needs identities -> blocked on the licence rule

Skin band (RGB, normalised 0-1), the classic rule, stated explicitly so it can be argued with:
    R > 0.36  and  R > G  and  G > B  and  (R - G) > 0.02  and  (R - B) > 0.06
    and not (G > 0.85 and B > 0.85)          # excludes bright white/neutral walls
"""
import numpy as np


def _rgb(img):
    a = np.asarray(img, dtype=np.float32)
    if a.max() > 1.5:
        a = a / 255.0
    return a


def skin_mask(img):
    """The exact skin rule above, applied per pixel. Returns a boolean mask."""
    a = _rgb(img)
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    m = (R > 0.36) & (R > G) & (G > B) & ((R - G) > 0.02) & ((R - B) > 0.06)
    m &= ~((G > 0.85) & (B > 0.85))
    return m


def largest_blob(mask):
    """Largest 4-connected True region, iterative (no recursion limits). Returns (bbox, size, mask).

    Deliberately plain: BFS over the mask. This is O(pixels) and runs in milliseconds at 640x480
    after downscaling, and it is completely inspectable — no library, no magic.
    """
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best = (0, None, None)
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None, 0, np.zeros_like(mask, dtype=bool)
    # sample starts from actual True pixels so we do not scan the whole frame per blob
    for sy, sx in zip(ys[::max(1, len(ys) // 4000)], xs[::max(1, len(xs) // 4000)]):
        if seen[sy, sx] or not mask[sy, sx]:
            continue
        stack = [(sy, sx)]
        seen[sy, sx] = True
        comp = []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(comp) > best[0]:
            cy = [c[0] for c in comp]
            cx = [c[1] for c in comp]
            bbox = (min(cy), min(cx), max(cy), max(cx))
            bm = np.zeros_like(mask, dtype=bool)
            for y, x in comp:
                bm[y, x] = True
            best = (len(comp), bbox, bm)
    return best[1], best[0], best[2]


def fill_holes(blob):
    """Fill the interior holes of a blob (iterative flood fill from the bounding-box border).

    WHY THIS EXISTS: eyes and mouth are DARK, so they fail the skin rule and are therefore NOT part
    of the skin blob -- they are HOLES in it. The first version of the dark-band feature measured
    brightness only over skin-classified pixels, which mathematically EXCLUDED the eyes and mouth,
    so the feature could never fire (measured: every row came out 0.573, std 0.000). Filling the
    holes recovers the actual face region, and the hole FRACTION itself becomes a usable signal.
    """
    ys, xs = np.nonzero(blob)
    if len(ys) == 0:
        return blob.copy()
    y0, x0, y1, x1 = ys.min(), xs.min(), ys.max(), xs.max()
    sub = blob[y0:y1 + 1, x0:x1 + 1]
    h, w = sub.shape
    comp = ~sub
    outside = np.zeros_like(comp)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if comp[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if comp[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and comp[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    holes = comp & ~outside
    filled = sub | holes
    out = blob.copy()
    out[y0:y1 + 1, x0:x1 + 1] = filled
    return out


def face_features(img):
    """Every named feature, with its exact formula, for one frame.

    Returns a dict: name -> value, plus the intermediate masks needed to draw the result.
    Nothing is hidden: each value is a number you can recompute from the image by hand.
    """
    a = _rgb(img)
    h, w = a.shape[:2]
    m = skin_mask(a)
    bbox, size, blob = largest_blob(m)
    total = h * w
    feats = {
        "skin_fraction": float(m.sum()) / total,
        "skin_cluster": float(size) / max(float(m.sum()), 1.0),
    }
    if bbox is None or size < 50:
        for k in ("skin_centred", "blob_aspect", "blob_solidity", "vertical_symmetry",
                  "hole_fraction", "eye_dark_bands", "mouth_dark_band"):
            feats[k] = 0.0
        return feats, {"skin": m, "blob": blob, "bbox": None}

    y0, x0, y1, x1 = bbox
    bh, bw = y1 - y0 + 1, x1 - x0 + 1
    cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
    diag = np.hypot(h, w)
    feats["skin_centred"] = float(1.0 - min(np.hypot(cy - h / 2, cx - w / 2) / (diag / 2), 1.0))
    feats["blob_aspect"] = float(bh) / max(float(bw), 1.0)
    feats["blob_solidity"] = float(size) / max(float(bh * bw), 1.0)

    # --- measure INSIDE the filled face region (holes = eyes/mouth are part of the face) ---
    filled = fill_holes(blob)
    bmask = filled[y0:y1 + 1, x0:x1 + 1]
    grey = a[y0:y1 + 1, x0:x1 + 1].mean(axis=2)
    box_area = float(bmask.sum())
    feats["hole_fraction"] = float(bmask.sum() - size) / max(box_area, 1.0)

    # vertical symmetry over the FILLED region (grey levels, left vs right flipped)
    gm = np.where(bmask, grey, np.nan)
    flipped = gm[:, ::-1]
    n = min(gm.shape[1], flipped.shape[1])
    if n > 1:
        d = np.nanmean(np.abs(gm[:, :n] - flipped[:, :n]))
        feats["vertical_symmetry"] = float(max(0.0, 1.0 - (0.0 if np.isnan(d) else d) * 2.0))
    else:
        feats["vertical_symmetry"] = 0.0

    # dark bands: rows of the FILLED region that are notably darker than the region's mean.
    rows = []
    for r in range(bmask.shape[0]):
        idx = np.nonzero(bmask[r])[0]
        if idx.size >= max(3, bmask.shape[1] // 20):
            rows.append(float(grey[r, idx].mean()))
    if len(rows) > 6:
        rowmean = np.asarray(rows, dtype=np.float32)
        contrast = float(rowmean.std())
        feats["row_contrast"] = contrast
        if contrast < 0.01:
            feats["eye_dark_bands"] = 0.0
            feats["mouth_dark_band"] = 0.0
        else:
            thr = rowmean.mean() - max(0.8 * contrast, 0.03)
            dark = rowmean < thr
            up = dark[:int(len(dark) * 0.6)]
            low = dark[int(len(dark) * 0.66):]
            feats["eye_dark_bands"] = float(up.sum()) / max(len(up), 1) * 3.0
            feats["mouth_dark_band"] = float(low.sum()) / max(len(low), 1) * 3.0
    else:
        feats["eye_dark_bands"] = feats["mouth_dark_band"] = 0.0
    return feats, {"skin": m, "blob": blob, "filled": filled, "bbox": bbox}


# The detector: a TRANSPARENT weighted sum. Every weight is here, nothing is learned, nothing hidden.
# Chosen so that a face-like blob scores high and a hand/wall/object scores low. These are STATED
# constants, not trained parameters -- so there is no hidden fitting and no dataset requirement.
#
# FIXED 2026-10-09 after the negative-control test FAILED: the first weight set scored a featureless
# skin blob (8.20) ABOVE a real face with eyes and mouth (7.70) -- i.e. it was a COLOUR detector
# wearing a face detector's name. The cause: skin_cluster + blob_solidity + vertical_symmetry can
# all be satisfied by a plain ellipse, and the eye/mouth terms were too small to matter.
# The fix makes FACIAL STRUCTURE a REQUIREMENT, not a bonus:
#   * eye/mouth weights raised so a face without them cannot reach the threshold
#   * a hard GATE: if eye_dark_bands < 0.15 AND mouth_dark_band < 0.15, the score is halved
#     (a face has dark features; a skin-coloured wall does not)
WEIGHTS = {
    "skin_fraction": 1.0,      # want a decent amount of skin, not the whole frame
    "skin_cluster": 1.5,       # want ONE blob (a face) rather than scattered skin pixels
    "skin_centred": 1.0,       # want it near the middle of the frame
    "blob_solidity": 1.5,      # want it to fill its box
    "vertical_symmetry": 1.0,  # faces are near-symmetric
    "hole_fraction": 3.0,      # a face has HOLES in the skin region: eyes, mouth, nostrils.
                               # This is the feature that separates a face from a bare skin blob
                               # (measured: face 0.111 vs featureless blob 0.000)
    "eye_dark_bands": 2.0,     # dark rows in the upper part of the face region
    "mouth_dark_band": 1.0,
}
THRESHOLD = 3.6               # score above this = "a face is present"
ASPECT_OK = (0.7, 2.2)        # plausible face box shape
STRUCTURE_GATE = 0.15         # below this much dark structure, the score is halved
MIN_SKIN_FRACTION = 0.01      # a face must occupy at least 1% of the frame as skin.
                              # FIXED 2026-10-09 from the REAL-PHOTO test: 4 screenshots were
                              # flagged as faces with skin_fraction 0.001-0.017 (essentially no skin)
                              # and hole_fraction 0.000, carried by symmetry+solidity on tiny
                              # blobs. Below this floor there is not enough skin for the word
                              # "face" to mean anything, so detection is refused outright.


def detect(img):
    """Return {'found': bool, 'score': float, 'bbox': (y0,x0,y1,x1) or None, 'features': {...},
    'explain': [(name, value, weight, contribution), ...]} — fully inspectable."""
    feats, aux = face_features(img)
    contrib = []
    score = 0.0
    for k, wt in WEIGHTS.items():
        v = feats.get(k, 0.0)
        # squash skin_fraction: peak around 3-25% of the frame, penalise near-zero and near-total
        if k == "skin_fraction":
            v2 = 0.0 if v < 0.005 else (1.0 if v < 0.35 else 0.3)
        elif k == "eye_dark_bands":
            v2 = min(v, 1.0)
        elif k == "mouth_dark_band":
            v2 = min(v, 1.0)
        else:
            v2 = float(np.clip(v, 0.0, 1.0))
        c = wt * v2
        score += c
        contrib.append((k, round(v, 4), wt, round(c, 3)))
    asp = feats.get("blob_aspect", 0.0)
    if aux["bbox"] is not None and not (ASPECT_OK[0] <= asp <= ASPECT_OK[1]):
        score -= 1.0
        contrib.append(("blob_aspect_out_of_range", round(asp, 3), -1.0, -1.0))
    # STRUCTURE GATE: a face has dark features (eyes, mouth). A skin-coloured wall does not.
    # Without this, the score is dominated by colour+shape and the detector is just a colour rule.
    has_structure = (feats.get("eye_dark_bands", 0.0) >= STRUCTURE_GATE
                     or feats.get("mouth_dark_band", 0.0) >= STRUCTURE_GATE
                     or feats.get("hole_fraction", 0.0) >= 0.02)
    if not has_structure:
        contrib.append(("NO_DARK_STRUCTURE_halved", 0.0, 0.5, round(score * -0.5, 3)))
        score *= 0.5
    # SKIN FLOOR: refuse outright when there is not enough skin for "face" to mean anything.
    if feats.get("skin_fraction", 0.0) < MIN_SKIN_FRACTION:
        contrib.append(("SKIN_BELOW_FLOOR_refused", round(feats.get("skin_fraction", 0.0), 4),
                        MIN_SKIN_FRACTION, round(-score, 3)))
        score = 0.0
    return {"found": bool(score >= THRESHOLD), "score": round(float(score), 3),
            "bbox": aux["bbox"] if score >= THRESHOLD else None,
            "features": {k: round(v, 4) for k, v in feats.items()},
            "explain": contrib}
