"""diag_face_feats.py — print EVERY feature value for the synthetic cases, so the next fix is
driven by numbers rather than another guess. (Two fixes already went in blind; stop that.)
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import face_concepts as FC                              # noqa: E402
from test_face_detect import synth_face                 # noqa: E402

print("=" * 100)
print("FEATURE VALUES, CASE BY CASE")
print("=" * 100)
for name, img in (("face WITH eyes+mouth", synth_face(with_features=True)),
                  ("skin blob NO features", synth_face(with_features=False))):
    feats, aux = FC.face_features(img)
    print(f"\n  {name}")
    print(f"    bbox={aux['bbox']}  row_contrast={aux.get('row_contrast', 'n/a')}")
    for k, v in feats.items():
        print(f"    {k:<20} {v}")
    r = FC.detect(img)
    print(f"    -> score {r['score']} found={r['found']}")
    for n, v, w, c in r["explain"]:
        print(f"       {n:<26} {v} * {w} = {c}")

# what does the blob's row profile actually look like for the face?
img = synth_face(with_features=True)
a = np.asarray(img, dtype=np.float32)
m = FC.skin_mask(a)
bbox, size, blob = FC.largest_blob(m)
y0, x0, y1, x1 = bbox
grey = a[y0:y1 + 1, x0:x1 + 1].mean(axis=2)
bm = blob[y0:y1 + 1, x0:x1 + 1]
rows = []
for r in range(bm.shape[0]):
    idx = np.nonzero(bm[r])[0]
    if idx.size >= max(3, bm.shape[1] // 20):
        rows.append(round(float(grey[r, idx].mean()), 3))
print(f"\n  FACE blob row means ({len(rows)} rows): {rows}")
if rows:
    arr = np.asarray(rows)
    print(f"    mean {arr.mean():.3f} std {arr.std():.3f} "
          f"min {arr.min():.3f} max {arr.max():.3f}")
    print(f"    dark threshold would be {arr.mean() - max(0.6*arr.std(), 0.05):.3f}")
