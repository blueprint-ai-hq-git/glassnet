"""test_face_detect.py — does the from-scratch face detector actually work? Honest test with
POSITIVE and NEGATIVE controls (a detector that says "face" to everything is useless).

Controls:
  * real webcam frame        (may or may not contain a face — reported as-is, no pretending)
  * synthetic face           (skin ellipse + eye/mouth dark bands, drawn by hand — a KNOWN positive)
  * flat wall (grey)         KNOWN negative
  * random noise             KNOWN negative
  * a red rectangle          KNOWN negative (skin rule requires G>B and R>G ordering)
  * a skin-coloured rectangle with NO features (no eyes/mouth)  KNOWN negative — this is the hard one:
    colour alone must not be enough, or the detector is just a colour detector.
"""
import os
import subprocess
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import face_concepts as FC                              # noqa: E402

FFMPEG = "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
SCRATCH = "/home/yahya/.hermes/cache/scratch"


def synth_face(h=240, w=320, with_features=True, skin=(0.72, 0.55, 0.45)):
    img = np.ones((h, w, 3), dtype=np.float32) * 0.85          # light grey background (not skin)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    face = (((yy - cy) / (h * 0.34)) ** 2 + ((xx - cx) / (w * 0.26)) ** 2) <= 1.0
    img[face] = skin
    if with_features:
        # two dark eye bands
        for ey in (int(cy - h * 0.13),):
            band = face.copy()
            band[:int(cy - h * 0.06)] = False
            band[int(cy - h * 0.02):] = False
            band[:, :int(cx - w * 0.17)] = False
            band[:, int(cx + w * 0.17):] = False
            img[band] = (0.25, 0.2, 0.18)
        mouth = face.copy()
        mouth[:int(cy + h * 0.12)] = False
        mouth[int(cy + h * 0.20):] = False
        mouth[:, :int(cx - w * 0.10)] = False
        mouth[:, int(cx + w * 0.10):] = False
        img[mouth] = (0.35, 0.22, 0.22)
    return img


def read_png(path):
    """Read the probe png via ffmpeg -> raw rgb24 (no image library needed)."""
    raw = path + ".raw"
    r = subprocess.run([FFMPEG, "-y", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", raw],
                       capture_output=True, text=True, timeout=60)
    if not os.path.exists(raw):
        return None
    d = np.fromfile(raw, dtype=np.uint8)
    # the probe was 640x480
    if d.size == 640 * 480 * 3:
        return d.reshape(480, 640, 3).astype(np.float32) / 255.0
    return None


print("=" * 100)
print("FACE DETECTOR TEST — positives AND negatives (a detector that fires on everything is useless)")
print("=" * 100)

cases = []
probe = os.path.join(SCRATCH, "cam_probe.png")
if os.path.exists(probe):
    real = read_png(probe)
    if real is not None:
        cases.append(("REAL webcam frame (unknown content)", real, None))

cases += [
    ("synthetic face WITH eyes+mouth", synth_face(with_features=True), True),
    ("synthetic skin blob, NO features", synth_face(with_features=False), False),
    ("flat grey wall", np.ones((240, 320, 3), dtype=np.float32) * 0.85, False),
    ("random noise", np.random.default_rng(0).random((240, 320, 3)).astype(np.float32), False),
    ("red rectangle", np.tile(np.array([0.8, 0.2, 0.2], dtype=np.float32), (240, 320, 1)), False),
]

results = []
for name, img, expect in cases:
    r = FC.detect(img)
    verdict = "?" if expect is None else ("CORRECT" if r["found"] == expect else "WRONG")
    results.append((name, r["found"], r["score"], expect, verdict))
    print(f"\n  {name}")
    print(f"    found={r['found']}  score={r['score']}  expect={expect}  -> {verdict}")
    top = sorted(r["explain"], key=lambda t: -abs(t[3]))[:4]
    print(f"    top contributions: " + " | ".join(f"{n}={v}*{w}={c}" for n, v, w, c in top))

print("\n" + "-" * 100)
print("  SUMMARY")
for name, found, score, expect, verdict in results:
    print(f"    {name:<38} found={str(found):<5} score={score:<6} {verdict}")

n_known = sum(1 for r in results if r[3] is not None)
n_right = sum(1 for r in results if r[4] == "CORRECT")
print(f"\n  {n_right}/{n_known} known cases correct")
print("  NOTE: the synthetic positive is drawn BY HAND, so it tests the MACHINERY, not real-world")
print("  accuracy. Real accuracy needs real faces, which need either the owner's own captures or a")
print("  licence-clean dataset — stated honestly rather than claimed.")
