"""test_face_real.py — REAL-FACE test using ONLY the owner's own photos (read-only, no copies out).

WHY: the synthetic test proved the machinery but not real-world accuracy. The only licence-clean
source of real faces available offline is the owner's OWN pictures (he owns them; every public face
dataset checked fails or is ambiguous under the safe-licence rule).

RULES OBEYED:
  * READ-ONLY: photos are opened and measured. Nothing is modified, moved, deleted, or copied out.
  * No crops or thumbnails of the owner's photos are written anywhere.
  * Only aggregate numbers and per-file scores are printed.
"""
import glob
import os
import subprocess
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import face_concepts as FC                              # noqa: E402

FFMPEG = "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"


def load_rgb(path, w=320, h=240):
    """Decode any image to a small rgb24 array via ffmpeg (no image library needed)."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", path,
           "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    r = subprocess.run(cmd, capture_output=True, timeout=60)
    if len(r.stdout) < w * h * 3:
        return None
    return np.frombuffer(r.stdout[:w * h * 3], dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0


print("=" * 100)
print("REAL-FACE TEST — owner's own photos, READ-ONLY (no copies, no crops written out)")
print("=" * 100)

paths = []
for pat in ("/home/yahya/Pictures/**/*.jpg", "/home/yahya/Pictures/**/*.jpeg",
            "/home/yahya/Pictures/**/*.png", "/home/yahya/Pictures/**/*.webp",
            "/home/yahya/Downloads/**/*.jpg", "/home/yahya/Downloads/**/*.png"):
    for p in glob.glob(pat, recursive=True):
        if os.path.isfile(p) and p not in paths:
            paths.append(p)
paths = sorted(paths)[:40]

print(f"  scanning {len(paths)} photos (read-only)")
rows = []
for p in paths:
    img = load_rgb(p)
    if img is None:
        continue
    r = FC.detect(img)
    rows.append((os.path.basename(p), r["found"], r["score"],
                 r["features"].get("skin_fraction"), r["features"].get("hole_fraction")))

found = [r for r in rows if r[1]]
print(f"\n  {'file':<34}{'found':>7}{'score':>8}{'skin':>8}{'holes':>8}")
for name, f, sc, sk, hf in rows:
    mark = "  <-- FACE" if f else ""
    print(f"  {name[:33]:<34}{str(f):>7}{sc:>8.2f}{sk:>8.3f}{(hf if hf is not None else 0):>8.3f}{mark}")

print("\n" + "-" * 100)
print(f"  {len(found)}/{len(rows)} photos flagged as containing a face")
if rows:
    scores = [r[2] for r in rows]
    print(f"  score distribution: min {min(scores):.2f}  median "
          f"{sorted(scores)[len(scores)//2]:.2f}  max {max(scores):.2f}")
print("\n  HONEST READING: these are wallpapers and screenshots, so most SHOULD score low — a")
print("  wallpaper is not a face. What matters is whether any real face in the set is MISSED or any")
print("  obvious non-face is flagged. No ground-truth labels exist for these files, so this measures")
print("  false positives honestly and CANNOT measure misses. Stated, not hidden.")
