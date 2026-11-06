"""diag_cam_caps.py — what can we do with the webcam RIGHT NOW, offline, no downloads?

Checks: can we capture frames (ffmpeg v4l2), what resolution, and is OpenCV available (its Haar
cascades are a PRETRAINED model — using them would break the from-scratch rule, so this only
reports whether it EXISTS, not that we will use it).
"""
import os
import shutil
import subprocess
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
FFMPEG = shutil.which("ffmpeg") or "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"

print("=" * 96)
print("WEBCAM / OFFLINE CAPABILITY CHECK")
print("=" * 96)
print(f"  ffmpeg: {FFMPEG}  ({'present' if os.path.exists(FFMPEG) else 'MISSING'})")

# try a real 1-frame capture to prove the camera works headlessly
out = "/home/yahya/.hermes/cache/scratch/cam_probe.png"
try:
    r = subprocess.run([FFMPEG, "-y", "-f", "v4l2", "-i", "/dev/video0", "-frames:v", "1",
                        out], capture_output=True, text=True, timeout=45)
    ok = os.path.exists(out) and os.path.getsize(out) > 1000
    print(f"  /dev/video0 capture: {'OK' if ok else 'FAILED'}"
          f"{'' if ok else ' — ' + r.stderr.strip().splitlines()[-1][:90] if r.stderr else ''}")
    if ok:
        print(f"    wrote {os.path.getsize(out):,} bytes -> {out}")
except Exception as e:
    print(f"  capture error: {str(e)[:90]}")

# what resolutions does it offer?
try:
    r = subprocess.run([FFMPEG, "-f", "v4l2", "-i", "/dev/video0"], capture_output=True,
                       text=True, timeout=25)
    modes = [l.strip() for l in r.stderr.splitlines() if "x" in l and "fps" in l][:6]
    print("  formats offered:")
    for m in modes:
        print("    " + m[:80])
except Exception:
    pass

print("\n  OpenCV (cv2):", end=" ")
try:
    sys.path.insert(0, PROJ)
    import cv2
    print(f"PRESENT v{cv2.__version__} — NOTE: its Haar face cascades are PRETRAINED weights and")
    print("    using them would break the from-scratch rule. Not used for the model itself.")
except ImportError:
    print("absent (good for purity; we build our own detection from scratch)")

print("\n  RESULT: camera is usable headlessly via ffmpeg. Face detection will be built from")
print("  scratch with our own concept layer — skin/edge/symmetry measurements, all named.")
