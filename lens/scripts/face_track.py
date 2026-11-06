"""face_track.py — LIVE face tracking from the webcam, from scratch, no dataset, no downloads.

Tracking = detection (face_concepts) + temporal association (a simple, readable rule):
  * each frame is detected independently
  * a track is kept if the new box overlaps the previous box (IoU > 0.2) -- that is the whole
    association rule, printed in the code, no Kalman filter, no hidden state
  * a track survives up to N missing frames before it is dropped (occlusion tolerance)
  * each track has an id, an age, and a smoothed centre (exponential moving average, alpha printed)

Every number the tracker uses is a stated constant. Nothing is learned, nothing is hidden.

Usage:
  python3 face_track.py --seconds 20 --out track.mp4        # record a clip with boxes drawn
  python3 face_track.py --seconds 5 --out track.mp4 --stats # plus per-frame stats

Video is written through frameio's ffmpeg pipe (already proven in this project).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import face_concepts as FC                              # noqa: E402
import frameio                                          # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
W, H = 640, 480
FPS = 10

# --- tracker constants (all stated, all tunable, none hidden) ---
IOU_MATCH = 0.20        # overlap needed to call it the same face
MAX_MISSING = 8         # frames a track survives without a detection (~0.8 s at 10 fps)
EMA_ALPHA = 0.4         # centre smoothing: new = alpha*new + (1-alpha)*old
MIN_TRACK_AGE = 2       # frames before a track is reported (kills one-frame false positives)


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0, ix0 = max(ay0, by0), max(ax0, bx0)
    iy1, ix1 = min(ay1, by1), min(ax1, bx1)
    if iy1 <= iy0 or ix1 <= ix0:
        return 0.0
    inter = (iy1 - iy0) * (ix1 - ix0)
    aa = (ay1 - ay0) * (ax1 - ax0)
    bb = (by1 - by0) * (bx1 - bx0)
    return inter / float(aa + bb - inter + 1e-9)


class Track:
    _next = 1

    def __init__(self, box):
        self.id = Track._next
        Track._next += 1
        self.box = box
        self.age = 1
        self.missing = 0
        self.cy = (box[0] + box[2]) / 2.0
        self.cx = (box[1] + box[3]) / 2.0

    def update(self, box):
        self.box = box
        self.age += 1
        self.missing = 0
        ny, nx = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        self.cy = EMA_ALPHA * ny + (1 - EMA_ALPHA) * self.cy
        self.cx = EMA_ALPHA * nx + (1 - EMA_ALPHA) * self.cx

    def miss(self):
        self.missing += 1
        return self.missing <= MAX_MISSING


def read_frames(seconds, device="/dev/video0"):
    """Yield (frame_rgb_float01, t) from the camera via ffmpeg rawvideo. Generator = constant memory."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-framerate", str(FPS),
           "-i", device, "-t", str(seconds), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=W * H * 3)
    n = 0
    try:
        while True:
            buf = p.stdout.read(W * H * 3)
            if len(buf) < W * H * 3:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3).astype(np.float32) / 255.0, n
            n += 1
    finally:
        p.stdout.close()
        p.terminate()


def draw(frame_u8, tracks, det, show_explain=False):
    """Draw boxes + ids. Reuses frameio's own rect/text helpers (already proven in this project)."""
    img = frame_u8.copy()
    for t in tracks:
        if t.age < MIN_TRACK_AGE:
            continue
        y0, x0, y1, x1 = t.box
        col = (0, 255, 0) if t.missing == 0 else (255, 200, 0)
        frameio.draw_rect(img, int(x0), int(y0), int(x1 - x0), int(y1 - y0), col, 2)
        frameio.draw_text(img, f"face {t.id} age{t.age}", int(x0), max(0, int(y0) - 14),
                          scale=1, colour=col, bg=(0, 0, 0))
    label = f"faces {sum(1 for t in tracks if t.age >= MIN_TRACK_AGE)}  score {det['score']:.2f}"
    frameio.draw_text(img, label, 6, 6, scale=1, colour=(255, 255, 255), bg=(0, 0, 0))
    if show_explain and det["bbox"] is not None:
        y = 24
        for name, val, wt, contrib in det["explain"]:
            if abs(contrib) < 0.01:
                continue
            frameio.draw_text(img, f"{name} {val} x{wt} = {contrib}", 6, y, scale=1,
                              colour=(200, 255, 200), bg=(0, 0, 0))
            y += 12
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--out", default="track.mp4")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--explain", action="store_true")
    a = ap.parse_args()

    outdir = os.path.join(PROJ, "lens", "runs", "video")
    os.makedirs(outdir, exist_ok=True)
    out = a.out if os.path.isabs(a.out) else os.path.join(outdir, a.out)

    print("=" * 96)
    print(f"LIVE FACE TRACKING — {a.seconds:.0f}s from {a.device} @ {FPS} fps -> {out}")
    print(f"  rules: IoU match > {IOU_MATCH} · survive {MAX_MISSING} missing frames · "
          f"EMA alpha {EMA_ALPHA} · report after {MIN_TRACK_AGE} frames")
    print("=" * 96)

    tracks = []
    t0 = time.perf_counter()
    n_frames = n_found = 0
    writer = None
    rows = []
    for frame, i in read_frames(a.seconds, a.device):
        n_frames += 1
        det = FC.detect(frame)
        boxes = []
        if det["found"] and det["bbox"] is not None:
            n_found += 1
            boxes.append(det["bbox"])
        used = set()
        for t in tracks:
            best_j, best_ov = None, 0.0
            for j, b in enumerate(boxes):
                if j in used:
                    continue
                ov = iou(t.box, b)
                if ov > best_ov:
                    best_ov, best_j = ov, j
            if best_j is not None and best_ov >= IOU_MATCH:
                t.update(boxes[best_j])
                used.add(best_j)
            else:
                t.miss()
        for j, b in enumerate(boxes):
            if j not in used:
                tracks.append(Track(b))
        tracks = [t for t in tracks if t.missing == 0 or t.missing <= MAX_MISSING]

        img = draw((frame * 255).astype(np.uint8), tracks, det, show_explain=a.explain)
        if writer is None:
            writer = frameio.VideoWriter(out, W, H, FPS)
        writer.write(img)
        if a.stats:
            rows.append({"frame": i, "found": det["found"], "score": det["score"],
                         "tracks": len([t for t in tracks if t.age >= MIN_TRACK_AGE]),
                         "hole_fraction": det["features"].get("hole_fraction"),
                         "eye_dark_bands": det["features"].get("eye_dark_bands")})
    if writer:
        writer.close()

    dt = time.perf_counter() - t0
    print(f"\n  frames {n_frames}  frames-with-face {n_found} ({n_found/max(n_frames,1)*100:.1f}%)")
    print(f"  distinct tracks created {Track._next - 1}  live at end "
          f"{len([t for t in tracks if t.age >= MIN_TRACK_AGE])}")
    print(f"  wall time {dt:.1f}s -> {n_frames/max(dt,1e-9):.1f} fps processed")
    print(f"  wrote {out} ({os.path.getsize(out)/1e6:.2f} MB)" if os.path.exists(out)
          else "  NO VIDEO WRITTEN")
    if a.stats and rows:
        import json
        p = os.path.join(outdir, os.path.splitext(os.path.basename(out))[0] + "_stats.json")
        json.dump(rows, open(p, "w"), indent=1)
        print(f"  stats -> {p}")
    print("\n  HONEST NOTE: if the owner was not in front of the camera, 'frames-with-face 0' is the")
    print("  CORRECT answer, not a failure. Detection quality on real faces is not claimed until")
    print("  tested on real faces (owner's own captures or a licence-clean set).")


if __name__ == "__main__":
    main()
