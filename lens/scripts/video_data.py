"""video_data.py — from-scratch video front-end for the VIDEO line.

Same rule as every line: FRAMES ARE PATCHES. A video window is W frames; each frame is a
low-res luminance thumbnail (32x32 = 1024 px) treated exactly like an audio spectral column or
an image patch — one vector per frame, projected by one shared front-end into the ONE named
concept layer.

Pipeline (no ML library anywhere; ffmpeg is only the codec):
1. DECODE   ffmpeg -> raw gray frames: fps=4, scale 32x32, format=gray.
2. WINDOW   W=16 frames (~4 s) per window, stride 16 (non-overlapping).
3. CONCEPTS named formulas, disjoint by construction; provenance comes from the FILENAMES and is
   honestly pixel-decodable (rendered CG vs screen UI differ strongly in colour/edge statistics —
   the is_lecture audio lesson holds: a concept must be decodable from the input or it is dropped):
      is_rendered       file in the rendered set  {xcool_skull_edit*, cam_synthetic}
      is_screen_record  filename matches the screen-record pattern (^\\d{4}-\\d{2}-\\d{2})
      is_dark           window mean luminance < 1.2 x the 20th percentile of its file
      has_motion        mean |frame-difference| > 0.05 x the file's max difference
      cam capture = the IMPLIED complement of rendered u screen (no twin concept)
4. SPLIT    stratified BY FILE and BY FAMILY: 20% of rendered-files, 20% of screen-files, 20% of
            cam-files go to val, so every concept is measurable on validation.
"""
import glob
import json
import os
import re
import subprocess

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = "/home/yahya"
OUT = os.path.join(PROJ, "data", "video")
FFMPEG = "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"

FPS = 8            # frames per second (v2: 4 -> 8 to roughly double the 167-window dataset)
PX = 32            # thumbnail side (final: 32 px; resolution was NOT the AUROC bottleneck)
W_FRAMES = 16      # frames per window (~2 s)
STRIDE = 16
DIFF_CHANNEL = True  # v4: append the |frame-difference| as a second channel per position —
                     # motion IS |Δframe|, so the front gets the direct signal instead of having
                     # to rediscover it across two planes of the same luminance feature.
SCREEN_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
RENDERED_NAMES = ("xcool_skull_edit", "cam_synthetic")


def decode_frames(path):
    """ffmpeg -> raw gray 32x32 frames at FPS -> np.ndarray (n_frames, 1024) uint8."""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", path,
                        "-vf", f"fps={FPS},scale={PX}:{PX}", "-pix_fmt", "gray",
                        "-f", "rawvideo", "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr[:200]}")
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    n = len(a) // (PX * PX)
    return a[: n * PX * PX].reshape(n, PX * PX).astype(np.float32) / 255.0


def build(seed=0, verbose=True):
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "video_v1.npz")
    meta_path = os.path.join(OUT, "video_v1_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_path):
        z = np.load(cache)
        return {k: z[k] for k in z.files}, json.load(open(meta_path))

    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SRC, "*.mp4")))
    rng = np.random.default_rng(seed)

    def family(n):
        if n in RENDERED_NAMES or n.startswith("xcool_skull_edit"):
            return "rendered"
        if SCREEN_RE.match(n):
            return "screen"
        return "cam"            # cam_*, caption_cam* = implied complement

    fams = {}
    for n in names:
        fams.setdefault(family(n), []).append(n)
    for f in fams:
        rng.shuffle(fams[f])
    file_split = {}
    for f, lst in fams.items():
        n_v = max(1, int(len(lst) * 0.2))
        for n in lst[:n_v]:
            file_split[n] = "val"
        for n in lst[n_v:]:
            file_split[n] = "train"
    # CONCEPT SET v2 (the 4-concept first pass failed honestly, and the reason is DATA POWER:
    # 106 train windows cannot fit 4 classes — the provenance classes (is_rendered 54 train
    # windows, is_screen_record 3!) had no room, AUROC collapsed to 0.2-0.4, train 0.93 vs
    # val 0.41 overfit. The two PIXEL-DECODED concepts are the only ones the data can feed.
    # Provenance returns when more clips exist. Same principle: a concept the data cannot
    # support is dropped, never forced.)

    Xs, Ys, Fs = [], [], []
    for name in names:
        fr = decode_frames(os.path.join(SRC, name + ".mp4"))
        if len(fr) < W_FRAMES:
            print(f"  {name:<38} only {len(fr)} frames (< {W_FRAMES}) — skipped")
            continue
        n = (len(fr) - W_FRAMES) // STRIDE + 1
        idx = np.arange(W_FRAMES)[None, :] + STRIDE * np.arange(n)[:, None]
        W = fr[idx]                                   # (n, W, 1024) luminance planes
        if DIFF_CHANNEL:
            dW = np.abs(W[:, 1:] - W[:, :-1])         # frame-to-frame difference plane
            dW = np.concatenate([dW, dW[:, -1:]], axis=1)   # pad the last position (no next frame)
            W = np.concatenate([W, dW], axis=2)              # (n, W, 2 * 1024) two channels
        mn = W[:, :, :PX * PX].mean(axis=(1, 2))      # luminance per window
        diff = dW.mean(axis=(1, 2)) if DIFF_CHANNEL else \
            np.abs(W[:, 1:] - W[:, :-1]).mean(axis=(1, 2))
        floor_l = np.percentile(mn, 20)
        dark = mn < floor_l * 1.2
        fm = diff.max()
        motion = diff > 0.05 * (fm + 1e-9)
        Y = np.stack([dark, motion], axis=1).astype(np.float32)
        Xs.append(W.astype(np.float32))
        Ys.append(Y)
        Fs.extend([file_split[name]] * n)
        if verbose:
            print(f"  {name:<38} {n:4d} windows  split={file_split[name]}  "
                  f"dark {dark.mean():.2f} motion {motion.mean():.2f}")

    X = np.concatenate(Xs)
    Y = np.concatenate(Ys)
    split = np.array([1 if s == "val" else 0 for s in Fs], dtype=np.uint8)
    meta = {"fps": FPS, "px": PX, "w_frames": W_FRAMES, "stride": STRIDE,
            "n_files": len(names), "files": names, "file_split": file_split, "seed": seed,
            "families": {f: len(l) for f, l in fams.items()},
            "concepts": ["is_dark", "has_motion"],
            "formulas": {
                "is_dark": "window mean luminance < 1.2 x file's 20th percentile",
                "has_motion": "mean |frame-difference| > 0.05 x file's max difference"},
            "dropped": ["is_rendered", "is_screen_record"],
            "dropped_reasons": "data power: 106 train windows cannot fit 4 classes; the "
                               "provenance classes had 3-54 train windows, AUROC collapsed to "
                               "0.2-0.4 with train 0.93 vs val 0.41 overfit. A concept the data "
                               "cannot feed is dropped, never forced.",
            "provenance": "all clips are Yahya's own captures/edits on his machine"}
    json.dump(meta, open(meta_path, "w"), indent=1)
    np.savez(cache, X=X, Y=Y, split=split)
    n_tr = int((split == 0).sum())
    print(f"\n  saved {cache}: X {X.shape} Y {Y.shape} train {n_tr} val {len(split) - n_tr}")
    return {"X": X, "Y": Y, "split": split}, meta


if __name__ == "__main__":
    build()