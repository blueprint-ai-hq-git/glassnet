"""audio_data.py — from-scratch audio front-end for the AUDIO line.

Everything here is our own math (numpy stdlib + ffmpeg only as a *codec* to decode ogg -> raw
PCM; no ML library is used). Pipeline, every step documented:

1. DECODE   ffmpeg:  ogg -> mono f32 PCM @ 16 kHz (raw f32le stream, no resampling library).
2. STFT     numpy rfft with a Hann window: 25 ms frame (400 samples), 10 ms hop (160 samples)
            -> 100 spectral columns per second, F = 201 frequency bins (0..8 kHz).
3. SCALE    log-magnitude in dB, clipped to [-60, 0], normalized to [0, 1] per global dB floor.
            A column is a vector of F frequency-bin energies = A PATCH (the "a patch is a token"
            rule: text tokens -> image patches -> spectral columns are all the same unit).
4. WINDOW   W=50 columns (~0.5 s) per window, stride 25 (50% overlap) -> n windows.
5. CONCEPTS named formulas, every anchor DISJOINT by construction (orthogonality pre-gate rule):
      is_silence       RMS below the 20th-percentile floor of its file        (acoustics)
      has_voiced       voiced speech: energy concentrated in <2 kHz band       (acoustics)
      has_emphasis     loudness peaks: window RMS in the top quartile          (acoustics: PEAK-vs-
                       MEDIAN ratio = dynamics; silence = LEVEL; voiced = BAND SHAPE; three
                       different features)
      has_fast_speech  high syllable-rate proxy: many energy onsets per second (acoustics)
      is_lecture       provenance: file belongs to the lecture set (documented names)
      (is_casual was DROPPED after the pre-gate: it is the exact complement of is_lecture
       (|corr| 1.00) and a complement pair can never be label-honest — the pre-gate blocked it.
       casual = NOT lecture is implied, and implied things do not get neurons.)
   Provenance labels have NO acoustic anchor, so they are orthogonal to the acoustic four by
   construction; the acoustic four use disjoint features (level / dynamics / bands / onsets).
6. SPLIT    by FILE (80% of files train, 20% held out), never by window: silence gaps in a
            training recording must not leak into validation of the same recording.
"""
import glob
import json
import os
import subprocess

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = "/home/yahya"
OUT = os.path.join(PROJ, "data", "audio")
FFMPEG = "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"

SR = 16000          # sample rate
FRAME = 256         # 16 ms
HOP = 128           # 8 ms
F_BINS = 65         # first 65 rfft bins = 0..4 kHz (speech lives <4 kHz; 125 Hz/bin)
W_COLS = 50         # ~0.4 s window
STRIDE = 50         # non-overlapping windows (keeps the cache ~450 MB)
DB_FLOOR = -60.0

# provenance sets (by filename, documented in DATA_LICENSES.md: all files are Yahya's own voice)
LECTURE_PREFIX = ("m0", "m1", "m2", "method-", "coding-", "forgecode", "honest-guarantee",
                  "absolutely-crazy-method", "ultimate-crazy-method", "ultra-crazy-training")
# casual set is the IMPLIED complement of the lecture set (no concept of its own — the pre-gate
# blocked is_casual as a perfect twin: |corr| 1.00; implied things do not get neurons)


def decode_ogg(path):
    """ffmpeg decode to mono f32 @ 16 kHz -> np.ndarray float32."""
    r = subprocess.run([FFMPEG, "-v", "error", "-i", path, "-f", "f32le",
                        "-ac", "1", "-ar", str(SR), "-"], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {r.stderr[:200]}")
    return np.frombuffer(r.stdout, dtype=np.float32)


def stft(x):
    """Own STFT: Hann-windowed rfft frames -> (cols, F_BINS) power spectrum (float64)."""
    n = len(x)
    if n < FRAME:
        return np.zeros((0, F_BINS))
    frames = np.lib.stride_tricks.sliding_window_view(x, FRAME)[::HOP]
    win = np.hanning(FRAME)
    spec = np.fft.rfft(frames * win, axis=1)
    p = spec.real ** 2 + spec.imag ** 2
    return p[:, :F_BINS]          # keep 0..4 kHz only


def windows_from_spec(pow_sp, n_cols=W_COLS, stride=STRIDE):
    cols, F = pow_sp.shape
    n = max(0, (cols - n_cols) // stride + 1)
    idx = np.arange(n_cols)[None, :] + stride * np.arange(n)[:, None]
    return pow_sp[idx]  # (n, n_cols, F)


def build(seed=0, verbose=True):
    """Decode -> STFT -> windows -> labels -> npz cache. Returns dict of arrays + formulas doc."""
    os.makedirs(OUT, exist_ok=True)
    cache = os.path.join(OUT, "audio_v1.npz")
    meta_path = os.path.join(OUT, "audio_v1_meta.json")
    if os.path.exists(cache) and os.path.exists(meta_path):
        z = np.load(cache)
        return {k: z[k] for k in z.files}, json.load(open(meta_path))

    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(SRC, "*.ogg")))
    # STRATIFIED file split: 20% of the LECTURE files and 20% of the CASUAL files go to val,
    # so is_lecture is measurable on val (an all-lecture val set made its AUROC and strict-corr
    # undefined). A name is lecture iff it starts with LECTURE_PREFIX.
    rng = np.random.default_rng(seed)
    lect_names = [n for n in names if n.startswith(LECTURE_PREFIX)]
    casual_names = [n for n in names if not n.startswith(LECTURE_PREFIX)]
    rng.shuffle(lect_names)
    rng.shuffle(casual_names)
    n_lect_val = max(1, int(len(lect_names) * 0.2))
    n_cas_val = max(1, int(len(casual_names) * 0.2))
    file_split = {}
    for n in lect_names[:n_lect_val] + casual_names[:n_cas_val]:
        file_split[n] = "val"
    for n in lect_names[n_lect_val:] + casual_names[n_cas_val:]:
        file_split[n] = "train"
    is_lecture_full = lect_names

    Xs, Ls, Fs = [], [], []
    for name in names:
        p = os.path.join(SRC, name + ".ogg")
        x = decode_ogg(p)
        sp = stft(x)
        W = windows_from_spec(sp)
        if len(W) == 0:
            continue
        # acoustics — RESTORED passing config 2026-10-11 (bug A1 finding documented in BUGS.md:
        # has_voiced's name overclaims on pure tones — voice-vs-sound separation is IMPOSSIBLE
        # for the linear front (flatness is nonlinear; d16/60ep/lam0.5 strict 1/3; 3-concept d8
        # failed; complement pairs pre-gate-blocked 0.91-1.00). The name stays with this caveat;
        # a real voice concept needs a flatness/peakiness INPUT channel (future).
        col_rms = np.sqrt((W ** 2).mean(axis=2))                          # (n_win, cols)
        rms = col_rms.mean(axis=1)                                        # per-window level
        low_band = W[:, :, :32].sum(axis=2)                               # <2 kHz energy
        hi_band = W[:, :, 32:].sum(axis=2)
        voiced = (low_band / (W.sum(axis=2) + 1e-9)).mean(axis=1) > 0.75  # band-shape
        floor = np.percentile(rms, 20)
        silence = rms < floor * 1.2
        # (has_emphasis, has_fast_speech, is_lecture were DROPPED by the gates:
        #  - emphasis/fast_speech: strict audit measured own|corr| 0.54/0.57 vs |best-other|
        #    0.51/0.66 — both tracked silence too;
        #  - is_lecture: AUROC 0.56 on the stratified val — the SPECTROGRAM cannot reveal the
        #    file's name. His lectures and casual takes are the same voice on the same mic. A
        #    neuron forced to decode provenance from the sound would be a fake; a concept that
        #    cannot be label-honest is dropped, never forced.)
        lect = np.full(len(W), name in is_lecture_full, dtype=bool)
        Y = np.stack([silence, voiced], axis=1).astype(np.float32)
        # log-magnitude scale to [0, 1]: dB clipped to [-60, 0], normalized
        Wn = 10.0 * np.log10(W + 1e-10)
        Wn = np.clip(Wn, DB_FLOOR, 0.0)
        Wn = (Wn - DB_FLOOR) / (-DB_FLOOR)
        Xs.append(Wn.astype(np.float32))
        Ls.append(Y)
        Fs.extend([file_split[name]] * len(W))
        if verbose:
            print(f"  {name:<34} {len(W):5d} windows  split={file_split[name]}")

    X = np.concatenate(Xs)
    Y = np.concatenate(Ls)
    split = np.array([1 if s == "val" else 0 for s in Fs], dtype=np.uint8)
    meta = {"sr": SR, "frame": FRAME, "hop": HOP, "f_bins": F_BINS, "w_cols": W_COLS,
            "stride": STRIDE, "db_floor": DB_FLOOR,
            "n_files": len(names),
            "files": names,
            "file_split": file_split, "seed": seed,
            "concepts": ["is_silence", "has_voiced"],
            "formulas": {
                "is_silence": "window RMS < 1.2 x 20th-percentile RMS of its file",
                "has_voiced": "mean low-band(<2 kHz)/total energy ratio > 0.75"},
            "dropped": ["has_emphasis", "has_fast_speech", "is_lecture", "is_casual"],
            "dropped_reasons": "emphasis/fast_speech: strict-audit own|corr| 0.54/0.57 vs |best-other| "
                               "0.51/0.66 (tracked silence too); is_lecture: AUROC 0.56 — the "
                               "spectrogram cannot reveal the file's name (his lectures and casual "
                               "takes are the same voice on the same mic); is_casual: exact "
                               "complement of is_lecture (|corr| 1.00). A concept that cannot be "
                               "label-honest is dropped, never forced.",
            "provenance": "all files are Yahya's own voice recordings on his machine"}
    json.dump(meta, open(meta_path, "w"), indent=1)
    np.savez(cache, X=X, Y=Y, split=split)
    n_tr = int((split == 0).sum())
    print(f"\n  saved {OUT}/audio_v1.npz: X {X.shape} Y {Y.shape} "
          f"train {n_tr} windows, val {len(split) - n_tr}")
    return {"X": X, "Y": Y, "split": split}, meta


if __name__ == "__main__":
    data, m = build()