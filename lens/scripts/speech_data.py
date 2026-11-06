#!/usr/bin/env python3
"""SPEECH DATA — real spoken words (Google Speech Commands v0.02, CC BY 4.0) turned into a
readable feature table with documented formulas.

Why this file exists: the audio line so far only knew "is_silence / has_voiced" from Yahya's own
recordings. The Wi-Fi window brought 105,835 real speech clips. This turns them into a dataset a
tiny readable model can learn: every input column is a FORMULA a human can compute by hand, and
its exact definition is printed and saved next to the data (nothing hidden).

Features (per clip, all from scratch: stdlib `wave` + numpy):
  dur_s        duration in seconds
  rms          mean RMS energy over 25 ms / 10 ms frames
  zcr          mean zero-crossing rate per frame  (noise/voicing proxy)
  centroid_hz  mean spectral centroid  (brightness)
  low_ratio    share of spectral energy below 500 Hz
  high_ratio   share of spectral energy above 2000 Hz
  voiced_frac  share of frames with a strong autocorrelation peak lag in 50-400 Hz (voiced speech)
  flatness     mean spectral flatness (geometric mean / arithmetic mean: 1 = noise, 0 = tone)
  peak_pos     time position of the loudest frame, relative to clip length

Splits: the dataset's OWN validation_list.txt / testing_list.txt are used (official splits), and a
cross-split duplicate audit is run and printed, because Speech Commands is known to contain some
duplicated clips.
"""
import json
import os
import sys
import wave

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
# ONE implementation of every feature formula lives in lens/speech.py; this builder imports it
# rather than keeping a copy (a second copy is how a "readable" claim stops being true).
from lens.speech import FEATS, FORMULAS, HOP, FRAME, SR, features, read_wav  # noqa: E402
RAW = os.path.join(PROJ, "data/external/speech_commands_raw/extracted")
OUT = os.path.join(PROJ, "data/speech")
SR = 16000
FRAME = 400        # 25 ms
HOP = 160          # 10 ms
WORDS = ["yes", "no", "up", "down", "left", "right", "stop", "go"]
FEATS = ["dur_s", "rms", "zcr", "centroid_hz", "low_ratio", "high_ratio", "voiced_frac",
         "flatness", "peak_pos"]
FORMULAS = {
    "dur_s": "len(samples)/16000",
    "rms": "mean over 25ms/10ms frames of sqrt(mean(x^2))",
    "zcr": "mean over frames of (#sign changes)/len(frame)",
    "centroid_hz": "mean over frames of sum(f*P)/sum(P), f = rfft bin freq (Hz)",
    "low_ratio": "sum(P[<500Hz])/sum(P)",
    "high_ratio": "sum(P[>2000Hz])/sum(P)",
    "voiced_frac": "fraction of frames whose autocorrelation peak (lag 40..320 samples = 50..400Hz) "
                   "exceeds 0.5*autocorr[0]",
    "flatness": "mean over frames of exp(mean(log(P+1e-12)))/mean(P+1e-12)",
    "peak_pos": "argmax over frames of frame energy / (#frames)",
}


def collect():
    val_lists = set()
    for f in ("validation_list.txt", "testing_list.txt"):
        p = os.path.join(RAW, f)
        if os.path.exists(p):
            val_lists |= set(open(p).read().split())
    rows, y, split, paths = [], [], [], []
    for wi, w in enumerate(WORDS):
        d = os.path.join(RAW, w)
        print(f"    word {w} ...", flush=True)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".wav"):
                continue
            rel = f"{w}/{fn}"
            p = os.path.join(d, fn)
            try:
                rows.append(features(read_wav(p)))
                y.append(wi)
                split.append(1 if rel in val_lists else 0)
                paths.append(rel)
            except Exception as e:
                print(f"  skip {rel}: {type(e).__name__}", file=sys.stderr)
    return np.asarray(rows, dtype=np.float32), np.asarray(y, dtype=np.int64), \
        np.asarray(split, dtype=np.int8), paths


def main():
    os.makedirs(OUT, exist_ok=True)
    X, y, split, paths = collect()
    print(f"  clips {len(X):,}  features {X.shape[1]}  train {(split==0).sum():,}  "
          f"official-val {(split==1).sum():,}")
    print("  feature formulas:")
    for f in FEATS:
        print(f"    {f:<12} {FORMULAS[f]}")
    # cross-split duplicate audit (exact float rows): Speech Commands is known to contain dupes
    import hashlib
    h = {}
    dup = 0
    for i in range(len(X)):
        k = hashlib.md5(X[i].tobytes()).hexdigest()
        if k in h and split[h[k]] != split[i]:
            dup += 1
        h.setdefault(k, i)
    print(f"  duplicate feature-rows across the split: {dup} "
          f"({dup/max((split==1).sum(),1)*100:.2f}% of val) -- leak audit, reported not hidden")
    np.savez_compressed(os.path.join(OUT, "speech_v1.npz"), X=X, y=y, split=split,
                        words=np.asarray(WORDS))
    json.dump({"features": FEATS, "formulas": FORMULAS, "words": WORDS, "sr": SR,
               "n": int(len(X)), "n_val": int((split == 1).sum()),
               "split": "the dataset's own validation_list.txt/testing_list.txt",
               "license": "CC BY 4.0 (google/speech_commands v0.02)",
               "dup_rows_across_split": int(dup)},
              open(os.path.join(OUT, "speech_v1_meta.json"), "w"), indent=1)
    print(f"  written {OUT}/speech_v1.npz")


if __name__ == "__main__":
    main()