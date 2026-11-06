#!/usr/bin/env python3
"""make_split.py - write the base train/val split for a corpus that has none yet.

Why this is needed: repair_split.py (which makes val disjoint from train at the training window) READS a
split file and rewrites it. A freshly built corpus has no split file, so the repair step failed with
FileNotFoundError on data/chat/prepared_kn/split_v12000.npy. This creates the base split first.

The split is per conversation (1 = held out), 2% held out, seeded, so it is reproducible. repair_split.py
then moves any held-out conversation whose text appears verbatim in train.

Usage: python3 lens/scripts/make_split.py <corpus-dir> [val-frac] [seed]
"""
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(ROOT)

src = sys.argv[1] if len(sys.argv) > 1 else "data/chat/prepared_kn"
frac = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

bounds = np.load(f"{src}/bounds_v12000.npy")
n_conv = int(len(bounds) - 1)
rng = np.random.default_rng(seed)
split = (rng.random(n_conv) < frac).astype(np.int8)
np.save(f"{src}/split_v12000.npy", split)
print(f"{src}: wrote split_v12000.npy for {n_conv:,} conversations "
      f"({int(split.sum()):,} held out = {split.mean()*100:.2f}%, seed {seed})")
print("next: repair_split.py moves any held-out conversation whose text appears verbatim in train")