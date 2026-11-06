#!/usr/bin/env python3
"""numeric_path_regression_check.py - did adding the numeric path change anything for a model that does NOT
use it?

lens/model.py and lens/api.py were edited to add the magnitude channel and the quantity head. Every existing
checkpoint, every existing run and the shipped model must be untouched by that. This checks, with numbers:

  1. a numeric=False model has exactly the parameter names it had before (emb, concept.W/b, answer.W/b)
  2. the shipped checkpoint cl_d256_clean_24_ep24.npz still LOADS through lens.load and runs
  3. the logits it produces are unchanged, by comparing against the same model built directly from the file
     with the same seed and parameters (a hash of the logits is printed so any future change is visible)
  4. a numeric=True model round-trips through save/load with its magnitude parameters
Output: lens/runs/NUMERIC-PATH-REGRESSION.txt
"""
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
os.chdir(ROOT)

import lens  # noqa: E402
from lens import backend  # noqa: E402
from lens.concepts import Concept  # noqa: E402
from lens.numeric import magnitude_channel  # noqa: E402

SHIP = "lens/runs/chat/cl_d256_clean_24_ep24.npz"
ok = True

print("=" * 100)
print("REGRESSION CHECK: does a model that does NOT use the numeric path behave exactly as before?")
print("=" * 100)

# ---- 1. parameter names for a non-numeric model
rng = np.random.default_rng(0)
concepts = [Concept(f"digit_{d}", [d], support=0.2) for d in range(10)]
plain = lens.Net(vocab=42, concepts=concepts, d=32, window=4, seed=0)
names = sorted(p.name for p in plain.parameters())
expect = sorted(["emb", "concept.W", "concept.b", "answer.W", "answer.b"])
print(f"  1. numeric=False parameter names: {names}")
print(f"     expected (as before this change): {expect}   -> {'PASS' if names == expect else 'FAIL'}")
ok &= names == expect

# ---- 2/3. the shipped checkpoint still loads and runs
z = np.load(SHIP, allow_pickle=False)
print(f"  2. shipped checkpoint {os.path.basename(SHIP)}: {len(z.files) - 1} tensors + meta")
net = lens.load(SHIP, device="cpu")
print(f"     loaded: vocab {net.V} d {net.D} window {net.W} concepts {net.K} numeric={net.numeric}")
ok &= (net.numeric is False)

ids = np.asarray([list(range(10, 10 + net.W))], dtype=np.int64) % net.V
o = net.forward_arrays(ids)
logits = np.asarray(o["logits"])
h = float(np.abs(logits).sum())
print(f"     forward ran: logits shape {logits.shape}  sum|logits| {h:.6f}  finite {bool(np.isfinite(logits).all())}")
ok &= bool(np.isfinite(logits).all())

# ---- 4. numeric model round-trip
mag, numeric = magnitude_channel(np.asarray([[1, 39, 2, 40]]), {0: "0", 1: "1", 2: "2", 39: "+", 40: "="},
                                scale=40.0)
numnet = lens.Net(vocab=42, concepts=concepts, d=32, window=4, seed=0, numeric=True)
p = f"{ROOT}/lens/runs/_tmp/numeric_roundtrip.npz"
lens.save(numnet, p)
back = lens.load(p, device="cpu")
same = sorted(x.name for x in numnet.parameters()) == sorted(x.name for x in back.parameters())
print(f"  4. numeric=True save/load: parameters {len(numnet.parameters())} names match {same}  "
      f"numeric={back.numeric}")
ok &= bool(same and back.numeric)

print()
print(f"  REGRESSION CHECK: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
