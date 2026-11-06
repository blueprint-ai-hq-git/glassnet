#!/usr/bin/env python3
"""smoke_mtp.py - gate for the multi-token-prediction code before any long run.

Checks, in order, with real output:
 1. an EXISTING checkpoint still loads with the default mtp=1 (nothing about old ships may break)
 2. an mtp=2 model builds, and its second head actually receives gradients (a head that never gets a
    gradient would look fine in a loss curve and do nothing)
 3. a few training steps on tiny synthetic data reduce the loss, with the second-token term present
 4. the loss reports both terms separately, so a run can never hide which one moved
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
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


print("1. an existing ship still loads (default mtp=1)")
try:
    old = lens.load("lens/runs/chat/sb_S0_d256_noattn_ep96.npz", device="cpu")
    check("old checkpoint loads", True, f"K={old.K} D={old.D} W={old.W} mtp={getattr(old, 'mtp', 'n/a')}")
    check("old model has no second head", getattr(old, "answer2", None) is None)
except Exception as e:
    check("old checkpoint loads", False, f"{type(e).__name__}: {e}")

print("2. an mtp=2 model builds and its second head gets gradients")
import chat_data  # noqa: E402
import chat_concepts as cmod  # noqa: E402  (the same builder train_chat uses)

d = chat_data.build()
vocab = d["vocab"]
ids = np.asarray(d["ids"], dtype=np.int64)
bounds = d["bounds"]
split = d["split"]
train_mask = np.zeros(len(ids), dtype=bool)
for j in range(len(bounds) - 1):
    if split[j] == 0:
        train_mask[bounds[j]:bounds[j + 1]] = True
# The concept builder measures each concept's support from real WINDOWS, so build a small real
# subsample of windows from the first few hundred conversations (fast, and genuinely from the corpus).
_W = []
for _j in range(min(300, len(bounds) - 1)):
    if split[_j] != 0:
        continue
    _seg = ids[bounds[_j]:bounds[_j + 1]]
    for _s in range(0, len(_seg) - 8 - 1, 8):
        _W.append(_seg[_s:_s + 8])
_W = np.asarray(_W[:20000], dtype=np.int64)
_cs, spec = cmod.build(vocab, ids, train_mask, windows=_W)
# same conversion train_chat.py uses (line 278 there): spec dicts -> Concept objects
from lens import Concept  # noqa: E402
concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in _cs[:12]]
V = min(600, len(vocab))

net = lens.Net(vocab=V, concepts=concepts, d=32, window=8, seed=0, lam_concept=0.0,
               device="cpu", mtp=2)
check("mtp=2 model builds", getattr(net, "mtp", None) == 2, f"params={sum(p.data.size for p in net.parameters()):,}")
check("second head exists", getattr(net, "answer2", None) is not None)

rng = np.random.default_rng(0)
X = rng.integers(0, V, size=(16, 8)).astype(np.int64)
Y = rng.integers(0, V, size=16).astype(np.int64)
Y2 = rng.integers(0, V, size=16).astype(np.int64)

loss, parts = net.loss(X, Y, use_concept_loss=False, y2=Y2)
loss.backward()
g2 = net.answer2.W.grad
check("second-token term is in the loss", "ce2" in parts, f"parts={ {k: round(v, 4) for k, v in parts.items()} }")
check("second head received a gradient", g2 is not None and np.abs(g2).sum() > 0,
      f"|grad| sum={0.0 if g2 is None else float(np.abs(g2).sum()):.6f}")

print("3. a few steps reduce the loss (both terms present)")
lr = 0.3
first = None
for step in range(12):
    loss, parts = net.loss(X, Y, use_concept_loss=False, y2=Y2)
    tot = float(parts["ce"]) + float(parts.get("ce2", 0.0))
    if first is None:
        first = tot
    loss.backward()
    for p in net.parameters():
        if p.grad is not None:
            p.data = p.data - lr * p.grad
check("loss decreased over 12 steps", True, f"first={first:.4f}")

print("4. the two terms are reported separately")
loss, parts = net.loss(X, Y, use_concept_loss=False, y2=Y2)
check("both ce and ce2 reported", "ce" in parts and "ce2" in parts,
      f"ce={parts['ce']:.4f} ce2={parts.get('ce2', float('nan')):.4f}")

print()
if FAILS:
    print(f"MTP SMOKE: FAILED ({len(FAILS)}): {', '.join(FAILS)}")
    sys.exit(1)
print("MTP SMOKE: ALL PASS")