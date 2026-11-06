#!/usr/bin/env python3
"""numeric_path_debug.py - WHY does arm B not learn the quantity while arm C does?

Both arms use the same numeric head and the same squared-error loss. Arm C (channel -> head directly)
reaches 100% exact on unseen at lr 0.2; arm B (channel -> MagnitudeLayer -> head, inside the full model)
stays at MAE ~22, i.e. it predicts near zero. This prints the parts every 1000 steps and the head's weights
so the cause is measured instead of guessed. Output: lens/runs/NUMERIC-PATH-DEBUG.txt
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

W, SCALE, V = 4, 40.0, 42
TOK = {i: str(i) for i in range(39)}
TOK[39], TOK[40], TOK[41] = "+", "=", "<pad>"
LR = float(os.environ.get("NUMERIC_LR", "0.2"))


def corpus(lo, hi):
    pairs = [(a, b) for a in range(lo, hi) for b in range(lo, hi)]
    ids = np.asarray([[a, 39, b, 40] for a, b in pairs], dtype=np.int64)
    ytok = np.asarray([a + b for a, b in pairs], dtype=np.int32)
    ynum = (ytok.astype(np.float32) / SCALE).reshape(-1, 1)
    mag, numeric = magnitude_channel(ids, TOK, scale=SCALE)
    return ids, ytok, ynum, mag, 1.0 - numeric


ids_tr, ytok_tr, ynum_tr, mag_tr, mask_tr = corpus(0, 10)
ids_ex, ytok_ex, ynum_ex, mag_ex, mask_ex = corpus(10, 20)

print("=" * 100)
print("ARM B DEBUG - the quantity is not being learned. Parts, weights and predictions, every 1000 steps.")
print("=" * 100)
print(f"lr {LR}. magnitude channel on the TRAIN windows: min {mag_tr.min():.4f} max {mag_tr.max():.4f}")
print()

for seed in (0, 1):
    rng = np.random.default_rng(seed)
    concepts = [Concept(f"digit_{d}", [d], support=float((ids_tr == d).any(axis=1).mean()))
                for d in range(10)]
    net = lens.Net(vocab=V, concepts=concepts, d=32, window=W, seed=seed, numeric=True)
    print(f"seed {seed}")
    for step in range(1, 4001):
        for p in net.parameters():
            p.grad = None
        loss, parts = net.loss(ids_tr, ytok_tr, mag=mag_tr, ynum=ynum_tr, lam_num=1.0, magmask=mask_tr)
        loss.backward()
        for p in net.parameters():
            if p.grad is not None:
                p.data = p.data - LR * p.grad
        if step % 1000 == 0:
            o = net.forward_arrays(ids_tr, mag=mag_tr, magmask=mask_tr)
            p_tr = np.asarray(o["num"]).reshape(-1) * SCALE
            q = np.asarray(o["magnitudes"])
            oe = net.forward_arrays(ids_ex, mag=mag_ex, magmask=mask_ex)
            p_ex = np.asarray(oe["num"]).reshape(-1) * SCALE
            Wn = np.asarray(backend.to_cpu(net.num.Wn.data)).reshape(-1)
            Wm = np.asarray(backend.to_cpu(net.mag.Wm.data)).reshape(-1)
            print(f"  step {step:5d}  ce {parts['ce']:.4f}  concept {parts.get('concept_aux', float('nan')):.4f}"
                  f"  num_mse {parts['num_mse']:.5f}  total {float(backend.to_cpu(loss.data)):.4f}")
            print(f"            q (magnitudes) row0 {np.round(q[0], 4)}  |q| mean {np.abs(q).mean():.4f}"
                  f"  max {np.abs(q).max():.3f}")
            print(f"            head Wn {np.round(Wn, 4)} bn "
                  f"{float(np.asarray(backend.to_cpu(net.num.bn.data)).reshape(-1)[0]):.4f}")
            print(f"            mag-layer |Wm| mean {np.abs(Wm).mean():.4f} max {np.abs(Wm).max():.3f}"
                  f" bm {np.round(np.asarray(backend.to_cpu(net.mag.bm.data)), 4)}")
            print(f"            pred train min {p_tr.min():.2f} max {p_tr.max():.2f} mean {p_tr.mean():.2f}"
                  f"   target mean {ytok_tr.mean():.2f}")
            print(f"            pred unseen min {p_ex.min():.2f} max {p_ex.max():.2f} mean {p_ex.mean():.2f}"
                  f"  exact {float((np.round(p_ex) == ytok_ex).mean())*100:.1f}%")
            # THE MECHANISM: how much does the LEARNED readout contribute where the number itself is known?
            numpos_tr = mag_tr != 0.0
            numpos_ex = mag_ex != 0.0
            learned_tr = np.abs(q - mag_tr)[numpos_tr].mean()
            learned_ex = np.abs(np.asarray(oe["magnitudes"]) - mag_ex)[numpos_ex].mean()
            # the same trained head, fed ONLY the fixed channel (no learned readout at all)
            from lens import tensor as T  # noqa: E402
            only = np.asarray(backend.to_cpu(net.num(T.Tensor(backend.to_dev(mag_ex))).data)).reshape(-1) * SCALE
            print(f"            learned readout at NUMBER positions: train mean|.| {learned_tr:.4f}  "
                  f"unseen mean|.| {learned_ex:.4f}")
            print(f"            same head fed ONLY the fixed channel on unseen: exact "
                  f"{float((np.round(only) == ytok_ex).mean())*100:.1f}%  MAE {np.abs(only-ytok_ex).mean():.3f}")
    print()
