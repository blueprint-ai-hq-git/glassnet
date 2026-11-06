#!/usr/bin/env python3
"""enforce_drop.py - the project's own doctrine applied to the readability failure.

DOCTRINE: a named unit that cannot be honest about its name is DROPPED, never excused. The random
readability gate found 2334 fires with no evidence present (only 17.2% of fires were honest), and the
soft lever (lam_act, a penalty on mean activation) barely moved it: 2197 false fires at lam_act 0.1.

WHY A DROP IS THE RIGHT TOOL HERE: lam_act penalises the MEAN activation, so it pushes every unit down a
little. It does not make units selective. A unit that fires on 31% of windows while its own tokens appear
in 5% of them needs to be removed or silenced, not nudged.

What this does, in order, and it reports every step:
 1. Measures every unit on the SAME random draw the gate uses: fires, honest fires, false fires, and the
    false-fire rate. A unit is HONEST if, whenever it fires, its declared tokens are present.
 2. Lists the units that fail, worst first. Nothing is dropped silently: the list is printed and saved.
 3. Zeroes the outgoing weights of the dropped units in the answer layer, so the model's answers can no
    longer use them, and prints how much of the answer layer that removed.
 4. Re-measures the model and writes the before/after table.

It does NOT retrain the answer layer here (that is a separate, deliberate step). Zeroing first shows how
much of the answer depended on the lying units, which is the number worth knowing before retraining.

Usage: SHIP=... python3 lens/scripts/enforce_drop.py [--max-false-rate 0.5] [--out path.npz]
"""
import argparse
import json
import os
import sys
from collections import Counter

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
import chat_data  # noqa: E402

N = int(os.environ.get("N", "600"))
SEED = int(os.environ.get("SEED", "20261018"))


def measure(net, X):
    """Per-unit fire / honest-fire / false-fire counts on the same random draw the gate uses."""
    K = len(net.concepts)
    fires = np.zeros(K, dtype=np.int64)
    honest = np.zeros(K, dtype=np.int64)
    for w in X:
        h = np.asarray(net.forward_arrays(w[None, :])["concepts"])[0]
        wt = set(int(t) for t in w)
        for i in np.flatnonzero(h > 0.5):
            fires[i] += 1
            declared = set(int(t) for t in np.asarray(net.concepts[i].ids).reshape(-1))
            if declared & wt:
                honest[i] += 1
    false = fires - honest
    return fires, honest, false


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ship", default=os.environ.get(
        "SHIP", "lens/runs/chat/rx_lamact_d256_24_ep24.npz"))
    ap.add_argument("--max-false-rate", type=float, default=0.5,
                    help="a unit is dropped when this share of its fires have no evidence present")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    net = lens.load(a.ship, device="cpu")
    d = chat_data.build()
    vocab = d["vocab"]
    ids = np.asarray(d["ids"], dtype=np.int64)
    Xva, _ = chat_data.make_windows(ids, d["bounds"], d["split"], net.W, which=1)
    rng = np.random.default_rng(SEED)
    X = Xva[rng.choice(len(Xva), size=min(N, len(Xva)), replace=False)]

    fires, honest, false = measure(net, X)
    print(f"enforce_drop on {os.path.basename(a.ship)}")
    print(f"  measured on {len(X)} random held-out windows (seed {SEED})")
    print(f"  units: {len(fires)}   total fires {int(fires.sum())}   honest {int(honest.sum())}   "
          f"false {int(false.sum())}   honest share "
          f"{honest.sum()/max(fires.sum(),1)*100:.1f}%")

    rate = np.divide(false, np.maximum(fires, 1), dtype=np.float64)
    drop = [i for i in range(len(fires)) if fires[i] > 0 and rate[i] > a.max_false_rate]
    silent = [i for i in range(len(fires)) if fires[i] == 0]
    print(f"\n  units failing the honesty bar (> {a.max_false_rate:.0%} of fires with no evidence): "
          f"{len(drop)} of {len(fires)}")
    print(f"  units that never fired on this draw: {len(silent)} "
          f"({', '.join(net.concepts[i].name for i in silent[:8])}{'...' if len(silent) > 8 else ''})")
    print("  the drop list, worst first:")
    for i in sorted(drop, key=lambda j: -rate[j])[:20]:
        print(f"    {net.concepts[i].name:<24} fires {fires[i]:>4}  false {false[i]:>4}  "
              f"false-rate {rate[i]*100:5.1f}%")

    if not drop:
        print("\n  nothing to drop at this bar")
        return 0

    # zero the dropped units' outgoing weights in the answer layer: the answer can no longer use them
    W = np.asarray(net.answer.W.data)
    touched = float(np.abs(W[:, drop]).sum() / max(np.abs(W).sum(), 1e-12))
    before = W.copy()
    W[:, drop] = 0.0
    net.answer.W.data = W
    print(f"\n  zeroed the outgoing weights of {len(drop)} units in the answer layer: "
          f"{touched*100:.1f}% of that layer's weight mass removed")

    fires2, honest2, false2 = measure(net, X)
    print(f"  after the drop: fires {int(fires2.sum())}  honest {int(honest2.sum())}  "
          f"false {int(false2.sum())}  honest share "
          f"{honest2.sum()/max(fires2.sum(),1)*100:.1f}%")
    print(f"  (units still firing: {int((fires2 > 0).sum())}; the dropped units can no longer contribute)")
    out = a.out or f"lens/runs/chat/{os.path.basename(a.ship).replace('.npz','')}_dropped.npz"
    lens.save(net, out)
    print(f"  saved: {out}")
    json.dump({"ship": a.ship, "n": len(X), "seed": SEED, "max_false_rate": a.max_false_rate,
               "dropped": [net.concepts[i].name for i in drop],
               "honest_share_before": float(honest.sum()/max(fires.sum(), 1)),
               "honest_share_after": float(honest2.sum()/max(fires2.sum(), 1)),
               "weight_mass_removed": touched, "saved": out},
              open(f"lens/runs/drop_{os.path.basename(a.ship).replace('.npz','')}.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())