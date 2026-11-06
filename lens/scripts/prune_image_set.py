"""prune_image_set.py — pick the largest image concept subset whose formula-formula
|corr| stays under a bound, using the SHIP's own datum (the medical method that took
the medical line from 2/15 to 4/4 strict).

Reads the ship named by VISION_SHIP, rebuilds the val datum exactly as
audit_strict_vision.py does, computes the K x K formula-incidence correlation matrix,
then brute-forces the largest subset with max-pair |corr| <= BOUND (default 0.60),
reporting the dropped names with their max correlation (the dossier).
"""
import itertools
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import train_seetalk as TS            # noqa: E402
from lens import backend as B         # noqa: E402

BOUND = float(os.environ.get("PRUNE_BOUND", "0.60"))
SHIP = os.environ.get("VISION_SHIP", "seetalk_k21d.npz")


def main():
    lj = os.path.join(PROJ, "lens", "runs", "seetalk",
                      SHIP.replace(".npz", "_log.json"))
    args = json.load(open(lj))["args"] if os.path.exists(lj) else {}
    n = int(args.get("n", 60000))
    data = TS.build_caption_data(n=n, seed=0, img_start=0)
    net, meta = TS.SeeTalkNet.load(os.path.join(PROJ, "lens", "runs", "seetalk", SHIP))
    Xva = np.asarray(data["Xva"], dtype=np.float32)
    K = len(net.concepts)
    names = [c.name for c in net.concepts]
    inc = np.zeros((len(Xva), K), dtype=np.float64)
    for k, c in enumerate(net.concepts):
        v = c.fn(Xva)
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 1 and len(v) != len(Xva):
            v = v.reshape(len(Xva), -1).mean(axis=1)
        inc[:, k] = v
    C = np.abs(np.corrcoef(inc.T))
    np.fill_diagonal(C, 0)
    print(f"ship {SHIP}: K={K}, datum {len(Xva)} windows")
    print(f"formula |corr|: max {C.max():.3f}, mean {C[np.triu_indices(K,1)].mean():.3f}, "
          f"pairs>0.8 {int((C[np.triu_indices(K,1)]>0.8).sum())}")

    # largest subset under BOUND: start from the full set, drop the concept with the worst
    # max-corr to the others until the bound holds (deterministic, no randomness)
    keep = list(range(K))
    dropped = []
    while True:
        sub = C[np.ix_(keep, keep)]
        if sub.max() <= BOUND:
            break
        worst, worst_i = -1.0, None
        for i in keep:
            others = [j for j in keep if j != i]
            m = C[i, others].max()
            if m > worst:
                worst, worst_i = m, i
        dropped.append((names[worst_i], worst))
        keep.remove(worst_i)
    kept_names = [names[i] for i in keep]
    print(f"\nKEPT ({len(kept_names)}) at max-pair <= {BOUND}:")
    for nm in kept_names:
        print(f"    {nm}")
    print(f"\nDROPPED ({len(dropped)}) with their max |corr| to the kept set:")
    for nm, m in dropped:
        print(f"    {nm:<28} max |corr| {m:.3f}")
    out = {"ship": SHIP, "bound": BOUND, "kept": kept_names,
           "dropped": [{"name": nm, "max_corr": round(m, 4)} for nm, m in dropped],
           "max_pair_kept": float(C[np.ix_(keep, keep)].max())}
    json.dump(out, open(os.path.join(PROJ, "lens", "runs", "seetalk", "image_prune.json"),
                        "w"), indent=1)
    print("\nwritten lens/runs/seetalk/image_prune.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())