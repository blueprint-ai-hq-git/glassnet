"""formula_ceiling.py — is the last margin shortfall a property of the MODEL or of the FORMULAS?

The strict audit asks: |corr(neuron_i, formula_i)| - max_{j!=i} |corr(neuron_i, formula_j)| > 0.10.
Four independent architecture changes (capacity d128/d160/d224, raw-pixel access) left that one
margin pinned at ~0.05 with own ~0.80 and rival ~0.75. So measure the rival PAIR directly: how
correlated are the two FORMULAS' own incidence labels on the same val windows? If |corr(formula_i,
formula_j)| is already ~0.75, then a neuron that tracks formula_i faithfully cannot avoid tracking
formula_j, and the 0.10 bar is unreachable BY CONSTRUCTION -- no architecture can fix a property of
the label definitions. That is the difference between 'the model failed' and 'the bar is not
achievable for this pair', and it must be measured, not asserted.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import train_seetalk as TS                     # noqa: E402

SHIP = os.environ.get("VISION_SHIP", "seetalk_k21n_direct.npz")
N = int(os.environ.get("VISION_AUDIT_N", "60000"))


def main():
    ckpt = os.path.join(PROJ, "lens", "runs", "seetalk", SHIP)
    data = TS.build_caption_data(n=N, seed=0, img_start=0)
    net, meta = TS.SeeTalkNet.load(ckpt)
    Xv = data["Xva"]
    names = [c.name for c in net.concepts]
    L = np.stack([c.labels(Xv) for c in net.concepts], axis=1).astype(np.float64)
    K = L.shape[1]
    C = np.corrcoef(L.T)
    off = ~np.eye(K, dtype=bool)
    print(f"ship {SHIP}  val windows {len(Xv)}  concepts {K}")
    print(f"  FORMULA-FORMULA |corr|: mean {np.abs(C[off]).mean():.4f}  max {np.abs(C[off]).max():.4f}  "
          f"pairs > 0.5: {int((np.abs(C[off]) > 0.5).sum() // 2)}  pairs > 0.7: "
          f"{int((np.abs(C[off]) > 0.7).sum() // 2)}")
    # the audit's own failing pair, named
    i = names.index("left_right_symmetric_rel")
    rivals = [(abs(C[i, j]), names[j]) for j in range(K) if j != i]
    rivals.sort(reverse=True)
    print(f"  formula left_right_symmetric_rel vs its rivals (|corr| of the LABELS themselves):")
    for v, nm in rivals[:3]:
        print(f"      {nm:<28} {v:.4f}")
    print(f"  -> the audit's failing rival was top_left_mass at 0.753.")
    print(f"  CEILING CHECK: a neuron that tracks formula_i at correlation A cannot score a margin")
    print(f"  above (A - |corr(formula_i, formula_j)|) unless it also fits NOISE. With A ~ 0.80 and")
    print(f"  the measured pair correlation above, the largest honest margin is approximately")
    print(f"  {0.80 - rivals[0][0]:.3f} -- compare the audit's 0.10 bar.")


if __name__ == "__main__":
    main()
