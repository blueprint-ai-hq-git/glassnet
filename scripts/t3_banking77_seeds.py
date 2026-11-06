"""t3_banking77_seeds.py — is the Banking77 dead-neuron finding robust?

The iter018 re-score differs slightly from the recorded fidelity values, which
suggests the surviving checkpoint is not the exact audited model. So the finding
"3 neurons pass at fidelity >= 0.55 while firing on 0.0000 of validation windows"
must be checked across ALL THREE surviving seeds, not just seed 0.

If a neuron fires on 0.0000 of validation windows, its fidelity AUROC is computed
from a constant-ish signal — the meter is measuring nothing.
"""
import os
import re
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import numpy as np
from glassnet import loop_lib
from glassnet.loop_audit import audit_table
from glassnet.phase2_model import GlassNetPhase2

names = [s.strip() for s in re.search(
    r"^set \(\d+\):\s*(.+)$",
    open(os.path.join(PROJ, "runs/loop/iter018/audit.md")).read(),
    re.M).group(1).split(",") if s.strip()]
print(f"Banking77: {len(names)} neurons")

Xtr, ytr, Xval, yval, Xte, yte = loop_lib.load_split(task="banking77")
Ctr, Cval, Cte, lo, hi = loop_lib.load_concept_set(
    Xtr, ytr, Xval, yval, Xte, yte, names, task="banking77")

print(f"\n{'seed':>4} {'pass(old)':>10} {'pass(2%)':>10} {'zero-fire neurons':>34} "
      f"{'fidelity of zero-fire ones':>28}")
for seed in (0, 1, 2):
    ck = os.path.join(PROJ, f"runs/loop/iter018/final_seed{seed}.npz")
    if not os.path.exists(ck):
        print(f"{seed:>4}  missing checkpoint")
        continue
    net = GlassNetPhase2.load(ck, K=len(names))
    rows, _ = audit_table(net, Xval, Cval, names)
    acts = net.predict(Xval)["concepts"].data
    firing = [float((np.asarray(acts[:, k]) > 0.5).mean()) for k in range(len(names))]
    old = sum(1 for r in rows if r["fidelity"] >= 0.55)
    new = sum(1 for r, f in zip(rows, firing)
              if r["fidelity"] >= 0.55 and f >= 0.02)
    zero = [(r["name"], round(r["fidelity"], 3)) for r, f in zip(rows, firing)
            if f == 0.0]
    znames = ", ".join(n for n, _ in zero) if zero else "-"
    zfid = ", ".join(f"{fid}" for _, fid in zero) if zero else "-"
    print(f"{seed:>4} {old:>6}/16 {new:>8}/16   {znames:>34} {zfid:>28}")
print("\nAny neuron listed as zero-fire with fidelity >= 0.55 is a PASS the meter")
print("was granting to a neuron that never fires on validation data.")
