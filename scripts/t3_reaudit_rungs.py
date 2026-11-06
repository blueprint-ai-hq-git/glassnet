"""t3_reaudit_rungs.py — re-score rungs 1/2/3 under Meter-02.

Their meter is a fidelity AUROC (each neuron against its own named concept on
held-out data), not the T2 AUROC+contribution meter. Meter-02 adds the same
firing-rate floor: the neuron must fire on >= 2% of validation windows.

Names are parsed from each rung's OWN audit file so they cannot drift from what was
originally audited. Models load from the surviving checkpoints; nothing is retrained.

Run: python3 scripts/t3_reaudit_rungs.py
"""
import os
import re
import sys
import json

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)

import numpy as np
from glassnet import loop_lib
from glassnet.loop_audit import audit_table
from glassnet.phase2_model import GlassNetPhase2

FLOORS = [0.0, 0.01, 0.02, 0.05]

CASES = [
    ("Fashion-MNIST (rung 1)", "fashion-mnist",
     "runs/phase2/final_seed0.npz", "runs/phase2/audit.md"),
    ("PathMNIST (rung 2, iter014)", "pathmnist",
     "runs/loop/iter014/final_seed0.npz", "runs/loop/iter014/audit.md"),
    ("Banking77 (rung 3, iter018)", "banking77",
     "runs/loop/iter018/final_seed0.npz", "runs/loop/iter018/audit.md"),
]


def names_from_audit(path):
    """Recover the neuron names the rung's own audit used.

    NOTE: phase2/audit.md contains TWO tables — the honest model AND a lie-model
    control. Parsing the whole file yields 28 names for a 14-neuron checkpoint
    (which crashed the loader). So the phase-2 style parse stops at the next
    section heading once it has started collecting.
    """
    txt = open(os.path.join(PROJ, path)).read()
    m = re.search(r"^set \(\d+\):\s*(.+)$", txt, re.M)
    if m:
        return [s.strip() for s in m.group(1).split(",") if s.strip()]
    out = []
    for line in txt.splitlines():
        if line.startswith("## ") and out:
            break                       # first table only
        mm = re.match(r"\|\s*([A-Za-z0-9_]+)\s*\|", line)
        if mm and mm.group(1) not in ("neuron",):
            out.append(mm.group(1))
    return out


out = {}
print("=" * 96)
print("RUNGS 1/2/3 RE-AUDIT UNDER METER-02  (fidelity meter + firing floor)")
print("=" * 96)

for label, task, ckpt, auditmd in CASES:
    print(f"\n{label}   task={task}")
    ck_path = os.path.join(PROJ, ckpt)
    if not os.path.exists(ck_path):
        print("  checkpoint MISSING -> NOT RE-AUDITABLE")
        out[label] = {"status": "not_reauditable", "reason": "no checkpoint"}
        continue
    names = names_from_audit(auditmd)
    if not names:
        print(f"  could not recover names from {auditmd} -> NOT RE-AUDITABLE")
        out[label] = {"status": "not_reauditable", "reason": "no names"}
        continue
    print(f"  {len(names)} named neurons recovered from {auditmd}")
    # guard: the names must match the checkpoint's actual neuron count, or the
    # loader will fail deep inside with an unhelpful broadcast error
    zk = np.load(ck_path)
    Kck = int(zk["concept.W"].shape[1])
    if len(names) != Kck:
        print(f"  !! name count {len(names)} != checkpoint K {Kck} -> NOT RE-AUDITABLE")
        out[label] = {"status": "not_reauditable",
                      "reason": f"{len(names)} names vs K={Kck} in checkpoint"}
        continue

    Xtr, ytr, Xval, yval, Xte, yte = loop_lib.load_split(task=task)
    Ctr, Cval, Cte, lo, hi = loop_lib.load_concept_set(
        Xtr, ytr, Xval, yval, Xte, yte, names, task=task)
    net = GlassNetPhase2.load(ck_path, K=len(names))
    rows, _ = audit_table(net, Xval, Cval, names)

    acts = net.predict(Xval)["concepts"].data
    firing = [(float((np.asarray(acts[:, k]) > 0.5).mean())) for k in range(len(names))]

    print(f"  {'neuron':<22} {'fidelity':>9} {'mean_act':>9} {'firing':>8} "
          f"{'old':>4} {'2%':>4}  flags")
    old_pass = new_pass = 0
    changed = []
    for k, r in enumerate(rows):
        f = firing[k]
        old = bool(r["fidelity"] >= 0.55)
        new = bool(old and f >= 0.02)
        old_pass += int(old)
        new_pass += int(new)
        if old != new:
            changed.append((r["name"], r["fidelity"], f))
        print(f"  {r['name']:<22} {r['fidelity']:>9.3f} {r['mean_act']:>9.3f} "
              f"{f:>8.4f} {'PASS' if old else '    '} {'PASS' if new else '    '}  "
              f"{r['flags']}")
    n = len(names)
    print(f"  -> old {old_pass}/{n} = {old_pass/n*100:.1f}%   "
          f"floor=2% {new_pass}/{n} = {new_pass/n*100:.1f}%   "
          f"changed: {len(changed)}")
    for nm, fid, f in changed:
        print(f"     CHANGED: {nm} fidelity {fid:.3f} but fires only {f:.4f}")
    out[label] = {
        "status": "ok", "task": task, "checkpoint": ckpt, "n": n,
        "names": names, "firing": firing,
        "fidelity": [r["fidelity"] for r in rows],
        "mean_act": [r["mean_act"] for r in rows],
        "flags": [r["flags"] for r in rows],
        "old_pass": old_pass, "new_pass_2pct": new_pass,
        "changed": changed,
    }

json.dump(out, open(os.path.join(PROJ, "runs", "t3", "reaudit_rungs.json"), "w"),
          indent=1)
print("\n" + "=" * 96)
print("saved runs/t3/reaudit_rungs.json")
print("=" * 96)
