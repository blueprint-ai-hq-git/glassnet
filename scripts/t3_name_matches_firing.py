"""t3_name_matches_firing.py — is the "name matched to its own firings" mechanism
general, or was it only the lie-probe column?

The probe showed a dead column (fires on 1.15% of windows) whose name's support was
0.0115 — the same number. If that is the general mechanism, then across ALL passing
rows the name's support (pos_frac) should track the column's own firing rate, and
most tightly for the sparsest columns.

No retraining: audit values come from the recorded JSONs, firing rates from the
saved checkpoints. Columns matched by index.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))
os.environ.setdefault("T3A_BACKEND", "numpy")
os.environ.setdefault("T3A_K", "32")

import numpy as np
from glassnet.tensor import set_backend
import t3_audit as A
import run_t2_iter003 as M

set_backend("numpy")
CKPT = os.path.join(PROJ, "runs", "t3", "ckpt")
RUNDIR = os.path.join(PROJ, "runs", "t3")

CASES = [
    ("baseline lc=0  lr=1.0", "audit_K32_lr10_pa_w20.json", 1.0, "_w20", None),
    ("lc=0.1  lr=1.0      ", "audit_K32_lr10_pa_lc01.json", 1.0, "_lc01", 16),
    ("lc=0.5  lr=1.0      ", "audit_K32_lr10_pa_lc05.json", 1.0, "_lc05", 16),
    ("lc=1.0  lr=0.5      ", "audit_K32_lr05_pa_lc10.json", 0.5, "_lc10", 16),
]

d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
V, K, D, WIN = d["V"], A.K, M.D, A.WIN

print("=" * 96)
print("DOES THE NAME JUST TRACK THE COLUMN'S OWN FIRINGS?  (all passing rows)")
print("=" * 96)
print(f"  {'case':<22} {'n pass':>7} {'corr(pos_frac,fire)':>20} "
      f"{'med|diff|':>10} {'max|diff|':>10}")

allrows = {}
for label, jf, lr, suffix, n_sup in CASES:
    jp = os.path.join(RUNDIR, jf)
    if not os.path.exists(jp):
        print(f"  {label:<22}  MISSING {jf}")
        continue
    aud = json.load(open(jp))
    seeds = [k for k in aud if k.startswith("seed")]
    rows = []
    for s in seeds:
        si = int(s.replace("seed", ""))
        ck = os.path.join(CKPT, f"readable_K32_lr{lr}{suffix}_seed{si}.npz")
        if not os.path.exists(ck):
            continue
        net = M.TinyLM3(V, K, D, WIN, rng=np.random.default_rng(si))
        z = np.load(ck)
        for p in net.parameters():
            if p.name in z:
                p.data = np.array(z[p.name])
        h = np.asarray(net.forward(d["Xva"])["concepts"].data)
        fire = (h > 0.5).mean(axis=0)
        for i, r in enumerate(aud[s]["rows"]):
            if r["flag"] == "-" and r.get("pos_frac") is not None:
                rows.append({"pos_frac": r["pos_frac"], "fire": float(fire[i]),
                             "auroc": r["auroc"], "col": i,
                             "sup": (n_sup is not None and i < n_sup)})
    allrows[label] = rows
    if not rows:
        print(f"  {label:<22}  no passing rows")
        continue
    pf = np.array([r["pos_frac"] for r in rows])
    fr = np.array([r["fire"] for r in rows])
    diff = np.abs(pf - fr)
    corr = float(np.corrcoef(pf, fr)[0, 1]) if pf.std() > 0 and fr.std() > 0 else float("nan")
    print(f"  {label:<22} {len(rows):>7} {corr:>20.3f} {np.median(diff):>10.4f} "
          f"{diff.max():>10.4f}")

print("\n" + "=" * 96)
print("SPARSEST PASSING COLUMNS (pos_frac < 5%) — name support vs actual firing")
print("=" * 96)
print(f"  {'case':<22} {'n':>4} {'median pos_frac':>16} {'median firing':>14} "
      f"{'median|diff|':>13}")
for label, rows in allrows.items():
    sp = [r for r in rows if r["pos_frac"] < 0.05]
    if not sp:
        print(f"  {label:<22} {0:>4}   (none)")
        continue
    pf = np.array([r["pos_frac"] for r in sp])
    fr = np.array([r["fire"] for r in sp])
    print(f"  {label:<22} {len(sp):>4} {np.median(pf):>16.4f} {np.median(fr):>14.4f} "
          f"{np.median(np.abs(pf-fr)):>13.4f}")

print("\n  If median|diff| is near zero on the sparse rows, the name's support IS the")
print("  column's firing rate — i.e. the namer is describing the column's own firings,")
print("  and AUROC on those rows is high by construction rather than by detection.")
print("=" * 96)
