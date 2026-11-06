"""t3_reaudit_t2.py — re-score the T2 concept audits under Meter-02.

Loads each recorded audit's checkpoints, re-derives the names with the SAME namer
and support setting the audit used, and re-runs the meter at floors 0 / 1% / 2% / 5%.
One naming pass per seed; all four floors come from the recorded firing rates.

Writes runs/t3/reaudit_t2.json and prints old vs new side by side.
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
FLOORS = [0.0, 0.01, 0.02, 0.05]

# The results the owner named, plus the K-sweep selections that were the ladder.
TARGETS = [
    ("T2 baseline (unsupervised, lam_w 2.0)", "audit_K32_lr10_pa_w20.json"),
    ("T2 supervised lc=0.1", "audit_K32_lr10_pa_lc01.json"),
    ("T2 supervised lc=0.5", "audit_K32_lr10_pa_lc05.json"),
    ("T2 supervised lc=1.0", "audit_K32_lr05_pa_lc10.json"),
    ("T2 K=8 selection", "audit_K8_lr10.json"),
    ("T2 K=16 selection", "audit_K16_lr10.json"),
    ("T2 K=24 selection", "audit_K24_lr05_pa.json"),
    ("T2 K=32 selection", "audit_K32_lr10_pa.json"),
]

d = A.tiny_data.build(win=A.WIN, vocab_cap=A.VOCAB_CAP)
V, D, WIN = d["V"], M.D, A.WIN

# index available checkpoints: (K, lr_string, suffix) -> {seed: path}
avail = {}
for fn in os.listdir(CKPT):
    if not fn.startswith("readable_") or not fn.endswith(".npz"):
        continue
    base = fn[len("readable_"):-len(".npz")]
    parts = base.split("_")
    Kp = int(parts[0][1:])
    lrp = parts[1][2:]                       # "lr0.5" -> "0.5"
    seed = int([p for p in parts if p.startswith("seed")][0][4:])
    suffix = "_" + "_".join(parts[2:-1]) if len(parts) > 3 else ""
    avail.setdefault((Kp, lrp, suffix), {})[seed] = os.path.join(CKPT, fn)

out = {}
OUTJSON = os.path.join(RUNDIR, "reaudit_t2.json")
# resumable: a silent kill must not cost the whole run
if os.path.exists(OUTJSON):
    try:
        prev = json.load(open(OUTJSON))
        out.update({k: v for k, v in prev.items() if v.get("status") == "ok"})
        if out:
            print(f"resuming: {len(out)} target(s) already scored")
    except Exception:
        pass
print("=" * 100)
print("T2 RE-AUDIT UNDER METER-02  (re-score of the SAME checkpoints; no retraining)")
print("=" * 100)
print(f"  {'result':<40} {'Meter-01':>9} {'1%':>7} {'2% (rule)':>10} {'5%':>7}  note")

for label, jf in TARGETS:
    if label in out:
        print(f"  {label:<40}  (already scored — skipped)")
        continue
    jp = os.path.join(RUNDIR, jf)
    if not os.path.exists(jp):
        print(f"  {label:<40}  MISSING {jf}")
        out[label] = {"status": "missing_file", "file": jf}
        continue
    rec = json.load(open(jp))
    seeds = sorted(int(k[4:]) for k in rec if k.startswith("seed"))
    # which namer / support did this audit use?
    namer_kind = "pa" if "_pa" in jf else "old"
    sup = 0.01
    for tag in ("_s005", "_s01", "_s02"):
        if tag in jf:
            sup = {"_s005": 0.005, "_s01": 0.01, "_s02": 0.02}[tag]
    lam = 0.2
    for tag in ("_lam05", "_lam10"):
        if tag in jf:
            lam = {"_lam05": 0.5, "_lam10": 1.0}[tag]
    lc = 0.0
    for tag in ("_lc01", "_lc05", "_lc10"):
        if tag in jf:
            lc = {"_lc01": 0.1, "_lc05": 0.5, "_lc10": 1.0}[tag]
    w = 1.0
    for tag in ("_w20", "_w50"):
        if tag in jf:
            w = {"_w20": 2.0, "_w50": 5.0}[tag]
    Kaud = int(jf.split("_")[1][1:])

    def ck_suffix(lam, w, lc):
        parts = []
        if lam != 0.2:
            parts.append(f"lam{str(lam).replace('.', '')}")
        if w != 1.0:
            parts.append(f"w{str(w).replace('.', '')}")
        if lc != 0:
            parts.append(f"lc{str(lc).replace('.', '')}")
        return "_" + "_".join(parts) if parts else ""

    sfx = ck_suffix(lam, w, lc)
    # audit filenames encode lr with the dot stripped ("lr10" = 1.0), so compare the
    # same way. Strip ".json" too: a filename with no namer tag (audit_K8_lr10.json)
    # otherwise yields "10.json" and never matches.
    lr_tag = jf.split("_")[2][2:].replace(".json", "")
    lrs = sorted({lrp for (Kp, lrp, s2) in avail
                  if Kp == Kaud and s2 == sfx
                  and str(float(lrp)).replace(".", "") == lr_tag})
    if not lrs:
        print(f"  {label:<40}  no checkpoint matches (K={Kaud}, lr={lr_tag}, "
              f"lam={lam}, w={w}, lc={lc}) -> NOT RE-AUDITABLE")
        out[label] = {"status": "not_reauditable", "file": jf,
                      "reason": f"no checkpoint for K={Kaud} lr={lr_tag} "
                                f"lam={lam} w={w} lc={lc}"}
        continue
    lrp = lrs[0]
    ckmap = avail[(Kaud, lrp, sfx)]

    old_counts, floor_counts = [], {f: [] for f in FLOORS}
    rows_all = []
    for seed in seeds:
        ck = ckmap.get(seed)
        if ck is None:
            continue
        net = M.TinyLM3(V, Kaud, D, WIN, rng=np.random.default_rng(seed))
        z = np.load(ck)
        for p in net.parameters():
            if p.name in z:
                p.data = np.array(z[p.name])
        # the audit's own naming configuration
        base_namer = M.auto_name_t2c_pa if namer_kind == "pa" else M.auto_name_t2c
        names = base_namer(net, d, d["Xtr"], min_support=sup)
        rows, _ = M.audit_t2c(net, names, d, d["Xva"], min_firing=0.0)
        rows_all.append(rows)
        old_counts.append(rec[f"seed{seed}"]["n_pass"])
        for f in FLOORS:
            floor_counts[f].append(sum(
                1 for r in rows
                if r["flag"] == "-" and (r.get("firing") or 0.0) >= f))

    if not floor_counts[0.0]:
        print(f"  {label:<40}  checkpoints unreadable")
        continue
    o = float(np.mean(old_counts))
    vals = {f: float(np.mean(floor_counts[f])) for f in FLOORS}
    note = ""
    if abs(vals[0.0] - o) > 0.05:
        note = f"!! re-score floor=0 gives {vals[0.0]:.1f} vs recorded {o:.1f}"
    print(f"  {label:<40} {o:>6.1f}/{Kaud:<3} {vals[0.01]:>5.1f}/{Kaud:<3} "
          f"{vals[0.02]:>6.1f}/{Kaud:<3} {vals[0.05]:>5.1f}/{Kaud:<3}  {note}")
    out[label] = {
        "status": "ok", "file": jf, "K": Kaud, "seeds": seeds,
        "meter01": o, "meter01_reproduced": vals[0.0],
        "floor_1pct": vals[0.01], "floor_2pct": vals[0.02], "floor_5pct": vals[0.05],
        "pct_2pct": vals[0.02] / Kaud * 100,
        "per_seed_firing": [float(np.mean([(r.get("firing") or 0) for r in rw]))
                            for rw in rows_all],
        "note": note,
    }
    # write after EVERY target so a kill cannot lose finished work
    json.dump(out, open(OUTJSON, "w"), indent=1)
    sys.stdout.flush()

json.dump(out, open(os.path.join(RUNDIR, "reaudit_t2.json"), "w"), indent=1)
print("\n" + "=" * 100)
print("Meter-01 column is the recorded number; floor=0 column re-derives it from the")
print("checkpoints (they must agree). 2% is the pre-registered rule.")
print(f"saved {os.path.join(RUNDIR, 'reaudit_t2.json')}")
print("=" * 100)
