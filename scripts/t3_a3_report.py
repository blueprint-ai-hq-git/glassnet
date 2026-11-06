"""t3_a3_report.py — TARGET A3 evaluation: per-column diagnosis, the four clauses, the
margin clause, the predictions scoreboard, and TARGET-A3-RESULTS.md.

Every number is read from an artifact. Nothing is typed from memory.
"""
import json
import math
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet.tensor import set_backend
from glassnet import tiny_data
import t3_audit_m3 as M3

RUNDIR = os.path.join(PROJ, "runs", "t3")
K, N_TAUGHT, N_HOLD = 16, 10, 4          # TARGET-A3.md: teach 10 qualifying, hold out 4

print("=" * 116)
print("TARGET A3 — EVALUATION")
print("=" * 116)

sel = json.load(open(os.path.join(RUNDIR, "a3_selection.json")))
LR, LC = sel["lr"], sel["lc"]
LCS = LC.replace(".", "")
PAT = f"readable_K16_lr{LR}_lc{LCS}_clusters16_seed{{s}}.npz"
print(f"selected setting: lr={LR} lam_concept={LC}  ({sel['n_seeds']}-seed mean "
      f"ppl {sel['ppl']:.4f} acc {sel['acc']:.2f}%)")

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
set_backend("numpy")
targets = M3.load_targets("clusters16")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters16.json")))
supports = [c["support"] for c in CLU["clusters"]]

per_col = {c: [] for c in range(K)}
for seed in (0, 1, 2):
    path = os.path.join(RUNDIR, "ckpt", PAT.format(s=seed))
    if not os.path.exists(path):
        print(f"  seed{seed}: MISSING {path}")
        continue
    net = M3.load_ckpt(M3.build_net(seed), path)
    h, L, A = M3.prep(net, X)
    for c in range(K):
        per_col[c].append(M3.audit_col(net, X, h, c, targets[c][1], L, A))


def clause_of(fr, au, co):
    if fr < 0.02:
        return "FIRING (<2%)"
    if au < 0.55:
        return "AUROC (<0.55)"
    if co < 0.25:
        return "CONTRIB (<0.25)"
    return "-"


rows = {}
for c in range(K):
    rs = per_col[c]
    if not rs:
        continue
    rows[c] = {"col": c, "support": supports[c], "firing": float(np.mean([r["firing"] for r in rs])),
               "contrib": float(np.mean([r["contrib"] for r in rs])),
               "auroc": float(np.mean([r["auroc"] for r in rs])),
               "words": "_".join(CLU["clusters"][c]["words"][:4])}
    rows[c]["clause"] = clause_of(rows[c]["firing"], rows[c]["auroc"], rows[c]["contrib"])


def band(lo, hi, title):
    tally = {"firing": 0, "auroc": 0, "contrib": 0, "pass": 0}
    print(f"\n--- {title}")
    print(f"{'col':>3} {'support':>8} {'firing':>8} {'contrib':>9} {'auroc':>7} "
          f"{'verdict':>8}  clause            target")
    for c in range(lo, hi):
        r = rows.get(c)
        if not r:
            continue
        cl = r["clause"]
        key = ("pass" if cl == "-" else
               "firing" if cl.startswith("FIRING") else
               "auroc" if cl.startswith("AUROC") else "contrib")
        tally[key] += 1
        print(f"{c:>3} {r['support']:>8.4f} {r['firing']:>8.4f} {r['contrib']:>+9.4f} "
              f"{r['auroc']:>7.4f} {('-' if cl == '-' else 'BROKEN'):>8}  {cl:<16} "
              f"{r['words'][:24]}")
    n = hi - lo
    print(f"    tally over {n}: FIRING={tally['firing']} AUROC={tally['auroc']} "
          f"CONTRIB={tally['contrib']} pass={tally['pass']} -> "
          f"{tally['pass']/n*100:.1f}%")
    return tally


t_taught = band(0, N_TAUGHT, f"TAUGHT columns 0-{N_TAUGHT-1} (A3-1 uses these)")
t_hold = band(N_TAUGHT, N_TAUGHT + N_HOLD,
              f"OFFICIAL HOLD-OUT columns {N_TAUGHT}-{N_TAUGHT+N_HOLD-1} "
              f"(4 highest-support below 2.5%)")
t_extra = band(N_TAUGHT + N_HOLD, K, f"EXTRA untaught columns {N_TAUGHT+N_HOLD}-{K-1}")

# ------------------------------------------------------------------ the four clauses
m3p = os.path.join(RUNDIR, "meter3_clusters16.json")
agg = None
if os.path.exists(m3p):
    m = json.load(open(m3p))
    key = [k for k in m["models"] if k.startswith("a3 ")][0]
    agg = m["models"][key]["agg"]

taught_rate = t_taught["pass"] / N_TAUGHT
A3_1 = taught_rate >= 0.5

probe_path = os.path.join(RUNDIR, "a3_lieprobe.json")
probe = json.load(open(probe_path)) if os.path.exists(probe_path) else None
A3_2 = bool(probe and probe["caught"] and probe["live"])

A3_3 = bool(agg and agg["random_sup"] < 5.0 and agg["random_uns"] < 5.0)

norm = json.load(open(os.path.join(RUNDIR, "models_a3_normal.json")))
read_ = json.load(open(os.path.join(RUNDIR, f"models_a3_cfm_lc{LC}_lr{LR}.json")))
scr = json.load(open(os.path.join(RUNDIR, "a3_screen.json")))
seed0 = [s for s in scr["screened"] if s["lc"] == LC and s["lr"] == LR][0]
read_all = [{"val": seed0["acc"], "val_ppl": seed0["ppl"]}] + \
           [x for x in read_ if x["model"] == "readable"]
n_all = [x for x in norm if x["model"] == "normal"]
ra = float(np.mean([x["val"] for x in read_all]))
rp = float(np.mean([x["val_ppl"] for x in read_all]))
na = float(np.mean([x["val"] for x in n_all]))
np_ = float(np.mean([x["val_ppl"] for x in n_all]))
rs_ = float(np.std([x["val"] for x in read_all], ddof=1))
ns_ = float(np.std([x["val"] for x in n_all], ddof=1))
rps = float(np.std([x["val_ppl"] for x in read_all], ddof=1))
nps = float(np.std([x["val_ppl"] for x in n_all], ddof=1))
dacc, dppl = ra - na, np_ - rp
pool_a, pool_p = math.sqrt(rs_**2 + ns_**2), math.sqrt(rps**2 + nps**2)
A3_4 = (dacc >= -2 * pool_a) and (dppl >= -2 * pool_p)

A3_MET = A3_1 and A3_2 and A3_3 and A3_4

print("\n" + "=" * 116)
print("TARGET A3 — THE FOUR CLAUSES")
print("=" * 116)
print(f"A3-1 taught pass >= 50%          : {t_taught['pass']}/{N_TAUGHT} = "
      f"{taught_rate*100:.1f}%  -> {'PASS' if A3_1 else 'FAIL'}")
print(f"A3-2 lie probe caught while live : "
      f"{'PASS' if A3_2 else 'FAIL'}"
      + (f" (auroc {probe['own']['auroc']:.4f} contrib {probe['own']['contrib']:+.4f} "
         f"firing {probe['own']['firing']:.4f})" if probe else ""))
print(f"A3-3 random control < 5%         : "
      f"{agg['random_sup']:.1f}%/{agg['random_uns']:.1f}% -> {'PASS' if A3_3 else 'FAIL'}"
      if agg else "A3-3 random control < 5%         : no audit artifact")
print(f"A3-4 margin (both clauses)       : dacc {dacc:+.2f} (2*pooled {2*pool_a:.3f}), "
      f"dppl {dppl:+.2f} (2*pooled {2*pool_p:.3f}) -> {'PASS' if A3_4 else 'FAIL'}")
print(f"\nhold-out: {t_hold['pass']}/{N_HOLD} untaught targets ever passed "
      f"| extra untaught: {t_extra['pass']}/{K-N_TAUGHT-N_HOLD}")
print(f"\n*** TARGET A3 = {'MET' if A3_MET else 'NOT MET'} ***")

out = {"selected": {"lr": LR, "lc": LC, "n_seeds": sel["n_seeds"]},
       "taught": t_taught, "holdout": t_hold, "extra": t_extra,
       "rows": rows, "agg": agg,
       "A3_1": bool(A3_1), "A3_2": A3_2, "A3_3": A3_3, "A3_4": bool(A3_4),
       "A3_MET": bool(A3_MET),
       "margin": {"readable_acc": ra, "readable_ppl": rp, "normal_acc": na,
                  "normal_ppl": np_, "dacc": dacc, "dppl": dppl,
                  "pool_a": pool_a, "pool_p": pool_p},
       "probe": probe}
json.dump(out, open(os.path.join(RUNDIR, "a3_eval.json"), "w"), indent=1)
print("\nsaved runs/t3/a3_eval.json")
print("=" * 116)
