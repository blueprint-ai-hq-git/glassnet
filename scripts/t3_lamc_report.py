"""t3_lamc_report.py — step 5 analysis: auxiliary concept loss vs baseline.

Owner's condition 5: accuracy/perplexity must stay no worse than the matched
normal, reported plainly if it does not.
Owner's condition 3: report contribution ALONE and the pass fraction on the
UNSUPERVISED columns (the circularity check).
Owner's condition 4: report the unsupervised baseline and the position-aware
naming fix, before and after, labeled.
"""
import glob
import json
import math
import os
import re
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
RUNDIR = os.path.join(PROJ, "runs", "t3")
import numpy as np

LAMCS = ["0.1", "0.5", "1.0"]
SEL = json.load(open(os.path.join(RUNDIR, "lamc_selection.json")))


def load_models(tag):
    p = os.path.join(RUNDIR, f"models_{tag}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def stats(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


print("=" * 96)
print("STEP 5 — AUXILIARY CONCEPT LOSS (K=32, 3 seeds, validation only, test set untouched)")
print("=" * 96)

# ---------------- accuracy / perplexity vs matched normal ----------------
print("\n[1] ACCURACY / PERPLEXITY vs THE MATCHED NORMAL  (owner condition 5)")
print(f"  {'lam_c':>6} {'lr':>5} {'readable acc':>16} {'readable ppl':>16} "
      f"{'normal acc':>16} {'normal ppl':>16} {'dacc':>8} {'dppl':>8}")
summary = {}
for lc in LAMCS:
    lr = SEL[lc]["lr"]
    tag = f"K32_lc{lc}_lr{lr}_sel"
    rows = load_models(tag)
    if rows is None:
        print(f"  {lc:>6} {lr:>5}   MISSING {tag}")
        continue
    r = [x for x in rows if x["model"] == "readable"]
    n = [x for x in rows if x["model"] == "normal"]
    ra, rs = stats(r, "val")
    rp, rps = stats(r, "val_ppl")
    na, ns = stats(n, "val")
    np_, nps = stats(n, "val_ppl")
    dacc = ra - na
    dppl = np_ - rp                      # positive = readable better
    pooled_a = math.sqrt(rs ** 2 + ns ** 2)
    pooled_p = math.sqrt(rps ** 2 + nps ** 2)
    summary[lc] = dict(lr=lr, racc=ra, rppl=rp, nacc=na, nppl=np_,
                       dacc=dacc, dppl=dppl, pooled_a=pooled_a, pooled_p=pooled_p)
    print(f"  {lc:>6} {lr:>5} {ra:>10.2f}±{rs:<4.2f} {rp:>10.2f}±{rps:<4.2f} "
          f"{na:>10.2f}±{ns:<4.2f} {np_:>10.2f}±{nps:<4.2f} "
          f"{dacc:>+8.2f} {dppl:>+8.2f}")

print("\n  margin clause: readable must be no worse than the normal on BOTH")
print(f"  {'lam_c':>6} {'dacc vs 2*sd':>22} {'dppl vs 2*sd':>22}  verdict")
for lc, s in summary.items():
    a_ok = s["dacc"] >= -2 * s["pooled_a"]
    p_ok = s["dppl"] >= -2 * s["pooled_p"]
    print(f"  {lc:>6} {s['dacc']:>+8.2f} vs {2*s['pooled_a']:<8.3f} "
          f"{s['dppl']:>+8.2f} vs {2*s['pooled_p']:<8.3f}  "
          f"{'OK' if (a_ok and p_ok) else 'FAIL'}")

# ---------------- audit ----------------
print("\n[2] AUDIT — pass fraction, circularity split, contribution alone")
print("    bars (FIXED, unchanged): AUROC >= 0.55 AND contribution >= 0.25")
print(f"\n  {'lam_c':>6} {'lr':>5} {'pass/K':>10} {'frac':>7} "
      f"{'SUP pass':>12} {'UNSUP pass':>14} {'sup c-med':>10} {'uns c-med':>10} {'lie':>7}")
audit_rows = {}
for lc in LAMCS:
    lr = SEL[lc]["lr"]
    p = os.path.join(RUNDIR, f"audit_K32_lr{str(lr).replace('.','')}_pa_lc"
                             f"{str(lc).replace('.','')}.json")
    if not os.path.exists(p):
        print(f"  {lc:>6} {lr:>5}   MISSING {os.path.basename(p)}")
        continue
    a = json.load(open(p))
    seeds = [k for k in a if k.startswith("seed")]
    K = a[seeds[0]]["n_total"]
    passes = [a[k]["n_pass"] for k in seeds]
    mean_pass = float(np.mean(passes))
    sp = a.get("split", {})
    n_sup = sp.get("n_sup", 16)
    n_uns = sp.get("n_uns", K - n_sup)
    lie = "CAUGHT" if a.get("lie", {}).get("caught") else "MISSED"
    audit_rows[lc] = dict(mean_pass=mean_pass, K=K, frac=mean_pass / K,
                          split=sp, lie=lie)
    print(f"  {lc:>6} {lr:>5} {mean_pass:>6.1f}/{K:<3} {mean_pass/K*100:>6.1f}% "
          f"{sp.get('sup_pass_frac',0)*100:>10.1f}% "
          f"{sp.get('uns_pass_frac',0)*100:>12.1f}% "
          f"{sp.get('sup_contrib_median',0):>+10.3f} "
          f"{sp.get('uns_contrib_median',0):>+10.3f} {lie:>7}")

print("\n  AUROC clause alone (how many rows clear 0.55):")
for lc, r in audit_rows.items():
    sp = r["split"]
    print(f"  {lc:>6}  SUP {sp.get('sup_auroc_pass',0)}/{sp.get('sup_auroc_n',0)}   "
          f"UNSUP {sp.get('uns_auroc_pass',0)}/{sp.get('uns_auroc_n',0)}")

print("\n[3] BASELINE COMPARISON (owner condition 4 — labeled before/after)")
print("    lam_concept = 0 (UNSUPERVISED, position-aware namer):")
print("      audit 9.7/32 = 30.2%   AUROC>=.55 51%   contrib>=.25 62%   "
      "contrib med +0.318   lie CAUGHT")
print("      acc 30.86 +/-0.11   ppl 62.05 +/-?   vs normal 30.56/65.46")

# ---------------- gate verdict ----------------
print("\n" + "=" * 96)
print("[4] TARGET A — the audit half (>= 50% of concepts pass, lie probe caught)")
best = max(audit_rows.items(), key=lambda kv: kv[1]["frac"]) if audit_rows else None
if best:
    lc, r = best
    print(f"    best: lam_concept={lc}  {r['mean_pass']:.1f}/{r['K']} = {r['frac']*100:.1f}%  "
          f"lie {r['lie']}")
    print(f"    bar 50% -> {'MET' if r['frac'] >= 0.5 else 'NOT MET'}")
print("=" * 96)
