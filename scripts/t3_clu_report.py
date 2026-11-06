"""t3_clu_report.py — condition 4 results: word-cluster targets under Meter-02.

Reports, for each selected lam_concept:
  * accuracy / perplexity vs the matched normal, with the margin clause
  * the Meter-02 audit at floors 0/1/2/5%, with the SUPERVISED vs UNSUPERVISED split
  * the cluster support table (from CONCEPT-TARGETS-02.md method)
  * explicit checks of the pre-registered predictions
  * a side-by-side against v1 single words and the unsupervised baseline
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
SEL = json.load(open(os.path.join(RUNDIR, "clu_selection.json")))
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters.json")))


def load(tag):
    p = os.path.join(RUNDIR, f"models_{tag}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def st(rows, key):
    v = [r[key] for r in rows]
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


print("=" * 100)
print("CONDITION 4 — WORD-CLUSTER CONCEPT TARGETS, Meter-02 audit")
print("=" * 100)

# ---------------- cluster supports ----------------
above = CLU["n_above_floor"]
multi = sum(1 for c in CLU["clusters"] if len(c["words"]) > 1)
print(f"\n[0] TARGETS: {above}/{CLU['k']} clusters have TRAIN support >= 2% "
      f"(v1 single words: 8/16); {multi}/{CLU['k']} have more than one member")
print(f"    PREDICTION 1 (>= 12/16 clear the floor): "
      f"{'CONFIRMED' if above >= 12 else 'NOT CONFIRMED'}")

# ---------------- accuracy / perplexity ----------------
print(f"\n[1] ACCURACY / PERPLEXITY vs THE MATCHED NORMAL")
print(f"  {'lam_c':>6} {'lr':>5} {'readable acc':>16} {'readable ppl':>16} "
      f"{'normal acc':>16} {'normal ppl':>16} {'dacc':>8} {'dppl':>8}")
summary = {}
for lc in LAMCS:
    if lc not in SEL:
        continue
    lr = SEL[lc]["lr"]
    rows = load(f"K32_clu{lc}_lr{lr}_sel")
    if rows is None:
        print(f"  {lc:>6} {lr:>5}   MISSING")
        continue
    r = [x for x in rows if x["model"] == "readable"]
    n = [x for x in rows if x["model"] == "normal"]
    ra, rs = st(r, "val"); rp, rps = st(r, "val_ppl")
    na, ns = st(n, "val"); npp, nps = st(n, "val_ppl")
    summary[lc] = dict(ra=ra, rp=rp, na=na, npp=npp, dacc=ra - na, dppl=npp - rp,
                       pa=math.sqrt(rs ** 2 + ns ** 2),
                       pp=math.sqrt(rps ** 2 + nps ** 2))
    print(f"  {lc:>6} {lr:>5} {ra:>10.2f}±{rs:<4.2f} {rp:>10.2f}±{rps:<4.2f} "
          f"{na:>10.2f}±{ns:<4.2f} {npp:>10.2f}±{nps:<4.2f} "
          f"{ra-na:>+8.2f} {npp-rp:>+8.2f}")
print(f"\n  margin clause (readable no worse than normal on BOTH):")
for lc, s in summary.items():
    ok = (s["dacc"] >= -2 * s["pa"]) and (s["dppl"] >= -2 * s["pp"])
    print(f"    lc={lc}: dacc {s['dacc']:+.2f} vs {2*s['pa']:.3f} | "
          f"dppl {s['dppl']:+.2f} vs {2*s['pp']:.3f}  -> {'OK' if ok else 'FAIL'}")

# ---------------- Meter-02 audit ----------------
print(f"\n[2] METER-02 AUDIT (AUROC>=0.55 AND contrib>=0.25 AND firing>=floor)")
print(f"  {'lam_c':>6} {'floor':>6} {'pass/K':>10} {'frac':>7} {'SUP':>7} {'UNSUP':>7} "
      f"{'sup c-med':>10} {'uns c-med':>10} {'lie':>8}")
audit_rows = {}
for lc in LAMCS:
    if lc not in SEL:
        continue
    lr = SEL[lc]["lr"]
    p = os.path.join(RUNDIR, f"audit_K32_lr{str(lr).replace('.','')}_pa_lc"
                             f"{str(lc).replace('.','')}_clusters_ff002.json")
    if not os.path.exists(p):
        print(f"  {lc:>6}   MISSING {os.path.basename(p)}")
        continue
    a = json.load(open(p))
    seeds = [k for k in a if k.startswith("seed")]
    K = a[seeds[0]]["n_total"]
    sp = a.get("split", {})
    lie = a.get("lie", {})
    n_sup = sp.get("n_sup", 16)
    for floor in (0.0, 0.01, 0.02, 0.05):
        counts = []
        for s in seeds:
            counts.append(sum(1 for r in a[s]["rows"]
                              if r["flag"] == "-"
                              and (r.get("firing") or 0.0) >= floor))
        mp = float(np.mean(counts))
        sup = uns = None
        if floor == 0.02:
            audit_rows[lc] = dict(pass_K=mp, K=K, frac=mp / K, split=sp, lie=lie)
        print(f"  {lc:>6} {floor*100:>5.0f}% {mp:>6.1f}/{K:<3} {mp/K*100:>6.1f}% "
              + (f"{sp.get('sup_pass_frac',0)*100:>6.1f}% "
                 f"{sp.get('uns_pass_frac',0)*100:>6.1f}% "
                 f"{sp.get('sup_contrib_median',0):>+10.3f} "
                 f"{sp.get('uns_contrib_median',0):>+10.3f} "
                 f"{('CAUGHT' if lie.get('caught') else 'MISSED'):>8}"
                 if floor == 0.02 else ""))

# ---------------- prediction checks ----------------
print(f"\n[3] PRE-REGISTERED PREDICTION CHECKS (CONCEPT-TARGETS-02.md §4)")
if audit_rows:
    best = max(audit_rows.items(), key=lambda kv: kv[1]["frac"])
    lc, r = best
    uns = r["split"].get("uns_pass_frac", 0) * 100
    sup = r["split"].get("sup_pass_frac", 0) * 100
    print(f"  best at 2% floor: lam_concept={lc}  {r['pass_K']:.1f}/{r['K']} = "
          f"{r['frac']*100:.1f}%")
    print(f"  PRED 2 (supervised >= 90%): {sup:.1f}% -> "
          f"{'CONFIRMED' if sup >= 90 else 'NOT CONFIRMED'}")
    print(f"  PRED 3 (unsupervised 25-40%): {uns:.1f}% -> "
          f"{'CONFIRMED' if 25 <= uns <= 40 else 'NOT CONFIRMED'}")
    print(f"  PRED 4 (50% bar NOT met on the unsupervised half): "
          f"{'CONFIRMED' if uns < 50 else 'NOT CONFIRMED — the bar IS met'}")
    print(f"  lie probe: {'CAUGHT' if r['lie'].get('caught') else 'MISSED'}"
          + (f" (floor only: {r['lie'].get('caught_by_floor_only')})"
             if r['lie'] else ""))

print(f"\n[4] SIDE BY SIDE (2% floor)")
print(f"  {'run':<34} {'pass/K':>10} {'frac':>8} {'SUP':>7} {'UNSUP':>7}")
print(f"  {'v1 single words, lc=1.0':<34} {'12.7/32':>10} {'39.6%':>8} "
      f"{'31.2%':>7} {'47.9%':>7}")
print(f"  {'unsupervised baseline (lc=0)':<34} {'9.3/32':>10} {'29.2%':>8} "
      f"{'—':>7} {'—':>7}")
for lc, r in audit_rows.items():
    print(f"  {'v2 clusters, lc=' + lc:<34} "
          f"{r['pass_K']:>7.1f}/{r['K']:<3} {r['frac']*100:>7.1f}% "
          f"{r['split'].get('sup_pass_frac',0)*100:>6.1f}% "
          f"{r['split'].get('uns_pass_frac',0)*100:>6.1f}%")
print("=" * 100)
