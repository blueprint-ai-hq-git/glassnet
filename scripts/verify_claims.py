"""verify_claims.py — does every number I reported trace to a file on disk?

Written because the owner demands proof and hates oversold status. This checks the
specific figures quoted in STATUS.md / REAUDIT-METER-02.md / LOG.md against the
artifacts that produced them, and reports any mismatch. It does NOT trust any summary:
it re-derives the numbers from the raw JSON.
"""
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")
ok, bad = [], []


def check(label, claimed, actual, tol=0.06):
    # 0.06 not 0.05: the reported figures are rounded to 1 decimal, so a true value
    # of 31.25 shows as 31.2 and a 0.05 tolerance fails on float representation.
    good = abs(claimed - actual) <= tol
    (ok if good else bad).append(f"{label}: claimed {claimed} vs artifact {actual}")
    return good


print("=" * 92)
print("CLAIM-vs-ARTIFACT CHECK")
print("=" * 92)

# ---- 1. Meter-02 re-audit numbers -------------------------------------------
re_t2 = json.load(open(os.path.join(R, "reaudit_t2.json")))
EXPECT = {
    "T2 baseline (unsupervised, lam_w 2.0)": (9.3, 9.3),
    "T2 supervised lc=0.1": (19.7, 9.3),
    "T2 supervised lc=0.5": (22.0, 12.0),
    "T2 supervised lc=1.0": (23.0, 12.7),
    "T2 K=8 selection": (0.7, 0.3),
    "T2 K=16 selection": (2.3, 2.3),
    "T2 K=24 selection": (7.0, 6.7),
    "T2 K=32 selection": (9.7, 9.7),
}
for k, (m1, m2) in EXPECT.items():
    v = re_t2.get(k, {})
    if v.get("status") != "ok":
        bad.append(f"reaudit {k}: status {v.get('status')}")
        continue
    check(f"reaudit[{k}].meter01", m1, v["meter01"])
    check(f"reaudit[{k}].floor_2pct", m2, v["floor_2pct"])
    # floor=0 must reproduce the recorded Meter-01 value exactly
    if abs(v["meter01_reproduced"] - v["meter01"]) > 0.05:
        bad.append(f"reaudit {k}: floor=0 gave {v['meter01_reproduced']} "
                   f"vs recorded {v['meter01']} (NOT a pure re-score)")

# ---- 2. rung re-audit --------------------------------------------------------
rr = json.load(open(os.path.join(R, "reaudit_rungs.json")))
RUNG = {"Fashion-MNIST (rung 1)": (13, 12), "PathMNIST (rung 2, iter014)": (8, 8),
        "Banking77 (rung 3, iter018)": (3, 0)}
for k, (old, new) in RUNG.items():
    v = rr.get(k, {})
    if v.get("status") != "ok":
        bad.append(f"rung {k}: status {v.get('status')}")
        continue
    check(f"rung[{k}].old_pass", old, v["old_pass"], 0)
    check(f"rung[{k}].new_pass_2pct", new, v["new_pass_2pct"], 0)

# ---- 3. cluster sweep -------------------------------------------------------
sel = json.load(open(os.path.join(R, "clu_selection.json")))
check("clu selection count", 3, len(sel), 0)
for lc, v in sel.items():
    if v["lr"] != 0.5:
        bad.append(f"cluster selection lc={lc}: lr {v['lr']} != claimed 0.5")
CLU = json.load(open(os.path.join(R, "concept_clusters.json")))
check("clusters above 2% floor", 13, CLU["n_above_floor"], 0)
multi = sum(1 for c in CLU["clusters"] if len(c["words"]) > 1)
check("clusters with >1 member", 9, multi, 0)

# ---- 4. cluster audit pass fractions + split --------------------------------
EXPECT_CLU = {"0.1": (16.3, 39.6, 62.5), "0.5": (17.3, 45.8, 62.5),
              "1.0": (16.7, 41.7, 62.5)}
for lc, (pk, uns, sup) in EXPECT_CLU.items():
    lr = sel[lc]["lr"]
    p = os.path.join(R, f"audit_K32_lr{str(lr).replace('.','')}_pa_lc"
                        f"{str(lc).replace('.','')}_clusters_ff002.json")
    if not os.path.exists(p):
        bad.append(f"cluster audit missing: {os.path.basename(p)}")
        continue
    a = json.load(open(p))
    seeds = [k for k in a if k.startswith("seed")]
    K = a[seeds[0]]["n_total"]
    mp = sum(sum(1 for r in a[s]["rows"] if r["flag"] == "-"
                 and (r.get("firing") or 0) >= 0.02) for s in seeds) / len(seeds)
    check(f"cluster lc={lc} pass/K", pk, mp)
    sp = a.get("split", {})
    check(f"cluster lc={lc} UNSUP", uns, sp.get("uns_pass_frac", 0) * 100)
    check(f"cluster lc={lc} SUP", sup, sp.get("sup_pass_frac", 0) * 100)
    if a.get("lie", {}).get("caught"):
        bad.append(f"cluster lc={lc}: lie probe reported CAUGHT, "
                   f"but the report says MISSED")
    if not a.get("lie", {}).get("live"):
        bad.append(f"cluster lc={lc}: probe not live, but reported live at 8.5%")

# ---- 5. the v1 numbers still on disk (must not have been overwritten) -------
V1 = os.path.join(R, "audit_K32_lr05_pa_lc10_ff002.json")
if os.path.exists(V1):
    a = json.load(open(V1))
    seeds = [k for k in a if k.startswith("seed")]
    mp = sum(a[s]["n_pass"] for s in seeds) / len(seeds)
    check("v1 lc=1.0 pass/K (must still be 12.7)", 12.7, mp)
    check("v1 lc=1.0 SUP (must still be 31.2)",
          31.2, a.get("split", {}).get("sup_pass_frac", 0) * 100)
else:
    bad.append("v1 audit file MISSING — it may have been overwritten")

# ---- 6. live probe records --------------------------------------------------
for f, tag in [("audit_K32_lr05_pa_lc10_ff002.json", "v1 supervised"),
               ("audit_K32_lr10_pa_ff002.json", "v1 baseline")]:
    p = os.path.join(R, f)
    if not os.path.exists(p):
        bad.append(f"live probe artifact missing: {f}")
        continue
    L = json.load(open(p)).get("lie", {})
    print(f"  {tag}: meter={L.get('meter')} floor={L.get('firing_floor')} "
          f"firing={L.get('firing')} caught={L.get('caught')} "
          f"floor_only={L.get('caught_by_floor_only')}")

# ---- 7. METER-03 claims -----------------------------------------------------
M3_EXPECT = {
    ("words", "baseline lc=0 (unsupervised)"): (0.0, 0.0),
    ("words", "single words lc=0.1"): (31.2, 0.0),
    ("words", "single words lc=0.5"): (31.2, 0.0),
    ("words", "single words lc=1.0"): (31.2, 0.0),
    ("clusters", "clusters lc=0.1"): (62.5, 0.0),
    ("clusters", "clusters lc=0.5"): (62.5, 2.1),
    ("clusters", "clusters lc=1.0"): (62.5, 2.1),
}
for (kind, model), (sup, uns) in M3_EXPECT.items():
    p = os.path.join(R, f"meter3_{kind}.json")
    if not os.path.exists(p):
        bad.append(f"meter3_{kind}.json missing")
        break
    a = json.load(open(p))["models"].get(model, {}).get("agg")
    if a is None:
        bad.append(f"meter3 {kind}/{model}: model not in artifact")
        continue
    check(f"meter3[{kind}/{model}].sup", sup, a["sup_pass"])
    check(f"meter3[{kind}/{model}].uns", uns, a["uns_pass_fixed"])
    # the random control MUST be 0% -- that is the claim "no free passes"
    if a["random_sup"] > 0.001 or a["random_uns"] > 0.001:
        bad.append(f"meter3 {kind}/{model}: random control "
                   f"{a['random_sup']}/{a['random_uns']} != 0 -> free passes exist")

# ---- 8. the Meter-03 lie probe ---------------------------------------------
lp = os.path.join(R, "meter3_lieprobe.json")
if os.path.exists(lp):
    L = json.load(open(lp))
    for case, v in L.items():
        if not v.get("caught"):
            bad.append(f"meter3 lie probe {case}: NOT CAUGHT, but reported CAUGHT")
        if v.get("own", {}).get("firing", 0) < 0.02:
            bad.append(f"meter3 lie probe {case}: not live -> the catch would be a "
                       f"floor artifact, not a real catch")
        # best_is_caught means the probe is ALSO caught on its best-scoring target,
        # i.e. the catch is robust to the pairing. The bad case is the opposite:
        # some target is NOT caught, so the "catch" was just an unlucky pairing.
        if not v.get("best_is_caught"):
            bad.append(f"meter3 lie probe {case}: passes its BEST target "
                       f"({v.get('best_target')}) -> the catch was an unlucky pairing")
        print(f"  meter3 lie probe {case}: auroc={v['own']['auroc']:.4f} "
              f"contrib={v['own']['contrib']:+.4f} firing={v['own']['firing']:.4f} "
              f"caught={v['caught']} best_target_caught={v['best_is_caught']}")
else:
    bad.append("meter3_lieprobe.json missing")

# ---- 9. TARGET A2 claims ----------------------------------------------------
a2f = os.path.join(R, "a2_faildiag32.json")
if os.path.exists(a2f):
    d = json.load(open(a2f))
    t = d["taught_tally"]
    check("A2 taught pass count", 8, t["pass"], 0)
    check("A2 taught pass rate", 33.3, t["pass"] / 24 * 100)
    check("A2 held-out pass count", 0, d["held_tally"]["pass"], 0)
    # the central claim: EVERY taught failure is the firing floor
    if t["auroc"] or t["contrib"]:
        bad.append(f"A2 diagnosis: taught failures include AUROC={t['auroc']} "
                   f"CONTRIB={t['contrib']} -- the 'all firing' claim is WRONG")
    if t["firing"] + t["pass"] != 24:
        bad.append("A2 diagnosis: tallies do not sum to 24")
    if d["a2_1_pass"]:
        bad.append("A2-1 recorded as PASS but 33.3% < 50%")
    print(f"  A2: taught {t['pass']}/24 = {t['pass']/24*100:.1f}% "
          f"(FIRING={t['firing']} AUROC={t['auroc']} CONTRIB={t['contrib']}) | "
          f"held-out {d['held_tally']['pass']}/8")
else:
    bad.append("a2_faildiag32.json missing")

a2l = os.path.join(R, "a2_lieprobe.json")
if os.path.exists(a2l):
    L = json.load(open(a2l))
    if not (L.get("caught") and L.get("live")):
        bad.append("A2-2: probe not caught-while-live, but reported PASS")
    if not L.get("best_is_caught"):
        bad.append("A2-2: probe passes its BEST target -- catch was unlucky pairing")
    print(f"  A2-2: caught={L['caught']} live={L['live']} "
          f"best_target_caught={L['best_is_caught']} auroc={L['own']['auroc']:.4f}")
else:
    bad.append("a2_lieprobe.json missing")

a2m = os.path.join(R, "a2_margin.json")
if os.path.exists(a2m):
    m = json.load(open(a2m))
    if not m["A2_4_pass"]:
        bad.append("A2-4 recorded FAIL but reported PASS")
    print(f"  A2-4: dacc {m['dacc']:+.2f} dppl {m['dppl']:+.2f} "
          f"pass={m['A2_4_pass']}")
else:
    bad.append("a2_margin.json missing")

a2a = os.path.join(R, "meter3_clusters32.json")
if os.path.exists(a2a):
    agg = json.load(open(a2a))["models"]["a2 clusters32 (24 taught / 8 held out)"]["agg"]
    if agg["random_sup"] > 0.001 or agg["random_uns"] > 0.001:
        bad.append(f"A2-3: random control {agg['random_sup']}/{agg['random_uns']} "
                   f"!= 0 -> free passes exist")
    print(f"  A2-3: random control SUP {agg['random_sup']:.1f}% "
          f"UNSUP {agg['random_uns']:.1f}%")
else:
    bad.append("meter3_clusters32.json missing")

# ---- 10. Target A must still be recorded NOT MET ---------------------------
st = os.path.join(PROJ, "STATUS.md")
if os.path.exists(st):
    txt = open(st).read()
    if "Target A = NOT MET" not in txt and "Target A NOT MET" not in txt:
        bad.append("STATUS.md no longer records Target A as NOT MET")
    else:
        print("  Target A still recorded NOT MET in STATUS.md")

print(f"\n  checks passed: {len(ok)}")
if bad:
    print(f"  MISMATCHES ({len(bad)}):")
    for b in bad:
        print("    !! " + b)
else:
    print("  MISMATCHES: none — every reported number traces to an artifact")
print("=" * 92)
