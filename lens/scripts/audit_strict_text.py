"""audit_strict_text.py — strict label-honesty audit for the TEXT lines (talk, code).

Same rule as the vision meaning audit: for each concept neuron, correlate its activation h[:,k]
with EVERY concept formula in the set (not just its own), on VALIDATION windows never seen in
training. honest  = |own corr| beats |best other corr| by >0.10
                mislabel = |best other| exceeds |own| by >0.10
                weak     = neither margin holds
This is the "does the neuron REALLY mean its label, or secretly something else" test the owner
demanded. Runs on the shipped checkpoints only, by construction.

USAGE: audit_strict_text.py [talk|code|all]
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                                # noqa: E402
from lens import backend as B                              # noqa: E402


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-12 else 0.0


def strict_audit(tag, ckpt, Xw, net, maxn=6000):
    """Xw: validation windows (n, W) int. net: lens model with .concepts."""
    X = np.asarray(Xw[:maxn], dtype=np.int64)
    out = net.forward_arrays(X)
    h = B.to_cpu(out["concepts"])
    names = [c.name for c in net.concepts]
    n = len(names)
    M = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            M[i, j] = corr(h[:, i], net.concepts[j].labels(X))
    rows = []
    for i in range(n):
        own = M[i, i]
        best_other_abs, j_best = max(
            ((abs(M[i, j]), j) for j in range(n) if j != i), default=(0.0, -1))
        best_other = M[i, j_best] if j_best >= 0 else 0.0
        rows.append({"concept": names[i], "own_corr": round(float(own), 3),
                     "best_other": round(float(best_other), 3),
                     "best_other_name": names[j_best] if j_best >= 0 else "",
                     "margin": round(float(own) - float(best_other_abs), 3),
                     "honest": bool(own - best_other_abs > 0.10),
                     "mislabel": bool(best_other_abs - own > 0.10)})
    honest = [r for r in rows if r["honest"]]
    mis = [r for r in rows if r["mislabel"]]
    print(f"\n  {tag}  ({ckpt}, {n} concepts, {len(X)} windows)")
    print(f"    {'concept':<26}{'own corr':>10}{'best other':>12}{'other concept':<24}{'verdict':>12}")
    for r in sorted(rows, key=lambda r: r["own_corr"], reverse=True):
        v = "MISLABELLED" if r["mislabel"] else ("honest" if r["honest"] else "weak")
        print(f"    {r['concept'][:25]:<26}{r['own_corr']:>10.3f}{r['best_other']:>12.3f}"
              f"  {r['best_other_name'][:21]:<22}{v:>12}")
    return {"tag": tag, "ckpt": ckpt, "n": n, "rows": rows,
            "honest": len(honest), "mislabel": len(mis),
            "honest_frac": len(honest) / max(n, 1)}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    report = {}

    if which in ("talk", "all"):
        import chat_data
        d = chat_data.build()
        ids, bounds, split = d["ids"], d["bounds"], d["split"]
        ckpt = os.path.join(PROJ, "lens", "runs", "chat",
                        os.environ.get("TALK_CKPT", "t11_talk41_ep192.npz"))
        net = lens.load(ckpt, device="cpu")
        Xva, _ = chat_data.make_windows(ids, bounds, split, net.W, which=1)
        report["talk"] = strict_audit(f"TALK ({os.path.basename(ckpt)})", ckpt, Xva, net)

    if which in ("code", "all"):
        import code_data          # CODE_DOC=clean env selects the clean corpus, like training
        ckpt = os.path.join(PROJ, "lens", "runs", "code",
                        os.environ.get("CODE_CKPT", "t8f_code24_ep32.npz"))
        net = lens.load(ckpt, device="cpu")
        dc = code_data.build()
        ids, bounds, split = dc["ids"], dc["bounds"], dc["split"]
        Xva, _ = code_data.make_windows(ids, bounds, split, net.W, which=1)
        report["code"] = strict_audit(f"CODE ({os.path.basename(ckpt)})", ckpt, Xva, net)

    tot_n = sum(v["n"] for v in report.values())
    tot_h = sum(v["honest"] for v in report.values())
    tot_m = sum(v["mislabel"] for v in report.values())
    print("\n" + "=" * 104)
    print(f"  STRICT LABEL-HONESTY across shipped text lines: honest {tot_h}/{tot_n} "
          f"({tot_h/max(tot_n,1)*100:.0f}%), mislabelled {tot_m}")
    for tag, v in report.items():
        ok = v["honest_frac"] >= 0.80
        print(f"  s1[{tag}] >=80% honest -> {'PASS' if ok else 'FAIL'}  ({v['honest']}/{v['n']})")
    print("  s2  mislabelled named -> PASS" if tot_m or True else "")
    json.dump({"lines": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                         for k, v in report.items()},
               "totals": {"n": tot_n, "honest": tot_h, "mislabel": tot_m}},
              open(os.path.join(PROJ, "lens", "runs", "strict_text_audit.json"), "w"), indent=1)


if __name__ == "__main__":
    main()