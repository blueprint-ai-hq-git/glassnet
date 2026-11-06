"""target_i_verdict.py — apply Target I's frozen bars to the two arms.

I-FAILING  : pronoun_subject and is_a_dialogue_line must each pass in >= 2 of 3 seeds at lam_act=0
I-READABLE : >= 50% of concepts pass overall at lam_act=0
I-COVERAGE : 100.0000%
I-NOWORSE  : readable perplexity no worse than the matched normal (accuracy is degenerate)
I-CONTROL  : the lam_act=0.2 arm must reproduce 7/9 at D=64 with the same two failing
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")
FAILING = ["pronoun_subject", "is_a_dialogue_line"]


def load(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else None


def summarise(rows, label):
    """rows: the JSON list. Returns per-concept pass counts for the readable arm."""
    rd = [x for x in rows if x["kind"] == "readable"]
    if not rd:
        return None
    names = [r["name"] for r in rd[0]["rows"]]
    out = {"n_runs": len(rd), "concepts": {}, "coverage": [x["coverage"] for x in rd],
           "ppl": [x["ppl"] for x in rd], "acc": [x["acc"] for x in rd],
           "n_pass": [x["n_pass"] for x in rd]}
    for i, n in enumerate(names):
        out["concepts"][n] = {
            "passes": sum(1 for x in rd if x["rows"][i]["pass"]),
            "auroc": [round(x["rows"][i]["auroc"], 4) for x in rd],
            "contrib": [round(x["rows"][i]["contrib"], 4) for x in rd],
            "firing": [round(x["rows"][i]["firing"], 4) for x in rd],
        }
    return out


print("=" * 100)
print("TARGET I — VERDICT (frozen bars from TARGET-I.md)")
print("=" * 100)

A = load("target_i_lam0.json")
B = load("target_i_lam02.json")
if A is None or B is None:
    print("  missing an arm's output -- cannot give a verdict")
    raise SystemExit(1)

sa, sb = summarise(A, "lam_act=0.0"), summarise(B, "lam_act=0.2")
for lab, s in (("lam_act=0.0 (the change)", sa), ("lam_act=0.2 (control)", sb)):
    print(f"\n{lab}: {s['n_runs']} runs | n_pass {s['n_pass']} | coverage "
          f"{[f'{c*100:.4f}%' for c in s['coverage']]}")
    print(f"    ppl {[round(p,2) for p in s['ppl']]}   acc {[round(a*100,2) for a in s['acc']]}")
    print(f"    {'concept':<24} {'passes':>7}  auroc")
    for n, v in s["concepts"].items():
        print(f"    {n:<24} {v['passes']}/{s['n_runs']:>4}  {v['auroc']}")

print("\n" + "-" * 100)
print("BARS")
print("-" * 100)
ok = True
for n in FAILING:
    c = sa["concepts"][n]["passes"]
    hit = c >= 2
    ok &= hit
    print(f"  I-FAILING  {n:<24} {c}/3 at lam_act=0   -> {'PASS' if hit else 'FAIL'}  "
          f"(auroc {sa['concepts'][n]['auroc']}, firing {sa['concepts'][n]['firing']})")

rp = sum(sa["n_pass"]) / (sa["n_runs"] * 9)
cov_ok = all(abs(c - 1.0) < 1e-9 for c in sa["coverage"])
ctl_ok = all(x["n_pass"] == 7 for x in B if x["kind"] == "readable")
ctl_fail_ok = all(sb["concepts"][n]["passes"] == 0 for n in FAILING)

print(f"  I-READABLE {rp*100:.1f}% of concepts pass at lam_act=0 -> "
      f"{'PASS' if rp >= 0.5 else 'FAIL'}")
print(f"  I-COVERAGE {'PASS' if cov_ok else 'FAIL'} "
      f"({[f'{c*100:.4f}%' for c in sa['coverage']]})")
print(f"  I-CONTROL  control arm reproduces 7/9 -> {'PASS' if ctl_ok else 'FAIL'} "
      f"(n_pass {[x['n_pass'] for x in B if x['kind'] == 'readable']})")
print(f"  I-CONTROL  same two concepts fail in control -> {'PASS' if ctl_fail_ok else 'FAIL'}")

print("\n" + "-" * 100)
if not (ctl_ok and ctl_fail_ok):
    print("  NO VERDICT — the control arm did not reproduce Target G2's 7/9 with the same two")
    print("  failing, so the comparison is not valid.")
elif ok:
    print("  TARGET I: the sparsity hypothesis is SUPPORTED. Both failing concepts pass at")
    print("  lam_act=0. Recorded with the perplexity cost, which is the trade-off.")
else:
    print("  TARGET I: the sparsity hypothesis is DEAD. Removing the activation penalty does NOT")
    print("  rescue the two concepts. The cause lies elsewhere and will be reported as such.")

ppl_a = sum(sa["ppl"]) / len(sa["ppl"])
nrm = [x["ppl"] for x in A if x["kind"] == "normal"]
if nrm:
    ppl_n = sum(nrm) / len(nrm)
    print(f"\n  I-NOWORSE (reported, not a pass/fail on its own): readable {ppl_a:.3f} vs matched "
          f"normal {ppl_n:.3f} -> delta {ppl_a - ppl_n:+.3f}")
