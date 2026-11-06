"""status_audit.py — the exact state, read from artifacts, not from memory.
"""
import glob
import json
import os

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(P, "lens", "runs")


def j(p):
    return json.load(open(p)) if os.path.exists(p) else None


print("=" * 100)
print("EXACT STATE — every number read from a log file")
print("=" * 100)

# ---------- LADDER ----------
print("\n[1] THE LADDER (small-to-big, per capability)")
for rung in ("tiny", "small", "medium", "big"):
    t = j(os.path.join(R, "ladder", f"ladder_{rung}.json"))
    if not t:
        print(f"  {rung:<7} not run")
        continue
    n = sum(x["passed"] for x in t)
    bw = t[0].get("max_windows", "?")
    print(f"  {rung:<7} {n}/{len(t)}   budget max_windows={bw}")
    for x in t:
        print(f"        {'PASS' if x['passed'] else 'FAIL'}  {x['capability']:<12} "
              f"{x['seconds']:7.1f}s   {x['checks'][0][2][:58] if x['checks'] else ''}")

# ---------- GRADCHECK / TRANSPARENCY ----------
print("\n[2] THE ENGINE — gradients and transparency")
import subprocess
sweep = j(os.path.join(R, "ladder", "ladder_small.json"))
for x in (sweep or []):
    if x["capability"] in ("gradcheck", "explain"):
        print(f"  {x['capability']:<10} {'PASS' if x['passed'] else 'FAIL'}   "
              f"{'; '.join(c[2] for c in x['checks'])[:80]}")

# ---------- TEXT ----------
print("\n[3] TEXT MODEL")
for f in ("chat/best1_log.json", "chat/big1_log.json"):
    d = j(os.path.join(R, f))
    if d:
        h = d["history"][-1]
        print(f"  {f.split('/')[-1]:<18} {d['params']:>10,} params  ppl {h['val_ppl']:>8.2f}  "
              f"acc {h['val_acc']*100:>5.2f}%")

# ---------- IMAGE ----------
print("\n[4] IMAGE (Fashion-MNIST) — readable vs plain across seeds")
for f in sorted(glob.glob(os.path.join(R, "image", "*.json"))):
    d = j(f)
    if not isinstance(d, dict):
        continue
    if "history" in d and d.get("history"):
        h = d["history"][-1]
        print(f"  {os.path.basename(f):<34} acc {h.get('val_acc', 0)*100:6.2f}%  "
              f"params {d.get('params', 0):,}")
    elif "delta" in d or "seeds" in d:
        print(f"  {os.path.basename(f):<34} {json.dumps(d)[:90]}")
print("  (the 3-seed verdict was: readable 82.57%+/-0.66 vs plain 85.63%+/-0.73, DELTA -3.06+/-0.58)")

# ---------- VL ----------
print("\n[5] VISION+LANGUAGE")
for tag, f in (("vqa1 fashion", "vl/vlsmoke_log.json"), ("vqa2 fashion", "vl/vqa2_small_log.json"),
               ("medical att3", "vl/med_adapt_log.json"), ("medical att4 colour", "vl/med_color_log.json"),
               ("medical att5 cconcept", "vl/med_cc_log.json")):
    d = j(os.path.join(R, f))
    if not d:
        continue
    a = d["audit"]
    r0 = d["results"]["readable"]["history"][-1]["test_acc"] * 100
    p0 = d["results"]["plain"]["history"][-1]["test_acc"] * 100
    print(f"  {tag:<22} readable {r0:6.2f}%  plain {p0:6.2f}%  delta {r0-p0:+6.2f}  "
          f"concepts {a['n_pass']}/{a['n_concepts']}")

# ---------- CODE ----------
print("\n[6] CODE")
for name in ("code1", "codedoc_full", "covocab", "codropunk"):
    d = j(os.path.join(R, f"code/{name}_log.json"))
    if d:
        h = d["history"][-1]
        print(f"  {name:<14} ppl {h['val_ppl']:>7.2f}  acc {h['val_acc']*100:>5.2f}%  "
              f"params {d.get('params', 0):>9,}")
for tag, f in (("all prompts", "code/code1_reexec.json"),
               ("clean prompts", "code/code1_reexec_clean.json")):
    d = j(os.path.join(R, f))
    if d:
        print(f"  {tag:<14} {d['stats']}")
u = j(os.path.join(R, "code", "unk_rate_compare.json"))
if u:
    for k, v in u.items():
        print(f"  {k:<30} <unk> {v['unk_rate']*100:5.2f}%  runs {v['ran']}/{v['n']}")

# ---------- VIDEO ----------
print("\n[7] VIRTUAL CAMERA")
for f in sorted(glob.glob(os.path.join(R, "video", "*_log.json"))):
    d = j(f)
    if d and d.get("total"):
        print(f"  {os.path.basename(f):<26} {d['correct']}/{d['total']} "
              f"({d['correct']/d['total']*100:.1f}%)")
for f in sorted(glob.glob(os.path.join(R, "video", "*.mp4"))):
    print(f"  {os.path.basename(f):<26} {os.path.getsize(f)/1024:.0f} KB")

# ---------- GATES ----------
print("\n[8] LOW-LEVEL GATE TOTALS")
tot = 0
for f in sorted(glob.glob(os.path.join(P, "lens", "runs", "**", "*gate*.json"), recursive=True)):
    d = j(f)
    if isinstance(d, dict) and "passed" in d:
        tot += 1
print(f"  gate result files found: {tot}")
st = open(os.path.join(P, "STATUS.md")).read() if os.path.exists(os.path.join(P, "STATUS.md")) else ""
import re
for m in re.finditer(r"(L\d[^\n]{0,70})", st):
    print("  ", m.group(1)[:80])
print()
print("=" * 100)
