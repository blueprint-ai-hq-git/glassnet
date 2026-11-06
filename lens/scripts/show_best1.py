"""show_best1.py — the best model's output, and the honest comparison.
"""
import json
import math
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
r = json.load(open(os.path.join(P, "best1_replies.json")))
CPW = 60949068 / 13318447
h = r["history"][-1]
a = r["audit"]

print("=" * 100)
print("BEST1 — the best configuration found")
print("=" * 100)
print(f"  params {r['params']:,}   d=128  window=16  vocab=3,500  concepts={r['concepts']['n_kept']}")
print(f"  bottleneck ratio {128*16}/{r['concepts']['n_kept']} = "
      f"{128*16/r['concepts']['n_kept']:.0f}:1")
print(f"  val ppl {h['val_ppl']:.2f}   acc {h['val_acc']*100:.2f}%   "
      f"bits/char {math.log2(h['val_ppl'])/CPW:.3f}")
print()
print("  versus everything else:")
for tag, ppl, npar in [("baseline v12k W8", 148.33, 2029925),
                       ("A vocab3500 W8", 90.43, 618925),
                       ("B window16 v12k", 109.95, 2124009),
                       ("big1 d384 K48", 147.79, 5177701),
                       ("BEST1", h["val_ppl"], r["params"])]:
    print(f"    {tag:<20}{ppl:>9.2f}  {npar:>10,} params")

print()
print("=" * 100)
print("WHAT IT SAYS")
print("=" * 100)
for x in r["replies"]:
    print(f"\n  PROMPT   {x['prompt'].splitlines()[0]}")
    print(f"  GREEDY   {x['greedy'][:190]}")
    print(f"  TOPK40   {x['sampled'][:190]}")
    print(f"  TOPK20   {x['sampled_topk20'][:190]}")

print()
print("=" * 100)
print("CONCEPT AUDIT")
print("=" * 100)
passing = [c for c in a["concepts"] if c["pass"]]
failing = [c for c in a["concepts"] if not c["pass"]]
print(f"  {a['n_pass']}/{a['n_concepts']} pass   coverage {a['coverage']*100:.2f}%")
print()
print("  PASSING:")
for c in passing:
    print(f"    {c['name']:<24} AUROC {c['auroc']:.4f}  fires {c['fires']:.3f}")
print()
perf = [c for c in failing if c["auroc"] is not None and c["auroc"] > 0.99]
print(f"  FAILING: {len(failing)}, of which {len(perf)} have AUROC > 0.99 (perfect, but too rare)")
print("=" * 100)
