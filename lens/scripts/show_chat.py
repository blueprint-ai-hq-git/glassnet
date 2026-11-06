"""show_chat.py — read what the talking model actually says.
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
d = json.load(open(os.path.join(P, "real1_replies.json")))

print("=" * 96)
print("FINAL TRAINING CURVE")
print("=" * 96)
for h in d["history"]:
    print(f"  ep {h['epoch']}  loss {h['train_loss']:.4f}  concept {h['concept_loss']:.4f}  "
          f"val ppl {h['val_ppl']:9.2f}  val acc {h['val_acc']*100:6.2f}%   ({h['seconds']:.0f}s)")
print(f"  parameters: {d['params']:,}")

print()
print("=" * 96)
print("WHAT IT SAYS")
print("=" * 96)
for r in d["replies"]:
    print(f"  PROMPT   {r['prompt'].splitlines()[0]}")
    print(f"  GREEDY   {r['greedy']}")
    print(f"  SAMPLED  {r['sampled']}")
    print()

print("=" * 96)
print("CONCEPT AUDIT — does each name mean what it says?")
print("=" * 96)
a = d["audit"]
for c in a["concepts"]:
    au = "None" if c["auroc"] is None else f"{c['auroc']:.4f}"
    fr = "None" if c["fires"] is None else f"{c['fires']:.3f}"
    print(f"  {c['name']:<22} AUROC {au:>7}  fires {fr:>6}  {'PASS' if c['pass'] else 'FAIL'}")
print(f"  -> {a['n_pass']}/{a['n_concepts']} pass   coverage {a['coverage']*100:.2f}%")
print("=" * 96)
