"""diag_chat_state.py — SMALL TEST FIRST: what do the existing chat checkpoints actually do?

Read-only. No training. Answers: which vocab does each checkpoint use, what ppl/acc did it reach,
and what is the <unk> rate in its GENERATED replies (the thing we actually care about for "talk").
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

R = os.path.join(PROJ, "lens", "runs", "chat")

print("=" * 96)
print("CHAT CHECKPOINT STATE (read-only)")
print("=" * 96)

for tag in ("best1", "big1"):
    lf = os.path.join(R, f"{tag}_log.json")
    if not os.path.exists(lf):
        print(f"{tag}: no log")
        continue
    d = json.load(open(lf))
    a, h = d["args"], d["history"]
    print(f"\n{tag}: d={a['d']} window={a['window']} epochs={a['epochs']} "
          f"lr_decay={a.get('lr_decay')}")
    print(f"   final val_ppl {h[-1]['val_ppl']:.2f}  val_acc {h[-1]['val_acc']*100:.2f}%  "
          f"train_loss {h[-1]['train_loss']:.3f}  {h[-1]['seconds']:.0f}s/epoch")

# which vocab does each checkpoint's output width imply?
print("\n" + "-" * 96)
print("VOCAB WIDTH IMPLIED BY EACH CHECKPOINT (net.V)")
print("-" * 96)
import lens
from lens import backend as B
for tag in ("best1", "big1"):
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        continue
    last = os.path.join(R, eps[-1])
    try:
        net = lens.load(last, device="cpu")
        print(f"  {tag:6} last ep {eps[-1]:<18} V={getattr(net, 'V', '?')} "
              f"W={getattr(net, 'W', '?')} d={getattr(net, 'd', '?')}")
    except Exception as e:
        print(f"  {tag}: load failed: {str(e)[:80]}")

print("\n" + "-" * 96)
print("VOCAB FILES ON DISK")
print("-" * 96)
for v in ("v3500", "v5000", "v12000"):
    p = os.path.join(PROJ, "data", "chat", "prepared", f"vocab_{v}.json")
    if os.path.exists(p):
        w = json.load(open(p))
        print(f"  {v}: {len(w):,} words   first 8: {w[:8]}")
