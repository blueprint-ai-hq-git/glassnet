#!/bin/bash
# Where is the CURRENT arm? Reads the arm's own log, not the queue's filtered output.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import json, os, glob
for r in ["sb_B1_d512_attn", "sb_S1_d256_attn", "sb_B0_d512_noattn", "sb_S0_d256_noattn"]:
    p = f"lens/runs/chat/{r}_log.json"
    if not os.path.exists(p):
        print(f"{r:<22} no log"); continue
    j = json.load(open(p)); h = j.get("history") or []
    last = h[-1] if h else {}
    print(f"{r:<22} epochs={len(h):>3}  last ep{last.get('epoch')}  "
          f"masked_acc={last.get('val_masked_acc')}  masked_ppl={last.get('val_masked_ppl')}  "
          f"sec/ep={last.get('seconds'):.0f}" if h else f"{r} empty")
# how many checkpoints exist for the running arm -> how much work is actually committed to disk
for r in ["sb_B1_d512_attn"]:
    n = len(glob.glob(f"lens/runs/chat/{r}_ep*.npz"))
    newest = sorted(glob.glob(f"lens/runs/chat/{r}_ep*.npz"))[-1] if n else "none"
    print(f"{r}: {n} checkpoints on disk, newest {os.path.basename(newest)}")
PY