#!/bin/bash
# RESUME VERIFICATION (CPU, tiny config -- never touches the GPU queue).
#   A: train 3 epochs uninterrupted
#   B: train 1 epoch, then resume from that checkpoint for epochs 2-3
# Claim under test: with the RNG sidecar restored, B's epochs 2 and 3 are BIT-IDENTICAL to A's,
# not merely similar. A near-miss is a FAIL here -- "close enough" is how a corrupt resume hides.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
CFG="--d 32 --window 8 --batch 32 --epochs 3 --max-windows 2000 --device cpu --seed 0 --out"
rm -f lens/runs/chat/rtest_a_* lens/runs/chat/rtest_b_*

echo "--- A: 3 epochs uninterrupted"
.venv-gpu/bin/python3 -u lens/scripts/train_chat.py $CFG rtest_a > /tmp/resume_a.log 2>&1
echo "--- B: 1 epoch, then resume"
.venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 32 --window 8 --batch 32 --epochs 1 \
    --max-windows 2000 --device cpu --seed 0 --out rtest_b > /tmp/resume_b1.log 2>&1
.venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 32 --window 8 --batch 32 --epochs 3 \
    --max-windows 2000 --device cpu --seed 0 --out rtest_b \
    --resume lens/runs/chat/rtest_b_ep1.npz --resume-ep 1 > /tmp/resume_b2.log 2>&1

grep -E "^  ep |RESUME" /tmp/resume_a.log | tail -4
echo "---"
grep -E "^  ep |RESUME" /tmp/resume_b1.log | tail -2
grep -E "^  ep |RESUME" /tmp/resume_b2.log | tail -4

.venv-gpu/bin/python3 -u - <<'PY'
import json
A = json.load(open("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat/rtest_a_log.json"))["history"]
B = json.load(open("/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat/rtest_b_log.json"))["history"]
print()
print(f"  A epochs {[h['epoch'] for h in A]}   B (post-resume) epochs {[h['epoch'] for h in B]}")
ok = True
for b in B:
    a = next((h for h in A if h["epoch"] == b["epoch"]), None)
    if a is None:
        print(f"    ep{b['epoch']}: missing from A"); ok = False; continue
    same = (a["train_loss"] == b["train_loss"] and a["val_ppl"] == b["val_ppl"]
            and a["train_acc"] == b["train_acc"] and a["val_acc"] == b["val_acc"])
    print(f"    ep{b['epoch']}: A loss {a['train_loss']:.6f} ppl {a['val_ppl']:.4f} acc "
          f"{a['val_acc']*100:.2f}%   B loss {b['train_loss']:.6f} ppl {b['val_ppl']:.4f} acc "
          f"{b['val_acc']*100:.2f}%   {'IDENTICAL' if same else 'DIFFERENT'}")
    ok &= same
print("  VERDICT:", "PASS - resume is bit-identical (params AND shuffle order restored)" if ok
      else "FAIL - resumed epochs differ from the uninterrupted run")
PY