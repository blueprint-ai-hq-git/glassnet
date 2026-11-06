#!/bin/bash
# run_followup.sh — after the code model: push the chat model further and find its plateau.
#
# best1 (vocab 3,500 + W=16 + lr decay, d=128, 41 concepts, 679,009 params) was still improving at
# epoch 10 of 24 (133 -> 74). So the obvious question is: how far does it go, and where does it stop?
#
# Two follow-ups, each one change from best1:
#   F1  best1 config, 60 epochs       -> where does it plateau? Does longer training keep paying?
#   F2  best1 config, vocab 5,000     -> is 3,500 the sweet spot, or was it just "smaller is better"?
#
# Pre-registered predictions:
#   1. F1 keeps improving but with clear diminishing returns; by epoch 60 the gain per epoch is
#      small. I expect it to land between 55 and 70 perplexity. If it is still falling sharply at
#      epoch 60, then this model is data-limited, not capacity-limited, and more epochs is the lever.
#   2. F2 lands BETWEEN A (vocab 3,500, 1.420) and the 12,000 baseline (1.576) -- I expect ~1.44.
#      If F2 BEATS F1, then 3,500 was too aggressive and the <unk> cost was real.
#   3. Replies stay non-conversational. Better perplexity has not produced a chatbot so far and
#      there is no reason to expect it here.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_chat.py|train_code.py" > /dev/null; do
  sleep 30
done
echo "GPU free; starting the follow-ups"

echo
echo "################################################################"
echo "# F1 — best1 config, 60 epochs (find the plateau)"
echo "################################################################"
CD_VOCAB=3500 TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 16 --epochs 60 --batch 256 --lr 0.4 \
  --lr-decay 0.5 --lam-concept 0.3 --device gpu --out f1_long --max-windows 1500000 2>&1 \
  | grep -E "^  ep |TOPK20|final" | tail -30

echo
echo "################################################################"
echo "# F2 — best1 config, vocab 5,000 (is 3,500 the sweet spot?)"
echo "################################################################"
CD_VOCAB=5000 TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 16 --epochs 24 --batch 256 --lr 0.4 \
  --lr-decay 0.5 --lam-concept 0.3 --device gpu --out f2_vocab5000 --max-windows 1500000 2>&1 \
  | grep -E "^  ep |TOPK20|final" | tail -30

echo
echo "################################################################"
echo "# FOLLOW-UPS — the whole picture"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, math, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CPW = 60949068 / 13318447
rows = [("BASELINE v12k W8", "grid_d128_k48"), ("A vocab3500 W8", "sweepA_vocab3500"),
        ("B window16", "sweepB_win16"), ("BEST combined", "best1"),
        ("F1 60 epochs", "f1_long"), ("F2 vocab5000", "f2_vocab5000")]
print(f"  {'variant':<22}{'params':>11}{'ppl':>9}{'bits/char':>11}{'acc':>8}{'concepts':>10}")
best = None
for tag, out in rows:
    f = None
    for suffix in ("_replies.json", "_log.json"):
        p = os.path.join(P, out + suffix)
        if os.path.exists(p):
            f = p
            break
    if f is None:
        print(f"  {tag:<22}{'-- not run --':>11}")
        continue
    r = json.load(open(f))
    h = r["history"][-1]
    bc = math.log2(h["val_ppl"]) / CPW
    K = r["concepts"]["n_kept"]
    a = r.get("audit", {})
    ct = f"{a.get('n_pass','?')}/{a.get('n_concepts',K)}"
    print(f"  {tag:<22}{r['params']:>11,}{h['val_ppl']:>9.2f}{bc:>11.3f}{h['val_acc']*100:>7.2f}%{ct:>10}")
    if best is None or bc < best[1]:
        best = (tag, bc, r["params"], h["val_ppl"])
if best:
    print(f"\n  BEST: {best[0]}  ppl {best[3]:.2f}  {best[1]:.3f} bits/char  ({best[2]:,} params)")
    print(f"  baseline 1.576  ->  {(1.576-best[1])/1.576*100:.1f}% better")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
