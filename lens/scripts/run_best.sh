#!/bin/bash
# run_best.sh — combine every winning option and push to the best output this machine can make.
#
# The sweep measured five variants against the baseline (d=128, K=37, vocab 12,000, W=8, 12 epochs,
# 2,029,925 params, ppl 148.33, 1.576 bits/char):
#
#   A  vocab 3,500    ppl  90.43   1.420 bits/char   618,925 params   <- best ppl, 3.3x smaller
#   B  window 16      ppl 109.95   1.482 bits/char   24/41 concepts   <- best accuracy 24.69%
#   C  24ep + decay   ppl 143.19   1.565 bits/char
#   D  d=256          ppl 147.26   1.574 bits/char
#
# ALL FIVE BEAT THE BASELINE. Two winners on different metrics, so this run takes BOTH:
#   vocab 3,500 (A) + window 16 (B) + lr decay (C) + 24 epochs, at d=128 (D showed width buys least)
#
# Also new: top-k sampling and a repetition penalty in the decoder, because greedy decoding loops
# ("i'm not sure i'm not sure") and pure sampling drifts. That is a DECODER change -- the model's
# perplexity and accuracy are untouched by it.
#
# Pre-registered predictions (written BEFORE the run):
#   1. Combined beats A on bits per character (below 1.420), because A's win came from freed
#      parameters and B's from context, and they do not compete for the same resource.
#   2. Concepts pass ABOVE 24/41, because window 16 gives each concept more room to appear.
#   3. The TOPK20 replies are the most coherent output this project has produced so far: real
#      sentences, correct grammar, and no single-phrase loop. Still not a chatbot -- it will
#      contradict itself across turns.
#   4. Greedy decoding STILL loops, even with the repetition penalty, because greedy always takes
#      the same argmax. If it does not loop, that is worth knowing.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

echo "waiting for the GPU..."
while pgrep -f "train_chat.py|run_sweep.sh|run_distill.py" > /dev/null; do
  sleep 20
done
echo "GPU free; starting the combined best run"

CD_VOCAB=3500 TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
  lens/scripts/train_chat.py --d 128 --window 16 --epochs "${BEST_EPOCHS:-24}" --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --device gpu --out best1 \
  --max-windows 1500000 2>&1 | tail -75

echo
echo "################################################################"
echo "# EVERYTHING, SIDE BY SIDE — bits per character"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, math, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CPW = 60949068 / 13318447
rows = [("BASELINE d128 K37 v12k W8", "grid_d128_k48", CPW),
        ("A vocab3500", "sweepA_vocab3500", CPW),
        ("B window16", "sweepB_win16", CPW),
        ("C 24ep+decay", "sweepC_long", CPW),
        ("D d=256", "sweepD_d256", CPW),
        ("big1 d384 K48", "big1", CPW),
        ("BEST combined", "best1", CPW),
        ("char d64 W32 K35", "char1", 1.0)]
print(f"  {'variant':<26}{'params':>11}{'ppl':>9}{'bits/char':>11}{'acc':>8}{'concepts':>10}")
best = None
for tag, out, cpt in rows:
    f = None
    for suffix in ("_replies.json", "_log.json"):
        p = os.path.join(P, out + suffix)
        if os.path.exists(p):
            f = p
            break
    if f is None:
        print(f"  {tag:<26}{'-- not run --':>11}")
        continue
    r = json.load(open(f))
    h = r["history"][-1]
    bc = math.log2(h["val_ppl"]) / cpt
    K = r["concepts"]["n_kept"]
    a = r.get("audit", {})
    ct = f"{a.get('n_pass','?')}/{a.get('n_concepts',K)}"
    print(f"  {tag:<26}{r['params']:>11,}{h['val_ppl']:>9.2f}{bc:>11.3f}{h['val_acc']*100:>7.2f}%{ct:>10}")
    if not tag.startswith("char") and (best is None or bc < best[1]):
        best = (tag, bc, r["params"])
if best:
    print(f"\n  BEST: {best[0]}  {best[1]:.3f} bits/char  ({best[2]:,} params)")
    print(f"  baseline was 1.576  ->  {(1.576-best[1])/1.576*100:.1f}% better")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
