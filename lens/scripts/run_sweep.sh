#!/bin/bash
# run_sweep.sh — try ALL the optimisation options and find the best one.
#
# Owner: "continue on till you get the perfect output ... if there is we have to choose between them
# left them all and find the best one."
#
# So: not one option. Every option, measured, and the best one chosen on evidence.
#
# BASELINE (measured, 12 epochs): d=128, K=37, window 8, vocab 12,000, lr 0.4 constant
#     val ppl 148.33   acc 21.21%   concepts 16/37   bits/char 1.576   2,029,925 params
#
# Each variant changes ONE thing from that baseline, so the winner is attributable:
#   A  vocab 3,500      the embedding is 93% of the model at d=128; a smaller vocab frees it
#   B  window 16        the replies drift; an 8-token window cannot track a conversation
#   C  24 epochs + lr decay 0.5   perplexity was still falling at epoch 12
#   D  d=256            width is the secondary lever (measured 1.68x on its own)
#
# Pre-registered predictions (written BEFORE the sweep):
#   1. A (vocab 3,500) gives the best perplexity per parameter, because it frees the largest block
#      of parameters. It may LOSE on absolute perplexity because <unk> rises above 1.92%.
#   2. B (window 16) gives the best RESPONSES, because drift is a context problem, not a
#      perplexity problem. Its perplexity may barely move.
#   3. C improves perplexity the most in absolute terms, since training was still descending.
#   4. D improves less than A, B or C, because the grid already showed d=128 K=37 TIES d=384 K=48.
#   5. No variant produces a coherent chatbot. The best one produces more fluent phrases.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
EP="${SW_EPOCHS:-12}"

run () {  # $1=tag  $2=vocab  $3=d  $4=window  $5=epochs  $6=lrdecay
  echo
  echo "################################################################"
  echo "# $1   vocab=$2 d=$3 W=$4 epochs=$5 lrdecay=$6"
  echo "################################################################"
  CD_VOCAB="$2" TC_WIDE=1 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u \
    lens/scripts/train_chat.py --d "$3" --window "$4" --epochs "$5" --batch 256 --lr 0.4 \
    --lr-decay "$6" --lam-concept 0.3 --device gpu --out "$1" --max-windows 1500000 2>&1 \
    | grep -E "concept module|concepts:|model:|characters per|tokens per|bottleneck|^  ep |SAMPLED|final" \
    | tail -22
}

run sweepA_vocab3500 3500 128 8  "$EP" 0.0
run sweepB_win16     12000 128 16 "$EP" 0.0
run sweepC_long      12000 128 8  24    0.5
run sweepD_d256      12000 256 8  "$EP" 0.0

echo
echo "################################################################"
echo "# THE SWEEP — all options, side by side"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, math, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
CPW = 60949068 / 13318447
rows = [("BASELINE d128 K37 v12k W8", "grid_d128_k48", 1),
        ("A vocab 3500", "sweepA_vocab3500", 1),
        ("B window 16", "sweepB_win16", 1),
        ("C 24ep + lrdecay", "sweepC_long", 1),
        ("D d=256", "sweepD_d256", 1)]
print(f"  {'variant':<26}{'params':>11}{'ppl':>9}{'bits/char':>11}{'acc':>8}{'concepts':>10}")
best = None
for tag, out, cpt in rows:
    for suffix in ("_replies.json", "_log.json"):
        f = os.path.join(P, out + suffix)
        if os.path.exists(f):
            break
    else:
        print(f"  {tag:<26}{'-- not run --':>11}")
        continue
    r = json.load(open(f))
    h = r["history"][-1]
    bc = math.log2(h["val_ppl"]) / cpt
    K = r["concepts"]["n_kept"]
    ct = f"{r.get('audit',{}).get('n_pass','?')}/{r.get('audit',{}).get('n_concepts',K)}"
    print(f"  {tag:<26}{r['params']:>11,}{h['val_ppl']:>9.2f}{bc:>11.3f}"
          f"{h['val_acc']*100:>7.2f}%{ct:>10}")
    if best is None or bc < best[1]:
        best = (tag, bc, r["params"])
if best:
    print(f"\n  BEST on bits per character: {best[0]}  {best[1]:.3f}  ({best[2]:,} params)")
    print(f"  baseline was 1.576")
    print(f"  {'IMPROVED' if best[1] < 1.576 else 'NO VARIANT BEAT THE BASELINE'}")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
