#!/bin/bash
# run_optimise.sh — the optimisation grid: does widening the concept layer fix it?
#
# The diagnostic measured the bottleneck: the concept layer takes 1,024 numbers in and puts 9 out
# (114:1). The scaling experiment's WINNING arm had a 50:1 ratio (41 concepts for a 2,048-number
# input). So 9 concepts for 1,024 numbers is more compressed than every arm that FAILED.
#
# A 2x2 grid isolates which change does the work:
#
#   run 1 (DONE)   d=128  K= 9   114:1   val ppl 353.28   acc 11.08%   concepts 5/8
#   run 2          d=384  K=48    64:1   <- queued already
#   run 3          d=384  K= 9   341:1   <- isolates d at the bad ratio
#   run 4          d=128  K=48    21:1   <- isolates K at the good ratio
#
# Pre-registered predictions (written BEFORE these runs):
#   1. Run 4 (d=128, K=48) gives the BEST perplexity of the four, because it has the best
#      bottleneck ratio (21:1, better than the 50:1 that won the scaling test).
#   2. Run 3 (d=384, K=9) is the WORST, because d=384 with 9 concepts is 341:1 -- nearly 7x worse
#      than anything that failed before. If run 3 beats run 4, the bottleneck theory is WRONG and
#      that gets said plainly.
#   3. Concept pass rate rises with K: the wide runs pass noticeably more than 5/8, and the extra
#      concepts pass at lower fidelity than the core ones, because many are rare topics.
#   4. Replies stay incoherent in all four. Widening the layer should improve perplexity, not
#      produce a coherent chatbot -- that needs more training, not a wider layer.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
EP="${OPT_EPOCHS:-12}"

run () {  # $1=d  $2=K  $3=wide  $4=tag
  echo
  echo "################################################################"
  echo "# d=$1  K=$2  (wide=$3)  tag=$4"
  echo "################################################################"
  TC_WIDE=$3 nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d "$1" --window 8 --epochs "$EP" --batch 256 --lr 0.4 --lam-concept 0.3 \
    --device gpu --out "$4" --max-windows 1500000 2>&1 \
    | grep -E "concept module|concepts:|model:|tokens per|^  ep |GREEDY|final|-> " | tail -24
}

run 384 9  0 grid_d384_k9
run 128 48 1 grid_d128_k48

echo
echo "################################################################"
echo "# THE GRID"
echo "################################################################"
.venv-gpu/bin/python3 - <<'PY'
import json, os
P = "/run/media/yahya/HARD DISK 4/GlassNet/lens/runs/chat"
rows = [("d=128 K= 9", "real1"), ("d=384 K=48", "big1"),
        ("d=384 K= 9", "grid_d384_k9"), ("d=128 K=48", "grid_d128_k48")]
print(f"  {'config':<14}{'params':>11}{'K':>5}{'ratio':>8}{'val ppl':>10}{'acc':>9}{'concepts':>10}")
res = []
for tag, out in rows:
    f = os.path.join(P, f"{out}_replies.json")
    if not os.path.exists(f):
        f2 = os.path.join(P, f"{out}_log.json")
        if os.path.exists(f2):
            r = json.load(open(f2))
            h = r["history"][-1]
            print(f"  {tag:<14}{r['params']:>11,}{r['concepts']['n_kept']:>5}"
                  f"{'--':>8}{h['val_ppl']:>10.2f}{h['val_acc']*100:>8.2f}%{'--':>10}")
        else:
            print(f"  {tag:<14}{'-- not run --':>11}")
        continue
    r = json.load(open(f))
    h = r["history"][-1]
    K = r["concepts"]["n_kept"]
    D = 128 if "128" in tag else 384
    ratio = D * 8 / max(K, 1)
    a = r["audit"]
    ct = f"{a['n_pass']}/{a['n_concepts']}"
    print(f"  {tag:<14}{r['params']:>11,}{K:>5}{ratio:>8.0f}{h['val_ppl']:>10.2f}"
          f"{h['val_acc']*100:>8.2f}%{ct:>10}")
    res.append((tag, h["val_ppl"], ratio))
if len(res) >= 2:
    print()
    best = min(res, key=lambda x: x[1])
    worst = max(res, key=lambda x: x[1])
    print(f"  best:  {best[0]}  ppl {best[1]:.2f}  ratio {best[2]:.0f}:1")
    print(f"  worst: {worst[0]}  ppl {worst[1]:.2f}  ratio {worst[2]:.0f}:1")
    corr = [r for _, p, r in res]
    print(f"  ratio vs ppl: best ratio {min(corr):.0f}:1, worst ratio {max(corr):.0f}:1")
    print("  PREDICTION 2 CHECK: run 3 (d=384 K=9, 341:1) should be the WORST.")
PY

cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "docs copied"
