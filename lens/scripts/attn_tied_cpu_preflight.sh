#!/bin/bash
# CPU PRE-FLIGHT for the attention smoke (P79): does attention=True + direct_mode='tied' TRAIN through the real
# data path, mask and loss, on this machine, WITHOUT a GPU and WITHOUT competing with the running trainer?
#
# WHY IT EXISTS: lens/runs/ATTN-TIED-STRUCTURAL.txt checked construction/forward only (d=64, dummies, no data,
# no loss, no optimiser). A queued GPU arm that dies on a wiring error costs the queue a slot; this costs minutes
# of CPU. It is capped in its own systemd scope (MemoryMax=4G) so that if it OOMs it dies ALONE and cannot take
# the ours-method run (pid 351490, the flagship run in flight) with it.
#
# TWO TINY ARMS, same seed, same 500 windows, same step rule, one variable apart:
#   A: --attention   B: no --attention
# Small on purpose: d=64, 500 windows, batch 32, 1 epoch, CPU. Nothing here is a quality claim.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest"
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/ATTN-TIED-CPU-PREFLIGHT.txt
COMMON="--d 64 --window 32 --epochs 1 --batch 32 --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 \
--seed 0 --device cpu --assistant-only --train-turn-end --direct --direct-mode tied --max-windows 500
--method standard"

{
  echo "=== ATTENTION CPU PRE-FLIGHT $(date '+%F %H:%M') ==="
  echo "predictions P79a-f: lens/runs/ATTN-TIED-SMOKE-PREDICTIONS-P79.txt (this pre-flight is not one of them;"
  echo "it is the wiring check that runs BEFORE the queued GPU smoke is allowed to claim anything)"
  echo ""

  for ARM in "attn_cpu_smoke_1 --attention" "notied_cpu_smoke_1" ; do
    set -- $ARM
    TAG=$1; EXTRA=$2
    echo "--- CPU ARM $TAG  flags: $EXTRA" 
    systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=1G \
      --unit="glassnet-attncpu-$TAG-$(date +%s)" \
      .venv-gpu/bin/python3 -u lens/scripts/train_chat.py $COMMON $EXTRA --out "$TAG"
    echo "--- CPU ARM $TAG exit=$?"
    echo ""
  done
  echo "=== DONE $(date '+%F %H:%M') ==="
} > "$LOG" 2>&1

tail -40 "$LOG"
