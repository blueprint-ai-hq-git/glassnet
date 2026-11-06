#!/bin/bash
# T12 LADDER — talk line, BIGGER models, from scratch, ONE variable per run.
# Baseline (measured, shipping): t11_talk41_ep192, d256 w32 v12000, 40 concepts, 192 ep,
# masked val acc 80.59%, strict honesty 40/40.
# PREDICTIONS (written before any run):
#   P1: d384 (L1) masked acc gain <= +1.5 pts      (data-bound: 13.3M tokens vs +1.5M embed params)
#   P2: d512 (L2) masked acc gain <= +1.5 pts      (same reason, larger gap)
#   P3: w48  (L3) masked acc gain +1..+3 pts       (multi-turn context truncated at w32)
#   P4: all three stay strict 40/40 (concept spec unchanged) — the audit IS the gate; a bigger
#       model that loses honesty does not ship.
# GATE per member: strict 40/40 (binary) AND the measured acc reported next to the baseline.
# Everything else identical to t11: v12000, batch 256, lr 0.4, lr-decay 0.5, lam-concept 0.3,
# seed 0, gpu, assistant-only, max-windows 1500000. Checkpoints + logs EVERY epoch.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1

run_one() { # $1=label  $2..=train args
  local label=$1; shift
  echo "=== [$label] training $(date +%H:%M:%S)"
  nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/train_chat.py \
    --d 256 --window 32 --epochs 96 --batch 256 --lr 0.4 --lr-decay 0.5 \
    --lam-concept 0.3 --seed 0 --device gpu --assistant-only --max-windows 1500000 \
    --out "t12_${label}" "$@" 2>&1 | tail -25
  local ckpt=$(ls -t lens/runs/chat/t12_${label}_ep*.npz | head -1)
  echo "=== [$label] audit of $(basename $ckpt)"
  TALK_CKPT="$(basename $ckpt)" .venv-gpu/bin/python3 -u lens/scripts/audit_strict_text.py 2>&1 | grep -E "STRICT|honest|mislabelled|PASS|FAIL" | head -4
}

run_one d384 --d 384
run_one d512 --d 512
run_one w48  --window 48
echo "=== LADDER DONE $(date +%H:%M)"