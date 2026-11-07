#!/bin/bash
# THE MISSING CORNER: plain SGD, gates off, at the 500k rung - the configuration the 37.37% headline used.
#
# Three configurations exist at d=256: (b) ours+gates = 20.81 at 500k, (c) ours, gates off = 22.05, and this arm
# (a) standard/plain SGD/gates off. Only (a) vs (c) isolates the step rule; today only the gates were isolated.
# Predictions, written before the launch: docs/history/SGD-500K-PREDICTIONS.md
#   S1 VAL 20.55-23.55   S2 ppl 120-175   S3 ~72-75 s/epoch   S4 mechanical (0 gate lines, "STEP RULE: plain SGD")
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/SGD-500K.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="sgd_500k_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy of $(basename "$0") holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ -e "$SAVED/${TAG}_ep3.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: $TAG already has an epoch-3 checkpoint and a save line; nothing to do"
  exit 0
fi
if trainer_alive; then
  say "===== BLOCKED: a real train_chat.py run is alive; not competing with it"
  trainer_alive_print | cut -c1-140 >> "$LOG"
  exit 3
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
free_mb=${free_mb:-0}
if [ "$free_mb" -lt 3000 ]; then
  say "===== BLOCKED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)"
  exit 3
fi
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_sgd_500k_s0" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_sgd_500k_s0\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_sgd_500k_s0" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== SGD-500K start (predictions: docs/history/SGD-500K-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --method standard (plain SGD, gates off) --max-windows 500000  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 500000 \
  --out "$TAG" --method standard >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL CHECK: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0)"
grep -m1 "STEP RULE:" "$LOG" | cut -c1-130 >> "$LOG"
grep -m1 "METHOD:" "$LOG" | cut -c1-130 >> "$LOG"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== SGD-500K DONE $(date '+%F %H:%M')"
else
  say "===== SGD-500K INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
