#!/bin/bash
# Does the data lever keep paying past 200k? 500k windows, seed 0, d=256, batch 256, 3 epochs.
#
# WHY: seed 0 goes 12.18% (20k) -> 17.43% (200k). Is the curve still rising at 200k, or has it flattened?
# The only larger rungs in the project used a different batch (1024), so they cannot be paired with these.
# 500k is ~50 min and one flag away from the 200k arm.
# Predictions, written before the launch: docs/history/DATASCALE-500K-PREDICTIONS.md
#   C1 VAL >= 18.93% (200k + 1.5 pp)   C2 ppl < 207.29   C3 concepts reported   C4 ep3 >= ep2
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-500K.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_500k_s0"

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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_500k_s0" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_500k_s0\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_500k_s0" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== DATASCALE-500K start (predictions: docs/history/DATASCALE-500K-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --max-windows 500000 (5x the 200k arm, same batch 256 and 3 epochs)  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 500000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  say "----- epoch 3 and final result (read from the log above)"
  grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== DATASCALE-500K DONE $(date '+%F %H:%M')"
else
  say "===== DATASCALE-500K INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
