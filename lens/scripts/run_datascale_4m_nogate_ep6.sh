#!/bin/bash
# Six epochs at 4M windows, gates off - same data, double the updates. Is the plateau steps or data?
#
# The 3-epoch 4M arm went ep1 23.11 -> ep2 23.75 -> ep3 23.74 (accuracy flat, ppl still falling), i.e. it looked
# step-limited. This run holds data fixed and doubles epochs: 46,875 -> 93,750 updates.
# Predictions, written before the launch: docs/history/NOGATE-4M-EP6-PREDICTIONS.md
#   W1 best VAL >= 24.04 (23.74 + 0.3)   W2 best in 24.0-25.5   W3 ppl < 95.99   W4 mechanical   W5 concepts >= 36
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-4M-NOGATE-EP6.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_4m_nogate_ep6_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ -e "$SAVED/${TAG}_ep6.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: $TAG already has an epoch-6 checkpoint and a save line; nothing to do"
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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_4m_nogate_ep6" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_4m_nogate_ep6\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_4m_nogate_ep6" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-4M-EP6 start (predictions: docs/history/NOGATE-4M-EP6-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --method ours --no-gates --max-windows 4000000 --epochs 6 (3-epoch twin 23.74 at DATASCALE-4M-NOGATE.txt:69)  $(date '+%H:%M:%S')"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 6 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 4000000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 --no-gates >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0)"
grep -m1 "METHOD:" "$LOG" | cut -c1-130 >> "$LOG"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep6.npz" ]; then
  grep -E "^  ep [0-9]/6|^  final VAL" "$LOG" | tail -3 >> "$LOG"
  say "===== NOGATE-4M-EP6 DONE $(date '+%F %H:%M')"
else
  say "===== NOGATE-4M-EP6 INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
