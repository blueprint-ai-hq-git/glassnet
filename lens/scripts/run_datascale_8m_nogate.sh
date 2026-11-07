#!/bin/bash
# 8M windows, gates off - the data axis, which is worth ~3.4x steps per doubling at 4M.
#
# Predictions, written before the launch: docs/history/NOGATE-8M-PREDICTIONS.md
#   X1 VAL > 23.74 (strict, the 3-epoch 4M arm)   X2 lands 24.0-24.6   X3 ppl < 95.99
#   X4 mechanical (0 gate lines)   X5 concepts >= 36
# Corpus = the ladder's own honest mix and cores 0,1 (NOT the dialogue-only corpus / cores 2,3 of the failed
# inline attempt recorded in lens/runs/chain_-c.out).
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-8M-NOGATE.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_8m_nogate_s0"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_8m_nogate" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_8m_nogate\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_8m_nogate" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-8M start (predictions: docs/history/NOGATE-8M-PREDICTIONS.md) vram_free=${free_mb}MiB"
say "----- off seed 0 : --method ours --no-gates --max-windows 8000000 (4M 3-ep twin 23.74 at DATASCALE-4M-NOGATE.txt:69)  $(date '+%H:%M:%S')"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 8000000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 --no-gates >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  MECHANICAL: agreement-step lines=$ag  gate-step lines=$gs  (both must be 0)"
grep -m1 "METHOD:" "$LOG" | cut -c1-130 >> "$LOG"
grep -m1 "tokens per parameter" "$LOG" | cut -c1-110 >> "$LOG"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ]; then
  grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  say "===== NOGATE-8M DONE $(date '+%F %H:%M')"
else
  say "===== NOGATE-8M INCOMPLETE $(date '+%F %H:%M') rc=$rc - a killed arm is not a result"
fi
exit 0
