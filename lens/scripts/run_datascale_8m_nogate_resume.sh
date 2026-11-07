#!/bin/bash
# Resume the 8M ungated arm from its own epoch-1 checkpoint after the machine rebooted at ~18:52.
#
# WHY RESUME AND NOT RESTART: epoch 1 finished at 18:37 and its checkpoint, rng sidecar, best.npz and log json are
# all on disk (datascale_8m_nogate_s0_ep1.npz, 15,262,368 B). Starting over would throw away 17 minutes of measured
# work; train_chat.py has a documented resume path (--resume <npz> --resume-ep N, start_ep = N+1, shuffle RNG
# restored) and it refuses on a corpus mismatch, so the risk is bounded and checkable: this runner records the
# RESUME line and the epoch it starts at, and if either is missing the arm is INCOMPLETE, not a result.
# Predictions unchanged: docs/history/NOGATE-8M-PREDICTIONS.md (X1 VAL > 23.74 strict; X2 24.0-24.6; X3 ppl < 95.99;
# X4 0 gate lines; X5 concepts >= 36). Epoch 1 already measured: VAL 23.46%, ASST 25.29%, ppl 95.71.
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-8M-NOGATE.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"
TAG="datascale_8m_nogate_s0"
CKPT="$SAVED/${TAG}_ep1.npz"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

if [ ! -f "$CKPT" ]; then
  say "===== REFUSED: no epoch-1 checkpoint at $CKPT - nothing to resume; use the from-scratch runner"
  exit 2
fi
if [ -e "$SAVED/${TAG}_ep3.npz" ] && grep -q "saved to .*${TAG}_\*" "$LOG" 2>/dev/null; then
  say "===== SKIP: epoch-3 checkpoint already present; the arm is finished"
  exit 0
fi
if trainer_alive; then
  say "===== BLOCKED: a real train_chat.py run is alive"
  trainer_alive_print | cut -c1-140 >> "$LOG"
  exit 3
fi
free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
free_mb=${free_mb:-0}
if [ "$free_mb" -lt 3000 ]; then
  say "===== BLOCKED: only ${free_mb} MiB VRAM free"
  exit 3
fi
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_8m_nogate_resume" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_8m_nogate_resume\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_8m_nogate_resume" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-8M RESUME start (reboot at ~18:52 killed the run after epoch 1) vram_free=${free_mb}MiB"
say "----- resume seed 0 from $(basename "$CKPT") : --resume-ep 1 -> epochs 2..3  $(date '+%H:%M:%S')"

CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
  --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed 0 --device gpu --assistant-only \
  --train-turn-end --max-windows 8000000 --gate-tol 0.05 \
  --out "$TAG" --method ours --opt-lr 1.0 --no-gates \
  --resume "$CKPT" --resume-ep 1 >> "$LOG" 2>&1
rc=$?
say "  exit=$rc"

res=$(grep -c "RESUME: loaded" "$LOG" 2>/dev/null || true); res=${res:-0}
ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
gs=$(grep -c "    gate step" "$LOG" 2>/dev/null || true); gs=${gs:-0}
say "  RESUME line present: $res (must be >= 1)   agreement-step lines: $ag   gate-step lines: $gs"

if [ "$rc" -eq 0 ] && [ -e "$SAVED/${TAG}_ep3.npz" ] && [ "$res" -ge 1 ]; then
  grep -E "^  ep [23]/3|^  final VAL" "$LOG" | tail -3 >> "$LOG"
  say "===== NOGATE-8M DONE (resumed) $(date '+%F %H:%M')"
else
  say "===== NOGATE-8M INCOMPLETE after resume rc=$rc resume_line=$res - not a result"
fi
exit 0
