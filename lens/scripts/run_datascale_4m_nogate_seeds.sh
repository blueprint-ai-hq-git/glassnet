#!/bin/bash
# Seed replication of the 4M ungated rung: seeds 1 and 2, back to back, one unit.
#
# Every gates-off number in the project is seed 0 only, and the 200k rung's three-seed spread (3.07 pp) is larger
# than either gates-off delta, so this arm gives the rung an error bar of its own.
# Predictions, written before the launch: docs/history/NOGATE-4M-SEEDS-PREDICTIONS.md
#   R1 each seed VAL in 22.2-25.3   R2 seeds differ <= 3.1 pp   R3 three-seed mean 22.6-24.9 (reported with range)
#   R4 mechanical (0 gate lines each)   R5 concepts >= 36 each
set -u
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG="lens/runs/DATASCALE-4M-NOGATE-SEEDS.txt"
CLAIM="lens/runs/GPU-CLAIM"
SAVED="lens/runs/chat"

exec 9>"lens/runs/.lock_$(basename "$0" .sh)"
if ! flock -n 9; then
  echo "$(date '+%F %T') REFUSED: another copy holds the lock; nothing done" >> "$LOG"
  exit 0
fi
source lens/scripts/lib_trainer_alive.sh
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

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
if [ -e "$CLAIM" ] && ! grep -q "^owner: run_datascale_4m_nogate_seeds" "$CLAIM" 2>/dev/null; then
  say "===== BLOCKED: GPU-CLAIM held by $(head -1 "$CLAIM")"
  exit 3
fi
printf 'owner: run_datascale_4m_nogate_seeds\npid: %s\nsince: %s\n' "$$" "$(date '+%F %T')" > "$CLAIM"
cleanup() { [ -e "$CLAIM" ] && grep -q "^owner: run_datascale_4m_nogate_seeds" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap cleanup EXIT INT TERM

say "===== NOGATE-4M-SEEDS start (predictions: docs/history/NOGATE-4M-SEEDS-PREDICTIONS.md) vram_free=${free_mb}MiB"
done_n=0
for seed in 1 2; do
  tag="datascale_4m_nogate_s${seed}"
  if [ -e "$SAVED/${tag}_ep3.npz" ] && grep -q "saved to .*${tag}_\*" "$LOG" 2>/dev/null; then
    say "----- seed $seed SKIP: checkpoint and save line already present"
    done_n=$((done_n+1))
    continue
  fi
  say "----- off seed $seed : --method ours --no-gates --max-windows 4000000 --epochs 3  $(date '+%H:%M:%S')"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed "$seed" --device gpu --assistant-only \
    --train-turn-end --max-windows 4000000 --gate-tol 0.05 \
    --out "$tag" --method ours --opt-lr 1.0 --no-gates >> "$LOG" 2>&1
  rc=$?
  say "  seed $seed exit=$rc"
  if [ "$rc" -eq 0 ] && [ -e "$SAVED/${tag}_ep3.npz" ]; then
    done_n=$((done_n+1))
    grep -E "^  ep 3/3|^  final VAL" "$LOG" | tail -2 >> "$LOG"
  else
    say "  seed $seed INCOMPLETE rc=$rc - a killed arm is not a result; continuing to the next seed"
  fi
done

ag=$(grep -c "agreement step" "$LOG" 2>/dev/null || true); ag=${ag:-0}
say "  MECHANICAL across both arms: agreement-step lines=$ag  (must be 0)"
if [ "$done_n" -eq 2 ]; then
  say "===== NOGATE-4M-SEEDS DONE $(date '+%F %H:%M') arms_done=2 of 2"
else
  say "===== NOGATE-4M-SEEDS INCOMPLETE $(date '+%F %H:%M') arms_done=$done_n of 2"
fi
exit 0
