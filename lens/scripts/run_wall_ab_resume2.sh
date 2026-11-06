#!/bin/bash
# WALL A/B, RESUMED (2nd attempt) - same six arms, same seeds, byte-identical trainer commands.
#
# WHAT WENT WRONG IN ATTEMPT 1 (lens/scripts/run_wall_ab_resume.sh, 17:04:26-17:10, MEASURED, my bug):
#   its skip test was `grep "saved to .*gpu_wall_${NAME}_s${SEED}_"`, copied from the d512 run whose tags
#   really do start with gpu_. These tags do not: the save line is ".../lens/runs/chat/wall_off_s0_*", so
#   the test never matched, and after the orphaned arm 1 finished legitimately (final VAL ppl 399.50
#   acc 12.18% concepts 20/36, line 201; saved .../chat/wall_off_s0_*, line 202) the supervisor started
#   arm 1 again from scratch. It was stopped by hand at 17:10 ~3 min in, before its own save could
#   overwrite the finished arm's files; its interrupt checkpoint is its only write. Nothing was deleted.
#   The pattern below is the one the log actually contains, and it is ALSO checked against the .npz files
#   that must exist on disk - a log line alone cannot make an arm count as done.
#
# WHY THE COMMANDS ARE UNCHANGED: P1-P5 (docs/history/WALL-AB-PREDICTIONS.md) were written before the
# first launch and grade this design. Supervision is not an arm. P6 (added before the 17:04 launch) says
# that a resume must not move any number; the same applies here.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/WALL-AB.txt
CLAIM=lens/runs/GPU-CLAIM
LOCK=lens/runs/GPU-WALLAB.lock
SAVED=lens/runs/chat

if ! mkdir "$LOCK" 2>/dev/null; then
  old=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then
    echo "===== REFUSED $(date '+%F %H:%M:%S'): $LOCK held by live pid $old" >> "$LOG"
    exit 0
  fi
  rm -rf "$LOCK" 2>/dev/null
  if ! mkdir "$LOCK" 2>/dev/null; then
    echo "===== REFUSED $(date '+%F %H:%M:%S'): could not take $LOCK" >> "$LOG"
    exit 0
  fi
fi
echo $$ > "$LOCK/pid"

release_claim() {
  [ -e "$CLAIM" ] && grep -q "pid $$" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"
  [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ] && rm -rf "$LOCK"
}
trap release_claim EXIT

echo "===== WALL-AB RESUME2 $(date '+%F %H:%M:%S')" >> "$LOG"
echo "pid $$ ; skip test corrected to the save line actually written (wall_<arm>_s<seed>_) plus an on-disk .npz check." >> "$LOG"
echo "pid $$ run_wall_ab_resume2.sh wall_ab claimed $(date '+%F %H:%M:%S')" > "$CLAIM"
echo "===== GPU-CLAIM written (pid $$)" >> "$LOG"

stop_resident() {
  for u in $(systemctl --user list-units 'glassnet-res-gpu_trainer-*' --state=active --no-legend --plain 2>/dev/null | awk '{print $1}'); do
    systemctl --user stop "$u" 2>>"$LOG"
    echo "----- stopped resident unit $u (graceful SIGTERM; the trainer saves its own interrupt checkpoint)" >> "$LOG"
  done
}

trainer_pids() {
  for p in $(pgrep -f 'train_chat\.py' 2>/dev/null); do
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null)
    case "$exe" in *python3*) echo "$p" ;; esac
  done
}

WAITED=0
GRACEFUL=0
while [ -n "$(trainer_pids)" ]; do
  sleep 15; WAITED=$((WAITED+15))
  stop_resident
  if [ -e lens/runs/STOP-NIGHT ]; then echo "===== ABORT: STOP-NIGHT appeared while waiting" >> "$LOG"; exit 0; fi
  if [ "$WAITED" -ge 1200 ] && [ "$GRACEFUL" -eq 0 ]; then
    PIDS=$(trainer_pids | tr '\n' ' ')
    echo "----- 20 min elapsed with a trainer alive (pid $PIDS): ONE graceful SIGTERM (it checkpoints itself)" >> "$LOG"
    for p in $PIDS; do kill -TERM "$p" 2>>"$LOG"; done
    GRACEFUL=1
  fi
  if [ "$WAITED" -ge 1500 ]; then
    echo "===== ABORT: trainer still alive 25 min after the claim; not starting a second one" >> "$LOG"
    exit 3
  fi
done

# ---- an arm is done ONLY if train_chat.py printed its save line AND the weights are on disk ----
arm_done() {
  local NAME=$1 SEED=$2
  grep -q "saved to .*wall_${NAME}_s${SEED}_" "$LOG" 2>/dev/null || return 1
  ls "$SAVED"/wall_${NAME}_s${SEED}_ep3.npz >/dev/null 2>&1 || return 2
  return 0
}

run() { NAME=$1; SEED=$2; shift 2
  if arm_done "$NAME" "$SEED"; then
    echo "----- SKIP $NAME seed $SEED: recorded (save line + ep3.npz present), not rerun" >> "$LOG"
    return 0
  fi
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free_mb=${free_mb:-0}
  if [ "$free_mb" -lt 3000 ]; then
    echo "----- SKIP $NAME seed $SEED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)" >> "$LOG"
    return 0
  fi
  echo "----- $NAME seed $SEED : $*  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB" >> "$LOG"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed "$SEED" --device gpu --assistant-only \
    --train-turn-end --max-windows 20000 --gate-tol 0.05 \
    --out "wall_${NAME}_s${SEED}" --method ours --opt-lr 1.0 "$@" >> "$LOG" 2>&1
  echo "  $NAME seed $SEED exit=$?" >> "$LOG"
}

for S in 0 1; do
  run off "$S"
  run on  "$S" --direct --direct-mode tied
  run r64 "$S" --direct --direct-mode lowrank --direct-rank 64
done

MISSING=0
for S in 0 1; do for A in off on r64; do arm_done "$A" "$S" || { echo "----- MISSING arm $A seed $S (no save line or no ep3.npz)" >> "$LOG"; MISSING=$((MISSING+1)); }; done; done
echo "===== WALL-AB DONE $(date '+%F %H:%M') arms_missing=$MISSING" >> "$LOG"
grep -n "^  final VAL" "$LOG"
exit 0
