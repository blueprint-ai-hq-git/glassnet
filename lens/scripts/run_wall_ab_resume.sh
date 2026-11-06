#!/bin/bash
# WALL A/B, RESUMED - same six arms, same seeds, same commands as run_wall_ab.sh.
#
# WHY THIS FILE EXISTS (measured, 2026-11-06 17:0x):
#   Three launches of run_wall_ab.sh were made by three agent turns; each was started inside the agent's
#   own process group, and each was killed by SIGTERM when that turn's tool call timed out
#   (chain rc=143, lens/runs/chain_wall_ab.out 0 bytes at 16:39). The trainer itself survived every time
#   (orphaned, reparented to pid 1) and kept training, but the loop that starts the OTHER five arms died
#   with the parent - so at 17:02 exactly one arm (off seed 0, pid 3015702) was running and nothing was
#   left alive to run the remaining five. Re-running run_wall_ab.sh unchanged would have started a
#   SECOND trainer on the same card (the OOM class in BUGS.md) and overwritten the arm already running.
#
# WHAT THIS DOES DIFFERENTLY:
#   1. takes the same atomic lock + GPU-CLAIM, so no other launcher can start a trainer beside it;
#   2. WAITS for the arm already in flight to finish instead of killing it (up to 20 min, then ONE
#      graceful SIGTERM -> the trainer writes its own interrupt checkpoint, nothing is deleted);
#   3. SKIPS any arm that already has its completion marker in WALL-AB.txt ("saved to ... gpu_wall_<arm>_s<seed>_"),
#      which is the line train_chat.py prints only after a full run, so no finished arm is overwritten;
#   4. runs the rest with byte-identical commands (same data dir, d256/w32/3ep/b256/lr0.4, ours+opt-lr1.0).
# Predictions P1-P5 were written before the first launch (docs/history/WALL-AB-PREDICTIONS.md); the
# resume addendum there records that this run is the same experiment, not a new one.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/WALL-AB.txt
CLAIM=lens/runs/GPU-CLAIM
LOCK=lens/runs/GPU-WALLAB.lock

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

echo "===== WALL-AB RESUME $(date '+%F %H:%M:%S')" >> "$LOG"
echo "pid $$ ; resumed after three SIGTERM'd launchers (rc 143). Same arms/seeds/commands as run_wall_ab.sh." >> "$LOG"
echo "pid $$ run_wall_ab_resume.sh wall_ab claimed $(date '+%F %H:%M:%S')" > "$CLAIM"
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

# ---- 1. wait for the arm already in flight; never kill what is producing a real result ----
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
echo "===== card free after ${WAITED}s wait; checking which arms are already recorded" >> "$LOG"

# ---- 2. an arm counts as done only if train_chat.py printed its post-run save line ----
arm_done() {
  grep -q "saved to .*gpu_wall_${1}_s${2}_" "$LOG" 2>/dev/null
}

run() { NAME=$1; SEED=$2; shift 2
  if arm_done "$NAME" "$SEED"; then
    echo "----- SKIP $NAME seed $SEED: already recorded (completion marker present)" >> "$LOG"
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

# ---- 3. arm order identical to run_wall_ab.sh (seeds outer, arms inner) ----
for S in 0 1; do
  run off "$S"
  run on  "$S" --direct --direct-mode tied
  run r64 "$S" --direct --direct-mode lowrank --direct-rank 64
done

MISSING=0
for S in 0 1; do for A in off on r64; do arm_done "$A" "$S" || { echo "----- MISSING arm $A seed $S (no completion marker)" >> "$LOG"; MISSING=$((MISSING+1)); }; done; done
echo "===== WALL-AB DONE $(date '+%F %H:%M') arms_missing=$MISSING" >> "$LOG"
grep -n "^  final VAL" "$LOG"
exit 0
