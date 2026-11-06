#!/bin/bash
# THE CONCEPT WALL, MEASURED (question = docs/history/PROBLEMS.md OPEN #1).
# Predictions P1-P5 and the falsifiers are in docs/history/WALL-AB-PREDICTIONS.md, written BEFORE this ran.
#
# Three arms, one change each against each other, seeds 0 and 1, everything else matched to the stored
# 20k-window reference in lens/runs/CTX-ATTN.txt (final val 11.96/10.08, mean 11.02%):
#   wall_off : concepts only, NO bypass  (the concept wall with nothing beside it)
#   wall_on  : --direct --direct-mode tied            (+2,097,152 params, the current default)
#   wall_r64 : --direct --direct-mode lowrank -rank 64 (+1,292,416 params, PROBLEMS.md's "low-rank residual")
#
# GPU OWNERSHIP (2026-11-06 16:4x, MEASURED - AND THIS IS WHY THE QUEUED ATTEMPT WOULD HAVE STOOD NO CHANCE).
# This used to be a plain wait loop. watchdog.sh's resident keeper ("gpu_trainer" -> run_resume_best.sh) is
# re-launched by the 15-min tick whenever no train_chat.py is alive, and RESUME-BEST.txt shows it finishing the
# same 3-epoch run and being restarted over and over. Measured 16:37: a fresh trainer (pid 2896786) started 15 s
# after the previous one died, so "wait until the GPU is free" could not be satisfied for more than a second and
# the 2-hour abort would have fired having trained nothing. Two trainers on a 6 GB card is also the OOM class
# in BUGS.md. The fix touches no results: claim the card with lens/runs/GPU-CLAIM (watchdog.sh now honours that
# file in BOTH the resident guard and the queue launcher, so nothing else starts a trainer while we hold it),
# stop the resident unit GRACEFULLY (systemctl --user stop -> the trainer writes its interrupt checkpoint and
# exits cleanly; nothing is deleted), then wait for the card. On exit the claim is removed so his keeper policy
# resumes on the next tick.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2
LOG=lens/runs/WALL-AB.txt
CLAIM=lens/runs/GPU-CLAIM

release_claim() { [ -e "$CLAIM" ] && grep -q "^pid $$" "$CLAIM" 2>/dev/null && rm -f "$CLAIM"; }
trap release_claim EXIT

echo "===== WALL-AB START $(date '+%F %H:%M:%S')" >> "$LOG"
echo "pid $$ ; predictions in docs/history/WALL-AB-PREDICTIONS.md (written before this run, not edited since)" >> "$LOG"

echo "pid $$ run_wall_ab.sh wall_ab claimed $(date '+%F %H:%M:%S')" > "$CLAIM"
echo "===== GPU-CLAIM written (pid $$)" >> "$LOG"
for u in $(systemctl --user list-units 'glassnet-res-gpu_trainer-*' --state=active --no-legend --plain 2>/dev/null | awk '{print $1}'); do
  systemctl --user stop "$u" 2>>"$LOG"
  echo "----- stopped resident unit $u (graceful SIGTERM; the trainer saves its own interrupt checkpoint)" >> "$LOG"
done
WAITED=0
while pgrep -f 'lens/scripts/train_chat' >/dev/null 2>&1; do
  sleep 5; WAITED=$((WAITED+5))
  if [ -e lens/runs/STOP-NIGHT ]; then echo "===== ABORT: STOP-NIGHT appeared while waiting" >> "$LOG"; exit 0; fi
  if [ "$WAITED" -ge 900 ]; then
    echo "===== ABORT: a trainer was still alive 15 min after the claim; not starting a second one" >> "$LOG"
    exit 3
  fi
done
echo "===== GPU free after ${WAITED}s wait; starting arms" >> "$LOG"

run() { NAME=$1; SEED=$2; shift 2
  free_mb=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  free_mb=${free_mb:-0}
  if [ "$free_mb" -lt 3000 ]; then
    echo "----- SKIP $NAME seed $SEED: only ${free_mb} MiB VRAM free (guard; nothing killed for it)" >> "$LOG"
    return 0
  fi
  echo "----- $NAME seed $SEED : $*  $(date '+%H:%M:%S')  vram_free=${free_mb}MiB" >> "$LOG"
  CHAT_DATA_DIR="$PWD/data/chat/prepared_mix_ctrl_honest" taskset -c 0,1 nice -n 19 ionice -c3 \
    .venv-gpu/bin/python3 -u lens/scripts/train_chat.py --d 256 --window 32 --epochs 3 --batch 256 \
    --lr 0.4 --lr-decay 0.5 --lam-concept 0.3 --lam-act 0.0 --seed $SEED --device gpu --assistant-only \
    --train-turn-end --max-windows 20000 --gate-tol 0.05 \
    --out "wall_${NAME}_s$SEED" --method ours --opt-lr 1.0 "$@" >> "$LOG" 2>&1
  echo "  $NAME seed $SEED exit=$?" >> "$LOG"
}

for S in 0 1; do
  run off $S
  run on  $S --direct --direct-mode tied
  run r64 $S --direct --direct-mode lowrank --direct-rank 64
done
echo "===== WALL-AB DONE $(date '+%F %H:%M')" >> "$LOG"
