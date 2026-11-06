#!/bin/bash
# stop_agree_arm_after_ep4.sh -- stop the AGREEMENT ARM (agree_sgd04_s2) at an EPOCH BOUNDARY, on a stated
# measured reason, so the ship-path arm (run_sentstate_arm.sh, now at the FRONT of lens/runs/QUEUE.txt) can
# have the GPU while the machine is still up. The watchdog starts the queue front once no train_chat runs.
#
# THE REASON, with its numbers, written before the stop:
#   * the arm's decisive verdict is ALREADY ON THE RECORD and is a FAIL: P61a is a BAND MISS (refusals 0.3647
#     at ep1 against its own 0.005-0.08 band = 4.6x high) and P61b is a FAIL at 29.43x / 30.87x the ungated
#     twin's per-epoch time against a declared 1.5-3.0x band, i.e. 6.5-6.9x above the arm's own 4.5x
#     falsifier whose stated consequence is that the FREE variant becomes required
#     (lens/runs/GRADE-P61.txt; rows lens/runs/chat/agree_sgd04_s2_history.jsonl).
#   * it CANNOT reach its own decidable epoch: at ~2,000 s/epoch it needs ~5.5 h for the remaining epochs and
#     the machine goes down at ~21:41, so P61c as worded is undecidable tonight whatever I do here.
#   * what the remaining epochs buy is MORE POINTS ON THE REFUSAL CURVE for a conclusion that is fixed. What
#     they cost is the only ship-path experiment in the queue (the sentstate arm, 40-80 min + deciders).
# The arm is stopped at a boundary so its per-epoch row for the current epoch is written, and
# lens/scripts/run_agree_numbers.sh then appends a fresh settings block from the rows that DO exist.
#
# It stops NOTHING unless the arm is still running, and it prints the row count it saw on both sides.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY=.venv-gpu/bin/python3
HIST=lens/runs/chat/agree_sgd04_s2_history.jsonl
OUT=lens/runs/AGREE-ARM-STOPPED.txt
TARGET=${1:-4}          # the epoch row this stop waits for (default 4)

rows_now() {
  [ -f "$HIST" ] && grep -c . "$HIST" || echo 0
}

{
  echo "===================================================================================================="
  echo "AGREE ARM STOPPED AT AN EPOCH BOUNDARY  $(date '+%F %H:%M:%S')"
  echo "predictions: lens/runs/AGREE-GRADE-PREDICTIONS-P113.txt; grade so far lens/runs/GRADE-P61.txt"
  echo "reason: P61a BAND MISS (0.3647 refusals at ep1 vs its own 0.005-0.08) and P61b FAIL (29.43x/30.87x the"
  echo "        ungated twin per epoch vs a 1.5-3.0x band, 6.5-6.9x above its own 4.5x falsifier); ~5.5 h would"
  echo "        be needed to reach its decidable epoch against ~2.5 h of machine time left. The GPU goes to the"
  echo "        ship-path arm at the front of the queue instead."
  echo "rows before: $(rows_now)   waiting for row $TARGET (max ~9 min)"
} >> "$OUT"

for i in $(seq 1 108); do
  n=$(rows_now)
  if [ "$n" -ge "$TARGET" ]; then
    echo "  $(date '+%H:%M:%S') row $n written -> stopping the arm now" >> "$OUT"
    PID=$(pgrep -f "train_chat.py .*--out agree_sgd04_s2" | head -1)
    if [ -z "$PID" ]; then
      echo "  NO train_chat PROCESS for agree_sgd04_s2 -- nothing to stop (it may have finished or died)" >> "$OUT"
    else
      echo "  sending SIGTERM to pid $PID (the trainer's own checkpoint/resume path handles a mid-epoch stop;" >> "$OUT"
      echo "  at a boundary the epoch row and its ep-checkpoint are already on disk anyway)" >> "$OUT"
      kill -TERM "$PID"
      sleep 20
      if pgrep -f "train_chat.py .*--out agree_sgd04_s2" >/dev/null; then
        echo "  still alive after 20 s: sending SIGTERM again, once" >> "$OUT"
        kill -TERM "$PID"
        sleep 15
      fi
      if pgrep -f "train_chat.py .*--out agree_sgd04_s2" >/dev/null; then
        echo "  STILL ALIVE -- leaving it alone rather than sending SIGKILL: a killed trainer loses its"
        echo "  mid-epoch state and this arm's remaining value does not justify that. Report it instead."
        echo "  STILL ALIVE after 2 SIGTERMs -- NOT killed; the watchdog will not start the next arm while it runs." >> "$OUT"
      else
        echo "  arm is down. rows after: $(rows_now)" >> "$OUT"
      fi
    fi
    break
  fi
  sleep 5
done
{
  echo "  final rows: $(rows_now)   train_chat for this arm running: $(pgrep -f 'out agree_sgd04_s2' | wc -l)"
  echo "  NEXT: bash lens/scripts/run_agree_numbers.sh appends a fresh GRADE block from the rows that exist;"
  echo "        the watchdog starts the queue FRONT (run_sentstate_arm.sh) when it sees no train_chat."
  echo "=== DONE $(date '+%F %H:%M:%S') ==="
} >> "$OUT"
tail -12 "$OUT"
