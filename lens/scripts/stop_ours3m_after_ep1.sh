#!/bin/bash
# stop_ours3m_after_ep1.sh -- stop our method's 3M-window arm ONCE IT HAS WRITTEN EPOCH 1, so the queued
# ATTENTION ARM (lens/scripts/run_attn_full.sh, the project's declared last architectural lever) gets the
# GPU tonight instead of 30 minutes before the machine is switched off.
#
# THE DECISION, WITH ITS NUMBERS (2026-10-23 19:0x self-wake):
#   * the 3M ours arm (pid started 17:55) measures 1.97 steps/s (lens/runs/chat/ours_d256_3M_2ep_agreement.jsonl:
#     5,437 steps at 18:42:20 -> 7,987 at 19:03:53) = 11,719 steps/epoch -> epoch 1 lands ~19:35, epoch 2 ~21:05.
#     The machine is normally stopped by 21:41, so its epoch 2 would finish with ~35 min to spare and the
#     attention arm would then get ~30 min and write NOTHING (its checkpoint interval is 1800 s).
#   * all four of that arm's own predictions are answerable at epoch 1 (an epoch completes; the refusal rate
#     stays high; held-out low against the 35.26% twin; the talk test decides). Epoch 2 adds only the second
#     point of the curve.
#   * the attention arm is ~1 h for 3 epochs and decides P82e, whose falsifier closes the last remaining
#     architectural lever. Stopping after a COMPLETED epoch loses no artifact: the epoch's checkpoint
#     ({out}_ep1.npz), the best checkpoint and the history line are all written at the epoch boundary.
#   * NOTHING IS DELETED and the run can be re-queued by hand (its line in QUEUE.txt was consumed by the
#     watchdog at 17:55); if a later turn wants the second epoch it must re-queue run_ours_3m.sh BEHIND the
#     attention arm, never ahead of it.
#
# RETIRED RUNS ARE NOT TOUCHED: the legacy master pipeline stays retired, and no other train_chat is signalled
# -- the pattern below matches this run's --out tag and nothing else. If epoch 1 has not landed by the
# deadline, this script logs that and STOPS NOTHING rather than killing a run with nothing on disk.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
LOG="lens/runs/OURS3M-STOP.txt"
HIST="lens/runs/chat/ours_d256_3M_2ep_history.jsonl"
EP1="lens/runs/chat/ours_d256_3M_2ep_ep1.npz"
PAT='^\.venv-gpu/bin/python3 -u lens/scripts/train_chat\.py .*--out ours_d256_3M_2ep$'
DEADLINE=$(( $(date +%s) + 4500 ))     # ~75 min from start; epoch 1 is due ~35 min in

{
echo "=== OURS-3M STOP WAITER START $(date '+%F %H:%M:%S')  pid $$"
echo "waiting for: $HIST to contain an epoch-1 record (written AFTER the ep1 checkpoint and the best checkpoint)"
echo "deadline: $(date -d @$DEADLINE '+%F %H:%M:%S')"
} >> "$LOG"

while true; do
  if [ -f "$HIST" ] && grep -q '"epoch": 1' "$HIST"; then
    sleep 6                              # let the epoch's own writes finish
    pids=$(pgrep -f "$PAT" | tr '\n' ' ')
    if [ -z "$pids" ]; then
      { echo "$(date '+%F %H:%M:%S') epoch 1 IS on disk but no matching trainer is running any more --"
        echo "  nothing to stop (it finished or was stopped elsewhere). ep1 file:"; ls -l --time-style=+%H:%M:%S "$EP1" 2>&1; } >> "$LOG"
      break
    fi
    { echo "$(date '+%F %H:%M:%S') EPOCH 1 LANDED. epoch line from the log:"
      grep -E "^  ep 1/2 " lens/runs/OURS-METHOD-RUN2.txt | tail -1
      ls -l --time-style=+%H:%M:%S "$EP1" lens/runs/chat/ours_d256_3M_2ep_best.npz 2>&1
      echo "  signalling pids: $pids (SIGTERM -- the epoch's artifacts are already on disk)"
      kill -TERM $pids 2>&1
    } >> "$LOG"
    sleep 20
    still=$(pgrep -f "$PAT" | tr '\n' ' ')
    { echo "$(date '+%F %H:%M:%S') still running after SIGTERM: '${still:-none}'"
      echo "  the watchdog (5-min tick) will now start the queue front: lens/scripts/run_attn_full.sh"
      echo "=== WAITER DONE $(date '+%F %H:%M:%S')"; } >> "$LOG"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    { echo "$(date '+%F %H:%M:%S') DEADLINE REACHED and epoch 1 is NOT on disk -> STOPPING NOTHING (no artifact"
      echo "  would survive). Check the run by hand before deciding anything."
      echo "=== WAITER DONE $(date '+%F %H:%M:%S')"; } >> "$LOG"
    break
  fi
  sleep 10
done
