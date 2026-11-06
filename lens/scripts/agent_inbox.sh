#!/bin/bash
# ==================================================================================================
# AGENT INBOX / TWO-LOCK WATCHER  (2026-10-27)
# Lock 1 = this script (cheap, every 10 min, no LLM): it looks for the six things that actually went wrong
# in this project, writes them to a durable inbox my next turn reads FIRST, and prints a short line for
# Telegram ONLY when something is wrong or new. Empty output = nothing sent, so it cannot spam.
# Lock 2 = the self-wake agent (every 30 min): it reads the same inbox and acts. Two independent checks,
# so one missing something does not mean it is missed.
# WHY: twice this project lost hours with nothing running - the queue drained at 10:16 and the loops were
# paused, and a launcher bug stalled the queue for hours while the watchdog said "nothing wrong".
# ==================================================================================================
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
INBOX=lens/runs/AGENT-INBOX.md
STATE=lens/runs/.inbox_state
now=$(date '+%s')
# JOBS, not "train_chat only" (fix 2026-11-06, BUG-TRAINER-DETECT in BUGS.md). The old line was
#   trainers=$(pgrep -cf "venv-gpu/bin/python3 -u lens/scripts/train_chat")
# which is wrong in BOTH directions, both measured today on the live table:
#   (a) INVISIBLE REAL JOB - stable_vs_plain.py (PID 2504549, 4 cores, started 15:17) does not contain
#       "train_chat", so at 15:30 and 15:52 this line reported trainers=0 while it ran; the watchdog then
#       launched a SECOND job (systemd unit glassnet-res-gpu_trainer-20261106-153258.service, 15:32:58->
#       15:35:02, died 143, 2min01s CPU, 937.6M peak) and the queue alert was false.
#   (b) PHANTOM TRAINER - pgrep -f matches ANY cmdline containing the string, including the agent's own
#       `bash -c ...` wrapper for the command that runs it (observed: PID 2718236, /usr/bin/bash -c ...).
# Count instead every process whose argv[0] IS a project interpreter AND that runs a lens/scripts/*.py.
# A bash wrapper's argv[1] is -c/taskset, so wrappers, pgrep's own shell and graders' wrappers are excluded.
read -r trainers tr_el < <(ps -eo etimes=,args= 2>/dev/null | awk '
  $2 ~ /(^|\/)\.venv(-gpu)?\/bin\/python3$/ {
    for (i = 3; i <= NF; i++) if ($i ~ /lens\/scripts\/[^ ]*\.py$/) { n++; if ($1 > m) m = $1; break }
  }
  END { printf "%d %d\n", n + 0, int(m / 60) + 0 }')
trainers=${trainers:-0}; tr_el=${tr_el:-0}
queued=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null); queued=${queued:-0}
newest=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1)
age=9999
[ -n "$newest" ] && age=$(( (now - $(date -r "$newest" '+%s')) / 60 ))
oom=$(grep -c 'oom-kill' /var/log/syslog 2>/dev/null); oom=${oom:-0}
oom_prev=$(cat "$STATE" 2>/dev/null | head -1); oom_prev=${oom_prev:-$oom}   # first run records, never alerts
vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1); vram=${vram:-9999}
ALERTS=""
add() { ALERTS="${ALERTS}$1"$'\n'; }
# LOCK 1: the six failure shapes, each with the numbers that prove it
[ "$trainers" -eq 0 ] && [ "$queued" -gt 0 ]  && add "QUEUE STALLED: nothing training but $queued job(s) waiting - the watchdog should start one; if it does not, its launch path is broken again."
[ "$trainers" -eq 0 ] && [ "$queued" -eq 0 ]  && add "IDLE WITH NO WORK: nothing training and the queue is EMPTY. New experiments must be written, or the machine sits unused (this cost 3.5 h today)."
# tr_el (elapsed minutes of the longest-running project job) now comes from the same scan above, so the
# STALLED RUN check and the "is anything running" check can never disagree (old line grep'd only train_chat).
# EXPECTED EPOCH TIME from the run's own log, so a big arm is not mistaken for a stalled one (2026-10-29: the
# 2M-window arm took 73.5 min per epoch and raised two false "STALLED RUN" alerts at 18:03 and 18:14).
last_ep=$(grep -hoE '\(([0-9]+)s\)' lens/runs/*.txt 2>/dev/null | tail -1 | tr -dc '0-9')
STALL_LIMIT=45
if [ -n "$last_ep" ] && [ "$last_ep" -gt 600 ]; then STALL_LIMIT=$(( last_ep * 3 / 2 / 60 )); fi
[ "$trainers" -gt 0 ] && [ "$tr_el" -gt "$STALL_LIMIT" ] && [ "$age" -gt "$STALL_LIMIT" ] && add "STALLED RUN: trainer alive ${tr_el} min, newest checkpoint ${age} min old (allowed ${STALL_LIMIT} min for this arm)."
[ "$oom" -gt "$oom_prev" ]                    && add "OOM KILL: $((oom - oom_prev)) new cgroup OOM kill(s) since the last check. Something overran memory (usually two trainers or a loaded model)."
recent=$(find lens/runs -maxdepth 1 -name '*.txt' -mmin -60 2>/dev/null | grep -vE 'SUPERVISOR|WATCHDOG|AGENT-INBOX|QUEUE')   # SUPERVISOR.txt is retired; its exit=1 is history, not a live failure
[ -n "$recent" ] && grep -hE 'exit=([1-9][0-9]*)' $recent 2>/dev/null | tail -1 | grep -q . && \
  add "NON-ZERO EXIT (last hour): $(grep -hE 'exit=([1-9][0-9]*)' ${recent:-/dev/null} 2>/dev/null | tail -1)"
[ "$vram" -lt 1500 ] && add "GPU OCCUPIED: only ${vram} MiB VRAM free - the queue is gated until it frees."
echo "$oom" > "$STATE"
# write the durable inbox: header always, alerts if any
{ echo "## $(date '+%F %H:%M')"; echo "trainers=$trainers queued=$queued newest_age=${age}min vram_free=${vram}MiB"; \
  [ -n "$ALERTS" ] && echo "$ALERTS"; } >> "$INBOX"
# keep the inbox readable: last 200 lines
tail -200 "$INBOX" > "$INBOX.tmp" && mv "$INBOX.tmp" "$INBOX"
# Telegram line: silent unless there is a real alert
if [ -n "$ALERTS" ]; then
  echo "GLASSNET ALERT $(date '+%H:%M')"
  echo "$ALERTS" | head -4
  echo ""
  echo "--- full check ---"
  bash lens/scripts/status_digest.sh 2>/dev/null | tail -n +2
fi
