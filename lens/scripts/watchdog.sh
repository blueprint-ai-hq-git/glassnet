#!/bin/bash
# RESIDENT JOBS: long runs that must be alive. If one dies, it restarts without a human noticing.
RESIDENT_1_NAME="long_compare"; RESIDENT_1_PAT="stable_vs_plain.py"
RESIDENT_2_NAME="gpu_trainer";  RESIDENT_2_PAT="train_chat.py"
resident() {
  local name="$1" pat="$2" cmd="$3" mem="${4:-8G}"
  # GPU CLAIM (2026-11-06 16:4x, MEASURED NEED). A chained experiment (run_wall_ab.sh) takes the card for
  # ~20 min and writes lens/runs/GPU-CLAIM while it holds it. Without this the 15-min tick restarted the
  # resident keeper mid-experiment: two trainers on the 6 GB card, which is the OOM class BUGS.md documents.
  # The claim is a plain file with its owner on line 1, so the skip is visible in WATCHDOG.txt.
  if [ "$name" = "gpu_trainer" ] && [ -e lens/runs/GPU-CLAIM ]; then
    echo "  - RESIDENT '$name' held off: GPU-CLAIM present ($(head -1 lens/runs/GPU-CLAIM 2>/dev/null))" >> lens/runs/WATCHDOG.txt
    return 0
  fi
  if ! pgrep -f "$pat" >/dev/null 2>&1; then
    echo "  - RESIDENT '$name' was dead -> restarting $(date '+%H:%M')" >> lens/runs/WATCHDOG.txt
    # OWN TRANSIENT SERVICE (2026-11-06 15:31, MEASURED). This used to be
    #   ( cd "$root" && setsid nohup bash -c "$cmd" >/dev/null 2>&1 & )
    # and setsid leaves the session but NOT the cgroup, so both keeper jobs ran inside the cron worker's
    # scope (KillMode=control-group, MemoryMax=4G). Measured 15:24: that scope at MemoryCurrent 2.8G of 4G
    # with 1.5G left and both jobs alive; and 2x "SIGTERM ... epoch 0" in lens/runs/RESUME-BEST.txt when a
    # scope was torn down. launch_resident.sh puts each job in its own service and logs the unit + cgroup to
    # lens/runs/RESIDENT-LAUNCH.txt, so a relaunch is verified instead of assumed.
    local res
    res=$(bash lens/scripts/launch_resident.sh "$name" "$mem" "$cmd" 2>&1)
    echo "    RESIDENT '$name': $res" >> lens/runs/WATCHDOG.txt
  fi
}
resident "$RESIDENT_1_NAME" "$RESIDENT_1_PAT" "taskset -c 4,5,6,7 nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/stable_vs_plain.py >> lens/runs/LONG-COMPARE.txt 2>&1" 3G
resident "$RESIDENT_2_NAME" "$RESIDENT_2_PAT" "bash lens/scripts/run_resume_best.sh" 9G

# WATCHDOG - stays updated on its own, and FIXES what it finds instead of reporting a problem and waiting.
#
# The user's rule (2026-10-18): "if you find any problem you can get solve it instead of irritating me
# afterwards that it failed and we are late" and "I don't need to update you, you need to be updated".
# So this script's job is: notice, repair, and only then say what it did.
#
# What it checks and what it DOES about each one:
#   1. master run not running and not finished        -> relaunch it (runners skip finished rungs and
#                                                        resume from the newest checkpoint, so nothing is
#                                                        recomputed and nothing is lost)
#   2. a scope exists but nothing has progressed for
#      longer than STALE minutes                      -> log it, and if the scope is alive but idle, leave
#                                                        it (a long training epoch is not a fault) -- only
#                                                        a fully dead process is relaunched
#   3. corpus build missing or empty                  -> relaunch the finish steps
#   4. disk below 5 GB free                           -> report loudly, do not start anything new
#   5. an OOM kill since the last check               -> record it with the victim's name
#   6. an eval/bench file missing that the queue says
#      should exist                                   -> relaunch the step that produces it
#
# It writes lens/runs/WATCHDOG.txt (the digest includes it) and prints a short summary.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=2

OUT=lens/runs/WATCHDOG.txt
STATE=lens/runs/.watchdog_state
STALE=${STALE:-75}          # minutes without progress before a run is called stalled
ACTIONS=""
note() { ACTIONS="${ACTIONS}  - $1"$'\n'; echo "  - $1"; }

now=$(date '+%s')
mkdir -p lens/runs

# ---------------------------------------------------------------- 0a. ONE WATCHDOG AT A TIME
# THE BUG THIS FIXES (2026-10-22 15:40). Two watchdog ticks overlapped - the machine had just booted and the
# scheduler ran the missed 5-minute occurrences together. Both found the machine idle, both read the SAME
# first line of QUEUE.txt, both deleted that line and both launched run_tied_alldata.sh. Result: two
# train_chat processes on one 6 GB GPU, memory pressure 68% for > 20 s, and systemd-oomd killed this cron
# scope ITSELF (journal 2026-10-22 15:41:18, "killed 2 process(es) in this unit"), taking the second trainer
# with it. WATCHDOG.txt and QUEUE-DONE.txt each got the line twice. A tick that runs longer than the tick
# interval must not be able to start a second copy of itself.
#
# AND WHY THIS IS A LOCK DIRECTORY AND NOT flock ON A FILE (2026-10-22 16:00, measured). The first version used
# flock, and it STALLED THE QUEUE: bash keeps the lock fd open in the children it starts, the queue job launched
# in section 0 inherited it, so the lock stayed held for the whole life of the training run while the pid file
# named a watchdog that had already exited. Four consecutive ticks wrote "lock is held (owner 341575) and
# freeing it failed" into WATCHDOG.txt (15:51, 15:57, 16:00 x2) and did NOTHING. mkdir is atomic, the owner is
# checkable, and a dead owner's lock is removed, so a leaked descriptor cannot stall the machine.
LOCK=lens/runs/.watchdog.lockdir
lock_holder() { cat "$LOCK/pid" 2>/dev/null; }
lock_alive() { h=$(lock_holder); [ -n "$h" ] && kill -0 "$h" 2>/dev/null; }
acquire() {
  mkdir "$LOCK" 2>/dev/null || return 1
  echo $$ > "$LOCK/pid"
  trap 'rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null' EXIT
  return 0
}
if ! acquire; then
  holder=$(lock_holder)
  age=$(( (now - $(date -r "$LOCK" '+%s' 2>/dev/null || echo "$now")) / 60 ))
  if [ -n "$holder" ] && kill -0 "$holder" 2>/dev/null && [ "${age:-0}" -lt 25 ]; then
    echo "=== WATCHDOG $(date '+%F %H:%M') another watchdog (pid ${holder}, running ${age} min) holds the lock: NOT touching the queue" >> "$OUT"
    exit 0
  fi
  echo "=== WATCHDOG $(date '+%F %H:%M') stale lock (owner ${holder:-none}, ${age} min old): taking it over" >> "$OUT"
  if [ -n "$holder" ]; then kill -TERM "$holder" 2>/dev/null; fi
  rm -f "$LOCK/pid"
  rmdir "$LOCK" 2>/dev/null
  sleep 2
  if ! acquire; then
    echo "=== WATCHDOG $(date '+%F %H:%M') could not take the lock even after clearing a stale one: NOT touching the queue" >> "$OUT"
    exit 0
  fi
fi

# ---------------------------------------------------------------- 1. is the master run alive or finished?
# the marker lives in the project: /tmp gets cleaned, and when that happened the watchdog saw
# "finished: 0" and relaunched the entire run, duplicating hours of work (2026-10-19 08:19).
master_done=$(grep -c "MASTER RUN DONE" /tmp/master_run.log 2>/dev/null)
[ -f lens/runs/MASTER-DONE.flag ] && master_done=1
chain_alive=$(pgrep -cf "[r]un_morning_chain|[r]un_corpus_then_rest|[r]un_finish_and_train" 2>/dev/null)
if [ "${chain_alive:-0}" -gt 0 ]; then
  echo "  a chain job is running: the watchdog will not relaunch the master run behind it"
  master_alive=1
fi
master_alive=$(pgrep -cf "[r]un_master.sh" 2>/dev/null)
if [ "${master_alive:-0}" -eq 0 ] && [ "${master_done:-0}" -eq 0 ]; then
  # THE MASTER PIPELINE IS RETIRED (2026-10-21). It used to be relaunched here whenever it was
  # "not running and not finished", which put TWO trainings on the GPU at once: the master chain and the
  # queue. At 15:48:45 that competition hit the memory wall and the kernel OOM-killed a 5.2 GB python
  # inside glassnet-master.scope. The queue is now the ONLY source of work; this script starts queued
  # jobs and repairs dead ones, and never launches a pipeline of its own.
  note "master run: retired (the queue is the only source of work)"
  sleep 5
fi

# ---------------------------------------------------------------- 2. progress check
# THE BUG THIS FIXES (2026-10-19): this used to relaunch the ENTIRE master sequence whenever no checkpoint had
# been written for STALE minutes, even with nothing queued. That is how the master run kept reappearing,
# redoing work and competing for memory with the real job, and getting OOM-killed. Relaunching is now gated on
# an explicit queue marker: if lens/runs/QUEUE.txt does not list work, the watchdog REPORTS idle instead of
# inventing work. Its job is to keep real jobs alive, not to start new ones on a timer.
newest=$(ls -t lens/runs/chat/*.npz 2>/dev/null | head -1)
QUEUED=$(grep -cvE "^\s*#|^\s*$" lens/runs/QUEUE.txt 2>/dev/null || echo 0)
if [ -n "$newest" ]; then
  age=$(( (now - $(date -r "$newest" '+%s')) / 60 ))
  if [ "$age" -gt "$STALE" ]; then
    if [ "${master_alive:-0}" -gt 0 ]; then
      note "no new checkpoint for ${age} min, but a run is alive: leaving it alone (a long epoch is not a fault)"
    elif [ "${QUEUED:-0}" -gt 0 ]; then
      # THE MASTER RELAUNCH THAT USED TO BE HERE IS REMOVED (2026-10-22). It was left behind when the master
      # pipeline was retired on 2026-10-21: section 1 stopped relaunching it but this branch still did, so at
      # 15:40 both fired on the same tick - the RAM-heavy corpus build (build_knowledge_corpus.py, 2.9 GB) and
      # the queued GPU job (train_chat, 8.6 GB) - with 14 GB of RAM in the machine and 1 GB left over. That is
      # the same competition that had already OOM-killed a 5.2 GB python (2026-10-21 15:48). The queue is the
      # only source of work now, and section 0 below starts its next entry; nothing here may start a pipeline.
      note "no new checkpoint for ${age} min and ${QUEUED} queued job(s): the master is RETIRED, the queue starts it in section 0"
    else
      note "no new checkpoint for ${age} min and NOTHING QUEUED: reporting idle rather than inventing work"
    fi
  fi
fi

# ---------------------------------------------------------------- 3. corpus present?
if [ ! -f data/chat/prepared_mix_ctrl_honest/ids_v12000.npy ]; then
  if ! pgrep -f "[r]un_mix_finish|[r]un_mix_build" > /dev/null; then
    note "combined corpus missing and no build running: relaunching the finish steps"
    setsid nohup bash lens/scripts/run_mix_finish_scope.sh >> /tmp/watchdog_relaunch.log 2>&1 < /dev/null &
    sleep 5
  fi
fi

# ---------------------------------------------------------------- 4. disk
free_gb=$(df -BG . | tail -1 | awk '{print $4}' | tr -d 'G')
if [ "${free_gb:-99}" -lt 5 ]; then
  note "DISK LOW: ${free_gb} GB free. Not starting anything new; old run logs and stale checkpoints are the safe things to clear, and that needs the user's call"
fi

# ---------------------------------------------------------------- 5. OOM kills since last check
oom=$(grep -c "oom-kill" /var/log/syslog 2>/dev/null || echo 0)
last_oom=$(cat "$STATE" 2>/dev/null | awk -F= '/oom/{print $2}')
if [ "${oom:-0}" != "${last_oom:-0}" ]; then
  victim=$(grep "oom-kill" /var/log/syslog 2>/dev/null | tail -1 | grep -oE "\([^)]+\)" | head -1)
  note "an OOM kill happened since the last check (count ${last_oom:-0} -> ${oom}); victim ${victim:-unknown}. Jobs now run in their own memory-capped scopes, so a repeat kills the job, not the desktop"
fi

# ---------------------------------------------------------------- 6. expected result files
for pair in "lens/runs/METHODS-BENCH.txt:lens/scripts/run_methods_bench.sh" \
            "lens/runs/REWARD-HACKING-AUDIT.txt:lens/scripts/run_reward_hacking_audit.sh"; do
  f="${pair%%:*}"; s="${pair##*:}"
  if [ ! -f "$f" ] && [ -f "$s" ]; then
    note "$(basename "$f") missing: running $(basename "$s")"
    setsid nohup bash "$s" >> /tmp/watchdog_relaunch.log 2>&1 < /dev/null &
    sleep 3
  fi
done

# ---------------------------------------------------------------- 0. START THE NEXT QUEUED JOB
# This is the "be active" rule (2026-10-19): the user should not have to notice that a run finished. If no
# training is running and the queue has a job, start it. The cron job that runs this script fires reliably
# (verified), so the queue is now driven by a mechanism that actually ticks, not by a hand-rolled loop.
# first, let the machine give itself the next job if the queue is empty
bash lens/scripts/feed_queue.sh 2>/dev/null

# WHY THIS PATTERN IS NOW EXACT (2026-10-22 16:50, MEASURED - THE QUEUE HAD STALLED ON IT).
# This line used to be `pgrep -cf "[t]rain_chat"`, which matches ANY process whose command line merely MENTIONS
# train_chat. A leftover Hermes tool wrapper (`/usr/bin/bash -c source .../hermes-snap-*.sh ...`) had the string
# in its arguments, so the count was 1 forever: the watchdog believed a trainer was running when `ps` showed 0,
# wrote "found nothing wrong: every job that should be running is running", and refused to start the next queued
# job. The queue stalled with nothing running and no error anywhere. Match the actual interpreter and script.
# FIXED AGAIN 2026-11-06 16:0x (BUG-TRAINER-DETECT): matching "train_chat\.py" made every OTHER project
# entry point invisible - on this machine stable_vs_plain.py held 4 cores (PID 2504549, 43 min) while this
# count read 0, so a tick would have started the next queue entry beside it (the class of double-launch the
# block below was written to prevent; it already fired once today: unit glassnet-res-gpu_trainer-20261106-153258
# at 15:32:58, killed 143 at 15:35:02). Count any project interpreter running a lens/scripts/*.py instead; a
# bash wrapper's argv[1] is -c/taskset, so wrappers and agent shells containing the string are NOT counted.
qjobs=$(ps -eo etimes=,args= 2>/dev/null | awk '$2 ~ /(^|\/)\.venv(-gpu)?\/bin\/python3$/ {
  for (i = 3; i <= NF; i++) if ($i ~ /lens\/scripts\/[^ ]*\.py$/) { n++; break } } END { print n + 0 }')
qrunning=${qjobs:-0}
# A QUEUED JOB MUST NOT START BESIDE ANY OTHER HEAVY JOB (2026-10-22). The guard used to look only for
# train_chat, so a tick that found the GPU idle while the retired master pipeline was already chewing 2.9 GB of
# RAM started the GPU job anyway: 11.5 GB of 14 GB gone, 1 GB left, and systemd-oomd killing scopes. The
# machine runs ONE heavy job at a time; a corpus build or a python trainer both count as heavy.
qbuild=$(pgrep -cf "[b]uild_knowledge_corpus|[r]un_mix_build|[r]un_mix_finish|[r]un_size_sweep|[r]un_ladder" 2>/dev/null); qbuild=${qbuild:-0}
qheavy=$(( ${qrunning:-0} + qbuild ))
if [ "${qheavy:-0}" -gt "${qrunning:-0}" ]; then
  note "the queue has work but ${qheavy} heavy process(es) are already running (not train_chat): NOT starting the next entry"
  qrunning=${qheavy}
fi
# A QUEUED JOB MUST NOT START IN THE GAP BETWEEN THE ARMS OF THE JOB ALREADY RUNNING (2026-10-24 self-wake).
# MEASURED HAZARD: a queue line is often a RUNNER SCRIPT THAT RUNS SEVERAL TRAINERS BACK TO BACK
# (run_ours_sweep.sh = seven arms; run_unk_drop.sh = the arm, then its graders and its talk tests;
# run_attn_grade_talk.sh = a waiter, a grader and two instruments). Between two arms of such a script there is
# a gap of about a second in which `pgrep train_chat` is 0 WHILE THE JOB IS VERY MUCH ALIVE, so a tick landing
# in that gap would start the NEXT queue entry beside it: two trainers on a 6 GB GPU, which is exactly what
# OOM-killed scopes on this machine before (2026-10-22 15:41 journal, and the 15:48:45 kill). The transient
# scope covers the WHOLE queue line, gaps included, so an ACTIVE glassnet-q-* scope is the honest
# "the machine is busy" test; stale units are excluded by asking for --state=active only.
# Checked before it was trusted: with the sweep running this reads 1
# (glassnet-q-3757950001-074809.scope) and with a pattern matching no unit it reads 0.
qscopes=$(systemctl --user list-units 'glassnet-q-*' --state=active --no-legend --plain 2>/dev/null | grep -c .)
if [ "${qscopes:-0}" -gt 0 ] && [ "${qrunning:-0}" -eq 0 ]; then
  note "the queue has work but ${qscopes} active glassnet-q-* scope(s) are running a multi-arm job: NOT starting the next entry (this is the between-arms gap, not idleness)"
  qrunning=1
fi
# VRAM GUARD (2026-10-26). Yahya loads his own model into VRAM (LM Studio held 4,408 MiB of 6,144). Starting
# a trainer beside that took the card over the limit and the run died. A queued job now waits while the free
# VRAM is under 1,500 MiB, and says so, instead of launching and dying.
vram_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
vram_free=${vram_free:-9999}
if [ "$vram_free" -lt 1500 ]; then
  note "VRAM GUARD: only ${vram_free} MiB free on the GPU (something else is loaded - LM Studio?). NOT starting '$qnext'"
  qnext=""
fi
# STANDING PLAN REFILL (2026-10-30). The queue used to drain and the machine sat idle three times in one day
# while I reported progress. Now an empty queue pulls the next experiment from PLAN.txt automatically, and if
# the plan is ALSO empty that is written to the inbox so it is visible instead of silent.
qcount=$(grep -cvE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null)
if [ "${qcount:-0}" -eq 0 ] && [ "${qrunning:-0}" -eq 0 ]; then
  pnext=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/PLAN.txt 2>/dev/null | head -1)
  if [ -n "$pnext" ]; then
    echo "$pnext" >> lens/runs/QUEUE.txt
    grep -vF "$pnext" lens/runs/PLAN.txt > lens/runs/PLAN.txt.tmp && mv lens/runs/PLAN.txt.tmp lens/runs/PLAN.txt
    note "STANDING PLAN: queue was empty, pulled '$pnext'"
  else
    note "STANDING PLAN EMPTY: nothing training and no queued or planned experiment - new work must be written"
  fi
fi
qnext=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' lens/runs/QUEUE.txt 2>/dev/null | head -1)
# GPU CLAIM, QUEUE SIDE (2026-11-06 16:4x). Same file as the resident guard above: while a chained job owns
# the card, a tick must not start the queue's own copy of the same work (double launch, OOM risk).
if [ -e lens/runs/GPU-CLAIM ]; then
  note "GPU-CLAIM present ($(head -1 lens/runs/GPU-CLAIM 2>/dev/null)): NOT starting '$qnext'"
  qnext=""
fi
if [ "${qrunning:-0}" -eq 0 ] && [ -n "$qnext" ]; then
  note "nothing running and the queue has work: starting '$qnext'"
  grep -vxF "$qnext" lens/runs/QUEUE.txt > lens/runs/QUEUE.tmp 2>/dev/null && mv lens/runs/QUEUE.tmp lens/runs/QUEUE.txt
  echo "$(date '+%F %H:%M') $qnext" >> lens/runs/QUEUE-DONE.txt
  # CONSUME ON LAUNCH (2026-10-31): a finished entry stayed in QUEUE.txt and the watchdog
  # relaunched it instead of advancing - that cost 78 idle minutes on 2026-10-30. Remove it here.
  grep -vxF "$qnext" lens/runs/QUEUE.txt > lens/runs/QUEUE.txt.tmp 2>/dev/null && mv lens/runs/QUEUE.txt.tmp lens/runs/QUEUE.txt
  # WHY THE JOB NOW GETS ITS OWN SYSTEMD SCOPE (2026-10-22, MEASURED). A queued job launched as a plain
  # `setsid nohup bash -c` inherits the CGROUP of whatever ran this script. The d=512 run started by the 15:46
  # tick lived in hermes-worker-cron-83c757488676-exec-....scope and was SIGTERM'd 2 min 48 s later
  # (exit=143, no epoch line, no checkpoint: lens/runs/TIED-D512.txt). A cgroup teardown kills every member and
  # setsid does not leave the cgroup. The corpus build that WAS launched in its own scope (glassnet-master via
  # systemd-run --user --scope) survived every tick. So each queued job gets a transient scope of its own: the
  # tick may end, the job keeps running. MemoryMax comes from a MEASURED peak - the full-corpus d=256 trainer
  # reached 8.6 GB RSS (ps on pid 296421, 2026-10-22 15:45) - so a job that overreaches dies inside its own
  # scope instead of taking the desktop down with it. The real fix for that 8.6 GB is to shrink the trainer's
  # window array; see AUTOPILOT-LOG 2026-10-22.
  # UNIQUE UNIT NAME PER LAUNCH (2026-10-22 16:5x, MEASURED BUG). The name used to come from the command text
  # alone, so a queue entry whose FIRST launch died could never be started again from the queue: at 16:52 this
  # tick spent the full-corpus line and systemd answered
  #     "Failed to start transient scope unit: Unit glassnet-q-2591701866.scope was already loaded or has a
  #      fragment file."
  # (lens/runs/SUPERVISOR.txt) - the launch failed, NOTHING ran, and the line was already consumed, so the work
  # was silently eaten (QUEUE-DONE.txt says "started" while the log has no new block). Verified afterwards: the
  # unit CAN be cleared (`reset-failed <unit>` -> LoadState=not-found), so this was a RACE between
  # reset-failed and systemd-run, not a permanent block - but a race that costs work is still a bug. The name
  # now carries the launch time and the launch is VERIFIED below instead of assumed.
  unit="glassnet-q-$(printf '%s' "$qnext" | cksum | cut -d' ' -f1)-$(date +%H%M%S)"
  systemctl --user reset-failed "$unit" >/dev/null 2>&1
  # KillMode=process + an infinite stop timeout (2026-10-23): WITHOUT these, a job can be SIGTERM'd
  # (exit 143) when the scope that launched it is torn down. Measured twice: the d=512 run yesterday and the
  # our-method run today, which died at 17:11 after 85 minutes with no epoch finished.
  # TRANSIENT SERVICE, NOT A SCOPE (2026-10-24, after FOUR runs were SIGTERM'd).
  # A --scope unit is a member of the caller's cgroup hierarchy, so when the cron worker that
  # launched it goes away the job goes with it: measured deaths at 17:11, 19:35 and 10:08 today
  # (exit 143, no epoch finished). A transient SERVICE is a first-class unit of its own, with its
  # own cgroup, so nothing that happens to the launcher can take it down.
  systemctl --user reset-failed "$unit" >/dev/null 2>&1
  systemd-run --user --unit="$unit" -p MemoryMax=10G -p KillMode=process \
    -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
    bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" \
    >> lens/runs/SUPERVISOR.txt 2>&1 < /dev/null
  # THE STRAY FALLBACK LINE THAT USED TO SIT HERE IS REMOVED (2026-10-25 self-wake, MEASURED).
  # It read:
  #   bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" >> lens/runs/SUPERVISOR.txt 2>&1 < /dev/null &
  # and it was NOT behind an `else`: it ran on EVERY launch, unconditionally, right after systemd-run. So every
  # queue line was started TWICE — one copy as the transient service, one copy in the TICK'S OWN CGROUP (which is
  # exactly the failure the comment above warns about). MEASURED at 2026-10-25 10:30:38 on
  # `bash lens/scripts/run_tied_w64.sh`: two identical trainers (pid 4020503 in scope
  # glassnet-q-2325387518-103038, pid 4020510 in hermes-worker-cron-83c757488676-exec-....scope), both writing
  # the same header and the same --out tag into lens/runs/TIED-W64.txt, QUEUE-DONE.txt carrying BOTH a plain
  # line and a "[scope ...]" line for one launch, and ~7 GB of a 14 GB machine held by a duplicate that would
  # have OOM-killed the real arm. This is the proven cause of [BUG-DOUBLE-LAUNCH-OF-ONE-QUEUE-LINE] (BUGS.md).
  # The launch is NOT left unverified without it: the 15 s check below still runs, and a job that did not start
  # is put back at the front of the queue — which is the documented behaviour this line was duplicating.
  sleep 15
  if systemctl --user is-active "$unit" >/dev/null 2>&1 || [ "$(pgrep -cf '[t]rain_chat')" -gt 0 ]; then
    echo "  scope: $unit  -> $(systemctl --user show -p ControlGroup --value "$unit" 2>/dev/null)" >> "$OUT"
    echo "$(date '+%F %H:%M') $qnext [scope $unit]" >> lens/runs/QUEUE-DONE.txt
  else
    # THE LINE GOES BACK TO THE FRONT OF THE QUEUE. A consumed line with nothing running is lost work; the
    # next tick then tries again with a different unit name, and the failure is written where it can be seen.
    # AND THE CHECK IS NOT ONLY THE SCOPE (2026-10-22 16:5x, MEASURED FALSE NEGATIVE): at 16:54 the scope
    # reported inactive 15 s after a launch while its python3 child was ALIVE and training - the launcher shell
    # had exited (the runner file was being rewritten by a parallel turn while bash was reading it), so the
    # scope was torn down and the child was orphaned but working. Reading only the unit would have called that
    # a failed launch and queued a DUPLICATE training run. A live train_chat is therefore taken as "started".
    # A UNIT THAT RAN AND EXITED 0 IS A SUCCESS, NOT A FAILED LAUNCH (2026-10-25). The check used to accept
    # only "unit active" or "a live train_chat", so every JOB THAT FINISHES IN UNDER 15 s - the collect and
    # decode scripts - was declared DID-NOT-START, put back at the front of the queue and launched again,
    # forever. That is what stalled the queue with the machine idle: the same line, 56 launches. Now a unit
    # whose recorded Result is success counts as started.
    unit_result=$(systemctl --user show -p Result --value "$unit" 2>/dev/null)
    unit_active=$(systemctl --user show -p ActiveState --value "$unit" 2>/dev/null)
    if [ "${unit_result:-}" = "success" ]; then
      note "job '$qnext' ran to completion (unit $unit result=success): NOT re-queued"
      echo "$(date '+%F %H:%M') $qnext [ran to completion]" >> lens/runs/QUEUE-DONE.txt
      sleep 1
    else
    note "QUEUED JOB DID NOT START ('$qnext', scope $unit): the line is put back at the front of the queue"
    printf '%s\n%s' "$qnext" "$(cat lens/runs/QUEUE.txt 2>/dev/null)" > lens/runs/QUEUE.tmp
    mv lens/runs/QUEUE.tmp lens/runs/QUEUE.txt
    echo "$(date '+%F %H:%M') $qnext [DID NOT START, line restored; unit result=${unit_result:-unknown} active=${unit_active:-unknown}]" >> lens/runs/QUEUE-DONE.txt
    fi
  fi
elif [ "${qrunning:-0}" -eq 0 ] && [ -z "$qnext" ]; then
  note "nothing running and the queue is EMPTY: idle. Put work in lens/runs/QUEUE.txt and it starts on the next tick"
fi

# ---------------------------------------------------------------- report
{
  echo "=== WATCHDOG $(date '+%F %H:%M')"
  if [ -n "$ACTIONS" ]; then echo "FOUND AND FIXED:"; printf '%s' "$ACTIONS"
  else echo "found nothing wrong: every job that should be running is running and progressing"; fi
  echo "  master run: $([ "${master_alive:-0}" -gt 0 ] && echo alive || echo "not running (finished: ${master_done:-0})")"
  echo "  newest checkpoint: $(basename "${newest:-none}") $([ -n "$newest" ] && echo "($(( (now - $(date -r "$newest" '+%s')) / 60 )) min old)")"
  echo "  disk free: ${free_gb:-?} GB   ram: $(free -g | awk 'NR==2{print $3"G/"$2"G"}')   gpu: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null)"
} >> "$OUT"
echo "oom=${oom:-0}" > "$STATE"
tail -8 "$OUT"