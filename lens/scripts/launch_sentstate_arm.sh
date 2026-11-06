#!/bin/bash
# launch_sentstate_arm.sh -- start the QUEUE FRONT (bash lens/scripts/run_sentstate_arm.sh) the way the
# watchdog does, but now rather than at the next 5-minute tick.
#
# SAME LAUNCH DISCIPLINE AS lens/scripts/watchdog.sh (all of it measured there, not chosen here):
#   * a TRANSIENT SERVICE, not `--scope`: a scope is a member of the caller's cgroup, so when the cron worker
#     that launched it goes away the job goes with it (four measured deaths, exit 143, watchdog.sh:229-233);
#   * MemoryMax=10G (the measured peak of a full-corpus d=256 trainer), KillMode=process,
#     TimeoutStopSec=infinity;
#   * the launch is VERIFIED afterwards, and a job that did not start is reported instead of assumed.
# The unit name carries this launch's timestamp so it can never collide with a loaded unit
# ([BUG-SCOPE-NAME-COLLISION], the 2026-10-22 16:52 silent failure).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unit="glassnet-q-$$-$(date '+%H%M%S')"
out=lens/runs/SUPERVISOR.txt
if [ "$(pgrep -cf '[t]rain_chat')" -gt 0 ]; then
  echo "$(date '+%F %H:%M:%S') LAUNCH REFUSED: a trainer is already running ($(pgrep -cf '[t]rain_chat') process(es)); the sentstate arm will be started by the watchdog's next tick instead" >> "$out"
  exit 2
fi
systemctl --user reset-failed "$unit" >/dev/null 2>&1
systemd-run --user --unit="$unit" -p MemoryMax=10G -p KillMode=process \
  -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
  bash -c "cd \"/run/media/yahya/HARD DISK 4/GlassNet\" && exec taskset -c 0,1 nice -n 19 ionice -c3 bash lens/scripts/run_sentstate_arm.sh" \
  >> "$out" 2>&1 < /dev/null
echo "$(date '+%F %H:%M:%S') sentstate arm launch requested as unit $unit (queue front)" >> "$out"
sleep 20
if systemctl --user is-active "$unit" >/dev/null 2>&1 || [ "$(pgrep -cf '[t]rain_chat')" -gt 0 ]; then
  echo "$(date '+%F %H:%M:%S') LAUNCH OK: unit $unit active, train_chat processes: $(pgrep -cf '[t]rain_chat')" >> "$out"
else
  echo "$(date '+%F %H:%M:%S') LAUNCH FAILED: unit $unit is not active and no train_chat is running -- the queue line must be re-queued by hand" >> "$out"
  exit 1
fi
tail -4 "$out"
