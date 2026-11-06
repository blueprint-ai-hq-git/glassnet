#!/bin/bash
# launch_resident.sh - start a long GlassNet job in ITS OWN transient systemd service and prove it landed there.
#
# WHY THIS FILE EXISTS (measured 2026-11-06 15:19-15:31):
#   watchdog.sh's resident() relaunched its two keeper jobs with
#       ( cd "$root" && setsid nohup bash -c "$cmd" >/dev/null 2>&1 & )
#   setsid leaves the SESSION but NOT the CGROUP. Both jobs therefore lived inside the cron worker's scope
#   hermes-worker-cron-...-exec-....scope, which has KillMode=control-group and MemoryMax=4G (read with
#   `systemctl --user show <scope> -p KillMode -p MemoryMax`). Two measured consequences:
#     1. when that scope is torn down the jobs get SIGTERM: "SIGTERM: saving an interrupt checkpoint for epoch 0"
#        appears 2x in lens/runs/RESUME-BEST.txt, i.e. a full corpus load thrown away at epoch 0;
#     2. both keeper jobs share ONE 4 GB cap: at 15:24 the scope read MemoryCurrent 2.8 GB / MemoryMax 4 GB,
#        1.5 GB left, with the d=256 trainer (2.1 GB RSS) and the stable-vs-plain comparison both alive, so an
#        OOM there loses both.
#   The QUEUE launch path in watchdog.sh was given a transient service on 2026-10-24 for exactly this reason
#   ("A --scope unit is a member of the caller's cgroup hierarchy ... A transient SERVICE is a first-class unit
#   of its own, with its own cgroup"). The resident path never got that fix. This is that fix.
#
# USAGE: lens/scripts/launch_resident.sh <name> <MemoryMax> <command string>
#   name       short tag, becomes the unit glassnet-res-<name>-<YYYYmmdd-HHMMSS>.service
#   MemoryMax  a cap, e.g. 9G. A job that overreaches dies inside its own unit, not on the desktop.
#   command    shell command, exec'd inside the unit; its own redirects still apply.
# EXIT: 0 = unit active or the process pattern is visible; 1 = launch failed (reason printed and logged).
set -u
name="${1:?name}"
mem="${2:?MemoryMax}"
cmd="${3:?command}"
root="/run/media/yahya/HARD DISK 4/GlassNet"
unit="glassnet-res-${name}-$(date +%Y%m%d-%H%M%S)"
log="$root/lens/runs/RESIDENT-LAUNCH.txt"

systemctl --user reset-failed "$unit" >/dev/null 2>&1
out=$(systemd-run --user --unit="$unit" -p "MemoryMax=$mem" -p KillMode=process \
        -p TimeoutStopSec=infinity -p CPUWeight=20 -p IOWeight=20 \
        bash -c "cd \"$root\" && exec $cmd" 2>&1)
rc=$?
stamp="$(date '+%F %H:%M:%S')"
if [ "$rc" -ne 0 ]; then
  echo "$stamp LAUNCH FAILED name=$name unit=$unit rc=$rc :: $out" >> "$log"
  echo "LAUNCH_FAILED unit=$unit rc=$rc :: $out"
  exit 1
fi

sleep 8
cgroup=$(systemctl --user show -p ControlGroup --value "$unit" 2>/dev/null)
active=$(systemctl --user is-active "$unit" 2>/dev/null)
echo "$stamp LAUNCHED name=$name unit=$unit mem=$mem active=$active cgroup=$cgroup cmd=$cmd" >> "$log"
echo "LAUNCHED unit=$unit active=$active cgroup=$cgroup"
if [ "$active" = "active" ] || [ -n "$cgroup" ]; then
  exit 0
fi
echo "LAUNCH_UNVERIFIED unit=$unit (see lens/runs/RESIDENT-LAUNCH.txt)"
exit 1
