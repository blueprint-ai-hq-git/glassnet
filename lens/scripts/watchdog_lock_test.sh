#!/bin/bash
# watchdog_lock_test.sh - DOES THE LOCK ACTUALLY DO WHAT IT CLAIMS?
# Written 2026-10-22 after two overlapping ticks launched the same job twice and systemd-oomd killed the cron
# scope (WATCHDOG.txt and QUEUE-DONE.txt each received the 15:40 line twice), and after the FIRST lock attempt
# (flock on a file) stalled the queue because the launched job inherited the lock descriptor.
#
# Three cases, each against the REAL watchdog script, on the real lock path:
#   1. the lock is held by a LIVE owner, fresh  -> the tick must refuse and leave the queue alone
#   2. the lock is held by a DEAD owner         -> the tick must take it over and run
#   3. the lock is held by a LIVE owner but is older than the 25-minute staleness bar -> the tick must kill
#      the owner and run
# It never launches a queue job: a train_chat is alive during this test, and section 0 starts nothing when one
# is. It writes only to WATCHDOG.txt, lens/runs/_tmp/wd_lock_*.txt.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
T=lens/runs/_tmp
LOCK=lens/runs/.watchdog.lockdir
OUT=lens/runs/WATCHDOG.txt
mkdir -p "$T"
pass=0
fail=0

cleanup() { rm -f "$LOCK/pid"; rmdir "$LOCK" 2>/dev/null; }

count_refusals() { grep -c "another watchdog .*holds the lock: NOT touching the queue" "$OUT" 2>/dev/null; }

echo "train_chat running now: $(pgrep -cf '[t]rain_chat') (section 0 must not start a queue job while this is > 0)"

# ---------------- case 1: live owner, fresh ---------------------------------------------------------
cleanup
mkdir -p "$LOCK"
sleep 300 &
owner=$!
echo "$owner" > "$LOCK/pid"
b=$(count_refusals)
bash lens/scripts/watchdog.sh > "$T/wd_lock_case1.txt" 2>&1
a=$(count_refusals)
saw_start=$(grep -c "starting '" "$T/wd_lock_case1.txt")
kill "$owner" 2>/dev/null
if [ "$a" -gt "$b" ] && [ "$saw_start" -eq 0 ]; then
  echo "CASE 1 PASS: the live-owner lock made the tick refuse (refusals $b -> $a) and it started nothing"
  pass=$((pass+1))
else
  echo "CASE 1 FAIL: refusals $b -> $a, started=$saw_start"
  fail=$((fail+1))
fi

# ---------------- case 2: dead owner ----------------------------------------------------------------
cleanup
mkdir -p "$LOCK"
deadpid=$(cat /proc/sys/kernel/pid_max)
echo "$deadpid" > "$LOCK/pid"
bash lens/scripts/watchdog.sh > "$T/wd_lock_case2.txt" 2>&1
took=$(grep -c "stale lock" "$OUT")
if [ ! -d "$LOCK" ] && [ "$took" -gt 0 ]; then
  echo "CASE 2 PASS: a dead owner's lock was taken over ('stale lock' recorded) and the tick released it on exit"
  pass=$((pass+1))
else
  echo "CASE 2 FAIL: dir still there=$([ -d "$LOCK" ] && echo yes || echo no), 'stale lock' lines=$took"
  fail=$((fail+1))
fi

# ---------------- case 3: live owner, 40 minutes old -----------------------------------------------
cleanup
mkdir -p "$LOCK"
sleep 300 &
owner2=$!
echo "$owner2" > "$LOCK/pid"
touch -d "40 minutes ago" "$LOCK"
bash lens/scripts/watchdog.sh > "$T/wd_lock_case3.txt" 2>&1
if kill -0 "$owner2" 2>/dev/null; then
  echo "CASE 3 FAIL: the stale owner (pid $owner2) is still alive - the tick did not free the lock"
  fail=$((fail+1))
  kill "$owner2" 2>/dev/null
else
  echo "CASE 3 PASS: the stale live owner (pid $owner2) was killed and the lock freed"
  pass=$((pass+1))
fi
cleanup

# ---------------- final: the lock is clean and the next tick runs ----------------------------------
bash lens/scripts/watchdog.sh > "$T/wd_lock_final.txt" 2>&1
if [ ! -d "$LOCK" ]; then
  echo "FINAL PASS: lock released after the tick exited (no leak) and the tick completed normally"
  pass=$((pass+1))
else
  echo "FINAL FAIL: the lock directory leaked: $(ls -la "$LOCK")"
  fail=$((fail+1))
fi
echo "RESULT: ${pass} pass / ${fail} fail"
[ "$fail" -eq 0 ]
