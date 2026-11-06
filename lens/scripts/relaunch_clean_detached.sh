#!/bin/bash
# DETACHED relaunch of the clean-corpus training.
#
# LESSON (2026-10-19 16:10): the previous attempt launched the scope with a plain "&" from inside a script
# that the tool call later killed, and the scope died with its launcher after epoch 3. A scope must be started
# with setsid nohup so it belongs to nobody in this process tree. This is the same rule that the project
# already learned for long runs; it is repeated here because it cost a restart.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME

# the watchdog keeps relaunching the master run, which duplicates work and competes for memory: stop it
systemctl --user stop glassnet-master.scope 2>/dev/null
sleep 2
systemctl --user reset-failed 2>/dev/null

setsid nohup systemd-run --user --scope --unit=glassnet-clean3 -p MemoryMax=5G -p CPUWeight=30 \
  -p IOWeight=30 bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_clean_train.sh" \
  > /tmp/clean3_launch.log 2>&1 < /dev/null &
sleep 10
echo "launched detached; checking it survived the launch:"
pgrep -af "run_clean_train|train_chat" | grep -v hermes | cut -c1-70
systemctl --user list-units --type=scope --no-pager 2>/dev/null | grep -i glassnet | cut -c1-40