#!/bin/bash
# Restart the clean-corpus training with the fixed window builder.
# The running job still has the OLD code loaded in memory (it loops per conversation, 740k iterations), so it
# is stopped and restarted rather than left to grind. The new builder is one vectorised gather.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME

systemctl --user stop glassnet-clean.scope 2>/dev/null
sleep 3
systemctl --user reset-failed glassnet-clean 2>/dev/null

START=$(date +%s)
systemd-run --user --scope --unit=glassnet-clean2 -p MemoryMax=5G -p CPUWeight=30 -p IOWeight=30 \
  bash "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts/run_clean_train.sh" &
sleep 20
echo "restarted at $(date '+%H:%M:%S'), start epoch was $START"
echo "watching for the first epoch and the GPU engaging..."
for i in $(seq 1 20); do
  sleep 15
  if grep -qE "^  ep 1/" lens/runs/CLEAN-TRAIN.txt; then
    NOW=$(date +%s)
    echo "FIRST EPOCH REACHED after $((NOW-START)) seconds of startup"
    grep -E "^  windows|^  ep 1/|window cap" lens/runs/CLEAN-TRAIN.txt | tail -3
    break
  fi
done
ps -eo pcpu,rss,args | grep "[t]rain_chat" | awk '{printf "  cpu %s   ram %.2f GB\n", $1, $2/1048576}'
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader