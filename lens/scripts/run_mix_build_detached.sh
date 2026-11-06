#!/bin/bash
# Launch the combined-corpus build DETACHED. It waits for the code download, then builds and verifies the
# merged corpus in the standard chat format. CPU only, so the GPU ladders keep running.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_mix_build.sh > /tmp/mix_build_launch.log 2>&1 < /dev/null &
echo "combined corpus build launched detached, pid $!"
sleep 3
pgrep -af "[b]uild_knowledge_corpus|[m]erge_corpora" | cut -c1-60
