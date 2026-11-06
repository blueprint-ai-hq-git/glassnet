#!/bin/bash
# Launch the knowledge corpus build DETACHED. CPU-only work (pyarrow + numpy) at nice 19, so it runs
# alongside a GPU training job without fighting it for VRAM.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_build_knowledge.sh > /tmp/build_knowledge_launch.log 2>&1 < /dev/null &
echo "knowledge build launched detached, pid $!"
sleep 3
pgrep -af "[b]uild_knowledge_corpus" | cut -c1-60