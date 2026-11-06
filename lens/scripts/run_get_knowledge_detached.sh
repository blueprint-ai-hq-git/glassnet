#!/bin/bash
# Launch the knowledge download DETACHED. It already skips files that landed and re-fetches partials,
# so re-running is safe. Detached because a Hermes-owned background process gets reaped when the agent's
# worker scope is torn down (that is what killed the first attempt at 11:59).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
setsid nohup bash lens/scripts/run_get_knowledge.sh > /tmp/knowledge_dl_launch.log 2>&1 < /dev/null &
echo "download relaunched detached, pid $!"
sleep 3
pgrep -af "[r]un_get_knowledge" | cut -c1-60
ls -la data/knowledge/ | tail -4