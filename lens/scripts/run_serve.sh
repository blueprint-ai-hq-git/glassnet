#!/bin/bash
# Start the GlassNet chat server DETACHED (outside the agent's process tree, see
# references/long-run-lifecycle.md in the glassnet skill). Nothing about this server needs the GPU, so
# it runs on CPU and cannot disturb a training run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
export OPENBLAS_NUM_THREADS=3
export CHAT_DATA_DIR="$PWD/data/chat/prepared_ext_honest"
SHIP="${1:-lens/runs/chat/sb_S0_d256_noattn_ep96.npz}"
PORT="${2:-8199}"
setsid nohup nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/serve.py "$SHIP" \
  --port "$PORT" --device cpu > /tmp/serve_$PORT.log 2>&1 < /dev/null &
echo "server launching detached on port $PORT (log /tmp/serve_$PORT.log)"
sleep 2
pgrep -af "[l]ens/serve.py" | cut -c1-90