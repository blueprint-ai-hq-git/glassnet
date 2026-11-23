#!/bin/bash
# Linux: serve the app and open it. Nothing is installed and nothing leaves the machine.
cd "$(dirname "$0")" || exit 1
PORT=${PORT:-8781}
python3 -m http.server "$PORT" --bind 127.0.0.1 &
sleep 1
xdg-open "http://127.0.0.1:$PORT/index.html" 2>/dev/null || echo "open http://127.0.0.1:$PORT/index.html in a browser"
wait
