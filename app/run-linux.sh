#!/bin/bash
# GlassNet Runner on Linux: opens the app from this folder. No server, no internet, no install.
cd "$(dirname "$0")" || exit 1
xdg-open "index.html" 2>/dev/null || echo "open app/index.html in a browser"
