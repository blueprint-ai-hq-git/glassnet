#!/bin/bash
# FIRST PUSH - runs ONLY when Yahya says go. Public code lives in the org, never from a personal account.
set -e
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
source /home/yahya/.config/glassnet/keys.env
git init -q 2>/dev/null || true
git add -A
git -c user.name="blueprint-ai-bot" -c user.email="blueprintai.help1@gmail.com" commit -q -m "GlassNet: from-scratch engine, measured results and published failures" || true
git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://x-access-token:${GH_TOKEN}@github.com/blueprint-ai-hq-git/glassnet.git"
git push -u origin main
