#!/bin/bash
# How does the clean-corpus model actually talk? Verbatim answers, nothing selected.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
SHIP=$(ls -t lens/runs/chat/cl_d256_clean_24_ep24.npz 2>/dev/null | head -1)
bash lens/scripts/night_talk_test.sh "$SHIP" >> lens/runs/TALK-CLEAN.txt 2>&1
