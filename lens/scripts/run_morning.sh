#!/bin/bash
# MORNING RESUME — one command: health check first, then the queued talk experiments back-to-back.
# Bars (pre-registered in the run scripts): attn > 56.07% clean-val + 40/40 + no word salad;
# splitfix > 56.07% + 40/40. All outputs are saved under /tmp and lens/runs/chat/.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "===[1/3] SELFCHECK $(date +%H:%M)"
.venv-gpu/bin/python3 lens/scripts/selfcheck.py
echo "===[2/3] ATTENTION EXPERIMENT $(date +%H:%M)"
bash lens/scripts/run_t12_attn.sh
echo "===[3/3] SPLITFIX EXPERIMENT $(date +%H:%M)"
bash lens/scripts/run_t12_splitfix.sh
echo "=== ALL DONE $(date +%H:%M)"