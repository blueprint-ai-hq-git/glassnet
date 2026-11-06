#!/bin/bash
# run_record_decode.sh
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_decode.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md REBUILD-PLAN.md /home/yahya/ 2>/dev/null
cp lens/runs/decode/model_*.txt /home/yahya/ 2>/dev/null
echo "copied (including the model-as-words file)"
ls -la /home/yahya/model_*.txt 2>/dev/null | tail -2