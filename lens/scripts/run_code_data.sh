#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=4
CODE_MAX_PAIRS="${CODE_MAX_PAIRS:-60000}" nice -n 19 ionice -c3 \
  .venv-gpu/bin/python3 -u lens/scripts/code_data.py 2>&1 | tail -14
echo
echo "=== best1 so far ==="
.venv-gpu/bin/python3 -u -c "
import json,os
p='lens/runs/chat/best1_log.json'
if os.path.exists(p):
    r=json.load(open(p))
    for h in r['history']:
        print(f\"  ep {h['epoch']:>2}  ppl {h['val_ppl']:9.2f}  acc {h['val_acc']*100:6.2f}%  lr {h.get('lr',-1):.4f}  ({h['seconds']:.0f}s)\")
" 2>/dev/null || echo "  (none)"