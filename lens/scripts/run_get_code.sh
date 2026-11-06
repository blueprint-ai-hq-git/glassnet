#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export PYTHONUNBUFFERED=1
nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/scripts/get_code_data.py 2>&1 | tail -30
echo
echo "=== sweep progress ==="
.venv-gpu/bin/python3 -u -c "
import json,os,glob
for f in sorted(glob.glob('lens/runs/chat/sweep*_log.json')):
    r=json.load(open(f)); h=r['history'][-1]
    print(f\"  {os.path.basename(f):<28} ep {h['epoch']:>2}  ppl {h['val_ppl']:8.2f}  acc {h['val_acc']*100:6.2f}%  params {r['params']:,}\")
" 2>/dev/null || echo "  (no sweep logs yet)"
echo
echo "=== game guard ==="
tail -3 lens/runs/game_guard.log 2>/dev/null || echo "  (no guard log yet)"