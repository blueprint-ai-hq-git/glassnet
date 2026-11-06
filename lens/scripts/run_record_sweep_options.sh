#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_sweep_options.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== best1 progress ==="
.venv-gpu/bin/python3 -u -c "
import json,os
p='lens/runs/chat/best1_log.json'
if os.path.exists(p):
    r=json.load(open(p)); print(f\"  params {r['params']:,}  concepts {r['concepts']['n_kept']}/{r['concepts']['n_defined']} defined\")
    for h in r['history']: print(f\"  ep {h['epoch']:>2}  ppl {h['val_ppl']:9.2f}  acc {h['val_acc']*100:6.2f}%  lr {h.get('lr',0):.3f}  ({h['seconds']:.0f}s)\")
else: print('  (not started yet)')
" 2>/dev/null
echo "=== guard ==="; tail -2 lens/runs/game_guard.log 2>/dev/null