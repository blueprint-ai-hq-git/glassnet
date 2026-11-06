#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_code_run.py 2>&1 | tail -25
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo
echo "=== follow-ups ==="
.venv-gpu/bin/python3 -u -c "
import json,os
for out in ['f1_long','f2_vocab5000']:
    p=f'lens/runs/chat/{out}_log.json'
    if os.path.exists(p):
        r=json.load(open(p)); hs=r['history']
        print(f'  {out}: {len(hs)} epochs, last ppl {hs[-1][\"val_ppl\"]:.2f} acc {hs[-1][\"val_acc\"]*100:.2f}% params {r[\"params\"]:,}')
    else: print(f'  {out}: not started')
" 2>/dev/null
echo "=== guard ==="; tail -2 lens/runs/game_guard.log 2>/dev/null