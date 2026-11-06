#!/bin/bash
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_distill_seeds.py 2>&1 | tail -3
cp STATUS.md LOG.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo copied
echo
echo "=== big1 progress ==="
ls -la lens/runs/chat/big1_* 2>/dev/null | tail -4
.venv-gpu/bin/python3 -u -c "
import json,os
p='lens/runs/chat/big1_log.json'
if os.path.exists(p):
    r=json.load(open(p))
    print(f\"  params {r['params']:,}  concepts {r['concepts']['n_kept']}/{r['concepts']['n_defined']} defined\")
    for h in r['history']:
        print(f\"  ep {h['epoch']:>2}  loss {h['train_loss']:.4f}  concept {h['concept_loss']:.4f}  val ppl {h['val_ppl']:9.2f}  acc {h['val_acc']*100:6.2f}%  ({h['seconds']:.0f}s)\")
" 2>/dev/null || echo "  (no log yet)"