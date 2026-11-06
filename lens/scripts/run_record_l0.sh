#!/bin/bash
# run_record_l0.sh — record L0, copy key files to ~, and check the old Target I run.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_l0.py 2>&1 | tail -3
cp STATUS.md REBUILD-PLAN.md TARGET-LEDGER.md /home/yahya/ 2>/dev/null
echo "=== copied to ~ ==="
echo
echo "=== Target I (the old run — did it finish?) ==="
.venv-gpu/bin/python3 -c "
import json, os
for f in ('target_i_lam0.json', 'target_i_lam02.json'):
    p = os.path.join('runs/t3', f)
    if not os.path.exists(p):
        print(f'  {f}: not written'); continue
    d = json.load(open(p))
    rd = [x for x in d if x['kind'] == 'readable']
    for x in rd:
        names = [r['name'] for r in x['rows']]
        fails = [names[i] for i, r in enumerate(x['rows']) if not r['pass']]
        print(f\"  {f} seed {x['seed']}: {x['n_pass']}/9  ppl {x['ppl']:8.3f}  failing {fails}\")
"
echo
echo "=== anything running? ==="
ps -eo pid,etime,cmd | grep -E "g2_scaling|g_loop" | grep -v grep || echo "  nothing running"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader