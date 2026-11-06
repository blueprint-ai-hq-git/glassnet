#!/bin/bash
# finish_l2_ops.sh — record L2 completion.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
.venv-gpu/bin/python3 -u lens/scripts/record_l2_ops.py 2>&1 | tail -3
cp STATUS.md REBUILD-PLAN.md /home/yahya/ 2>/dev/null
echo "copied"
echo
echo "=== all gates so far ==="
for g in test_l0 test_l1 test_l2_smoke test_l2_ops; do
  if [ -f "lens/tests/$g.py" ]; then echo "  lens/tests/$g.py"; fi
done
echo
echo "=== GPU + jobs ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
ps -eo pid,cmd | grep -E "test_l2|g2_scaling" | grep -v grep || echo "  nothing running"