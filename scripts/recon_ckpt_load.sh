#!/bin/bash
# recon_ckpt_load.sh — how are the rung checkpoints loaded, and what data tasks exist?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== np.load / load_npz in loop/phase2 code ==="
grep -rn "np.load\|load_npz\|final_seed" scripts/run_iter019_discover.py glassnet/loop_lib.py glassnet/loop_audit.py glassnet/autopilot.py 2>/dev/null | head -14
echo
echo "=== load_split tasks ==="
sed -n '116,152p' glassnet/loop_lib.py
echo
echo "=== checkpoint keys: phase2 final_seed0 ==="
/home/yahya/.hermes/hermes-agent/venv/bin/python3 - <<'EOF'
import numpy as np
for p in ["runs/phase2/final_seed0.npz", "runs/loop/iter018/final_seed0.npz",
          "runs/loop/iter014/final_seed0.npz"]:
    try:
        z = np.load(p)
        print(p, "->", sorted(z.keys()))
    except Exception as e:
        print(p, "ERR", e)
EOF