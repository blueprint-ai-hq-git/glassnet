#!/bin/bash
# run_f_audit_fix.sh — prove the loader fix changes the numbers, then re-run the F audit.
# The old audit loaded f_train checkpoints with M3.load_ckpt, which matches by parameter NAME,
# so only the embedding loaded and the concept/answer layers stayed random.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
export T3M_IMPORT_ONLY=1
export T3M_K=9
PY=".venv-gpu/bin/python3"

echo "=== 1. does the load fix actually change the model? ==="
"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
os.environ["T3M_IMPORT_ONLY"] = "1"; os.environ["T3M_K"] = "9"
import numpy as np
import t3_audit_m3 as M3
from glassnet import tiny_data
from glassnet.tensor import set_backend
set_backend("numpy")
D = tiny_data.build(win=8, vocab_cap=1500); X = np.asarray(D["Xva"])
f = "runs/t3/ckpt/student_readable_all_f_seed0.npz"
a = M3.load_ckpt(M3.build_net(0), f)
ha = np.asarray(a.forward(X)["concepts"].data)
b = M3.build_net(0); z = np.load(f)
b.emb.data = z["emb"]; b.concept.W.data = z["Wc"]; b.concept.b.data = z["bc"]
b.answer.W.data = z["Wa"]; b.answer.b.data = z["ba"]
hb = np.asarray(b.forward(X)["concepts"].data)
print(f"  OLD load  concept col0: mean {ha[:,0].mean():.4f} sd {ha[:,0].std():.4f}")
print(f"  NEW load  concept col0: mean {hb[:,0].mean():.4f} sd {hb[:,0].std():.4f}")
print(f"  identical? {bool(np.allclose(ha, hb))}")
print("  -> if NOT identical, the old audit was measuring a model with random concept weights")
EOF

echo
echo "=== 2. re-run the audit with the fixed loader ==="
nice -n 19 ionice -c3 "$PY" -u scripts/f_audit.py 2>&1 \
  | sed -n '/TARGET F — DO THE CONCEPTS/,$p' | tail -34