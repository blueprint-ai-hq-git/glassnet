#!/bin/bash
# judge_cli smoke: score a real corpus window, the same window shuffled, a blank window, and a
# short typed sentence -- the judge must call the corruptions BAD and the real one GOOD.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1 CD_VOCAB=12000
PY=.venv-gpu/bin/python3
echo "=== real window (first val window ids) ==="
$PY -u - <<'EOF'
import os, sys, numpy as np
P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P); sys.path.insert(0, P + "/lens/scripts")
import chat_data
d = chat_data.build()
Xva, _ = chat_data.make_windows(d["ids"], d["bounds"], d["split"], 32, which=1)
w = np.asarray(Xva[7], dtype=np.int64)
print("REAL:"); print(" ".join(str(int(t)) for t in w))
np.save("/tmp/dj_smoke_ids.npy", w)
EOF
IDS=$(cat /tmp/dj_smoke_ids.txt 2>/dev/null || true)
$PY -u lens/scripts/judge_cli.py --window-ids "$(python3 - <<'EOF'
import numpy as np
w = np.load("/tmp/dj_smoke_ids.npy")
print(",".join(str(int(t)) for t in w))
EOF
)"
echo
echo "=== the SAME window shuffled (corruption) ==="
$PY -u lens/scripts/judge_cli.py --window-ids "$(python3 - <<'EOF'
import numpy as np
w = np.load("/tmp/dj_smoke_ids.npy")
rng = np.random.default_rng(0)
print(",".join(str(int(t)) for t in rng.permutation(w)))
EOF
)"
echo
echo "=== a BLANK window (corruption) ==="
$PY -u lens/scripts/judge_cli.py --window-ids "$(python3 -c "print(','.join(['0']*32))")"
echo
echo "=== a typed sentence ==="
$PY -u lens/scripts/judge_cli.py "hello how are you today i am fine thank you"