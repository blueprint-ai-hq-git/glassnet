#!/bin/bash
# smoke_aux_gpu.sh — 30-second end-to-end check on the GPU venv before the sweep.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=6
export PYTHONUNBUFFERED=1
PY="/run/media/yahya/HARD DISK 4/GlassNet/.venv-gpu/bin/python3"
export T2_LAM_CONCEPT=1.0

"$PY" - <<'EOF'
import os, sys
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
import numpy as np
import cupy
from glassnet.tensor import set_backend, backend_name
from glassnet.optim import SGD
import run_t2_iter003 as M

set_backend("cupy")
print("backend:", backend_name(), "| cupy", cupy.__version__)
print("LAM_CONCEPT:", M.LAM_CONCEPT,
      "| targets:", None if M.CONCEPT_TARGETS is None else len(M.CONCEPT_TARGETS["ids"]))

net = M.TinyLM3(1501, 32, 64, 8, rng=np.random.default_rng(0))
opt = SGD(net.parameters(), lr=0.5)
rng = np.random.default_rng(1)
ids = rng.integers(0, 1500, size=(64, 8))
y = rng.integers(0, 1500, size=(64,))

loss, parts = net.loss(ids, y)
print("loss parts:", {k: round(float(v.data.get() if hasattr(v.data, 'get') else v.data), 4)
                      for k, v in parts.items()})
loss.backward()
gnorms = {p.name: float(np.linalg.norm(p.grad.get() if hasattr(p.grad, "get") else p.grad))
          for p in net.parameters() if p.grad is not None}
print("grad norms:", {k: round(v, 4) for k, v in gnorms.items()})
opt.step()
print("one optimiser step OK")
print("SMOKE PASSED")
EOF
echo "smoke exit=$?"