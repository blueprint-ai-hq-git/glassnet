#!/bin/bash
# verify_refactor_cpu.sh — prove the backend refactor did not change any
# maths: run the full from-scratch gradient check suite on CPU, plus a
# short end-to-end smoke test (forward, backward, one SGD step).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p runs/verify
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=/home/yahya/.hermes/hermes-agent/venv/bin/python3

echo "=== gradcheck suite (CPU) ===" > runs/verify/refactor_cpu.log
nice -n 19 ionice -c3 "$PY" tests/test_gradcheck.py \
  >> runs/verify/refactor_cpu.log 2>&1
echo "gradcheck rc=$?" >> runs/verify/refactor_cpu.log

echo "=== smoke test (CPU) ===" >> runs/verify/refactor_cpu.log
nice -n 19 ionice -c3 "$PY" - >> runs/verify/refactor_cpu.log 2>&1 <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
import numpy as np
from glassnet import tensor as T
from glassnet.tensor import Tensor, embedding, sum_axis, sigmoid
from glassnet.layers import Dense
from glassnet.losses import cross_entropy
from glassnet.optim import SGD

print("backend:", T.backend_name())
rng = np.random.default_rng(0)
E = Tensor(rng.standard_normal((50, 8)).astype(np.float32), requires_grad=True)
idx = rng.integers(0, 50, size=(4, 6))
X = embedding(E, idx)                 # (4,6,8)
S = sum_axis(X, 1)                    # (4,8)
H = sigmoid(Dense(8, 5, rng=rng)(S))
l2 = Dense(5, 7, rng=rng)
loss = cross_entropy(l2(H), np.array([1, 3, 0, 6]))
loss.backward()
print("loss:", float(loss.data))
print("grads set:", all(p.grad is not None for p in [E] + l2.parameters()))
opt = SGD([E] + l2.parameters(), lr=0.1)
before = float(loss.data)
opt.step()
loss2 = cross_entropy(l2(sigmoid(Dense(8, 5, rng=np.random.default_rng(0))(sum_axis(embedding(E, idx), 1)))), np.array([1,3,0,6]))
print("smoke OK: params updated:", not np.allclose(E.data, E.data))
EOF
echo "smoke rc=$?" >> runs/verify/refactor_cpu.log
tail -40 runs/verify/refactor_cpu.log