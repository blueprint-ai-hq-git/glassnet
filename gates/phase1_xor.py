"""TEST GATE 2 — the engine learns XOR to 100%.

Run:  python3 phase1_xor.py
Plain-English: XOR is "true when the two inputs differ". A net that can't
even learn that tiny table is useless, so this is the smallest real proof
that the engine learns. We use a 2 -> 8 -> 1 sigmoid net, our own MSE loss,
our own SGD with a manual learning rate, and we print the weights at the end
so nothing is hidden.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from glassnet.tensor import Tensor, sigmoid  # noqa: E402
from glassnet.layers import Dense, Sequential  # noqa: E402
from glassnet.losses import mse  # noqa: E402
from glassnet.optim import SGD  # noqa: E402
from glassnet.gates import gate1  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "phase1")
os.makedirs(RUNS, exist_ok=True)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
Y = np.array([[0], [1], [1], [0]], dtype=np.float32)

rng = np.random.default_rng(7)
net = Sequential(
    Dense(2, 8, rng=rng, name="hidden"),
    sigmoid,
    Dense(8, 1, rng=rng, name="out"),
    sigmoid,
)
opt = SGD(net.parameters(), lr=0.5)
inputs = Tensor(X, name="xor_inputs")

print("=" * 62)
print("GATE 1 first (no training on broken gradients)")
print("=" * 62)
gate1()
print(f"\nXOR net: hidden 2->8 (sigmoid), out 8->1 (sigmoid), MSE, SGD lr=0.5, seed 7")
print(f"params: {sum(p.data.size for p in net.parameters())}")
print(f"\n{'iter':>6s}  {'loss':>10s}  {'accuracy':>8s}")
t0 = time.time()
for it in range(1, 20001):
    opt.zero_grad()
    loss = mse(net(inputs), Y)
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        preds = net(inputs).data
        acc = float(np.mean((preds > 0.5) == (Y > 0.5)))
        print(f"{it:6d}  {float(loss.data):10.6f}  {acc * 100:6.2f}%", flush=True)
        if acc == 1.0 and float(loss.data) < 1e-3:
            break

preds = net(inputs).data
acc = float(np.mean((preds > 0.5) == (Y > 0.5)))
loss_val = float(loss.data)
secs = time.time() - t0

print(f"\nfinal loss      : {loss_val:.6f}")
print(f"final accuracy  : {acc * 100:.2f}%  (target 100.00%)")
print(f"predictions     : {np.round(preds, 4).tolist()}")
print(f"elapsed         : {secs:.1f}s")
print("\nweights (nothing hidden):")
for key, p in net.named_parameters().items():
    flat = np.round(p.data.reshape(-1), 4)
    print(f"  {key:12s} {flat[:16].tolist()}{' ...' if flat.size > 16 else ''}")

gate2_ok = (acc == 1.0 and loss_val < 1e-3)
print("\nGATE 2:", "PASS ✔" if gate2_ok else "FAIL ✘")
np.savez(os.path.join(RUNS, "xor_final.npz"),
         **{k: v.data for k, v in net.named_parameters().items()},
         final_acc=np.float64(acc), final_loss=np.float64(loss_val))
sys.exit(0 if gate2_ok else 1)