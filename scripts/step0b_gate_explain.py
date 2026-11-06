"""STEP 0b follow-up (LOG.md Q2): how well do the NAMED CONCEPTS alone
predict the learned gate's outputs?

The gate ablation showed the gate path is necessary (removing it -> 10% =
prior). But the readability question is the reverse: is the gate's computed
feature state EXPLAINABLE by the named concepts? Measure: on VAL, take the
rung-1 model (iter 010, gate=128 -> 14 named concepts -> 10 classes),
record G = relu(gate(x)) (128-d) and h = named concept activations (14-d),
then regress each gate output dimension linearly on the named concepts and
report R^2. High R^2 -> the named concepts alone span/explain the gate's
feature space; low -> the gate carries information the named concepts do
not explain (a readability gap). 3 seeds.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glassnet.loop_lib import load_split, RUNS
from glassnet.phase2_model import GlassNetPhase2
from glassnet.tensor import Tensor

Xtr, ytr, Xval, yval, Xte, yte = load_split("fashion-mnist")
X = np.asarray(Xval)
N = len(X)

for s in (0, 1, 2):
    m = GlassNetPhase2.load(os.path.join(RUNS, f"iter010/final_seed{s}.npz"))
    f = m.forward(Tensor(X))
    G = f["gate_h"].data if "gate_h" in f else None
    if G is None:
        # forward returns concepts etc.; recompute gate outputs directly
        from glassnet.tensor import relu
        G = relu(m.concept_gate(Tensor(X))).data
    h = f["concepts"].data                       # named concept activations
    # linear regression: G ~ [1, h]  -> R2 per gate dim
    A = np.hstack([np.ones((N, 1)), h])
    coef, res, rank, sv = np.linalg.lstsq(A, G, rcond=None)
    pred = A @ coef
    ss_res = ((G - pred) ** 2).sum(axis=0)
    ss_tot = ((G - G.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    print(f"seed{s}: gate outputs={G.shape[1]} named concepts={h.shape[1]} "
          f"R2 mean={r2.mean():.3f} min={r2.min():.3f} max={r2.max():.3f} "
          f"share R2>=0.5: {(r2 >= 0.5).mean():.2%}")