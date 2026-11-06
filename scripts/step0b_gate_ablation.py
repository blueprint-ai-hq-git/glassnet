"""STEP 0b — gate-vs-concepts attribution on the rung-1 Fashion model.

The learned gate is an unnamed 128-unit ReLU layer feeding the named concept
readouts. Question: what fraction of the final answer comes through the gate
vs through the named concepts? Method (ablation, per instruction): take the
saved iter-010 models (3 seeds), evaluate four forward variants on VAL:
  full           : normal forward
  no-gate        : relu(gate(x)) forced to 0  -> concepts see gate bias only
  no-concepts    : concept activations forced to 0.5 -> answer sees constant
  no-both        : gate=0 AND concepts=0.5
Report accuracies, the drop each ablation causes, and the honest
interpretation in LOG.md terms.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from glassnet.loop_lib import load_split, RUNS
from glassnet.phase2_model import GlassNetPhase2
from glassnet.tensor import Tensor, relu, sigmoid

Xtr, ytr, Xval, yval, Xte, yte = load_split("fashion-mnist")
yv = np.asarray(yval)

def acc(preds):
    return float(np.mean(preds == yv)) * 100

rows = []
for s in (0, 1, 2):
    m = GlassNetPhase2.load(os.path.join(RUNS, f"iter010/final_seed{s}.npz"))
    # full
    f = m.forward(Tensor(Xval))
    full = acc(f["logits"].data.argmax(1))
    # no-gate: recompute forward with gate output zeroed
    hg = relu(m.concept_gate(Tensor(Xval)))
    zc = m.concept(hg * 0.0)          # gate path dead -> bias-only concepts
    h = sigmoid(zc)
    logits_nogate = m.answer(h)
    no_gate = acc(logits_nogate.data.argmax(1))
    # no-concepts: answer sees constant 0.5 concepts
    logits_noc = m.answer(Tensor(np.full((len(Xval), m.K), 0.5, np.float32)))
    no_concepts = acc(logits_noc.data.argmax(1))
    # no-both
    logits_nob = m.answer(Tensor(np.full((len(Xval), m.K), 0.5, np.float32)))
    no_both = acc(logits_nob.data.argmax(1))
    rows.append((full, no_gate, no_concepts, no_both))
    print(f"  seed{s}: full={full:.2f} | no-gate={no_gate:.2f} "
          f"| no-concepts={no_concepts:.2f} | no-both={no_both:.2f}")

full = np.mean([r[0] for r in rows]); ng = np.mean([r[1] for r in rows])
nc = np.mean([r[2] for r in rows]); nb = np.mean([r[3] for r in rows])
print(f"\n[step0b] VAL means: full={full:.2f} | gate-ablated={ng:.2f} "
      f"(-{full-ng:.2f}) | concepts-ablated={nc:.2f} (-{full-nc:.2f}) "
      f"| both={nb:.2f}")
print("[step0b] NOTE: both ablations degenerate to the answer layer's bias "
      "prior — the named concepts are the ONLY information path to the "
      "classes; the gate is the feature SOURCE feeding them. Readability "
      "claim: classes depend on named concepts 100%; the gate contributes "
      "to the answer only through the named concepts (0% independent path).")
out = os.path.join(RUNS, "step0b_gate_ablation.txt")
with open(out, "w") as fh:
    fh.write(f"full {full:.2f} | gate-ablated {ng:.2f} (-{full-ng:.2f}) | "
             f"concepts-ablated {nc:.2f} (-{full-nc:.2f}) | both {nb:.2f}\n")
print("saved", out)