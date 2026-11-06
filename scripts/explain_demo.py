"""explain_demo.py — show the readable model explaining itself, on real validation windows.

Includes a real proof rather than a demonstration: the explanation is exact by construction
(logits = h @ W + b), so the contributions + bias must rebuild the model's own score. This
script checks that on EVERY validation window and reports the worst residual.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet.tensor import set_backend
from glassnet import tiny_data
from glassnet.explain import explain, format_explanation, concept_word_table
import t3_audit_m3 as M3

R = os.path.join(PROJ, "runs", "t3")
sel = json.load(open(os.path.join(R, "c_selection.json")))
LR, LC = sel["lr"], sel["lc"]
CK = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_meaningful_c_seed0.npz"

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
Y = np.asarray(DATA["yva"])
VOCAB = DATA["vocab"]
set_backend("numpy")
names = [t["name"] for t in json.load(open(os.path.join(R, "meaningful_targets.json")))["targets"]]
names = names + [f"untaught_{i}" for i in range(16 - len(names))]
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", CK))

print("=" * 78)
print("GLASSNET explain() — the readable model explaining itself")
print("=" * 78)

# ---------------------------------------------------------------- the exactness proof
worst, worst_at = 0.0, -1
for i in range(len(X)):
    e = explain(net, X[i], VOCAB, names)
    if e["residual"] > worst:
        worst, worst_at = e["residual"], i
print(f"\nEXACTNESS PROOF over all {len(X)} validation windows:")
print(f"  worst |contributions + bias - model score| = {worst:.3e}  (window {worst_at})")
print(f"  -> {'EXACT everywhere' if worst < 1e-4 else 'NOT EXACT — the explanation would be wrong'}")

# ---------------------------------------------------------------- interesting examples
print("\n" + "=" * 78)
print("THREE REAL EXAMPLES (chosen to show different concepts doing the work)")
print("=" * 78)
picks = []
for target_concept, want_high in (("is_a_negation", True), ("is_archaic_address", True),
                                  ("pronoun_subject", True)):
    ci = names.index(target_concept)
    h = np.asarray(net.forward(X)["concepts"].data)
    idx = np.where(h[:, ci] > 0.8)[0]
    if len(idx):
        picks.append(int(idx[0]))
for i in picks[:3]:
    e = explain(net, X[i], VOCAB, names)
    print(format_explanation(e))
    print(f"  (the true next word was '{VOCAB[Y[i]] if Y[i] < len(VOCAB) else '<PAD>'}'"
          f"{' — correct' if Y[i] == e['predicted_id'] else ' — this one is wrong, and it is shown anyway'})")

# ---------------------------------------------------------------- what each concept means
print("\n" + "=" * 78)
print("WHAT EACH CONCEPT PUSHES UP (read straight off the answer weights)")
print("=" * 78)
for row in concept_word_table(net, VOCAB, names, top_n=10)[:6]:
    print(f"\n{row['name']:<24} top words: {', '.join(row['top_words'])}")
    print(f"{'':<24} weight range {row['min_weight']:+.3f} .. {row['max_weight']:+.3f}")
