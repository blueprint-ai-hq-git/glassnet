"""t3_coverage.py — READABILITY COVERAGE: how much of this model can a human actually read?

The question "100% readable" really asks is: when the model answers, how much of the reasoning
is done by concepts a human can name? This measures it directly and exactly, using the same
linear decomposition explain() uses (no approximation):

    logit[v] = bias[v] + sum over ALL concepts of (h[c] * W[c, v])

Split that sum into NAMED concepts (taught, with pre-registered targets) and UNNAMED concepts
(untaught columns). Report:

  * the share of total |contribution| mass that is named
  * how often the single LARGEST contributor is a named concept
  * the same, restricted to correct predictions (a wrong answer explained is not the goal)

No bar is set here -- this is a measurement, and it is the honest denominator for "100%".
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
import t3_audit_m3 as M3

R = os.path.join(PROJ, "runs", "t3")
sel = json.load(open(os.path.join(R, "c_selection.json")))
LR, LC = sel["lr"], sel["lc"]
CK = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_meaningful_c_seed0.npz"
N_TAUGHT = 6

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
Y = np.asarray(DATA["yva"])
VOCAB = DATA["vocab"]
set_backend("numpy")
names = [t["name"] for t in json.load(open(os.path.join(R, "meaningful_targets.json")))["targets"]]
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", CK))

out = net.forward(X)
h = np.asarray(out["concepts"].data)                 # (N, K)
logits = np.asarray(out["logits"].data)              # (N, V)
W = net.answer.W.data                                # (K, V)
pred = logits.argmax(1)
correct = pred == Y

# exact contribution of every concept to the CHOSEN word, for every window
contrib = h * W[:, pred].T                           # (N, K)
mass = np.abs(contrib)
named_mass = mass[:, :N_TAUGHT].sum(1)
unnamed_mass = mass[:, N_TAUGHT:].sum(1)
total = named_mass + unnamed_mass
share = named_mass / np.maximum(total, 1e-12)

top = np.abs(contrib).argmax(1)
top_is_named = top < N_TAUGHT

print("=" * 92)
print("READABILITY COVERAGE — how much of the model's reasoning is done by NAMED concepts?")
print("=" * 92)
print(f"model: {CK}  |  {N_TAUGHT} named columns, {16 - N_TAUGHT} UNNAMED columns")
print(f"validation windows: {len(X)}   accuracy on them: {correct.mean()*100:.2f}%\n")
print(f"  share of contribution mass from NAMED concepts : {share.mean()*100:6.2f}%  "
      f"(median {np.median(share)*100:6.2f}%)")
print(f"  largest contributor is a NAMED concept         : {top_is_named.mean()*100:6.2f}% of windows")
print(f"  ... on CORRECT predictions only                : "
      f"{top_is_named[correct].mean()*100:6.2f}% of {correct.sum()} windows")

print("\n  per-column share of the total contribution mass:")
col_mass = mass.mean(0)
col_share = col_mass / col_mass.sum() * 100
for c in range(16):
    nm = names[c] if c < len(names) else f"UNNAMED col{c}"
    mark = "" if c < N_TAUGHT else "   <- not readable"
    print(f"    {nm:<24} {col_share[c]:6.2f}%  (mean |contrib| {col_mass[c]:.4f}){mark}")

named_total = col_share[:N_TAUGHT].sum()
print(f"\n  NAMED columns carry {named_total:.2f}% of all contribution mass; "
      f"UNNAMED carry {100-named_total:.2f}%.")
print(f"  => a human can account for {named_total:.2f}% of this model's reasoning, "
      f"not 100%.")
print("\n  This is a measurement, not a bar. It is the honest denominator for the 100% goal.")

json.dump({"model": CK, "n_taught": N_TAUGHT, "n_columns": 16,
           "mean_named_share": float(share.mean()),
           "median_named_share": float(np.median(share)),
           "top_is_named_frac": float(top_is_named.mean()),
           "top_is_named_frac_correct": float(top_is_named[correct].mean()),
           "accuracy": float(correct.mean()),
           "per_column_share_pct": [float(x) for x in col_share],
           "column_names": [names[c] if c < len(names) else f"UNNAMED col{c}" for c in range(16)],
           "named_total_share": float(named_total)},
          open(os.path.join(R, "c_coverage.json"), "w"), indent=1)
print("\nsaved runs/t3/c_coverage.json")
