"""t3_readability_sheet.py — STEP 0: the readability sheet (NO training, NO retuning,
NO renaming). Reads the existing A3 readable model and reports, for each of the 10 taught
concepts: its target words, 5 validation windows where it fires, and whether the target is
actually present in those windows.

The "what a human would call it" line is written by hand from this evidence, honestly
marking meaningful ideas vs mere word lists. This script only produces the evidence.
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

# KNOWN WART: t3_audit_m3.py runs its entire audit at module level, so importing it re-runs
# that audit (~30 s) and rewrites runs/t3/meter3_words.json. It is deterministic, so the
# content is identical and nothing is corrupted -- but the real fix is to wrap that tail in
# main() with an __main__ guard. Logged, not done here, because touching the audit's
# execution path mid-milestone is riskier than the wart.
import t3_audit_m3 as M3

RUNDIR = os.path.join(PROJ, "runs", "t3")
N_TAUGHT = 10
SEED = 0                      # the readability sheet reads one trained model; stated, not hidden

sel = json.load(open(os.path.join(RUNDIR, "a3_selection.json")))
LR, LC = sel["lr"], sel["lc"]
LCS = LC.replace(".", "")
CK = f"readable_K16_lr{LR}_lc{LCS}_clusters16_seed{SEED}.npz"

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]


def decode(row):
    # VOCAB is the 1500-word list; ids can reach V-1 == 1500 (PAD), which is past the end.
    return " ".join(VOCAB[i] if i < len(VOCAB) else "<PAD>" for i in row)


set_backend("numpy")
targets = M3.load_targets("clusters16")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters16.json")))

net = M3.load_ckpt(M3.build_net(SEED), os.path.join(RUNDIR, "ckpt", CK))
h, L, A = M3.prep(net, X)
print(f"model: {CK}  (lr={LR}, lam_concept={LC})")
print(f"validation windows: {len(X)}")
print("=" * 108)

out = {"model": CK, "seed": SEED, "lr": LR, "lc": LC, "concepts": []}

for c in range(N_TAUGHT):
    words = CLU["clusters"][c]["words"]
    ids = set(int(i) for i in CLU["clusters"][c]["ids"])
    support = CLU["clusters"][c]["support"]
    act = np.asarray(h[:, c]).ravel()
    order = np.argsort(-act)[:5]
    fr_active = float((act > 0).mean())

    print(f"\n[{c}] TARGET: {words}")
    print(f"    TRAIN support {support:.4f} | activation: min {act.min():.3f} "
          f"median {np.median(act):.3f} max {act.max():.3f} | "
          f"active (>0) on {fr_active*100:.1f}% of windows")
    rec = {"col": c, "target_words": words, "support": support,
           "act_min": float(act.min()), "act_median": float(np.median(act)),
           "act_max": float(act.max()), "frac_active": fr_active, "examples": []}
    for rank, i in enumerate(order, 1):
        row = X[i]
        present = [VOCAB[j] if j < len(VOCAB) else "<PAD>" for j in row
                   if int(j) in ids]
        tag = "TARGET-PRESENT" if present else "no target word"
        print(f"    {rank}. act {act[i]:+.3f}  \"{decode(row)}\"   [{tag}]")
        rec["examples"].append({"rank": rank, "act": float(act[i]),
                                "window": decode(row), "target_present": bool(present),
                                "present_words": present})
    n_present = sum(1 for e in rec["examples"] if e["target_present"])
    print(f"    -> target present in {n_present}/5 of its top firings")
    rec["target_present_in_top5"] = n_present
    out["concepts"].append(rec)

json.dump(out, open(os.path.join(RUNDIR, "readability_evidence.json"), "w"), indent=1)
print("\n" + "=" * 108)
print("saved runs/t3/readability_evidence.json")
