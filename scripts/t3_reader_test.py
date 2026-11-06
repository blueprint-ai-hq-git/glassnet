"""t3_reader_test.py — METER-04 R4: build the blind reader-test cards.

For each taught concept, one card: the concept's NAME plus 10 validation windows in a fixed
shuffled order -- 5 the column fires on (activation > 0.5) and 5 it does not. The card shows
no activations and no labels.

The answer key goes to a SEPARATE file that the reader must not open.

Also reports the R2 threshold sensitivity (how many concepts would pass at other drop bars)
so the pre-registered 0.10 bar can be judged honestly -- WITHOUT moving it.
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

RUNDIR = os.path.join(PROJ, "runs", "t3")
N_TAUGHT, K = 10, 16
N_FIRE, N_NOT = 5, 5
SHUFFLE_SEED = 0          # fixed, pre-registered, so the same card can be re-scored

sel = json.load(open(os.path.join(RUNDIR, "a3_selection.json")))
LR, LC = sel["lr"], sel["lc"]
CK = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_clusters16_seed0.npz"

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]
set_backend("numpy")
CLU = json.load(open(os.path.join(RUNDIR, "concept_clusters16.json")))
net = M3.load_ckpt(M3.build_net(0), os.path.join(RUNDIR, "ckpt", CK))
h = np.asarray(net.forward(X)["concepts"].data)


def decode(row):
    return " ".join(VOCAB[i] if i < len(VOCAB) else "<PAD>" for i in row)


rng = np.random.default_rng(SHUFFLE_SEED)
cards, key = [], {}
for c in range(N_TAUGHT):
    words = CLU["clusters"][c]["words"]
    act = h[:, c]
    fired = np.where(act > 0.5)[0]
    notf = np.where(act <= 0.5)[0]
    # deterministic picks: strongest firings, and the LOWEST-activation non-firings
    f_pick = fired[np.argsort(-act[fired])][:N_FIRE]
    n_pick = notf[np.argsort(act[notf])][:N_NOT]
    idx = np.concatenate([f_pick, n_pick])
    labels = np.array([1] * len(f_pick) + [0] * len(n_pick))
    order = rng.permutation(len(idx))
    idx, labels = idx[order], labels[order]

    card = {"col": c, "name": "_".join(words), "windows": []}
    for n, (i, lab) in enumerate(zip(idx, labels), 1):
        card["windows"].append({"n": n, "text": decode(X[i])})
    cards.append(card)
    key[str(c)] = {"name": card["name"],
                   "answers": [{"n": n, "fires": bool(lab)}
                               for n, lab in enumerate(labels, 1)]}

# ------------------------------------------------------------------ the cards
L = []
L.append("# READER TEST CARDS — R4 (is the name enough to understand the neuron?)")
L.append("")
L.append("**This is the one metric in GlassNet judged by a human, not computed.**")
L.append("**Official reader: the owner or the external reviewer — NOT the agent**, which wrote")
L.append("the names and would be circular.")
L.append("")
L.append("## How to fill this in")
L.append("")
L.append("Each card below gives a concept's **name** and **10 text windows in random order**.")
L.append("For each window, mark **Y** if you believe that neuron fires on it, **N** if not.")
L.append("**Exactly 5 of the 10 do fire** — that is the only hint you get, and it is fair.")
L.append("")
L.append("Scoring: **8 of 10 correct (0.80) or better = the name is genuinely informative.**")
L.append("Chance is 5/10. The answer key is in a separate file (`runs/t3/reader_test_key.json`)")
L.append("— do not open it until you have filled in the cards.")
L.append("")
L.append("---")
L.append("")
for card in cards:
    L.append(f"## Concept {card['col']}: `{card['name']}`")
    L.append("")
    L.append("| # | window | fires? (Y/N) |")
    L.append("|---|---|---|")
    for w in card["windows"]:
        L.append(f"| {w['n']} | `{w['text']}` |  |")
    L.append("")
    L.append(f"Your score for this concept: ___ / 10")
    L.append("")
    L.append("---")
L.append("")
L.append("## Summary")
L.append("")
L.append("| concept | name | your score | ≥8/10? |")
L.append("|---|---|---|---|")
for card in cards:
    L.append(f"| {card['col']} | `{card['name']}` | ___/10 |  |")
L.append("")
L.append("**Readability score under R1–R4 = the number of concepts passing all four.**")
L.append("Report it as a count.")

open(os.path.join(PROJ, "READER-TEST-CARDS.md"), "w").write("\n".join(L))
json.dump(key, open(os.path.join(RUNDIR, "reader_test_key.json"), "w"), indent=1)
json.dump({"shuffle_seed": SHUFFLE_SEED, "n_fire": N_FIRE, "n_not": N_NOT,
           "cards": cards}, open(os.path.join(RUNDIR, "reader_cards.json"), "w"), indent=1)

print(f"wrote READER-TEST-CARDS.md ({len(cards)} cards, 10 windows each)")
print(f"wrote runs/t3/reader_test_key.json (the answer key -- reader must not open it)")

# ------------------------------------------------- R2 threshold sensitivity (NOT a rule change)
np_ = json.load(open(os.path.join(RUNDIR, "name_precision.json")))
print("\nR2 threshold SENSITIVITY — reported for honesty, the bar is NOT moved:")
print(f"  pre-registered bar 0.10 -> {np_['new_pass']}/10 concepts have fully-used names")
for bar in (0.01, 0.05, 0.10, 0.25, 0.50):
    ok = sum(1 for r in np_["rows"]
             if r.get("drops") and all(v >= bar for v in r["drops"].values()))
    print(f"  if the bar were {bar:.2f}      -> {ok}/10")
zero = [f"[{r['col']}] {w}" for r in np_["rows"] for w in r.get("unused", [])
        if r.get("drops", {}).get(w, 1) == 0.0]
print(f"  words contributing LITERALLY ZERO (drop 0.0000): {zero}")
border = [f"[{r['col']}] {w} (drop {r['drops'][w]:.4f})"
          for r in np_["rows"] for w in r.get("unused", [])
          if 0 < r.get("drops", {}).get(w, 0) < 0.10]
print(f"  borderline words just under the bar           : {border}")
