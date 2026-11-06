"""c_detail.py — Target C: per-column audit detail, plus R2 name precision and the R4 cards.

Reuses the exact same code paths as the A3 versions so the metrics are comparable:
  * Meter-03 statistics come from t3_audit_m3 (not re-implemented here)
  * name precision uses the pre-registered ablation rule (METER-04 section 1, drop >= 0.10)
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
TAG = f"readable_K16_lr{LR}_lc{LC.replace('.','')}_meaningful_c_seed{{s}}.npz"
K, N_TAUGHT, DROP = 16, 6, 0.10

DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]
set_backend("numpy")
names = [t["name"] for t in json.load(open(os.path.join(R, "meaningful_targets.json")))["targets"]]

# ---------------------------------------------------------------- per-column audit (seed 0)
print("=" * 112)
print(f"TARGET C — PER-COLUMN AUDIT (seed 0)  model lr={LR} lc={LC}")
print("=" * 112)
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", TAG.format(s=0)))
tgt = M3.load_targets("meaningful")
h, L, A = M3.prep(net, X)
h = np.asarray(h) if not isinstance(h, np.ndarray) else h
print(f"{'col':>3} {'concept':<22} {'AUROC':>7} {'contrib':>9} {'firing':>8}  verdict")
for c in range(N_TAUGHT):
    r = M3.audit_col(net, X, h, c, tgt[c][1], L, A)
    ok = (r["auroc"] >= 0.55) and (r["contrib"] >= 0.25) and (r["firing"] >= 0.02)
    print(f"{c:>3} {names[c]:<22} {r['auroc']:>7.4f} {r['contrib']:>9.4f} "
          f"{r['firing']:>8.4f}  {'PASS' if ok else 'FAIL:' + r['flag']}")

# ------------------------------------------------------------------- R2 name precision
print("\n" + "=" * 112)
print("R2 — NAME PRECISION BY ABLATION (pre-registered drop bar 0.10)")
print("=" * 112)
tgt_ids = [t["ids"] for t in json.load(open(os.path.join(R, "meaningful_targets.json")))["targets"]]
emb = net.emb.data
backup = emb.copy()
rows = []
for c in range(N_TAUGHT):
    ids = tgt_ids[c]
    words = [VOCAB[i] if i < len(VOCAB) else "?" for i in ids]
    act = h[:, c]
    fired = act > 0.5
    if fired.sum() == 0:
        print(f"[{c}] {names[c]:<22} NO FIRING WINDOWS")
        rows.append({"col": c, "name": names[c], "precision": 0.0, "unused": words})
        continue
    A0 = float(act[fired].mean())
    drops = {}
    for w, wid in zip(words, ids):
        row = emb[wid].copy()
        emb[wid] = 0.0
        hw = np.asarray(net.forward(X)["concepts"].data)[:, c]
        emb[wid] = row
        drops[w] = float((A0 - float(hw[fired].mean())) / A0) if A0 > 0 else 0.0
    used = [w for w in words if drops[w] >= DROP]
    unused = [w for w in words if drops[w] < DROP]
    prec = len(used) / len(words)
    print(f"[{c}] {names[c]:<22} precision {len(used)}/{len(words)} = {prec:.2f}  "
          f"-> R2 {'PASS' if prec == 1.0 else 'FAIL'}"
          + (f"  (unused: {', '.join(unused)})" if unused else ""))
    rows.append({"col": c, "name": names[c], "words": words, "precision": prec,
                 "used": used, "unused": unused, "drops": drops})
emb[:] = backup

n_r2 = sum(1 for r in rows if r["precision"] == 1.0)
print(f"\n  R2 pass count: {n_r2}/{N_TAUGHT}")
print(f"  words contributing literally zero: "
      f"{[f'[{r[chr(99)+chr(111)+chr(108)]}] {w}' for r in rows for w in r.get('unused', []) if r.get('drops', {}).get(w, 1) == 0.0]}")

# ------------------------------------------------------------------------- R4 blind cards
print("\n" + "=" * 112)
print("R4 — BLIND READER CARDS")
print("=" * 112)
rng = np.random.default_rng(0)
cards, key = [], {}
for c in range(N_TAUGHT):
    act = h[:, c]
    fired = np.where(act > 0.5)[0]
    notf = np.where(act <= 0.5)[0]
    f_pick = fired[np.argsort(-act[fired])][:5]
    n_pick = notf[np.argsort(act[notf])][:5]
    idx = np.concatenate([f_pick, n_pick])
    labs = np.array([1] * len(f_pick) + [0] * len(n_pick))
    o = rng.permutation(len(idx))
    idx, labs = idx[o], labs[o]
    cards.append({"col": c, "name": names[c],
                  "windows": [{"n": n, "text": " ".join(
                      VOCAB[i] if i < len(VOCAB) else "<PAD>" for i in X[i2])}
                              for n, i2 in enumerate(idx, 1)]})
    key[str(c)] = {"name": names[c],
                   "answers": [{"n": n, "fires": bool(l)} for n, l in enumerate(labs, 1)]}
print(f"  {len(cards)} cards built, 5 firing + 5 non-firing each, fixed shuffle seed 0")
json.dump(key, open(os.path.join(R, "reader_test_key_c.json"), "w"), indent=1)
json.dump({"model": TAG.format(s=0), "rows": rows, "r2_pass": n_r2, "n_taught": N_TAUGHT,
           "cards": cards}, open(os.path.join(R, "name_precision_c.json"), "w"), indent=1)
print("  saved runs/t3/name_precision_c.json and runs/t3/reader_test_key_c.json")
