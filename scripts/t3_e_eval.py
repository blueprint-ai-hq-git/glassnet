"""t3_d_eval.py — TARGET D full evaluation.

  1. margin: readable (K=6, all taught) vs the matched normal (K=6, lam_concept=0)
  2. coverage: must be 100.00% by construction -- measured, not assumed
  3. R1: the Meter-03 audit on all six concepts, 3 seeds
  4. R5: METER-05 category ablation (all six are CATEGORICAL, declared in TARGET-D.md)
  5. R4: blind reader cards
"""
import json
import os
import statistics as st
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet.tensor import set_backend
from glassnet import tiny_data
import t3_audit_m3 as M3

R = os.path.join(PROJ, "runs", "t3")
sel = json.load(open(os.path.join(R, "e_selection.json")))
LR, LC = sel["lr"], sel["lc"]
K, N_TAUGHT = 9, 9
TAG = f"readable_K9_lr{LR}_lc{LC.replace('.','')}_meaningful_ext_e_seed{{s}}.npz"
DROP, N_CONTROL = 0.10, 20


def load(f):
    return json.load(open(os.path.join(R, f)))


rd = [x for x in load(f"models_e_scr_lc{LC}_lr{LR}.json") if x["model"] == "readable"]
rd += [x for x in load(f"models_e_cfm_lc{LC}_lr{LR}.json") if x["model"] == "readable"]
nm = [x for x in load("models_e_normal.json") if x["model"] == "normal"]
rd.sort(key=lambda x: x["seed"])
nm.sort(key=lambda x: x["seed"])

print("=" * 96)
print("TARGET E — READABLE-BY-CONSTRUCTION vs MATCHED NORMAL (same K=6, same params)")
print("=" * 96)
print(f"readable: {len(rd)} seeds  params {rd[0]['params']}  K={rd[0]['K']}  lc={LC} lr={LR}")
print(f"normal  : {len(nm)} seeds  params {nm[0]['params']}  K={nm[0]['K']}  lam_concept=0")
for tag, rows in (("readable", rd), ("normal", nm)):
    for x in rows:
        print(f"  {tag:<9} seed {x['seed']}  ppl {x['val_ppl']:8.4f}  acc {x['val']:6.2f}%")

rp, np_ = [x["val_ppl"] for x in rd], [x["val_ppl"] for x in nm]
ra, na = [x["val"] for x in rd], [x["val"] for x in nm]
pp, pa = ((st.stdev(rp) ** 2 + st.stdev(np_) ** 2) / 2) ** 0.5, \
         ((st.stdev(ra) ** 2 + st.stdev(na) ** 2) / 2) ** 0.5
d_ppl, d_acc = st.mean(np_) - st.mean(rp), st.mean(ra) - st.mean(na)
print(f"\n  readable mean ppl {st.mean(rp):8.4f} sd {st.stdev(rp):6.4f}  "
      f"acc {st.mean(ra):6.2f}% sd {st.stdev(ra):5.4f}")
print(f"  normal   mean ppl {st.mean(np_):8.4f} sd {st.stdev(np_):6.4f}  "
      f"acc {st.mean(na):6.2f}% sd {st.stdev(na):5.4f}")
print(f"\n  perplexity: readable {'BETTER' if d_ppl>0 else 'WORSE'} by {abs(d_ppl):.4f} "
      f"| bar {2*pp:6.4f} -> {'BEATS' if d_ppl > 2*pp else 'does NOT beat'}")
print(f"  accuracy  : readable {'BETTER' if d_acc>0 else 'WORSE'} by {abs(d_acc):.4f} "
      f"| bar {2*pa:6.4f} -> {'BEATS' if d_acc > 2*pa else 'does NOT beat'}")
verdict = ("WIN" if (d_ppl > 2 * pp and d_acc > 2 * pa)
           else "PARITY" if (abs(d_ppl) <= 2 * pp and abs(d_acc) <= 2 * pa)
           else "BEHIND" if (d_ppl < 0 or d_acc < 0) else "MIXED")
print(f"  VERDICT: {verdict}")

# ------------------------------------------------------------------ coverage
DATA = tiny_data.build(win=8, vocab_cap=1500)
X = np.asarray(DATA["Xva"])
Y = np.asarray(DATA["yva"])
VOCAB = DATA["vocab"]
set_backend("numpy")
names = [t["name"] for t in json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]]
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", TAG.format(s=0)))
out = net.forward(X)
h = np.asarray(out["concepts"].data)
logits = np.asarray(out["logits"].data)
W = net.answer.W.data
pred = logits.argmax(1)
correct = pred == Y
contrib = h * W[:, pred].T
mass = np.abs(contrib)
named = mass[:, :N_TAUGHT].sum(1)
total = mass.sum(1)
share = named / np.maximum(total, 1e-12)

print("\n" + "=" * 96)
print("READABILITY COVERAGE — measured, not assumed")
print("=" * 96)
print(f"  K={K}, columns with a name = {N_TAUGHT}, UNNAMED columns = {K - N_TAUGHT}")
print(f"  share of contribution mass from named concepts: {share.mean()*100:8.4f}%")
print(f"  minimum over all {len(X)} windows               : {share.min()*100:8.4f}%")
print(f"  largest contributor is a named concept         : "
      f"{(np.abs(contrib).argmax(1) < N_TAUGHT).mean()*100:8.2f}% of windows")
print(f"  => {'100% BY CONSTRUCTION, CONFIRMED BY MEASUREMENT' if share.min() > 0.9999 else 'NOT 100% -- the construction did not hold'}")

# ------------------------------------------------------------------ R1 audit + R5 category
print("\n" + "=" * 96)
print("R1 (Meter-03 audit) and R5 (METER-05 category ablation) — all six declared CATEGORICAL")
print("=" * 96)
tj = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]
tgt = M3.load_targets("meaningful_ext")
h2, L, A = M3.prep(net, X)
emb = net.emb.data
backup = emb.copy()
rng = np.random.default_rng(0)
print(f"{'col':>3} {'concept':<22} {'AUROC':>7} {'contrib':>9} {'firing':>8} {'R1':>6} "
      f"{'d_all':>8} {'z':>10} {'R5':>6}")
rows = []
for c in range(N_TAUGHT):
    r = M3.audit_col(net, X, h2, c, tgt[c][1], L, A)
    r1 = (r["auroc"] >= 0.55) and (r["contrib"] >= 0.25) and (r["firing"] >= 0.02)
    ids = [i for i in tj[c]["ids"] if 0 <= i < len(VOCAB)]
    act = h2[:, c]
    fired = act > 0.5
    A0 = float(act[fired].mean()) if fired.sum() else 0.0

    def drop_for(idset):
        for i in idset:
            emb[i] = 0.0
        hw = np.asarray(net.forward(X)["concepts"].data)[:, c]
        for i in idset:
            emb[i] = backup[i]
        return (A0 - float(hw[fired].mean())) / A0 if A0 > 0 else 0.0

    d_all = drop_for(ids)
    rand = [drop_for(rng.choice(len(VOCAB) - 1, size=len(ids), replace=False))
            for _ in range(N_CONTROL)]
    z = (d_all - float(np.mean(rand))) / (float(np.std(rand)) + 1e-9)
    r5 = (d_all >= 0.25) and (z >= 3.0)
    print(f"{c:>3} {names[c]:<22} {r['auroc']:>7.4f} {r['contrib']:>9.4f} {r['firing']:>8.4f} "
          f"{'PASS' if r1 else 'FAIL':>6} {d_all:>8.4f} {z:>+10.2f} {'PASS' if r5 else 'FAIL':>6}")
    rows.append({"col": c, "name": names[c], "auroc": r["auroc"], "contrib": r["contrib"],
                 "firing": r["firing"], "r1": bool(r1), "d_all": float(d_all), "z": float(z),
                 "r5": bool(r5)})
emb[:] = backup

# ------------------------------------------------------------------ R4 cards
cards, key = [], {}
for c in range(N_TAUGHT):
    act = h2[:, c]
    fired = np.where(act > 0.5)[0]
    notf = np.where(act <= 0.5)[0]
    fp = fired[np.argsort(-act[fired])][:5]
    npk = notf[np.argsort(act[notf])][:5]
    idx = np.concatenate([fp, npk])
    labs = np.array([1] * len(fp) + [0] * len(npk))
    o = rng.permutation(len(idx))
    idx, labs = idx[o], labs[o]
    cards.append({"col": c, "name": names[c],
                  "windows": [{"n": n, "text": " ".join(
                      VOCAB[i] if i < len(VOCAB) else "<PAD>" for i in X[i2])}
                              for n, i2 in enumerate(idx, 1)]})
    key[str(c)] = {"name": names[c],
                   "answers": [{"n": n, "fires": bool(l)} for n, l in enumerate(labs, 1)]}
json.dump(key, open(os.path.join(R, "reader_test_key_e.json"), "w"), indent=1)
json.dump({"model": TAG.format(s=0), "rows": rows, "cards": cards,
           "coverage": {"mean": float(share.mean()), "min": float(share.min())},
           "margin": {"d_ppl": d_ppl, "d_acc": d_acc, "ppl_bar": 2 * pp, "acc_bar": 2 * pa,
                      "verdict": verdict},
           "readable": {"ppl": rp, "acc": ra}, "normal": {"ppl": np_, "acc": na}},
          open(os.path.join(R, "e_eval.json"), "w"), indent=1)
print(f"\n  R1 pass: {sum(r['r1'] for r in rows)}/{N_TAUGHT}   "
      f"R5 pass: {sum(r['r5'] for r in rows)}/{N_TAUGHT}")
print("  saved runs/t3/e_eval.json and runs/t3/reader_test_key_e.json")
