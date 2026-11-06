"""t3_contrib_ceiling.py — DIAGNOSTIC (no training, no change).

The audit's contribution clause fails everywhere (14-35% vs the 50% bar) while
fidelity passes (77-90%). Before choosing a model change for step 4, measure the
CEILING: for each concept, what is the LARGEST contribution any single word
could give it? That separates two very different problems:

  * if even the best word cannot reach 0.25 -> the concept genuinely is not
    explained by any word: a MODEL problem.
  * if the best word easily reaches 0.25 but the NAME's words do not -> the
    concepts are fine and the NAMING is the bottleneck: a measurement problem.

Math (no (N,K,V) intermediate):
  numerator_i = sum_p (blocks[p] . emb_i) * #{fired windows with token i at p}
  contrib_i   = numerator_i / #fired windows
so for a concept: B = blocks @ emb.T  (W,V); counts (W,V); numerator = (B*counts).sum(0).

Env: CE_K (32), CE_LR (1.0), CE_SEED (0), CE_TOPNAME (6)
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from glassnet.tensor import set_backend
from run_t2_iter003 import TinyLM3, D, WIN, auto_name_t2c, STOP

VOCAB_CAP = 1500
K = int(os.environ.get("CE_K", "32"))
LR = float(os.environ.get("CE_LR", "1.0"))
SEED = int(os.environ.get("CE_SEED", "0"))
TOPN = int(os.environ.get("CE_TOPNAME", "6"))
set_backend("numpy")

RUNDIR = os.path.join(PROJ, "runs", "t3")
CKPT = os.path.join(RUNDIR, "ckpt", f"readable_K{K}_lr{LR}_seed{SEED}.npz")
OUT = os.path.join(RUNDIR, f"contrib_ceiling_K{K}.json")

d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
V = d["V"]
Xva, yva = np.asarray(d["Xva"]), d["yva"]

net = TinyLM3(V, K, D, WIN, rng=np.random.default_rng(SEED))
z = np.load(CKPT)
for p in net.parameters():
    if p.name in z:
        p.data = np.array(z[p.name])

h = np.asarray(net.forward(Xva)["concepts"].data)
Wm = np.asarray(net.answer.W.data)          # unused here, kept for clarity
blocks_all = np.asarray(net.concept.W.data)  # (D*W, K)
emb = np.asarray(net.emb.data)               # (V, D)
vocab = list(d["vocab"])

names = auto_name_t2c(net, d, d["Xtr"])      # the CURRENT naming
vmap = {w: i for i, w in enumerate(vocab)}

rows = []
for k in range(K):
    fired = h[:, k] > 0.5
    nf = int(fired.sum())
    if nf == 0:
        rows.append({"k": k, "fired": 0, "cur": None, "best": None, "best_word": None})
        continue
    Xf = Xva[fired]                                   # (nf, W)
    counts = np.zeros((WIN, V), dtype=np.float64)
    for p in range(WIN):
        np.add.at(counts[p], Xf[:, p], 1.0)
    blocks = blocks_all[:, k].reshape(WIN, D)          # (W, D)
    B = blocks @ emb.T                                 # (W, V) = blocks[p].emb_i
    num = (B * counts).sum(axis=0)                     # (V,)
    contrib_all = num / nf                             # (V,)

    # current name's contribution (exactly the audit's quantity)
    words = set(names[k].split("_")) - {"neuron"}
    ids = [vmap[w] for w in words if w in vmap]
    cur = float(sum(contrib_all[i] for i in ids)) if ids else None

    # ceiling: best single word, and best set of TOPN words.
    # `<unk>` is NOT a readable name, so also report the ceiling with it excluded
    # (the naming already excludes it: it iterates only over d["vocab"]).
    order = np.argsort(-contrib_all)
    best_i = int(order[0])
    best = float(contrib_all[best_i])
    top = float(contrib_all[order[:TOPN]].sum())
    c_nounk = contrib_all.copy()
    c_nounk[V - 1] = -np.inf
    o2 = np.argsort(-c_nounk)
    best2_i, best2 = int(o2[0]), float(c_nounk[o2[0]])
    top2 = float(c_nounk[o2[:TOPN]].sum())

    rows.append({"k": k, "fired": nf, "cur": cur, "best": best,
                 "best_word": (vocab[best_i] if best_i < len(vocab) else "<unk>"),
                 "best_nounk": best2,
                 "best_nounk_word": (vocab[best2_i] if best2_i < len(vocab) else "<unk>"),
                 "top%d" % TOPN: top, "top%d_nounk" % TOPN: top2,
                 "name": names[k]})

cur_v = [r["cur"] for r in rows if r["cur"] is not None]
best_v = [r["best"] for r in rows if r["best"] is not None]
top_v = [r["top%d" % TOPN] for r in rows if r.get("top%d" % TOPN) is not None]

print(f"CONTRIBUTION CEILING — readable K={K} lr={LR} seed={SEED} (validation)")
print(f"  checkpoint {CKPT}")
print(f"  concepts: {K}, fired windows per concept: min "
      f"{min(r['fired'] for r in rows)} max {max(r['fired'] for r in rows)}")
print()
print(f"  {'measure':38s} {'median':>9} {'>=0.25':>10} {'frac':>7}")
for label, vals in [(f"CURRENT naming (top{TOPN} words, as audited)", cur_v),
                    ("CEILING: best single word (any token)", best_v),
                    ("CEILING: best single REAL word (no <unk>)",
                     [r["best_nounk"] for r in rows if r.get("best_nounk") is not None]),
                    (f"CEILING: best {TOPN} words", top_v),
                    (f"CEILING: best {TOPN} real words (no <unk>)",
                     [r["top%d_nounk" % TOPN] for r in rows
                      if r.get("top%d_nounk" % TOPN) is not None])]:
    n_ok = sum(1 for v in vals if v >= 0.25)
    print(f"  {label:38s} {np.median(vals):>+9.3f} {n_ok:>5}/{len(vals):<4} "
          f"{n_ok/len(vals)*100:>6.1f}%")

print()
print("  per-concept detail (current name contrib vs best-word ceiling):")
for r in sorted(rows, key=lambda x: -(x["best"] or -9))[:12]:
    print(f"    k{r['k']:>3} fired {r['fired']:>5} | name {str(r['name'])[:34]:34s} "
          f"cur {('%.3f' % r['cur']) if r['cur'] is not None else '  n/a'} | "
          f"best {r['best']:.3f} ('{r['best_word']}')")

n_best = sum(1 for v in best_v if v >= 0.25)
json.dump({"K": K, "lr": LR, "seed": SEED,
           "cur_median": float(np.median(cur_v)),
           "cur_pass": int(sum(1 for v in cur_v if v >= 0.25)),
           "best_median": float(np.median(best_v)),
           "best_pass": int(n_best),
           "topn_pass": int(sum(1 for v in top_v if v >= 0.25)),
           "n": len(cur_v), "rows": rows}, open(OUT, "w"), indent=1)
print(f"\n  => with the BEST possible single word, {n_best}/{len(best_v)} "
      f"({n_best/len(best_v)*100:.0f}%) concepts reach 0.25")
print(f"saved {OUT}")
