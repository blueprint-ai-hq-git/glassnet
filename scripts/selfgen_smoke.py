"""selfgen_smoke.py — does our own model actually generate usable text, and how healthy is the
corpus it produces? Runs the generator for real and prints samples and statistics.
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
from glassnet.selfgen import generate, to_windows, teacher_confidence, select, corpus_stats
import t3_audit_m3 as M3

R = os.path.join(PROJ, "runs", "t3")
DATA = tiny_data.build(win=8, vocab_cap=1500)
Xva = np.asarray(DATA["Xva"])
VOCAB = DATA["vocab"]
set_backend("numpy")

print("=" * 96)
print("SELF-GENERATION SMOKE TEST — teacher = our own K=9 model, no outside weights")
print("=" * 96)
print("checkpoints available:")
for f in sorted(os.listdir(os.path.join(R, "ckpt"))):
    if "meaningful_ext" in f and "normal" in f:
        print("   ", f)

CK = "normal_K9_lr1.0_meaningful_ext_e_seed0.npz"
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", CK))
print(f"\nteacher: {CK}")

rng = np.random.default_rng(0)
seeds = Xva[rng.choice(len(Xva), size=12, replace=False)]
seqs = generate(net, seeds, n_new=24, temperature=1.0, rng=rng)

print("\n--- what the teacher writes (seed = real text, everything after is its own) ---")
for i in range(4):
    W = seeds.shape[1]
    real = " ".join(VOCAB[t] if t < len(VOCAB) else "<PAD>" for t in seqs[i, :W])
    gen = " ".join(VOCAB[t] if t < len(VOCAB) else "<PAD>" for t in seqs[i, W:])
    print(f"\n  [{i}] REAL: {real}")
    print(f"      GEN : {gen}")

X, y = to_windows(seqs, win=8)
print(f"\ngenerated windows: {len(X)}")
conf = teacher_confidence(net, X, y)
for rule in ("all", "conf"):
    Xs, ys, keep = select(X, y, conf, rule=rule, threshold=0.5)
    print(f"  rule '{rule}': kept {keep.sum()}/{len(X)} windows ({keep.mean()*100:.1f}%)")

Xs, ys, _ = select(X, y, conf, rule="conf", threshold=0.5)
stats = corpus_stats(Xs, ys, VOCAB, real_X=Xva)
print("\n--- corpus health: generated vs REAL text ---")
print(f"  {'measure':<32} {'generated':>12} {'real val':>12}")
print(f"  {'distinct tokens used':<32} {stats['n_distinct_tokens']:>12} "
      f"{stats.get('real_n_distinct_tokens', 0):>12}")
print(f"  {'target entropy (nats)':<32} {stats['target_entropy']:>12.4f} "
      f"{stats.get('real_target_entropy', 0):>12.4f}")
print(f"  {'top-10 token share':<32} {stats['top10_token_share']:>12.4f} "
      f"{stats.get('real_top10_token_share', 0):>12.4f}")
print(f"  top tokens in generated: {stats['top10_tokens']}")

json.dump({"teacher": CK, "n_windows": int(len(X)),
           "kept_all": int(len(X)), "kept_conf": int(len(Xs)),
           "stats_conf": stats,
           "conf_mean": float(conf.mean()), "conf_median": float(np.median(conf))},
          open(os.path.join(R, "selfgen_smoke.json"), "w"), indent=1)
print("\nsaved runs/t3/selfgen_smoke.json")
