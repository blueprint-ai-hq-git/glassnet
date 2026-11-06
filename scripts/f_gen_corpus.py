"""f_gen_corpus.py — TARGET F step 1: the teacher writes a corpus, and the pre-registered
selection rules are applied. Seeds come from TRAIN only; validation stays clean for evaluation.
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
N_SEEDS = int(os.environ.get("F_N_SEEDS", "4000"))
N_NEW = int(os.environ.get("F_N_NEW", "32"))
TEACHER = os.environ.get("F_TEACHER", "normal_K9_lr1.0_meaningful_ext_e_seed0.npz")

DATA = tiny_data.build(win=8, vocab_cap=1500)
Xtr = np.asarray(DATA["Xtr"])
VOCAB = DATA["vocab"]
set_backend("numpy")

print("=" * 96)
print("TARGET F — TEACHER GENERATES A CORPUS (seeds from TRAIN only, validation stays clean)")
print("=" * 96)
net = M3.load_ckpt(M3.build_net(0), os.path.join(R, "ckpt", TEACHER))
print(f"teacher: {TEACHER}")

rng = np.random.default_rng(0)
seeds = Xtr[rng.choice(len(Xtr), size=min(N_SEEDS, len(Xtr)), replace=False)]
seqs = generate(net, seeds, n_new=N_NEW, temperature=1.0, rng=rng)
X, y = to_windows(seqs, win=8)
conf = teacher_confidence(net, X, y)
print(f"seeds {len(seeds)} x {N_NEW} new tokens -> {len(X)} window pairs")

summary = {"teacher": TEACHER, "n_seeds": int(len(seeds)), "n_new": N_NEW,
           "n_windows": int(len(X)), "conf_mean": float(conf.mean()),
           "conf_median": float(np.median(conf)), "conf_p90": float(np.quantile(conf, 0.9)),
           "rules": {}}
for rule in ("all", "conf50"):
    Xs, ys, keep = select(X, y, conf, rule=rule)
    np.savez(os.path.join(R, f"f_corpus_{rule}.npz"), X=Xs, y=ys)
    st = corpus_stats(Xs, ys, VOCAB, real_X=Xtr)
    summary["rules"][rule] = {"kept": int(keep.sum()), "kept_frac": float(keep.mean()),
                              "stats": st}
    print(f"\n--- rule '{rule}': kept {keep.sum()}/{len(X)} ({keep.mean()*100:.1f}%)")
    print(f"    distinct tokens {st['n_distinct_tokens']} (real train: "
          f"{st.get('real_n_distinct_tokens')})")
    print(f"    target entropy {st['target_entropy']:.4f} nats (real train: "
          f"{st.get('real_target_entropy'):.4f})")
    print(f"    top-10 token share {st['top10_token_share']:.4f} (real train: "
          f"{st.get('real_top10_token_share'):.4f})")
    print(f"    most common tokens: {st['top10_tokens'][:8]}")

json.dump(summary, open(os.path.join(R, "f_gen.json"), "w"), indent=1)
print("\nsaved runs/t3/f_corpus_all.npz, f_corpus_conf50.npz, f_gen.json")
