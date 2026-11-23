"""CAN THE DIRECT PATH BE DELETED?  Inference-time ablation, no training needed.

WHY THIS EXISTS (2026-11-22 15:34): the aggregate accounting says the readable concept layer supplies only 0.2% to 18.2% of the
decision margin, while the bias plus the direct path supplies the other 81.8% to 99.8%. That is a share of the MARGIN, which is not the
same as a share of the ANSWER. So the practical question is the one a user actually cares about:

        if the direct path were removed, would the model still give the same answers?

Measured directly on three checkpoints, with no training and no labels needed:

  logits_full     = h @ W + b + direct          (what the model actually predicts)
  logits_concepts = h @ W + b                   (the readable route alone, direct path deleted)
  logits_direct   = b + direct                  (the unreadable route alone, concepts deleted)

and then, for each window:
  * does argmax(logits_concepts) equal argmax(logits_full)?      <- agreement
  * what rank does the full model's top token have under the concept-only logits?   <- graded version

PREDICTIONS, WRITTEN BEFORE THE RUN:
  P1 the concept-only route reproduces the full model's chosen token in FEWER THAN 50% of windows on all three checkpoints.
  P2 the mean rank of the full model's top token under concept-only logits is worse than 5 (1 = the concept route liked it best).
FALSIFIER: if the concept-only route agrees with the full model in 90% or more of windows, the direct path is redundant and could be
  deleted for free - which would be a large win for readability, and would contradict the margin-share picture.
"""
import os
import sys

import numpy as np

ROOT = os.environ.get("GLASSNET_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.chdir(ROOT)
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest"))

from concept_rank import load_windows  # reuse the exact same data path as the share measurements


def run(ckpt, n_windows):
    net, vocab, X, forward_arrays, backend = load_windows(ckpt, n_windows)
    W = backend.to_cpu(net.answer.W.data).astype(np.float64)
    b = backend.to_cpu(net.answer.b.data).astype(np.float64)
    agree_c = agree_d = 0
    ranks = []
    used = 0
    for i in range(len(X)):
        act = forward_arrays(net, np.asarray(X[i])[None, :])
        h = backend.to_cpu(act["concepts"][0]).astype(np.float64)
        logits = backend.to_cpu(act["logits"][0]).astype(np.float64)
        direct = logits - (h @ W + b)
        lg_c = h @ W + b                 # readable route only
        lg_d = b + direct                # unreadable route only
        top = int(np.argmax(logits))
        agree_c += int(np.argmax(lg_c) == top)
        agree_d += int(np.argmax(lg_d) == top)
        order = np.argsort(-lg_c)
        rank = int(np.where(order == top)[0][0]) + 1
        ranks.append(rank)
        used += 1
    return dict(ckpt=os.path.basename(ckpt), windows=used,
                agree_concepts=100.0 * agree_c / used, agree_direct=100.0 * agree_d / used,
                mean_rank=float(np.mean(ranks)), median_rank=float(np.median(ranks)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=200)
    a = ap.parse_args()
    n = a.windows
    ckpts = ["lens/runs/chat/d1024_s1_best.npz",
             "lens/runs/chat/d1024_e9_s1_best.npz",
             "lens/runs/chat/direct1024_d512_s0_best.npz"]
    print("=== can the direct path be deleted? %s ===" % __import__("time").strftime("%Y-%m-%d %H:%M:%S"))
    print("no training, no labels: does the readable route alone reproduce the model's own answer?")
    print()
    print("  %-30s %8s %16s %14s %10s" % ("checkpoint", "windows", "concept route", "direct route", "mean rank"))
    rows = []
    for ck in ckpts:
        if not os.path.exists(os.path.join(ROOT, ck)):
            print("  %-30s MISSING, skipped" % os.path.basename(ck))
            continue
        r = run(ck, n)
        rows.append(r)
        print("  %-30s %8d %15.1f%% %13.1f%% %10.1f"
              % (r["ckpt"], r["windows"], r["agree_concepts"], r["agree_direct"], r["mean_rank"]))
    if not rows:
        raise SystemExit("no checkpoints measured - refusing to report")
    worst = min(r["agree_concepts"] for r in rows)
    best = max(r["agree_concepts"] for r in rows)
    ranks_ok = all(r["mean_rank"] > 5 for r in rows)
    print()
    print("RESULT concept-only route reproduces the model's own top token in %.1f%% to %.1f%% of windows"
          % (worst, best))
    print("P1 concept route below 50%% on all three: %s" % ("MET" if best < 50 else "NOT MET"))
    print("P2 mean rank of the model's own top token worse than 5 on all three: %s" % ("MET" if ranks_ok else "NOT MET"))
    if best >= 90:
        print("FALSIFIER FIRED: the direct path is redundant - it could be deleted for free.")
        print("OUTCOME: DIRECT_PATH_REDUNDANT")
    elif best < 50:
        print("FALSIFIER not fired.")
        print("OUTCOME: DIRECT_PATH_CARRIES_THE_ANSWER")
    else:
        print("FALSIFIER not fired (50-90%% agreement): the direct path matters but the concept route is not negligible.")
        print("OUTCOME: BOTH_ROUTES_MATTER")
    print("DIRECT-ABLATION COMPLETE")


if __name__ == "__main__":
    main()
