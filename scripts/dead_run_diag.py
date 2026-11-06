"""dead_run_diag.py — WHY does the readable model sometimes die at init?

OBSERVED TWICE, identically:
  - Step 2, readable seed 1: val 16.32%, best epoch 1, excl-unk 0.00%
  - Step 4, readable K=72  : val 16.317661003555905 = the majority baseline
                             BIT-FOR-BIT, best epoch 1
In both cases the model predicts <unk> for every val example: it never learns.

HYPOTHESIS (written BEFORE running): the two sparsity penalties win at
initialisation on unlucky seeds and collapse the concept layer.
    lam_act * mean(concepts)    pushes the sigmoid concepts toward 0
    lam_w   * mean(|concept.W|) pushes the concept weights toward 0
If the concepts collapse to a constant, the 8001-way head receives a constant
input, so it can only learn the class prior (<unk> = 16.32%), and the gradient
flowing back into the concepts vanishes -> permanently stuck at the prior.

PRE-REGISTERED PREDICTIONS (decided before the run, not after):
  P1. the dead seed's mean concept activation is far smaller than the live
      seed's (collapse toward 0)
  P2. the dead seed predicts <unk> for ~100% of val examples, the live one
      far less
  P3. the dead seed's CE loss stays flat near ln(8001) = 8.99 while the live
      seed's falls
  P4. the two seeds are already distinguishable in the FIRST epoch (which is
      what best_epoch == 1 implies)

PROTOCOL: identical to the sweep (same seed -> same init, same shuffle, same
lr/batch/epochs). The ONLY addition is internal instrumentation. No setting is
changed, so the death is reproduced, not constructed.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from glassnet.optim import SGD
from run_t2_iter003 import K, D, WIN, LR, BATCH, LAM_A, LAM_W, TinyLM3

EPOCHS = 6
UNK = 8000
N_INSPECT = 2048


def stats(net, X, y):
    f = net.forward(X[:N_INSPECT])
    concepts = np.asarray(f["concepts"].data)
    logits = np.asarray(f["logits"].data)
    pred = logits.argmax(axis=1)
    z = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(z)
    p = ex / ex.sum(axis=1, keepdims=True)
    yy = np.asarray(y[:N_INSPECT])
    ce = float(-np.log(p[np.arange(len(yy)), yy] + 1e-12).mean())
    W = np.asarray(net.concept.W.data)          # (512, K)
    colmass = np.abs(W).sum(axis=0)             # per-concept incoming mass
    return {
        "mean_act": float(concepts.mean()),
        "act_std": float(concepts.std()),
        "pct_unk": float((pred == UNK).mean() * 100),
        "ce": ce,
        "dead_cols": int((colmass < 1e-3).sum()),
        "colmass_min": float(colmass.min()),
        "colmass_max": float(colmass.max()),
    }


def run(seed, label):
    d = tiny_data.build(win=WIN)
    V = d["V"]
    net = TinyLM3(V, K, D, WIN, rng=np.random.default_rng(seed))
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    Xva, yva = d["Xva"], d["yva"]
    rngd = np.random.default_rng(seed * 131 + 7)

    print(f"\n=== seed {seed} ({label}) ===")
    s0 = stats(net, Xva, yva)
    print(f"  init      mean_act {s0['mean_act']:.5f}  act_std {s0['act_std']:.5f}  "
          f"pct_unk {s0['pct_unk']:6.2f}%  ce {s0['ce']:.4f}  "
          f"dead_cols {s0['dead_cols']}/{K}")

    out = [{"epoch": 0, **s0}]
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            opt.step()
        s = stats(net, Xva, yva)
        out.append({"epoch": ep, **s})
        print(f"  ep {ep:2d}     mean_act {s['mean_act']:.5f}  act_std {s['act_std']:.5f}  "
              f"pct_unk {s['pct_unk']:6.2f}%  ce {s['ce']:.4f}  "
              f"dead_cols {s['dead_cols']}/{K}  ({time.time()-t0:.0f}s)")
    return out


def main():
    print("=" * 78)
    print("DEAD-RUN DIAGNOSTIC — readable model, seeds 1 (dead) vs 0 (live)")
    print(f"config: K={K} D={D} W={WIN} lr={LR} batch={BATCH} "
          f"lam_act={LAM_A} lam_w={LAM_W} epochs={EPOCHS}")
    print("=" * 78)

    dead = run(1, "DEAD in Step 2")
    live = run(0, "live in Step 2")

    print("\n" + "=" * 78)
    print("PRE-REGISTERED PREDICTIONS vs RESULT")
    print("=" * 78)
    d1, l1 = dead[1], live[1]
    print(f"P1 mean_act smaller when dead : dead {d1['mean_act']:.5f} vs "
          f"live {l1['mean_act']:.5f}  -> "
          f"{'CONFIRMED' if d1['mean_act'] < l1['mean_act'] else 'NOT CONFIRMED'}")
    print(f"P2 pct_unk ~100% when dead    : dead {d1['pct_unk']:.2f}% vs "
          f"live {l1['pct_unk']:.2f}%  -> "
          f"{'CONFIRMED' if d1['pct_unk'] > 95 and l1['pct_unk'] < 90 else 'NOT CONFIRMED'}")
    print(f"P3 ce flat when dead          : dead ep1 {d1['ce']:.4f} vs "
          f"live ep1 {l1['ce']:.4f}  (ln V = {np.log(8001):.4f})  -> "
          f"{'CONFIRMED' if d1['ce'] > 8.5 and l1['ce'] < d1['ce'] else 'NOT CONFIRMED'}")
    print(f"P4 distinguishable at epoch 1 : dead mean_act {d1['mean_act']:.5f} vs "
          f"live {l1['mean_act']:.5f}  -> "
          f"{'CONFIRMED' if abs(d1['mean_act'] - l1['mean_act']) > 1e-4 else 'NOT CONFIRMED'}")

    out = os.path.join(PROJ, "runs", "t2", "dead_diag.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import json
    with open(out, "w") as f:
        json.dump({"dead_seed1": dead, "live_seed0": live}, f, indent=1)
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
