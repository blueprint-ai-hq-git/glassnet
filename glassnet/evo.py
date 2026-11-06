"""glassnet/evo.py — Phase 6: the experimental trainer behind the switch.

Method: (1+1)-Evolution Strategy with the classic 1/5 success rule.
No gradients anywhere. Every evaluation:
  1. mutate the parent weight vector with Gaussian noise (size = sigma)
  2. evaluate the mutant AND the parent on the SAME fresh batch,
     using the IDENTICAL objective the SGD trainer minimizes
     (CE + alpha*conceptBCE + lam*L1)
  3. keep whichever scored better (mutant -> parent if it won)
  4. every `onefifth_window` evaluations, adapt sigma by the 1/5 rule:
     more than ~1/5 mutations improving  -> grow sigma
     fewer                                -> shrink sigma

Fairness contract vs the normal SGD auto-pilot (phase6_comparison.py):
same architecture, same seeds, same batch stream family, same objective,
same teaching budget in batch-evaluations, same pruning after training.
The normal SGD trainer is not touched — `optimizer="sgd"` stays the default.
"""
import time
from collections import deque

import numpy as np

from .tensor import Tensor
from .phase2_model import GlassNetPhase2
from .eval import accuracy

EVO_DEFAULTS = {
    # ---- budget (== SGD: 12 epochs x 469 batches of 128) ----
    "batch_size": 128,
    "budget_batches": 5628,
    # ---- mutation ----
    "sigma_start": 0.05,
    "sigma_min": 1e-4,
    "sigma_grow": 1.15,
    "sigma_shrink": 0.85,
    "onefifth_window": 50,
    "onefifth_high": 0.22,      # success rate above this -> grow sigma
    "onefifth_low": 0.18,       # below this -> shrink sigma
    # ---- same objective as the SGD trainer ----
    "concept_loss_alpha": 1.0,
    "l1_lambda": 0.005,
    # ---- cadence + post-processing ----
    "eval_every_batches": 469,
    "prune_keep_connections": 3,
}


def _layout(net):
    """Flat weight vector + write-back handles (positions into each param)."""
    pieces = [p.data for layer in (net.concept, net.rules, net.answer)
              for p in layer.parameters()]
    sizes = [int(np.prod(p.shape)) for p in pieces]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    flat = np.concatenate([p.ravel() for p in pieces]).astype(np.float32)
    return flat, pieces, offsets, sizes


def _write(flat, pieces, offsets, sizes):
    for p, off, size in zip(pieces, offsets, sizes):
        p[:] = flat[off:off + size].reshape(p.shape)


def train_evo(Xtr, ytr, Ctr, Xte, yte, Cte, seed=0, defaults=None,
              log_actions=True):
    """(1+1)-ES training of the identical readable architecture.
    Returns (net, info) — info has the same shape as the auto-pilot's."""
    d = dict(EVO_DEFAULTS)
    if defaults:
        d.update(defaults)
    actions = []

    def act(msg):
        actions.append(msg)
        if log_actions:
            print(f"[evo @eval {step:5d}] {msg}", flush=True)

    step = 0
    rng = np.random.default_rng(int(seed))
    data_rng = np.random.default_rng(int(seed) * 777 + 1)   # same family as SGD
    net = GlassNetPhase2(K=14, R=30, rng=rng)
    flat, pieces, offsets, sizes = _layout(net)
    n = len(Xtr)
    alpha, lam = d["concept_loss_alpha"], d["l1_lambda"]
    sigma = float(d["sigma_start"])
    win = deque(maxlen=int(d["onefifth_window"]))
    parent = flat.copy()
    best_test, best_flat = 0.0, None
    curve = []
    t0 = time.time()

    act(f"MISSION START (EVO) — seed {seed}, sigma {sigma}, "
        f"budget {d['budget_batches']} batch-evaluations, no gradients")

    for step in range(1, d["budget_batches"] + 1):
        idx = data_rng.choice(n, d["batch_size"], replace=False)
        to = Tensor(Xtr[idx], name="batch")
        yb, cb = ytr[idx], Ctr[idx]

        # 1. mutate
        child = (parent + rng.normal(0.0, sigma, size=parent.shape)
                 .astype(np.float32))
        # 2. evaluate both on the SAME batch (identical objective)
        _write(child, pieces, offsets, sizes)
        f_child = float(net.loss(to, yb, cb, alpha=alpha, lam=lam).data)
        _write(parent, pieces, offsets, sizes)
        f_parent = float(net.loss(to, yb, cb, alpha=alpha, lam=lam).data)
        # 3. selection
        if f_child < f_parent:
            parent = child
            win.append(1)
        else:
            win.append(0)
        # 4. sigma adaptation (1/5 rule)
        if step % d["onefifth_window"] == 0:
            sr = float(np.mean(win))
            if sr > d["onefifth_high"]:
                sigma = min(sigma * d["sigma_grow"], 1.0)
            elif sr < d["onefifth_low"]:
                sigma = max(sigma * d["sigma_shrink"], d["sigma_min"])
        # cadence: evaluate on the held-out test
        if step % d["eval_every_batches"] == 0:
            test_acc = accuracy(net, Xte, yte)
            curve.append({"step": step, "test_acc": test_acc, "sigma": sigma})
            if test_acc > best_test:
                best_test = test_acc
                best_flat = parent.copy()
            act(f"eval: parent fitness {f_parent:.3f}, sigma "
            f"{sigma:.4f}, test {test_acc*100:.2f}% "
            f"(best {best_test*100:.2f}%)")

    # ---- finish ------------------------------------------------
    if best_flat is not None:
        _write(best_flat, pieces, offsets, sizes)
    pre = accuracy(net, Xte, yte)
    net.prune_rule_connections(keep=d["prune_keep_connections"])
    post = accuracy(net, Xte, yte)
    stats = net.connection_stats()
    act(f"finished: best-test {best_test*100:.2f}%, "
            f"pre-prune {pre*100:.2f}% -> post-prune {post*100:.2f}%, "
            f"{stats['avg']:.2f} conns/rule")
    info = {"seed": int(seed), "pre_prune_acc": pre, "post_prune_acc": post,
            "best_test": best_test, "final_sigma": sigma,
            "seconds": time.time() - t0, "curve": curve,
            "actions": actions}
    return net, info