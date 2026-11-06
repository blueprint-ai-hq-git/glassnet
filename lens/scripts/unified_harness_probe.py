"""lens/scripts/unified_harness_probe.py - WHY is the 2+2 harness weak with the gate switched off?

UNIFIED-TRAINING.md says: "the smoke-test model itself is undertrained (9% exact even on the pairs it trained
on ...), so this harness is too weak to judge ANY gate. Fix the harness before drawing conclusions."

lens/unified.py STAGE 2 measured, with the gate OFF, 3000 steps and 3 seeds: 44.3% exact on the pairs it
trained on, 0.0% on unseen. This script asks which part is at fault, with the answer known in advance:

  P8  The corpus is exactly fit-able: y = a+b with X = [a/20, b/20, 1] has a closed-form least-squares
      solution with ZERO residual, so if the model cannot reach ~100% exact the fault is the OPTIMISER or
      the minibatch size, never the corpus.
  P9  Full-batch gradient descent at the same lr reaches >99% exact on trained pairs; the mini-batch of 16
      is what holds it back.

Measured here: closed-form solution, full-batch GD, mini-batch GD at two batch sizes, all 3 seeds.
"""
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def corpus():
    pairs = [(a, b) for a in range(10) for b in range(10)]
    X = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in pairs], dtype=np.float64)
    y = np.asarray([a + b for a, b in pairs], dtype=np.float64)
    unseen = [(a, b) for a in range(10, 20) for b in range(10, 20)]
    Xu = np.asarray([[a / 20.0, b / 20.0, 1.0] for a, b in unseen], dtype=np.float64)
    yu = np.asarray([a + b for a, b in unseen], dtype=np.float64)
    return X, y, Xu, yu


def gd(X, y, Xu, yu, *, steps=3000, lr=0.05, batch=None, seed=0, holdout_frac=0.25):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_held = max(8, int(len(X) * holdout_frac))
    hi, ti = order[:n_held], order[n_held:]
    Xt, yt = X[ti], y[ti]
    Xh, yh = X[hi], y[hi]
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        if batch is None:
            Xb, yb = Xt, yt
        else:
            idx = rng.integers(0, len(Xt), size=batch)
            Xb, yb = Xt[idx], yt[idx]
        err = Xb @ w - yb
        w = w - lr * (2.0 / len(Xb)) * (Xb.T @ err)
    exact_train = float((np.round(Xt @ w) == yt).mean())
    exact_unseen = float((np.round(Xu @ w) == yu).mean())
    held = float(np.mean((Xh @ w - yh) ** 2))
    return exact_train, exact_unseen, held, w


def main():
    X, y, Xu, yu = corpus()
    print("=" * 96)
    print("WHY IS THE 2+2 HARNESS WEAK? closed form vs full batch vs mini-batch, 3 seeds")
    print("=" * 96)
    print("P8: the corpus is exactly fit-able (closed form residual 0), so the fault is the optimiser, not data")
    print("P9: full-batch GD reaches >99% exact on trained pairs; the mini-batch of 16 holds it back")
    print()

    w_ls, res, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    print(f"closed-form least squares: w = {np.round(w_ls, 4).tolist()}  residual {res[0] if len(res) else 0.0:.3e}  "
          f"rank {rank} (input columns {X.shape[1]})")
    print(f"  exact on trained pairs {100*float((np.round(X @ w_ls) == y).mean()):.1f}%   "
          f"exact on UNSEEN pairs {100*float((np.round(Xu @ w_ls) == yu).mean()):.1f}%   "
          f"-> the corpus CAN be solved exactly (P8)")
    print()

    print(f"{'optimiser':<22}{'steps':>7}{'exact trained':>15}{'exact unseen':>14}{'held-out loss':>15}")
    print("-" * 96)
    for label, kw in (("full batch, lr 0.05", dict(batch=None, lr=0.05, steps=3000)),
                      ("mini-batch 16, lr 0.05", dict(batch=16, lr=0.05, steps=3000)),
                      ("mini-batch 16, lr 0.05 (30k)", dict(batch=16, lr=0.05, steps=30000))):
        tr, un, hl = [], [], []
        for seed in (0, 1, 2):
            a, b_, h, _w = gd(X, y, Xu, yu, seed=seed, **kw)
            tr.append(a); un.append(b_); hl.append(h)
        print(f"{label:<22}{kw['steps']:>7}{np.mean(tr)*100:>14.1f}%{np.mean(un)*100:>13.1f}%"
              f"{np.mean(hl):>15.5f}")
    print("-" * 96)
    print("read the table: if the full-batch row is near 100% and the mini-batch rows are not, the harness is")
    print("limited by its own gradient noise, not by the corpus, and a gate verdict taken on it is uninformative.")

    # ---- SECTION 2: the corpus and the optimiser are both fine, so the fault must be INSIDE the loop.
    # The loop multiplies every step by max(agreement, 0.05) - a cosine between per-example gradients.
    print("\n" + "=" * 96)
    print("SECTION 2 - inside the loop: what does the agreement scale do to learning? (train_unified, 3000 steps)")
    print("=" * 96)
    print("P10: the mean agreement on this corpus is below 0.25, so max(agreement,0.05) multiplies every step")
    print("     by <=0.25 and the loop is throttled ~4-20x; with scaling OFF the same loop reaches >99% exact")
    print("     on trained pairs and >90% on unseen 10..19.")
    print()
    from lens.unified import train_unified, GapGate as _GapGate

    def agreement_stats(X, y, seed, n=100, batch=16, holdout_frac=0.25):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(X))
        n_held = max(8, int(len(X) * holdout_frac))
        ti = order[n_held:]
        Xt, yt = X[ti], y[ti]
        w = np.zeros(X.shape[1])
        vals = []
        for _ in range(n):
            idx = rng.integers(0, len(Xt), size=batch)
            Xb, yb = Xt[idx], yt[idx]
            pred = Xb @ w
            peg = 2.0 * (pred - yb)[:, None] * Xb
            flat = peg / (np.linalg.norm(peg, axis=1, keepdims=True) + 1e-12)
            md = flat.mean(axis=0)
            md /= (np.linalg.norm(md) + 1e-12)
            vals.append(float((flat @ md).mean()))
            err = Xb @ w - yb
            w = w - 0.05 * (2.0 / len(Xb)) * (Xb.T @ err)
        return float(np.mean(vals)), float(np.min(vals))

    ag, agmin = agreement_stats(X, y, seed=0)
    print(f"measured agreement on this corpus (first 100 steps, seed 0): mean {ag:.3f}, min {agmin:.3f}")
    print()
    print(f"{'loop setting':<56}{'exact trained':>14}{'exact unseen':>14}{'refused':>9}")
    print("-" * 96)
    for label, kw in (
        ("agreement scale ON  + gap gate ON  (shipped)", dict()),
        ("agreement scale OFF + gap gate ON", dict(agreement_mode="one")),
        ("agreement scale ON  + gap gate OFF", dict(gate=_GapGate(variant="off"))),
        ("agreement scale OFF + gap gate OFF (plain loop)", dict(agreement_mode="one", gate=_GapGate(variant="off"))),
    ):
        tr, un, rf = [], [], []
        for seed in (0, 1, 2):
            kw_seed = dict(kw)
            if "gate" in kw_seed:
                kw_seed["gate"] = _GapGate(variant="off")
            (w, _b), log = train_unified(X, y, mode="data", steps=3000, lr=0.05, seed=seed, verbose=False,
                                         **kw_seed)
            tr.append(float((np.round(X @ w + _b) == y).mean()))
            un.append(float((np.round(Xu @ w + _b) == yu).mean()))
            rf.append(log["refused_gap"])
        print(f"{label:<56}{np.mean(tr)*100:>13.1f}%{np.mean(un)*100:>13.1f}%{np.mean(rf):>9.1f}")
    print("-" * 96)

    # ---- SECTION 3: the agreement throttle, calibrated on the same two corpora as the gap gate.
    print("\n" + "=" * 96)
    print("SECTION 3 - calibrate the step scale on a memorisable corpus and a learnable corpus")
    print("=" * 96)
    print("P11: with the gap gate ON, removing the agreement throttle (scale=1) gives the LOWEST held-out loss")
    print("     on the learnable corpus and no worse held-out loss on the memorisable corpus, because the")
    print("     throttle slows the honest steps and the gap gate is already refusing the memorising ones.")
    print()
    from lens.unified import GapGate as _G

    def corpora(kind, n=256, d=8, seed=1234):
        rng = np.random.default_rng(seed)
        Xc = rng.normal(size=(n, d))
        Xc[:, -1] = 1.0
        if kind == "noise":
            yc = rng.normal(size=n)
        else:
            wt = rng.normal(size=d)
            wt[-1] = 0.0
            yc = Xc @ wt + 0.05 * rng.normal(size=n)
        return Xc, yc

    print(f"{'step scale':<28}{'learnable: held-out':>22}{'memorisable: held-out':>24}{'pairs unseen':>14}")
    print("-" * 96)
    for label, scale_mode in (("max(agreement, 0.05) shipped", "floor"), ("agreement (no floor)", "raw"),
                              ("1.0 (throttle removed)", "one")):
        hl_rule, hl_noise = [], []
        for cseed in (1234, 7, 99):
            for tseed in (0, 1, 2):
                for kind, bucket in (("rule", hl_rule), ("noise", hl_noise)):
                    Xc, yc = corpora(kind, seed=cseed)
                    (w, b_), log = train_unified(Xc, yc, mode="data", steps=300, lr=0.05, seed=tseed,
                                                 gate=_G(beta=0.5, tol=0.0, variant="ema_band"),
                                                 verbose=False, agreement_mode=scale_mode)
                    bucket.append(log["heldout_last"])
        tr, un = [], []
        for seed in (0, 1, 2):
            (w, b_), log = train_unified(X, y, mode="data", steps=3000, lr=0.05, seed=seed, verbose=False,
                                         gate=_G(variant="off"), agreement_mode=scale_mode)
            tr.append(float((np.round(X @ w + b_) == y).mean()))
            un.append(float((np.round(Xu @ w + b_) == yu).mean()))
        print(f"{label:<28}{np.mean(hl_rule):>15.5f}+-{np.std(hl_rule):.5f}"
              f"{np.mean(hl_noise):>17.4f}+-{np.std(hl_noise):.4f}{np.mean(un)*100:>13.1f}%")
    print("-" * 96)


if __name__ == "__main__":
    main()
