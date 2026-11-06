#!/usr/bin/env python3
"""numeric_chat_readout.py - the decisive question for PLAN.txt item (d3), measured in CLOSED FORM.

WHY THIS TEST AND NOT THE FULL-MODEL ARM. The full-model arm (numeric_chat_test.py) went non-finite in
every configuration tried, and the cause was measured, not guessed (lens/runs/NUMERIC-CHAT-GRAD-SCALE.txt
+ lens/scripts/diagnose_numeric_chat.py): the quantity head is a LINEAR regression on magnitudes whose
dynamic range is 0..80, the windows that carry a numeric target are only ~4% of a batch (3 rows of 256), and
the numeric term's gradient feeds back into the shared embedding, so the run blows up. Before spending more
compute on conditioning a head, the question that decides whether any of it is worth doing is:

    IS THERE ANY LINEAR SIGNAL AT ALL, in the numbers already in the window, about the quantity of the
    next token - on the HELD-OUT split?

That question does not need a training loop. A ridge regression fitted on the TRAIN masked rows and scored
on the HELD-OUT masked rows answers it exactly, with no optimiser, no step size, and no seed noise - so a
FAIL cannot be an artefact of a learning rate. Predictions: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt
ADDENDUM 4 (P43l, P43m). Output: lens/runs/NUMERIC-CHAT-READOUT.txt + .json.

Arms
  ridge      W features (the window's magnitudes) + bias, fitted on the train masked rows
  shuffled   the SAME fit with the train targets permuted: the control. It must land at the constant.
  persist    predict the LAST number written in the window (the simplest real predictor of a next number)
  constants  predict 0 / the train mean / the train median

Usage: python3 lens/scripts/numeric_chat_readout.py
"""
import datetime
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
os.chdir(PROJ)
D_DIR = os.environ.get("CHAT_DATA_DIR") or os.path.join(PROJ, "data", "chat", "prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D_DIR

import chat_data                                                     # noqa: E402
from lens.numeric import magnitude_channel, numeric_targets          # noqa: E402

W = 32
SCALE = 100.0
TRAIN_WINDOWS = int(os.environ.get("NUMERIC_CHAT_TRAIN", "200000"))
VAL_WINDOWS = int(os.environ.get("NUMERIC_CHAT_VAL", "50000"))
LAMS = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
FOLDS = 5
OUT = os.path.join(PROJ, "lens", "runs", "NUMERIC-CHAT-READOUT.txt")
JSON_OUT = os.path.join(PROJ, "lens", "runs", "numeric_chat_readout.json")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def ridge_fit(Q, y, lam):
    """Closed form: (Q'Q + lam I)^-1 Q'y, with the bias column NOT penalised (standard practice)."""
    A = Q.T @ Q
    pen = lam * np.eye(A.shape[0])
    pen[-1, -1] = 0.0                       # the bias is not shrunk
    return np.linalg.solve(A + pen, Q.T @ y)


def cv_lambda(Q, y, lams, folds, rng):
    """Pick lambda by k-fold CV ON THE TRAIN ROWS ONLY. Nothing about the held-out split is used here."""
    n = len(y)
    idx = rng.permutation(n)
    parts = np.array_split(idx, folds)
    best, best_err = None, np.inf
    for lam in lams:
        errs = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            w = ridge_fit(Q[tr], y[tr], lam)
            errs.append(float(np.mean((Q[te] @ w - y[te]) ** 2)))
        e = float(np.mean(errs))
        if e < best_err:
            best, best_err = lam, e
    return best, best_err


def report(name, pred, tgt, consts):
    mae = float(np.abs(pred - tgt).mean())
    rmse = float(np.sqrt(((pred - tgt) ** 2).mean()))
    say(f"   {name:26s} MAE {mae:>10.4f}  RMSE {rmse:>10.4f}   "
        f"MAE {mae/consts['best_mae']:>6.3f}x  RMSE {rmse/consts['best_rmse']:>6.3f}x")
    return {"mae": mae, "rmse": rmse,
            "mae_ratio": mae / consts["best_mae"], "rmse_ratio": rmse / consts["best_rmse"]}


def main():
    say("=" * 100)
    say("CHAT NUMERIC TARGET - IS THERE LINEAR SIGNAL IN THE WINDOW'S NUMBERS? (closed form, held-out split)")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {D_DIR}")
    say("predictions first: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt ADDENDUM 4 (P43l, P43m)")
    say("=" * 100)
    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    Xtr, Ytr = chat_data.make_windows(ids, bounds, split, W, which=0,
                                      max_windows=TRAIN_WINDOWS, seed=0)
    Xva, Yva = chat_data.make_windows(ids, bounds, split, W, which=1,
                                      max_windows=VAL_WINDOWS, seed=0)
    id_to_token = {i: t for i, t in enumerate(vocab)}
    # magnitude_channel() loops over np.unique(ids) and compares the whole array per token id, which is
    # O(vocab x N) - on 200,000 windows that is minutes of pure indexing. The lookup below is the same
    # arithmetic (value/scale, mask where the token is written as a number) done in one gather, and it is
    # CHECKED against magnitude_channel on a 2,000-window sample before it is trusted.
    from lens.numeric import _parse_decimal
    vvals = np.zeros(len(vocab), dtype=np.float32)
    vmask = np.zeros(len(vocab), dtype=np.float32)
    for i, tok in enumerate(vocab):
        v = _parse_decimal(tok)
        if v is not None:
            vvals[i], vmask[i] = v, 1.0

    def fast_mag(X):
        return vvals[np.asarray(X)] / SCALE, vmask[np.asarray(X)]

    chk_mag, chk_num = magnitude_channel(Xtr[:2000], id_to_token, scale=SCALE)
    f_mag, f_num = fast_mag(Xtr[:2000])
    assert np.array_equal(np.asarray(chk_mag), f_mag) and np.array_equal(np.asarray(chk_num), f_num), \
        "the fast magnitude lookup disagrees with magnitude_channel - do not trust this run"
    say("   magnitude lookup checked against lens.numeric.magnitude_channel on 2,000 windows: identical")
    mag_tr, _ = fast_mag(Xtr)
    mag_va, _ = fast_mag(Xva)
    raw_tr, m_tr = numeric_targets(vocab, Ytr, scale=1.0)
    raw_va, m_va = numeric_targets(vocab, Yva, scale=1.0)
    tr, va = m_tr > 0.5, m_va > 0.5
    Qtr = np.concatenate([np.asarray(mag_tr[tr], dtype=np.float64), np.ones((int(tr.sum()), 1))], axis=1)
    Qva = np.concatenate([np.asarray(mag_va[va], dtype=np.float64), np.ones((int(va.sum()), 1))], axis=1)
    ytr, yva = np.asarray(raw_tr[tr], dtype=np.float64), np.asarray(raw_va[va], dtype=np.float64)
    say("")
    say(f"windows: train {len(Ytr):,} -> {len(ytr):,} with a numeric target   "
        f"val {len(Yva):,} -> {len(yva):,}")
    say(f"features: the {W} magnitudes of the window (number/100) + a bias   targets in ORIGINAL units")

    consts = {"zero": 0.0, "train_mean": float(ytr.mean()), "train_median": float(np.median(ytr))}
    say("")
    say("BASELINES on the held-out masked rows (original units)")
    err = {}
    for k, c in consts.items():
        err[k] = (float(np.abs(yva - c).mean()), float(np.sqrt(((yva - c) ** 2).mean())))
        say(f"   {k:26s} constant {c:>9.4f}  MAE {err[k][0]:>10.4f}  RMSE {err[k][1]:>10.4f}")
    best_mae = min(err, key=lambda k: err[k][0])
    best_rmse = min(err, key=lambda k: err[k][1])
    consts["best_mae"], consts["best_rmse"] = err[best_mae][0], err[best_rmse][1]
    say(f"   best for MAE: {best_mae}   best for RMSE: {best_rmse}")

    rng = np.random.default_rng(0)
    lam, cverr = cv_lambda(Qtr, ytr, LAMS, FOLDS, rng)
    say("")
    say(f"RIDGE: lambda {lam:g} chosen by {FOLDS}-fold CV ON THE TRAIN ROWS ONLY (train CV MSE {cverr:.4g})")
    w = ridge_fit(Qtr, ytr, lam)
    res = {}
    res["ridge"] = report("ridge (window numbers)", Qva @ w, yva, consts)
    wsh = ridge_fit(Qtr, ytr[rng.permutation(len(ytr))], lam)
    res["shuffled"] = report("shuffled targets (control)", Qva @ wsh, yva, consts)
    # the simplest real predictor of a next number: the last number written in the window
    last = np.zeros(len(yva))
    for i in range(len(yva)):
        nz = np.nonzero(np.asarray(mag_va[va][i]))[0]
        last[i] = mag_va[va][i][nz[-1]] * SCALE if len(nz) else 0.0
    res["persist"] = report("persist (last number seen)", last, yva, consts)
    exact = float(np.mean(np.isclose(last, yva)))
    say(f"   (the last number in the window IS the next number on {100.0*exact:.2f}% of held-out rows)")

    say("")
    say("VERDICTS (bars declared in ADDENDUM 4 before this ran; a gate is PASS or FAIL)")
    p43l = res["ridge"]["rmse_ratio"] <= 0.90
    p43m = res["shuffled"]["rmse_ratio"] >= 0.98
    say(f"   P43l ridge RMSE {res['ridge']['rmse']:.4f} <= 0.90 x {consts['best_rmse']:.4f} = "
        f"{0.90*consts['best_rmse']:.4f}   (ratio {res['ridge']['rmse_ratio']:.3f})   "
        f"-> {'PASS' if p43l else 'FAIL'}")
    say(f"   P43m shuffled RMSE ratio {res['shuffled']['rmse_ratio']:.3f} >= 0.98   "
        f"-> {'PASS' if p43m else 'FAIL'}")
    say("")
    say("NOT CLAIMED: this says nothing about chat quality, and nothing about the arithmetic path (which is")
    say("unchanged and still PASSes its own gate). It measures ONE thing: whether a linear readout of the")
    say("numbers in the window carries held-out signal about the quantity of the next token.")

    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"when": datetime.datetime.now().isoformat(), "corpus": D_DIR, "window": W, "scale": SCALE,
               "n_train_masked": int(tr.sum()), "n_val_masked": int(va.sum()),
               "constants": consts, "ridge_lambda": lam, "train_cv_mse": cverr,
               "results": res, "verdicts": {"P43l": bool(p43l), "P43m": bool(p43m)},
               "persist_exact_rate": exact}, open(JSON_OUT, "w"), indent=1)
    print(f"\nappended to {OUT}\njson -> {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
