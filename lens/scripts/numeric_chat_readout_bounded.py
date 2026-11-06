#!/usr/bin/env python3
"""numeric_chat_readout_bounded.py - PLAN item (d3b): does the BOUNDED representation carry held-out signal
about the next token's quantity, where the unbounded one carried none?

Predictions written BEFORE this ran: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44f, P44g, P44h).
Output: lens/runs/NUMERIC-CHAT-READOUT-BOUNDED.txt + .json. Never overwrites the stored run.

WHY THIS IS THE DECIDING MEASUREMENT. The stored run (lens/runs/NUMERIC-CHAT-READOUT.txt, P43l FAIL) measured
that a linear readout of the window's raw numbers carries NO held-out signal about the next token's quantity
(ridge RMSE 378.43 against the best constant's 362.63 = 1.044x). Bounding the representation changes the
FEATURES, so the same closed-form question must be asked again with the bounded features before any GPU run of
the numeric term over text is justified. A ridge fit needs no optimiser and no step size, so a FAIL cannot be
blamed on a learning rate - the whole reason this question is asked in closed form.

BOTH ARMS RUN IN THE SAME PROCESS ON THE SAME WINDOWS, and BOTH ARE SCORED IN ORIGINAL UNITS, each through its
own declared inverse:
    raw      features v/100   target v         readout: nothing to invert (the target is already original)
    bounded  features v/d     target v/d       readout: multiply by d, the window's own divisor
Scoring one arm in its own units against the other arm's constants would be a FALSE-PASS TRAP (a small
bounded-space RMSE divided by a large original-unit constant reads as a win) - it is avoided by construction
here, and the compared values are printed beside every verdict.
The raw arm must reproduce the stored ratio 1.044 (P44h); if it does not, the two runs are not comparable.

Usage: python3 lens/scripts/numeric_chat_readout_bounded.py
"""
import datetime
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lens", "scripts"))
os.chdir(ROOT)
D_DIR = os.environ.get("CHAT_DATA_DIR") or os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D_DIR

import chat_data                                                                      # noqa: E402
import numeric_chat_readout as ncr                                                    # noqa: E402  (ridge_fit, cv_lambda)
from lens.numeric import magnitude_channel, numeric_targets, window_divisor, _parse_decimal   # noqa: E402

W = 32
SCALE = 100.0
TRAIN_WINDOWS = int(os.environ.get("NUMERIC_CHAT_TRAIN", "200000"))
VAL_WINDOWS = int(os.environ.get("NUMERIC_CHAT_VAL", "50000"))
OUT = os.path.join(ROOT, "lens", "runs", "NUMERIC-CHAT-READOUT-BOUNDED.txt")
JSON_OUT = os.path.join(ROOT, "lens", "runs", "numeric_chat_readout_bounded.json")
lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def report(name, pred, tgt, consts):
    """MAE/RMSE in ORIGINAL units, with the ratios printed beside them so the verdict can be checked."""
    mae = float(np.abs(pred - tgt).mean())
    rmse = float(np.sqrt(((pred - tgt) ** 2).mean()))
    say(f"   {name:34s} MAE {mae:>10.4f}  RMSE {rmse:>10.4f}   "
        f"MAE {mae/consts['best_mae']:>6.3f}x  RMSE {rmse/consts['best_rmse']:>6.3f}x")
    return {"mae": mae, "rmse": rmse, "mae_ratio": mae / consts["best_mae"],
            "rmse_ratio": rmse / consts["best_rmse"]}


def main():
    say("=" * 100)
    say("CHAT NUMERIC TARGET WITH THE BOUNDED REPRESENTATION - held-out signal or not? (closed form)")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {D_DIR}")
    say("predictions first: lens/runs/NUMERIC-BOUNDED-PREDICTIONS.txt (P44f, P44g, P44h)")
    say("=" * 100)
    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    Xtr, Ytr = chat_data.make_windows(ids, bounds, split, W, which=0, max_windows=TRAIN_WINDOWS, seed=0)
    Xva, Yva = chat_data.make_windows(ids, bounds, split, W, which=1, max_windows=VAL_WINDOWS, seed=0)
    id_to_token = {i: t for i, t in enumerate(vocab)}

    vvals = np.zeros(len(vocab), dtype=np.float32)
    vmask = np.zeros(len(vocab), dtype=np.float32)
    for i, tok in enumerate(vocab):
        v = _parse_decimal(tok)
        if v is not None:
            vvals[i], vmask[i] = v, 1.0

    chk_mag, chk_num = magnitude_channel(Xtr[:2000], id_to_token, scale=1.0)
    f_raw, f_num = vvals[np.asarray(Xtr[:2000])], vmask[np.asarray(Xtr[:2000])]
    assert np.array_equal(np.asarray(chk_mag), f_raw) and np.array_equal(np.asarray(chk_num), f_num), \
        "the fast magnitude lookup disagrees with magnitude_channel - do not trust this run"
    say("   fast lookup checked against lens.numeric.magnitude_channel on 2,000 windows: identical")

    def load(X, Y):
        raw = vvals[np.asarray(X)]
        div = window_divisor(raw)
        tgt_raw, tgt_mask = numeric_targets(vocab, Y, scale=1.0)     # Y is (N,): the single next token
        return raw, div, tgt_raw, tgt_mask

    raw_tr, d_tr, tgt_tr, m_tr = load(Xtr, Ytr)
    raw_va, d_va, tgt_va, m_va = load(Xva, Yva)
    tr, va = m_tr > 0.5, m_va > 0.5
    yva_orig = np.asarray(tgt_va[va], dtype=np.float64)
    dv = d_va.reshape(-1)[va].astype(np.float64)
    say(f"windows: train {len(Ytr):,} -> {int(tr.sum()):,} with a numeric target   "
        f"val {len(Yva):,} -> {int(va.sum()):,}")
    say(f"   stored run, same protocol, must match: 200,000 -> 9,096 and 50,000 -> 2,031   "
        f"matches: {int(tr.sum()) == 9096 and int(va.sum()) == 2031}")

    consts = {"zero": 0.0, "train_mean": float(tgt_tr[tr].mean()),
              "train_median": float(np.median(tgt_tr[tr]))}
    err = {}
    say("")
    say("BASELINES on the held-out masked rows (ORIGINAL units; both arms are scored against these)")
    for k, c in consts.items():
        err[k] = (float(np.abs(yva_orig - c).mean()), float(np.sqrt(((yva_orig - c) ** 2).mean())))
        say(f"   {k:26s} constant {c:>9.4f}  MAE {err[k][0]:>10.4f}  RMSE {err[k][1]:>10.4f}")
    consts["best_mae"] = err[min(err, key=lambda k: err[k][0])][0]
    consts["best_rmse"] = err[min(err, key=lambda k: err[k][1])][1]
    say(f"   best for MAE {consts['best_mae']:.4f}   best for RMSE {consts['best_rmse']:.4f}")

    def fit(Qtr, ytr, tag):
        rng = np.random.default_rng(0)
        lam, cverr = ncr.cv_lambda(Qtr, ytr, ncr.LAMS, ncr.FOLDS, rng)
        w = ncr.ridge_fit(Qtr, ytr, lam)
        wsh = ncr.ridge_fit(Qtr, ytr[rng.permutation(len(ytr))], lam)
        say("")
        say(f"RIDGE [{tag}]: lambda {lam:g} by {ncr.FOLDS}-fold CV ON TRAIN ONLY (train CV MSE {cverr:.4g})")
        return w, wsh, lam

    # ---------------- RAW arm: the stored protocol (features v/100, target in original units)
    Qtr_raw = np.concatenate([np.asarray(raw_tr[tr] / np.float32(SCALE), dtype=np.float64),
                              np.ones((int(tr.sum()), 1))], axis=1)
    Qva_raw = np.concatenate([np.asarray(raw_va[va] / np.float32(SCALE), dtype=np.float64),
                              np.ones((int(va.sum()), 1))], axis=1)
    w_raw, wsh_raw, lam_raw = fit(Qtr_raw, np.asarray(tgt_tr[tr], dtype=np.float64), "raw, v/100")
    res_raw = {"ridge": report("ridge (raw, original units)", Qva_raw @ w_raw, yva_orig, consts),
               "shuffled": report("shuffled targets (raw)", Qva_raw @ wsh_raw, yva_orig, consts),
               "lambda": lam_raw}

    # ---------------- BOUNDED arm: features v/d, target v/d, readout x d  (scored in ORIGINAL units)
    Qtr_b = np.concatenate([np.asarray((raw_tr / d_tr)[tr], dtype=np.float64),
                            np.ones((int(tr.sum()), 1))], axis=1)
    Qva_b = np.concatenate([np.asarray((raw_va / d_va)[va], dtype=np.float64),
                            np.ones((int(va.sum()), 1))], axis=1)
    ytr_b = np.asarray(tgt_tr[tr], dtype=np.float64) / d_tr.reshape(-1)[tr].astype(np.float64)
    w_b, wsh_b, lam_b = fit(Qtr_b, ytr_b, "bounded, v/d")
    res_b = {"ridge": report("bounded + inverse x d", (Qva_b @ w_b) * dv, yva_orig, consts),
             "shuffled": report("shuffled targets (bounded + inv)", (Qva_b @ wsh_b) * dv, yva_orig, consts),
             "lambda": lam_b,
             "bounded_space_rmse": float(np.sqrt(np.mean(((Qva_b @ w_b) - yva_orig / dv) ** 2)))}
    say("")
    say(f"   feature range, raw     min {float(np.min(Qtr_raw[:, :-1])):>10.3f}  max "
        f"{float(np.max(Qtr_raw[:, :-1])):>10.3f}")
    say(f"   feature range, bounded min {float(np.min(Qtr_b[:, :-1])):>10.3f}  max "
        f"{float(np.max(Qtr_b[:, :-1])):>10.3f}   (bounded by 1 by construction)")
    say(f"   target  range, raw     min {float(np.min(np.asarray(tgt_tr[tr], dtype=np.float64))):>10.3f}  max "
        f"{float(np.max(np.asarray(tgt_tr[tr], dtype=np.float64))):>10.3f}")
    say(f"   target  range, bounded min {float(np.min(ytr_b)):>10.3f}  max {float(np.max(ytr_b)):>10.3f}   "
        f"(NOT bounded by 1: the next token lies OUTSIDE the window - P44e FAIL)")
    say(f"   persist baseline for reference (stored run): MAE 100.53, RMSE 412.36")

    # ---------------- ARM 3 (ADDENDUM 2): BOUNDED CHANNEL with the RAW target, NO inverse.
    # Exactly one change from the stored raw run: features v/100 -> v/d. The target is untouched, so nothing
    # amplifies the error and the control is meaningful again.
    Qtr_c = np.concatenate([np.asarray((raw_tr / d_tr)[tr], dtype=np.float64),
                            np.ones((int(tr.sum()), 1))], axis=1)
    Qva_c = np.concatenate([np.asarray((raw_va / d_va)[va], dtype=np.float64),
                            np.ones((int(va.sum()), 1))], axis=1)
    w_c, wsh_c, lam_c = fit(Qtr_c, np.asarray(tgt_tr[tr], dtype=np.float64), "bounded channel, RAW target")
    res_c = {"ridge": report("bounded channel, raw target", Qva_c @ w_c, yva_orig, consts),
             "shuffled": report("shuffled targets (channel arm)", Qva_c @ wsh_c, yva_orig, consts),
             "lambda": lam_c}

    say("")
    say("VERDICTS (bars declared in NUMERIC-BOUNDED-PREDICTIONS.txt before this ran)")
    p44f = res_b["ridge"]["rmse_ratio"] <= 0.90
    p44g = res_b["shuffled"]["rmse_ratio"] >= 0.98
    p44h = abs(res_raw["ridge"]["rmse_ratio"] - 1.044) <= 0.005
    say(f"   P44f bounded ridge RMSE {res_b['ridge']['rmse']:.4f} <= 0.90 x {consts['best_rmse']:.4f} = "
        f"{0.90*consts['best_rmse']:.4f}   (ratio {res_b['ridge']['rmse_ratio']:.3f})   "
        f"-> {'PASS' if p44f else 'FAIL'}")
    say(f"   P44g bounded shuffled RMSE ratio {res_b['shuffled']['rmse_ratio']:.3f} >= 0.98   "
        f"-> {'PASS' if p44g else 'FAIL'}")
    say(f"   P44h raw arm in this process reproduces the stored ratio 1.044 +-0.005: measured "
        f"{res_raw['ridge']['rmse_ratio']:.3f}   -> {'PASS' if p44h else 'FAIL'}")
    p44j = res_c["ridge"]["rmse_ratio"] <= 0.90
    p44k = 0.98 <= res_c["shuffled"]["rmse_ratio"] <= 1.02
    say(f"   P44j bounded CHANNEL, raw target RMSE {res_c['ridge']['rmse']:.4f} <= 0.90 x "
        f"{consts['best_rmse']:.4f} = {0.90*consts['best_rmse']:.4f}   (ratio "
        f"{res_c['ridge']['rmse_ratio']:.3f})   -> {'PASS' if p44j else 'FAIL'}")
    say(f"   P44k channel-arm control lands AT the constant (0.98..1.02): measured "
        f"{res_c['shuffled']['rmse_ratio']:.3f}   -> {'PASS' if p44k else 'FAIL'}")
    say(f"   P44g is recorded as PASSED AS WORDED AND DEFECTIVE (control {res_b['shuffled']['rmse_ratio']:.1f}x "
        f"ABOVE the constant, not at it) - see the predictions file ADDENDUM 2")
    say("")
    say("   NOT CLAIMED: nothing about chat quality, and nothing about the arithmetic path (its own gate is in")
    say("   lens/runs/NUMERIC-BOUNDED.txt). The numeric term stays OFF over chat text unless P44f passes.")
    say("=== DONE " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " ===")

    json.dump({"when": datetime.datetime.now().isoformat(), "corpus": D_DIR, "window": W,
               "n_train_masked": int(tr.sum()), "n_val_masked": int(va.sum()),
               "constants": consts, "raw": res_raw, "bounded": res_b, "channel_only": res_c,
               "verdicts": {"P44f": bool(p44f), "P44g": bool(p44g), "P44h": bool(p44h),
                            "P44j": bool(p44j), "P44k": bool(p44k)}},
              open(JSON_OUT, "w"), indent=1)
    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nappended to {OUT}\njson -> {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
