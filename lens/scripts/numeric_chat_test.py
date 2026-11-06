#!/usr/bin/env python3
"""numeric_chat_test.py - PLAN.txt item (d3): give the chat corpus a numeric target, and MEASURE it.

Predictions written BEFORE this ran: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt (P43a-h, including the dated
addendum that declares the MSE bar beside the MAE bar). Output: lens/runs/NUMERIC-CHAT.txt (appended,
dated). Raw numbers: lens/runs/numeric_chat.json.

THE TARGET IS NOT INVENTED. For every window the token classifier is already trained on the next TOKEN;
`numeric_targets` gives the SAME target AS A NUMBER (its value, /scale). So the question this test asks is
narrow and honest: can the quantity head learn "how much" from the same 32-token window, on the held-out
split, better than the best constant predictor?

ARMS (all on the SAME windows, the SAME seed, the SAME held-out split, only the numeric term differing)
  A   the model, numeric term ON, masked to the rows that have a numeric target
  B0  the same model, numeric term OFF - nothing else in the loss touches the quantity head, so it stays
      at its random initialisation. If B0 were already good, arm A's result would not be the term's doing.
  B1  the same model, numeric term ON but the TARGETS SHUFFLED across windows. A test that passes when
      the targets are random proves nothing, so this arm is the test's ability to fail.
  C   the FIXED part of the magnitude channel (the numbers written in the window) -> one linear head, no
      embedding path at all. This says how much of the signal is just "the numbers already in the window".

BASELINES, on the SAME held-out numeric windows: predict 0, predict the TRAIN mean, predict the TRAIN
median. MAE and RMSE are both reported because the loss is MSE and the target is heavy-tailed.

Usage: python3 lens/scripts/numeric_chat_test.py
Env: CHAT_DATA_DIR (default data/chat/prepared_mix_ctrl_honest), NUMERIC_CHAT_SEEDS, NUMERIC_CHAT_TRAIN
"""
import datetime
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
os.chdir(PROJ)

import chat_data                                                     # noqa: E402
import lens.tensor as T                                              # noqa: E402
from lens import backend as B                                        # noqa: E402
from lens.concepts import Concept                                    # noqa: E402
from lens.model import LensNet                                       # noqa: E402
from lens.numeric import magnitude_channel, numeric_targets          # noqa: E402

W = 32
D = 128
SCALE = 100.0
BATCH = 256
EPOCHS = 2
LR = 0.2
LAM_NUM = 1.0
TRAIN_WINDOWS = int(os.environ.get("NUMERIC_CHAT_TRAIN", "40000"))
VAL_WINDOWS = 30000
SEEDS = [int(x) for x in os.environ.get("NUMERIC_CHAT_SEEDS", "0,1,2").split(",")]
OUT = os.path.join(PROJ, "lens", "runs", "NUMERIC-CHAT.txt")
JSON_OUT = os.path.join(PROJ, "lens", "runs", "numeric_chat.json")
D_DIR = os.environ.get("CHAT_DATA_DIR") or os.path.join(PROJ, "data", "chat", "prepared_mix_ctrl_honest")
os.environ["CHAT_DATA_DIR"] = D_DIR

lines = []


def say(m=""):
    print(m, flush=True)
    lines.append(m)


def concepts_for(vocab):
    """A small, explicit, named concept set. lam_concept is 0 below, so these are the BOTTLENECK the
    architecture always has, not a trained auxiliary target: this test measures the numeric term."""
    vmap = {w: i for i, w in enumerate(vocab)}
    num = [i for i, w in enumerate(vocab) if w.replace(".", "").isdigit()]
    words = {"is_a_number": num,
             "mentions_a_person": [vmap[w] for w in ("he", "she", "they", "i", "you") if w in vmap],
             "is_a_question": [vmap["?"]] if "?" in vmap else []}
    return [Concept(n, ids) for n, ids in words.items() if ids]


def build_net(seed, concepts, vocab_size):
    return LensNet(vocab_size=vocab_size, concepts=concepts, d=D, window=W, seed=seed,
                   lam_concept=0.0, lam_act=0.0, lam_w=0.0, numeric=True)


def val_metrics(net, Xva, mag_va, keep_va, ymask_va, raw_va):
    """Held-out predictions of the quantity head, on the rows that HAVE a numeric target.

    Returns the predictions in the loss's own (standardised) units and the targets in the ORIGINAL units,
    so the caller converts one way only and the comparison is always against the number the token denotes.
    """
    preds = []
    for s in range(0, len(Xva), BATCH):
        idx = np.arange(s, min(s + BATCH, len(Xva)))
        out = net.forward_arrays(Xva[idx], mag=mag_va[idx], magmask=keep_va[idx])
        preds.append(np.asarray(out["num"], dtype=np.float64).reshape(-1))
    pred = np.concatenate(preds)
    m = ymask_va > 0.5
    return pred[m], raw_va[m]


def measure_numeric_lrs(net, Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr, base_lr):
    """Per-tensor step sizes from ONE real batch: equalise the FIRST step's size across tensors.

    Declared before the run in lens/runs/NUMERIC-CHAT-PREDICTIONS.txt ADDENDUM 3, with the measurement in
    lens/runs/NUMERIC-CHAT-GRAD-SCALE.txt. Why it is needed: mag.Wm's gradient is a sum over N*W rows per
    batch while the language tensors' gradients are sums over N, so one lr cannot serve both - at lr 0.2
    the first step on mag.Wm is ~0.118 in a tensor whose entries are ~0.1 and the run is nan by step 10.
    """
    idx = np.arange(min(BATCH, len(Xtr)))
    for p in net.parameters():
        p.grad = None
    total, _ = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=False,
                        mag=mag_tr[idx], magmask=keep_tr[idx],
                        ynum=ynum_tr[idx], ynum_mask=ymask_tr[idx], lam_num=LAM_NUM)
    total.backward()
    norms, lang = {}, []
    for p in net.parameters():
        if p.grad is None:
            continue
        n = float(np.linalg.norm(np.asarray(B.to_cpu(p.grad), dtype=np.float64)))
        norms[p.name] = n
        if not p.name.startswith(("mag.", "num.")):
            lang.append(n)
    med = float(np.median(lang)) if lang else 1.0
    out = {}
    for p in net.parameters():
        if p.name in norms and p.name.startswith(("mag.", "num.")):
            out[p.name] = base_lr * med / max(norms[p.name], 1e-12)
    return out, norms, med


def train_full(seed, concepts, vocab_size, Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr,
               lam_num, shuffle_targets=False, lr_map=None):
    """Plain SGD with the shipped chat rule, CE + the masked numeric term."""
    net = build_net(seed, concepts, vocab_size)
    rng = np.random.default_rng(seed)
    y_tr = ynum_tr
    m_tr = ymask_tr
    if shuffle_targets:
        # shuffle the VALUES across windows but keep the mask where it was, so the same rows are trained
        # and only the pairing between window and target is destroyed
        y_tr = ynum_tr[rng.permutation(len(ynum_tr))]
    for ep in range(EPOCHS):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(Xtr), BATCH):
            idx = order[s:s + BATCH]
            for p in net.parameters():
                p.grad = None
            total, _parts = net.loss(Xtr[idx], Ytr[idx], use_concept_loss=False,
                                     mag=mag_tr[idx], magmask=keep_tr[idx],
                                     ynum=(y_tr[idx] if lam_num > 0 else None),
                                     ynum_mask=(m_tr[idx] if lam_num > 0 else None),
                                     lam_num=lam_num)
            total.backward()
            for p in net.parameters():
                if p.grad is not None:
                    lr_p = LR if lr_map is None else lr_map.get(p.name, LR)
                    p.data = p.data - lr_p * p.grad
    return net


def arm_c(mag_tr, keep_tr, ynum_tr, ymask_tr, mag_va, ymask_va, raw_va, seed):
    """The FIXED numbers in the window -> one linear head. No embedding, no learned readout."""
    from lens.numeric import NumericHead, num_mse
    rng = np.random.default_rng(seed)
    head = NumericHead(W, rng, name="num_c")
    m = ymask_tr > 0.5
    q = T.Tensor(B.to_dev(mag_tr[m]))
    yt = ynum_tr[m]
    for _ in range(3000):
        for p in head.parameters():
            p.grad = None
        num_mse(head(q), yt).backward()
        for p in head.parameters():
            if p.grad is not None:
                p.data = p.data - 0.05 * p.grad
    mv = ymask_va > 0.5
    pred = np.asarray(B.to_cpu(head(T.Tensor(B.to_dev(mag_va[mv]))).data), dtype=np.float64).reshape(-1)
    return pred, raw_va[mv]


def main():
    say("=" * 100)
    say("NUMERIC TARGET ON THE REAL CHAT CORPUS - can the quantity head learn what the next token QUANTIFIES?")
    say(f"run {datetime.datetime.now():%Y-%m-%d %H:%M:%S}   corpus {D_DIR}")
    say(f"predictions first: lens/runs/NUMERIC-CHAT-PREDICTIONS.txt (P43a-j, incl. 2 dated addenda)")
    say("=" * 100)
    d = chat_data.build()
    ids, vocab, bounds, split = d["ids"], d["vocab"], d["bounds"], d["split"]
    Xtr, Ytr = chat_data.make_windows(ids, bounds, split, W, which=0,
                                      max_windows=TRAIN_WINDOWS, seed=0)
    Xva, Yva = chat_data.make_windows(ids, bounds, split, W, which=1,
                                      max_windows=VAL_WINDOWS, seed=0)
    id_to_token = {i: t for i, t in enumerate(vocab)}
    mag_tr, numeric_tr = magnitude_channel(Xtr, id_to_token, scale=SCALE)
    keep_tr = 1.0 - numeric_tr
    raw_tr, ymask_tr = numeric_targets(vocab, Ytr, scale=1.0)      # RAW: the number the token denotes
    mag_va, numeric_va = magnitude_channel(Xva, id_to_token, scale=SCALE)
    keep_va = 1.0 - numeric_va
    raw_va, ymask_va = numeric_targets(vocab, Yva, scale=1.0)
    mva = ymask_va > 0.5
    # TARGET STANDARDISATION, TRAIN STATISTICS ONLY (addendum 2; the val numbers never touch this)
    tmean = float(raw_tr[ymask_tr > 0.5].mean())
    tstd = float(raw_tr[ymask_tr > 0.5].std())
    ynum_tr = ((raw_tr - tmean) / tstd).astype(np.float32)
    ynum_va = ((raw_va - tmean) / tstd).astype(np.float32)

    def to_raw(p):
        return np.asarray(p, dtype=np.float64) * tstd + tmean

    say("")
    say(f"windows: train {len(Xtr):,} ({int(ymask_tr.sum()):,} with a numeric target, "
        f"{100.0*ymask_tr.mean():.2f}%)   val {len(Xva):,} ({int(ymask_va.sum()):,}, "
        f"{100.0*ymask_va.mean():.2f}%)")
    say(f"window {W}  d {D}  magnitude scale {SCALE:g}  batch {BATCH}  epochs {EPOCHS}  lr {LR}  "
        f"lam_num {LAM_NUM}  seeds {SEEDS}")
    say(f"target standardisation (TRAIN only): mean {tmean:.4f}  std {tstd:.4f}  "
        f"(raw units: the number the token denotes)")
    say(f"finite check: magnitudes {bool(np.isfinite(mag_tr).all() and np.isfinite(mag_va).all())}  "
        f"targets {bool(np.isfinite(ynum_tr).all() and np.isfinite(ynum_va).all())}")

    # ---- baselines on the SAME held-out numeric windows, in the ORIGINAL units
    v = raw_va[mva]
    base = {"predict_zero": 0.0, "predict_train_mean": tmean,
            "predict_train_median": float(np.median(raw_tr[ymask_tr > 0.5]))}
    say("")
    say("BASELINES on the held-out numeric windows, in the ORIGINAL units (the number the token denotes)")
    for k, c in base.items():
        say(f"   {k:22s} constant {c:>9.4f}   MAE {np.abs(v-c).mean():>9.4f}   "
            f"RMSE {np.sqrt(((v-c)**2).mean()):>9.4f}")
    best_mae_const = min(base, key=lambda k: np.abs(v - base[k]).mean())
    best_rmse_const = min(base, key=lambda k: np.sqrt(((v - base[k]) ** 2).mean()))
    b_mae = float(np.abs(v - base[best_mae_const]).mean())
    b_rmse = float(np.sqrt(((v - base[best_rmse_const]) ** 2).mean()))
    say(f"   best MAE constant: {best_mae_const} -> {b_mae:.4f}     "
        f"best RMSE constant: {best_rmse_const} -> {b_rmse:.4f}")

    concepts = concepts_for(vocab)
    say(f"   concept layer: {len(concepts)} named concepts (lam_concept 0 - the bottleneck, not a target)")

    # ---- per-tensor step sizes for the numeric path, MEASURED on one real batch (addendum 3)
    probe_net = build_net(0, concepts, len(vocab))
    lr_map, norms, med = measure_numeric_lrs(probe_net, Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr, LR)
    say("")
    say("STEP SIZES: language path keeps the shipped lr; each numeric tensor is matched to the language")
    say(f"tensors' first-step movement at lr {LR} (measurement: lens/runs/NUMERIC-CHAT-GRAD-SCALE.txt)")
    say(f"   median ||g|| over language tensors {med:.6g}")
    for k in sorted(norms):
        tag = f"  -> lr {lr_map[k]:.6g}" if k in lr_map else f"  -> lr {LR} (language path)"
        say(f"   {k:<12} ||g|| {norms[k]:>11.5g}{tag}")
    del probe_net

    rows = {"A": [], "B0": [], "B1": [], "C": []}
    for seed in SEEDS:
        t0 = time.perf_counter()
        net_a = train_full(seed, concepts, len(vocab), Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr,
                           LAM_NUM, shuffle_targets=False, lr_map=lr_map)
        pa, ta = val_metrics(net_a, Xva, mag_va, keep_va, ymask_va, raw_va)
        net_b0 = train_full(seed, concepts, len(vocab), Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr,
                            0.0, shuffle_targets=False, lr_map=lr_map)
        p0, _ = val_metrics(net_b0, Xva, mag_va, keep_va, ymask_va, raw_va)
        net_b1 = train_full(seed, concepts, len(vocab), Xtr, Ytr, mag_tr, keep_tr, ynum_tr, ymask_tr,
                            LAM_NUM, shuffle_targets=True, lr_map=lr_map)
        p1, _ = val_metrics(net_b1, Xva, mag_va, keep_va, ymask_va, raw_va)
        pc, tc = arm_c(mag_tr, keep_tr, ynum_tr, ymask_tr, mag_va, ymask_va, raw_va, seed)
        for key, (p, t) in (("A", (to_raw(pa), ta)), ("B0", (to_raw(p0), ta)),
                            ("B1", (to_raw(p1), ta)), ("C", (to_raw(pc), tc))):
            mae = float(np.abs(p - t).mean())
            rmse = float(np.sqrt(((p - t) ** 2).mean()))
            rows[key].append({"seed": seed, "mae": mae, "rmse": rmse,
                              "finite": bool(np.isfinite(p).all())})
        say("")
        say(f"  seed {seed} ({time.perf_counter()-t0:.0f}s)  n_held_out_numeric {len(ta):,}")
        for key in ("A", "B0", "B1", "C"):
            r = rows[key][-1]
            say(f"    {key:3s} MAE {r['mae']:>9.4f}  RMSE {r['rmse']:>9.4f}  "
                f"vs best MAE const {r['mae']/b_mae:>6.3f}x  vs best RMSE const {r['rmse']/b_rmse:>6.3f}x  "
                f"finite {r['finite']}")

    say("")
    say("-" * 100)
    mean = {k: {"mae": float(np.mean([r["mae"] for r in rows[k]])),
                "rmse": float(np.mean([r["rmse"] for r in rows[k]])),
                "finite": all(r["finite"] for r in rows[k])} for k in rows}
    say("MEANS OVER SEEDS")
    for k in ("A", "B0", "B1", "C"):
        say(f"   {k:3s} MAE {mean[k]['mae']:.5f}  RMSE {mean[k]['rmse']:.5f}  "
            f"MAE {mean[k]['mae']/b_mae:.3f}x  RMSE {mean[k]['rmse']/b_rmse:.3f}x  "
            f"all finite {mean[k]['finite']}")
    say("")
    say("VERDICTS, with the compared values beside every one (a gate is PASS or FAIL, never 'in spirit')")
    p43d = (mean["A"]["mae"] <= 0.90 * b_mae) and mean["A"]["finite"]
    p43g = (mean["A"]["rmse"] <= 0.90 * b_rmse) and mean["A"]["finite"]
    p43e = mean["B1"]["mae"] >= 0.98 * b_mae
    p43f = mean["B0"]["mae"] >= b_mae
    say(f"   P43d arm A MAE {mean['A']['mae']:.5f} <= 0.90 x {b_mae:.5f} = {0.90*b_mae:.5f}   "
        f"-> {'PASS' if p43d else 'FAIL'}")
    say(f"   P43g arm A RMSE {mean['A']['rmse']:.5f} <= 0.90 x {b_rmse:.5f} = {0.90*b_rmse:.5f}   "
        f"-> {'PASS' if p43g else 'FAIL'}")
    say(f"   P43e arm B1 (shuffled) MAE {mean['B1']['mae']:.5f} >= 0.98 x {b_mae:.5f} = {0.98*b_mae:.5f} "
        f"  -> {'PASS' if p43e else 'FAIL'}")
    say(f"   P43f arm B0 (term OFF) MAE {mean['B0']['mae']:.5f} >= baseline {b_mae:.5f}   "
        f"-> {'PASS' if p43f else 'FAIL'}")
    say(f"   arm C (the numbers already in the window, one linear head): MAE {mean['C']['mae']:.5f} "
        f"({mean['C']['mae']/b_mae:.3f}x), RMSE {mean['C']['rmse']:.5f} ({mean['C']['rmse']/b_rmse:.3f}x)"
        f"   [reported, no bar declared]")
    say("")
    say("NOT CLAIMED: no chat-quality claim of any kind. The metric here is the quantity of the next token")
    say("on the assistant/human text as it is; the honest masked accuracy numbers (27-28.5%) stand, and no")
    say("default or shipped recipe changed in this turn.")

    with open(OUT, "a") as f:
        f.write("\n".join(lines) + "\n")
    json.dump({"when": datetime.datetime.now().isoformat(), "corpus": D_DIR,
               "config": {"window": W, "d": D, "scale": SCALE, "batch": BATCH, "epochs": EPOCHS,
                          "lr": LR, "lam_num": LAM_NUM, "train_windows": TRAIN_WINDOWS,
                          "val_windows": VAL_WINDOWS, "seeds": SEEDS},
               "n_train_numeric": int(ymask_tr.sum()), "n_val_numeric": int(ymask_va.sum()),
               "baselines": base, "best_mae_const": [best_mae_const, b_mae],
               "best_rmse_const": [best_rmse_const, b_rmse],
               "arms": rows, "means": mean,
               "verdicts": {"P43d": p43d, "P43g": p43g, "P43e": p43e, "P43f": p43f}},
              open(JSON_OUT, "w"), indent=1)
    print(f"\nappended to {OUT}\njson -> {JSON_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
