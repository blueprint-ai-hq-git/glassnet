"""t3_audit.py — reviewer item (1): concept audit for the K=64 / lr=0.3 readable.

Trains the item-(3) best-vs-best readable config (K=64, lr=0.3, vocab 1501,
8-token window) for 3 seeds, then runs the STANDING readability meter on each:

  * auto_name_t2c  — position-aware, support-gated causal naming (words by
                     sum over position blocks of row_block_p @ emb_w, words
                     must appear in >=1% of the fired windows)
  * audit_t2c      — pass = residualized AUROC >= 0.55 AND logit
                     name-contribution >= 0.25 (residualized on window length,
                     so a neuron that only tracks length cannot pass)
  * lie probe      — a net trained with ONE concept column FROZEN (its gradient
                     zeroed, so it stays at init = pure random firing). The
                     audit MUST catch it, otherwise the meter proves nothing.

Validation only (naming uses train windows; every audited metric is on VAL).
The test split is never read.

Env: T3A_SEEDS (default 0,1,2), T3A_K (64), T3A_LR (0.3), T3A_BACKEND (cupy)
"""
import os
import sys
import json
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from glassnet.optim import SGD
from glassnet.tensor import set_backend, backend_name
from run_t2_iter003 import (TinyLM3, D, WIN, BATCH, auto_name_t2c, audit_t2c,
                            readable_params)
# step 4 idea #1: position-aware naming (auto_name_t2c credited every word for
# weight blocks at positions where it never appears). NAMER=old restores the
# old behaviour for A/B comparison.
try:
    from run_t2_iter003 import auto_name_t2c_pa
except ImportError:                                  # pragma: no cover
    auto_name_t2c_pa = None
NAMER = os.environ.get("T3A_NAMER", "pa")
MIN_SUPPORT = float(os.environ.get("T3A_MIN_SUPPORT", "0.01"))
LAM_A = float(os.environ.get("T2_LAM_A", "0.2"))
LAM_W = float(os.environ.get("T2_LAM_W", "1.0"))
LAM_CONCEPT = float(os.environ.get("T2_LAM_CONCEPT", "0.0"))
# number of supervised concept columns (the circularity control split)
N_SUP = int(os.environ.get("T3A_N_SUP", "0"))
# Meter-02 firing-rate floor (METER-RULE-02.md). 0.02 is the pre-registered
# primary floor; 0.01 and 0.05 are sensitivity settings. 0.0 reproduces Meter-01.
FIRING_FLOOR = float(os.environ.get("T3A_FIRING_FLOOR", "0.02"))
# Live lie probe: train normally until the probe column's val firing rate first
# reaches LIVE_TARGET, then freeze its gradients (owner condition 2).
LIVE_TARGET = float(os.environ.get("T3A_LIVE_TARGET", "0.02"))
# Which concept target set the model was trained with. MUST be part of both the
# checkpoint name and the output name, or a cluster run collides with single-word.
CONCEPT_SET = os.environ.get("T2_CONCEPT_SET", "words")


def namer():
    """Returns a naming function bound to the configured support gate."""
    base = auto_name_t2c_pa if (NAMER == "pa" and auto_name_t2c_pa is not None) \
        else auto_name_t2c

    def _f(net, d, Xtr, **kw):
        kw.setdefault("min_support", MIN_SUPPORT)
        return base(net, d, Xtr, **kw)
    return _f

VOCAB_CAP = 1500
K = int(os.environ.get("T3A_K", "64"))
LR = float(os.environ.get("T3A_LR", "0.3"))
SEEDS = [int(x) for x in os.environ.get("T3A_SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("T3A_EPOCHS", "96"))
MIN_EPOCH = int(os.environ.get("T3A_MIN_EPOCH", "30"))
PATIENCE = int(os.environ.get("T3A_PATIENCE", "15"))
LIE_COL = int(os.environ.get("T3A_LIE_COL", "2"))
BACKEND = os.environ.get("T3A_BACKEND", "cupy")
# If set, audit the SAVED best checkpoints from t3_models.py instead of
# retraining. Guarantees the audit runs on the exact model whose accuracy and
# perplexity are reported, and saves ~50 min per sweep.
LOAD_DIR = os.environ.get("T3A_LOAD_DIR", "")
set_backend(BACKEND)

RUNDIR = os.path.join(PROJ, "runs", "t3")
os.makedirs(RUNDIR, exist_ok=True)
OUT = os.path.join(RUNDIR,
                   f"audit_K{K}_lr{str(LR).replace('.','')}_{NAMER}"
                   f"{('_s' + str(MIN_SUPPORT).replace('.','')) if MIN_SUPPORT != 0.01 else ''}"
                   f"{('_lam' + str(LAM_A).replace('.','')) if LAM_A != 0.2 else ''}"
                   f"{('_w' + str(LAM_W).replace('.','')) if LAM_W != 1.0 else ''}"
                   f"{('_lc' + str(LAM_CONCEPT).replace('.','')) if LAM_CONCEPT != 0 else ''}"
                   # the concept SET must be in the name, or a cluster-target audit
                   # silently overwrites the single-word audit of the same shape
                   f"{('_' + CONCEPT_SET) if CONCEPT_SET != 'words' else ''}"
                   f"{('_ff' + str(FIRING_FLOOR).replace('.','')) if FIRING_FLOOR > 0 else ''}.json")
LOG = os.path.join(RUNDIR, "audit.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg, flush=True)


def to_cpu(net):
    """Move every parameter to host numpy so the audit's numpy maths runs."""
    set_backend("numpy")
    for p in net.parameters():
        if hasattr(p.data, "get"):
            p.data = p.data.get()


def eval_val(net, d, chunk=4096):
    X, y = d["Xva"], d["yva"]
    n = len(X)
    correct, ce_sum = 0, 0.0
    for i in range(0, n, chunk):
        xs, ys = X[i:i + chunk], y[i:i + chunk]
        z = net.forward(xs)["logits"].data
        if hasattr(z, "get"):
            z = z.get()
        z = np.asarray(z)
        z = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z)
        p = ex / ex.sum(axis=1, keepdims=True)
        correct += int((z.argmax(axis=1) == ys).sum())
        ce_sum += float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).sum())
    return correct / n * 100, ce_sum / n


def firing_rate(net, d, col, chunk=4096):
    """Fraction of VALIDATION windows where concept column `col` fires (> 0.5).

    This is the Meter-02 quantity (METER-RULE-02.md). Computed in chunks so it
    never allocates the full concept matrix.

    NOTE: this is called DURING training, when the backend may be cupy, so `h` can
    be a GPU array. Converting it with np.asarray() raises
    "Implicit conversion to a NumPy array is not allowed". Do the comparison on
    whichever backend is active and convert only the resulting SCALAR.
    """
    X = d["Xva"]
    n_fire = 0
    for i in range(0, len(X), chunk):
        h = net.forward(X[i:i + chunk])["concepts"].data
        s = (h[:, col] > 0.5).sum()
        if hasattr(s, "get"):        # cupy scalar -> host
            s = s.get()
        n_fire += int(s)
    return n_fire / float(len(X))


def train_readable(d, seed, freeze_col=None, tag="", live_freeze=False):
    V = d["V"]
    net = TinyLM3(V, K, D, WIN, rng=np.random.default_rng(seed))
    if freeze_col is not None and not live_freeze:
        net.freeze([freeze_col])
    opt = SGD(net.parameters(), lr=LR)
    X, y = d["Xtr"], d["ytr"]
    rngd = np.random.default_rng(seed * 131 + 7)
    best = {"val": -1.0, "epoch": 0, "ce": None}
    bad, stopped = 0, EPOCHS
    frozen_at = None
    for ep in range(EPOCHS):
        t0 = time.time()
        perm = rngd.permutation(len(X))
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss, _ = net.loss(X[idx], y[idx])
            loss.backward()
            if net.frozen:
                for k in net.frozen:
                    net.concept.W.grad[:, k] = 0.0
                    net.concept.b.grad[k] = 0.0
            opt.step()
        v, ce = eval_val(net, d)
        # LIVE PROBE (METER-RULE-02.md section 3): train the probe column NORMALLY
        # until it reaches a normal firing rate, and only then freeze its gradients.
        # The old probe froze from initialisation, so the column could die and then
        # pass the meter by being dead.
        if (live_freeze and freeze_col is not None and not net.frozen):
            fr = firing_rate(net, d, freeze_col)
            if fr >= LIVE_TARGET:
                net.freeze([freeze_col])
                frozen_at = ep + 1
                log(f"    [LIVE PROBE] col {freeze_col} firing {fr:.4f} >= "
                    f"{LIVE_TARGET:.4f} at epoch {ep+1} -> gradients FROZEN now")
        if v > best["val"]:
            best = {"val": v, "epoch": ep + 1, "ce": ce}
            bad = 0
        else:
            bad += 1
        log(f"    {tag} seed{seed} ep {ep+1:3d} val {v:6.2f}% ppl {np.exp(ce):8.2f} "
            f"(best {best['val']:.2f} @ {best['epoch']}, bad {bad}, {time.time()-t0:.0f}s)")
        if bad >= PATIENCE and (ep + 1) >= MIN_EPOCH:
            stopped = ep + 1
            log(f"    -> {tag} seed{seed} early stop at epoch {stopped}")
            break
    if live_freeze and freeze_col is not None and frozen_at is None:
        log(f"    !! LIVE PROBE col {freeze_col} never reached firing "
            f"{LIVE_TARGET:.4f} while unfrozen — probe INCONCLUSIVE")
    return net, best, stopped, frozen_at


def audit_one(net, d, label):
    """Name + audit. Returns the summary dict (numpy backend assumed)."""
    names = namer()(net, d, d["Xtr"])
    rows, n_pass = audit_t2c(net, names, d, d["Xva"], min_firing=FIRING_FLOOR)
    n_dead = sum(1 for r in rows if r.get("below_floor"))
    log(f"    [{label}] audit pass {n_pass}/{K} ({n_pass/K*100:.1f}%) "
        f"[Meter-02 firing floor {FIRING_FLOOR*100:.1f}%, excluded {n_dead}]")
    for r in rows:
        log(f"        {r['name'][:44]:44s} auroc={r['auroc']:.2f} "
            f"contrib={r['contrib']:+.3f} fire={r.get('firing', float('nan')):.3f} "
            f"{r['flag']}")
    return names, rows, n_pass


def main():
    d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
    V = d["V"]
    UNK = V - 1
    baseline = float((d["yva"] == UNK).mean() * 100)
    prior_train = float(np.bincount(d["ytr"], minlength=V).max() / len(d["ytr"]))

    log("=" * 78)
    log(f"ITEM (1) CONCEPT AUDIT — readable K={K} lr={LR}, vocab {V}, backend {backend_name()}")
    log(f"meter: resid AUROC >= 0.55 AND logit contrib >= 0.25 (the standing meter)")
    log(f"seeds {SEEDS} | stopping patience {PATIENCE}, max {EPOCHS}, no stop before {MIN_EPOCH}")
    log(f"val <unk> baseline {baseline:.2f}% | params {readable_params(V, D, K, WIN):,}")
    log("=" * 78)

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    result = done

    for seed in SEEDS:
        key = f"seed{seed}"
        if key in result and result[key].get("n_pass") is not None:
            log(f"  [{key}] already audited — skipped")
            continue
        log(f"  [{key}] training readable K={K} lr={LR}")
        t0 = time.time()
        if LOAD_DIR:
            ck = os.path.join(LOAD_DIR, f"readable_K{K}_lr{LR}"
                              f"{'' if LAM_A == 0.2 else '_lam' + str(LAM_A).replace('.','')}"
                              f"{'' if LAM_W == 1.0 else '_w' + str(LAM_W).replace('.','')}"
                              f"{'' if LAM_CONCEPT == 0 else '_lc' + str(LAM_CONCEPT).replace('.','')}"
                              f"{'' if CONCEPT_SET == 'words' else '_' + CONCEPT_SET}"
                              f"_seed{seed}.npz")
            if not os.path.exists(ck):
                log(f"    !! checkpoint missing: {ck} — skipping {key}")
                continue
            set_backend("numpy")
            net = TinyLM3(V, K, D, WIN, rng=np.random.default_rng(seed))
            z = np.load(ck)
            for p in net.parameters():
                if p.name in z:
                    p.data = np.array(z[p.name])
            v, ce = eval_val(net, d)
            best = {"val": v, "epoch": -1, "ce": ce}
            stopped = -1
            log(f"    loaded {ck} -> val {v:.2f}% ppl {np.exp(ce):.2f}")
        else:
            net, best, stopped = train_readable(d, seed, tag="readable")[0:3]
            to_cpu(net)
        names, rows, n_pass = audit_one(net, d, key)
        result[key] = {
            "val": best["val"], "val_ppl": float(np.exp(best["ce"])),
            "best_epoch": best["epoch"], "stopped_at": stopped,
            "n_pass": n_pass, "n_total": K, "pass_frac": n_pass / K,
            "names": names,
            "rows": [{"name": r["name"],
                      "auroc": (None if np.isnan(r["auroc"]) else r["auroc"]),
                      "contrib": (None if np.isnan(r["contrib"]) else r["contrib"]),
                      "pos_frac": r.get("pos_frac"),
                      "firing": r.get("firing"),
                      "below_floor": r.get("below_floor"),
                      "flag": r["flag"]} for r in rows],
            "secs": round(time.time() - t0)}
        json.dump(result, open(OUT, "w"), indent=1)
        log(f"    -> {key}: val {best['val']:.2f}% ppl {result[key]['val_ppl']:.2f} | "
            f"audit {n_pass}/{K} = {n_pass/K*100:.1f}% | {result[key]['secs']}s")

    # ---- lie probe: a LIVE column, frozen only once it fires normally ----
    # FORCE_PROBE re-runs the probe even when a verdict is already recorded, so its
    # checkpoint is saved for the Meter-03 re-audit (which needs the probe's
    # activations, not just its Meter-02 numbers).
    if ("lie" not in result or result["lie"].get("auroc") is None
            or os.environ.get("T3A_FORCE_PROBE") == "1"):
        set_backend(BACKEND)      # the audit left us on host numpy; train on GPU
        log("")
        log(f"  [LIE PROBE] METER-RULE-02: concept column {LIE_COL} trained "
            f"NORMALLY until its val firing rate reaches {LIVE_TARGET:.1%}, "
            f"THEN its gradients are frozen (it cannot pass by being dead)")
        t0 = time.time()
        probe, pbest, pstop, pfrozen = train_readable(
            d, 1, freeze_col=LIE_COL, tag="lieprobe", live_freeze=True)
        to_cpu(probe)
        # Save the probe's checkpoint so Meter-03 can audit it against a target it was
        # NEVER trained on (METER-RULE-03.md section 2). Previously the probe existed
        # only inside this process, so its activations could not be re-audited.
        probe_ck = os.path.join(
            RUNDIR, "ckpt",
            f"lieprobe_K{K}_lr{LR}_lc{str(LAM_CONCEPT).replace('.','')}"
            f"{'' if CONCEPT_SET == 'words' else '_' + CONCEPT_SET}"
            # T3_CKPT_TAG keeps a new milestone's probe from overwriting an earlier one's at
            # the same lr/lc/K. Without it, Target B's probe would have replaced A3's, which
            # A3's recorded verdict reads. Default empty = unchanged.
            f"{'' if not os.environ.get('T3_CKPT_TAG') else '_' + os.environ['T3_CKPT_TAG']}"
            f"_seed1.npz")
        probe.save(probe_ck)
        log(f"    probe checkpoint saved: {os.path.basename(probe_ck)}")
        pn = namer()(probe, d, d["Xtr"])
        pr, _ = audit_t2c(probe, pn, d, d["Xva"], min_firing=FIRING_FLOOR)
        lie = pr[LIE_COL]
        probe_fire = float(lie.get("firing", float("nan")))
        # The control's meaning: a neuron that cannot mean its name must NOT
        # pass the audit. Any non-clean outcome counts as caught -- including
        # NO-NAME, which the old test silently let through (a frozen neuron with
        # too few fired windows to be named has auroc = nan, and `nan < 0.55` is
        # False, so it was scored as MISSED). That hole is now closed.
        caught = bool(lie["flag"] != "-")
        # Did the AUROC/contribution clauses do the catching, or only the floor?
        caught_by_floor_only = bool(caught and lie.get("below_floor"))
        result["lie"] = {"col": LIE_COL, "name": pn[LIE_COL],
                         "auroc": (None if np.isnan(lie["auroc"]) else lie["auroc"]),
                         "contrib": (None if np.isnan(lie["contrib"]) else lie["contrib"]),
                         "flag": lie["flag"], "caught": caught,
                         "firing": probe_fire,
                         "frozen_at_epoch": pfrozen,
                         "caught_by_floor_only": caught_by_floor_only,
                         "live": bool(probe_fire >= LIVE_TARGET),
                         "meter": "Meter-02",
                         "firing_floor": FIRING_FLOOR,
                         "probe_val": pbest["val"], "secs": round(time.time() - t0)}
        json.dump(result, open(OUT, "w"), indent=1)
        log(f"    -> lie probe col {LIE_COL} ('{pn[LIE_COL][:40]}'): "
            f"auroc={lie['auroc']:.2f} contrib={lie['contrib']:+.3f} "
            f"firing={probe_fire:.4f} flag={lie['flag']} -> CAUGHT={caught}")
        log(f"       frozen at epoch {pfrozen}; live at audit time "
            f"{result['lie']['live']}; caught by the floor only: "
            f"{caught_by_floor_only}")

    # ---- summary ----
    log("")
    log("SUMMARY")
    ps = []
    for seed in SEEDS:
        k = f"seed{seed}"
        if k in result:
            ps.append(result[k]["n_pass"])
            log(f"  {k}: audit {result[k]['n_pass']}/{K} = "
                f"{result[k]['pass_frac']*100:.1f}%   val {result[k]['val']:.2f}%   "
                f"ppl {result[k]['val_ppl']:.2f}")
    if ps:
        log(f"  mean audit pass {np.mean(ps):.1f}/{K} = {np.mean(ps)/K*100:.1f}% "
            f"(min {min(ps)}, max {max(ps)})")
        log(f"  reviewer bar for the gate: >= 50% of concepts pass -> "
            f"{'MET' if np.mean(ps)/K >= 0.5 else 'NOT MET'} "
            f"(needs >= {int(np.ceil(K*0.5))} of {K})")

        # ---- SUPERVISED vs UNSUPERVISED split (the circularity control) ----
        # When the auxiliary concept loss is ON, columns 0..N_SUP-1 were TRAINED
        # to detect exactly the word the audit then asks about. A gain there is
        # circular. Columns N_SUP..K-1 were not supervised, so their pass
        # fraction is the honest check.
        if N_SUP > 0 and N_SUP < K:
            seeds = sorted(k for k in result if k.startswith("seed"))
            sup_p, uns_p, sup_c, uns_c, sup_a, uns_a = [], [], [], [], [], []
            for s in seeds:
                rows = result[s]["rows"]
                sup_p.append(sum(1 for i, r in enumerate(rows)
                                 if i < N_SUP and r["flag"] == "-"))
                uns_p.append(sum(1 for i, r in enumerate(rows)
                                 if i >= N_SUP and r["flag"] == "-"))
                sup_c += [r["contrib"] for i, r in enumerate(rows)
                          if i < N_SUP and r.get("contrib") is not None]
                uns_c += [r["contrib"] for i, r in enumerate(rows)
                          if i >= N_SUP and r.get("contrib") is not None]
                sup_a += [r["auroc"] for i, r in enumerate(rows)
                          if i < N_SUP and r.get("auroc") is not None]
                uns_a += [r["auroc"] for i, r in enumerate(rows)
                          if i >= N_SUP and r.get("auroc") is not None]
            n_uns = K - N_SUP
            log("  ---- CIRCULARITY CONTROL ----")
            log(f"  SUPERVISED   cols 0..{N_SUP-1}: pass {np.mean(sup_p):.1f}/{N_SUP} "
                f"= {np.mean(sup_p)/N_SUP*100:.1f}%   "
                f"contrib med {np.median(sup_c):+.3f}   "
                f"AUROC>=.55 {sum(1 for x in sup_a if x>=0.55)}/{len(sup_a)}")
            log(f"  UNSUPERVISED cols {N_SUP}..{K-1}: pass {np.mean(uns_p):.1f}/{n_uns} "
                f"= {np.mean(uns_p)/n_uns*100:.1f}%   "
                f"contrib med {np.median(uns_c):+.3f}   "
                f"AUROC>=.55 {sum(1 for x in uns_a if x>=0.55)}/{len(uns_a)}")
            result["split"] = {
                "n_sup": N_SUP, "n_uns": n_uns,
                "sup_pass_frac": float(np.mean(sup_p)/N_SUP),
                "uns_pass_frac": float(np.mean(uns_p)/n_uns),
                "sup_contrib_median": float(np.median(sup_c)),
                "uns_contrib_median": float(np.median(uns_c)),
                "sup_auroc_pass": int(sum(1 for x in sup_a if x >= 0.55)),
                "sup_auroc_n": len(sup_a),
                "uns_auroc_pass": int(sum(1 for x in uns_a if x >= 0.55)),
                "uns_auroc_n": len(uns_a),
                "lam_concept": LAM_CONCEPT,
            }
            json.dump(result, open(OUT, "w"), indent=1)
    if "lie" in result:
        log(f"  lie probe: {'CAUGHT' if result['lie']['caught'] else 'MISSED'} "
            f"(col {result['lie']['col']}, auroc {result['lie']['auroc']})")
    log(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
