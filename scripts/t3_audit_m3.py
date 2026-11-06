"""t3_audit_m3.py — METER-03: audit against PRE-REGISTERED targets, plus two controls.

Implements METER-RULE-03.md exactly. ONE change vs Meter-02: where the target comes
from. Fidelity (length-residualised AUROC), contribution (logit-gain formula), the
thresholds and the firing floor are byte-identical to Meter-02 -- only `ids` differs:
Meter-02 used the word-set parsed from the namer's name; Meter-03 uses the target that
was frozen BEFORE training.

Controls:
  * RANDOM-COLUMN: concept layer at random init, never trained (embedding/answer from
    the checkpoint), 10 draws -> the FREE PASS RATE. Trained results are reported as
    the EXCESS above it.
  * NAME-SWAP: every trained column audited against every OTHER column's target ->
    the rejection rate.

Re-scores existing checkpoints only. No model is retrained here.
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
import run_t2_iter003 as M

RUNDIR = os.path.join(PROJ, "runs", "t3")
K = int(os.environ.get("T3M_K", "32"))
N_SUP = int(os.environ.get("T3M_N_SUP", "16"))
WIN = 8
D = 64
VOCAB_CAP = 1500
AUROC_BAR, CONTRIB_BAR, FIRING_FLOOR = 0.55, 0.25, 0.02
N_RANDOM_DRAWS = 10
CKS = int(os.environ.get("T3M_CKS", "0"))   # set 1 to use the cluster checkpoints
A2 = int(os.environ.get("T3M_A2", "0"))     # set 1 for the TARGET-A2 run
A3 = int(os.environ.get("T3M_A3", "0"))     # set 1 for the TARGET-A3 run
B = int(os.environ.get("T3M_B", "0"))       # set 1 for the TARGET-B audit
CMODE = int(os.environ.get("T3M_C", "0"))   # set 1 for the TARGET-C audit


def auroc(score, target01):
    """Identical to run_t2_iter003._auroc."""
    pos = target01 > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load_targets(kind):
    """The pre-registered targets: (label, word_ids)."""
    if kind == "meaningful_ext":
        # TARGET-E.md: the extended set. ORDER MATTERS -- it must match the training order in
        # run_t2_iter003._load_concept_targets, or every column is audited against the wrong
        # concept. Target C's order differs, which is exactly the bug this guards against.
        j = json.load(open(os.path.join(RUNDIR, "meaningful_targets_ext.json")))
        return [(t["name"], list(t["ids"])) for t in j["qualifying"]]
    if kind == "meaningful":
        # TARGET-C.md: real ideas. The label is the concept's NAME; the ids are its word set.
        j = json.load(open(os.path.join(RUNDIR, "meaningful_targets.json")))
        return [(t["name"], list(t["ids"])) for t in j["targets"]]
    if kind in ("clusters", "clusters16", "clusters32"):
        fname = {"clusters32": "concept_clusters32.json",
                 "clusters16": "concept_clusters16.json"}.get(kind,
                                                              "concept_clusters.json")
        j = json.load(open(os.path.join(RUNDIR, fname)))
        return [(("_".join(c["words"][:6]) or "empty"), c["ids"]) for c in j["clusters"]]
    j = json.load(open(os.path.join(RUNDIR, "concept_targets.json")))
    return [(t["word"], [t["id"]]) for t in j["targets"]]


def audit_col(net, X, h, c, ids, L, A):
    """Meter-03 statistic for column c against target `ids`. Same maths as Meter-02."""
    vmap = {w: i for i, w in enumerate(DATA["vocab"])}
    ids = [i for i in ids if 0 <= i < len(DATA["vocab"])]
    firing = float((h[:, c] > 0.5).mean())
    if not ids:
        return dict(auroc=float("nan"), contrib=float("nan"), firing=firing,
                    pos_frac=0.0, flag="NO-TARGET")
    contains = np.stack([np.isin(X, i).any(axis=1) for i in ids]).any(0)
    coef, *_ = np.linalg.lstsq(A, h[:, c], rcond=None)
    resid = h[:, c] - A @ coef
    a = auroc(resid, contains)
    fired = h[:, c] > 0.5
    contrib = 0.0
    if fired.any():
        bk = net.concept.W.data[:, c].reshape(WIN, D)
        tot = 0.0
        for i in ids:
            wvec = bk @ net.emb.data[i]
            take = fired & np.isin(X, i).any(axis=1)
            if take.any():
                tot += float((wvec[None, :] * (X[take] == i)).sum())
        contrib = tot / float(fired.sum())
    ok = (a >= AUROC_BAR) and (contrib >= CONTRIB_BAR) and (firing >= FIRING_FLOOR)
    return dict(auroc=float(a), contrib=float(contrib), firing=firing,
                pos_frac=float(contains.mean()), flag="-" if ok else "BROKEN")


def build_net(seed):
    net = M.TinyLM3(len(DATA["vocab"]) + 1, K, D, WIN,
                    rng=np.random.default_rng(seed))
    return net


def load_ckpt(net, path):
    z = np.load(path)
    for p in net.parameters():
        if p.name in z:
            p.data = np.array(z[p.name])
    return net


def prep(net, X):
    h = np.asarray(net.forward(X)["concepts"].data)
    L = np.log1p(np.asarray([(x != DATA["V"] - 1).sum() for x in X]))
    A = np.vstack([np.ones_like(L), L]).T
    return h, L, A


def score_model(net, X, targets, label):
    """Diagonal (own target), unsupervised (fixed pairing + best-match), name-swap."""
    h, L, A = prep(net, X)
    diag, uns_fixed, uns_best, offdiag = [], [], [], []
    for c in range(K):
        if c < N_SUP:
            r = audit_col(net, X, h, c, targets[c][1], L, A)
            diag.append(r)
            for d in range(N_SUP):
                if d != c:
                    offdiag.append(audit_col(net, X, h, c, targets[d][1], L, A)["flag"])
        else:
            # A2: the k=32 set has a pre-registered target for EVERY column, so a column
            # past N_SUP is audited against ITS OWN target -- that is the hold-out check
            # (it was pre-registered but never supervised). For the k=16 sets there is no
            # matching target, so the fixed c mod N_SUP pairing is used instead.
            own_target = c if len(targets) >= K else c % N_SUP
            r = audit_col(net, X, h, c, targets[own_target][1], L, A)
            uns_fixed.append(r)
            cand = [audit_col(net, X, h, c, targets[d][1], L, A) for d in range(N_SUP)]
            best = max(cand, key=lambda z: (z["flag"] == "-",
                                            z["auroc"] if z["auroc"] == z["auroc"] else -1))
            uns_best.append(best)
    pf = lambda rs: (sum(1 for r in rs if r["flag"] == "-") / len(rs) * 100) if rs else 0.0
    return {
        "label": label,
        "sup_pass": pf(diag), "sup_n": len(diag),
        "sup_auroc_med": float(np.median([r["auroc"] for r in diag])),
        "uns_pass_fixed": pf(uns_fixed), "uns_pass_best": pf(uns_best),
        "uns_n": len(uns_fixed),
        "uns_auroc_med_fixed": float(np.median([r["auroc"] for r in uns_fixed])),
        "offdiag_reject": (sum(1 for f in offdiag if f != "-") / len(offdiag) * 100)
        if offdiag else 0.0,
        "offdiag_n": len(offdiag),
        "live_cols": int(sum(1 for c in range(K) if (h[:, c] > 0.5).mean() >= FIRING_FLOOR)),
        "rows_diag": [{"col": c, **r} for c, r in enumerate(diag)],
    }


print("=" * 104)
print("METER-03 — PRE-REGISTERED-TARGET AUDIT + RANDOM-COLUMN CONTROL + NAME-SWAP")
print("=" * 104)

DATA = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
X = np.asarray(DATA["Xva"])
KIND = ("meaningful" if CMODE else
        ("clusters16" if (A3 or B) else
         ("clusters32" if A2 else ("clusters" if CKS else "words"))))
targets = load_targets(KIND)
print(f"backend numpy | val windows {len(X)} | K={K} | {N_SUP} supervised columns")
print(f"targets: {KIND} ({len(targets)})")
print(f"bars: AUROC>={AUROC_BAR} contrib>={CONTRIB_BAR} firing>={FIRING_FLOOR}")
print(f"random control draws: {N_RANDOM_DRAWS}")

MODELS = []
if CMODE:
    # TARGET-C.md: the winner read from c_selection.json (the frozen rule chose it), so the
    # audit cannot hand-pick a convenient setting. C's checkpoints carry the _c tag AND the
    # concept-set name, so they cannot collide with A3's or B's.
    sel = json.load(open(os.path.join(RUNDIR, "c_selection.json")))
    lr, lc = sel["lr"], sel["lc"]
    print(f"C selected setting: lr={lr} lam_concept={lc} (from c_selection.json)")
    MODELS.append((f"c meaningful (6 taught / 10 untaught) lr{lr} lc{lc}",
                   f"readable_K16_lr{lr}_lc{str(lc).replace('.','')}"
                   f"_meaningful_c_seed{{s}}.npz"))
elif B:
    # TARGET-B.md: the winning readable setting, read from b_eval.json (which is produced by
    # t3_b_report.py applying the frozen rule to the full grid -- so the audited model cannot
    # be hand-picked). B's checkpoints carry the _b tag.
    sel = json.load(open(os.path.join(RUNDIR, "b_eval.json")))["readable_selected"]
    lr, lc = sel["lr"], sel["lc"]
    print(f"B selected setting: lr={lr} lam_concept={lc} (from b_eval.json)")
    MODELS.append((f"b clusters16 (10 taught / 6 untaught) lr{lr} lc{lc}",
                   f"readable_K16_lr{lr}_lc{str(lc).replace('.','')}"
                   f"_clusters16_b_seed{{s}}.npz"))
elif A3:
    # TARGET-A3.md: settings re-tuned for this target set by the frozen selection rule.
    # The winner is read from a3_selection.json, so the audit cannot pick a convenient one.
    sel = json.load(open(os.path.join(RUNDIR, "a3_selection.json")))
    lr, lc = sel["lr"], sel["lc"]
    print(f"A3 selected setting: lr={lr} lam_concept={lc} "
          f"(from a3_selection.json, {sel.get('rule','')})")
    MODELS.append((f"a3 clusters16 (10 taught / 6 untaught) lr{lr} lc{lc}",
                   f"readable_K16_lr{lr}_lc{str(lc).replace('.','')}"
                   f"_clusters16_seed{{s}}.npz"))
elif A2:
    # TARGET-A2.md: one setting (lc=0.5, lr=0.5, taken from the frozen cluster selection,
    # NOT re-tuned), 3 seeds. ONE entry with a {s} placeholder -- score_model loops the
    # seeds itself, so three entries would audit seed 0 three times.
    MODELS.append(("a2 clusters32 (24 taught / 8 held out)",
                   "readable_K32_lr0.5_lc05_clusters32_seed{s}.npz"))
elif CKS:
    # the cluster run's own frozen selection decides the lr per lam_concept
    sel = json.load(open(os.path.join(RUNDIR, "clu_selection.json")))
    for lc in ("0.1", "0.5", "1.0"):
        lr = sel[lc]["lr"]
        MODELS.append((f"clusters lc={lc}",
                       f"readable_K32_lr{lr}_lc{lc.replace('.','')}_clusters_seed{{s}}.npz"))
else:
    MODELS.append(("baseline lc=0 (unsupervised)", "readable_K32_lr1.0_seed{s}.npz"))
    # NOTE: the v1 selection is NOT uniform -- lc 0.1 and 0.5 selected lr 1.0, while
    # lc 1.0 selected lr 0.5. Reading it prevents loading the wrong checkpoint.
    sel = json.load(open(os.path.join(RUNDIR, "lamc_selection.json")))
    for lc in ("0.1", "0.5", "1.0"):
        lr = sel[lc]["lr"]
        MODELS.append((f"single words lc={lc}",
                       f"readable_K32_lr{lr}_lc{lc.replace('.','')}_seed{{s}}.npz"))

out = {"rule": "METER-RULE-03.md", "kind": KIND, "bars": dict(
    auroc=AUROC_BAR, contrib=CONTRIB_BAR, firing=FIRING_FLOOR), "models": {}}

for label, pat in MODELS:
    print(f"\n--- {label}")
    set_backend("numpy")          # this audit is CPU-only; never touch the GPU
    per_seed = []
    for seed in (0, 1, 2):
        path = os.path.join(RUNDIR, "ckpt", pat.format(s=seed))
        if not os.path.exists(path):
            print(f"    seed{seed}: MISSING {os.path.basename(path)}")
            continue
        net = load_ckpt(build_net(seed), path)
        r = score_model(net, X, targets, label)
        per_seed.append(r)
        print(f"    seed{seed}: SUP {r['sup_pass']:5.1f}%  "
              f"UNSUP fixed {r['uns_pass_fixed']:5.1f}%  best-match {r['uns_pass_best']:5.1f}%  "
              f"name-swap reject {r['offdiag_reject']:5.1f}%  live {r['live_cols']}/{K}")
        # ---- random-column control for this seed's architecture ----
        draws = []
        for dr in range(N_RANDOM_DRAWS):
            rn = load_ckpt(build_net(seed), path)
            rng = np.random.default_rng(1000 + dr)
            rn.concept.W.data = (rng.standard_normal(rn.concept.W.data.shape) *
                                 np.sqrt(2.0 / (WIN * D))).astype(np.float32)
            rn.concept.b.data = np.zeros_like(rn.concept.b.data)
            draws.append(score_model(rn, X, targets, "random"))
        rc_sup = float(np.mean([z["sup_pass"] for z in draws]))
        rc_uns = float(np.mean([z["uns_pass_fixed"] for z in draws]))
        rc_live = float(np.mean([z["live_cols"] for z in draws]))
        r["random_sup"], r["random_uns"], r["random_live"] = rc_sup, rc_uns, rc_live
        print(f"            random control: SUP {rc_sup:5.1f}%  UNSUP {rc_uns:5.1f}%  "
              f"live {rc_live:.1f}/{K}")
    if not per_seed:
        continue
    agg = {k: float(np.mean([r[k] for r in per_seed]))
           for k in ("sup_pass", "uns_pass_fixed", "uns_pass_best", "offdiag_reject",
                     "random_sup", "random_uns", "random_live", "live_cols")}
    agg["excess_sup"] = agg["sup_pass"] - agg["random_sup"]
    agg["excess_uns"] = agg["uns_pass_fixed"] - agg["random_uns"]
    agg["excess_uns_best"] = agg["uns_pass_best"] - agg["random_uns"]
    out["models"][label] = {"agg": agg, "per_seed": per_seed}
    print(f"    MEAN: SUP {agg['sup_pass']:.1f}% (excess {agg['excess_sup']:+.1f}) | "
          f"UNSUP {agg['uns_pass_fixed']:.1f}% (excess {agg['excess_uns']:+.1f}) | "
          f"name-swap reject {agg['offdiag_reject']:.1f}%")

json.dump(out, open(os.path.join(RUNDIR, f"meter3_{KIND}.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meter3_{KIND}.json")
print("=" * 104)
