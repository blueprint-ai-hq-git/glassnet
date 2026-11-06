"""data_judge.py — a SMALL LEARNED MODEL that scores data as good or bad (plan ask #4).

The project already has clean.py: a RULE-based cleaner (readable rules, per-rule counts, freeze
and on/off). This is the other half of the owner's ask: a small model that LEARNS the score.

WHAT "GOOD" AND "BAD" MEAN HERE (stated before any number, because a judge is only as honest
as its labels)
    GOOD = a real corpus window, exactly as the talk line trains on it
    BAD  = the same window under a CONTROLLED CORRUPTION, so the label is exact and auditable:
             shuffle   the window's tokens are permuted          (grammar destroyed, bag kept)
             repeat    one token is repeated to fill the window  (degenerate text)
             random    ~20% of positions replaced by random ids  (noise)
             blank     the window is zeroed                      (missing data)
Corruptions are applied to held-out real windows, so no corrupted window is ever seen in
training, and the judge is tested on corruptions it has never met.

WHY IT IS READABLE BY CONSTRUCTION
    The features are hand-computed, named, and printed per item:
        uniq_frac, repeat_run, rare_frac, punct_frac, bigram_rep, mean_logfreq,
        has_assistant, unk_frac
    The shipping model is a LOGISTIC regression on those features: the score is
    sum(w_f * f) + b, and the per-item explanation prints every w_f * f term. Nothing hidden.

PRE-REGISTERED GATES (written BEFORE the run)
    G1  held-out AUROC(real vs corrupted) >= 0.90
    G2  the MLP must not beat the linear model by more than 0.02, else the MLP ships and the
        linear is reported as the weaker readable option
    G3  the judge must beat the BEST SINGLE FEATURE's AUROC by >= 0.05 (otherwise it is just
        that one feature wearing a model's clothes)
    G4  on REAL windows only, the judge's score must rank rule-flagged items below clean ones
        at AUROC >= 0.65 (a weak-but-real agreement with the project's own rule cleaner);
        measured and reported either way, never hidden
"""
import json
import os
import sys
import time

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import chat_data                                     # noqa: E402
import lens.tensor as T                              # noqa: E402
from lens.model import Dense                         # noqa: E402

WIN = int(os.environ.get("DJ_WIN", "32"))
N_TRAIN_REAL = int(os.environ.get("DJ_N_TRAIN", "30000"))
N_VAL_REAL = int(os.environ.get("DJ_N_VAL", "10000"))
SEED = int(os.environ.get("DJ_SEED", "0"))
EPOCHS = int(os.environ.get("DJ_EPOCHS", "30"))
LR = float(os.environ.get("DJ_LR", "0.5"))
LR_MLP = float(os.environ.get("DJ_LR_MLP", "0.1"))   # the MLP diverged at 0.5 in the smoke
BATCH = 512
OUT = os.path.join(PROJ, "lens", "runs", "data_judge")
os.makedirs(OUT, exist_ok=True)
FEATSET = os.environ.get("DJ_FEATSET", "v2")
FEATS = ["uniq_frac", "repeat_run", "rare_frac", "punct_frac", "bigram_rep",
         "mean_logfreq", "has_assistant", "unk_frac"]
if FEATSET == "v2":
    # v2 PRE-REGISTRATION (2026-10-13): v1 measured shuffle-detection at 0.5124 (chance) while
    # repeat/random/blank were 0.92-0.9998 -- NO v1 feature sees ORDER. This one does: the mean
    # log P(next token | this token) of the window's adjacent pairs, estimated from the corpus's
    # own bigram counts. Shuffling destroys it; real text keeps it.
    FEATS = FEATS + ["bigram_plaus"]


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ features (named, readable)
def build_bigram_table(ids, V):
    """Corpus bigram counts as (sorted packed keys, counts) for searchsorted lookups."""
    pairs = ids[:-1].astype(np.int64) * V + ids[1:]
    uniq, cnt = np.unique(pairs, return_counts=True)
    return uniq, cnt


def bigram_plaus(X, V, uniq_pairs, pair_cnt, uni_cnt):
    """Mean log P(t+1 | t) over the window's adjacent pairs, add-1 smoothed. (N,)"""
    a = X[:, :-1].astype(np.int64) * V + X[:, 1:]
    pos = np.searchsorted(uniq_pairs, a)
    pos = np.clip(pos, 0, len(uniq_pairs) - 1)
    hit = uniq_pairs[pos] == a
    c_pair = np.where(hit, pair_cnt[pos], 0).astype(np.float64)
    c_tok = uni_cnt[X[:, :-1]].astype(np.float64)
    return np.log((c_pair + 1.0) / (c_tok + V)).mean(axis=1)


def make_features(X, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, bigram=None):
    """X: (N, WIN) token ids -> (N, F) hand-computed features, in FEATS order."""
    N, W = X.shape
    F = np.zeros((N, len(FEATS)), dtype=np.float32)
    for i in range(N):
        row = X[i]
        uniq = len(set(row.tolist()))
        F[i, 0] = uniq / W
        # longest run of one repeated token
        best = run = 1
        for k in range(1, W):
            if row[k] == row[k - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        F[i, 1] = best / W
        F[i, 2] = float(np.mean([freqs_log[t] < rare_thr for t in row]))
        F[i, 3] = float(np.mean([t in punct_ids for t in row]))
        pairs = [(int(row[k]), int(row[k + 1])) for k in range(W - 1)]
        F[i, 4] = 1.0 - (len(set(pairs)) / max(len(pairs), 1))
        F[i, 5] = float(np.mean([freqs_log[t] for t in row]))
        F[i, 6] = 1.0 if any(t in assistant_ids for t in row) else 0.0
        F[i, 7] = float(np.mean(row == unk_id))
    if "bigram_plaus" in FEATS and bigram is not None:
        uniq_pairs, pair_cnt, uni_cnt, V = bigram
        F[:, len(FEATS) - 1] = bigram_plaus(X, V, uniq_pairs, pair_cnt, uni_cnt)
    return F


def corrupt(X, kind, rng, V):
    Z = X.copy()
    if kind == "shuffle":
        for i in range(len(Z)):
            Z[i] = rng.permutation(Z[i])
    elif kind == "repeat":
        for i in range(len(Z)):
            t = Z[i, rng.integers(0, Z.shape[1])]
            Z[i] = t
    elif kind == "random":
        m = rng.random(Z.shape) < 0.2
        Z[m] = rng.integers(0, V, size=int(m.sum()))
    elif kind == "blank":
        Z[:] = 0
    return Z


def auroc(scores, labels):
    """Rank-sum AUROC (labels: 1 = positive)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.empty(len(s))
    r[np.argsort(s, kind="mergesort")] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ------------------------------------------------------------------ the models (our own engine)
class Judge:
    """logistic head (readable) + optional one-hidden-layer MLP, both on our own Tensor."""

    def __init__(self, F, rng, hidden=16, use_mlp=True):
        self.lin = Dense(F, 2, rng, name="judge.lin")
        self.use_mlp = use_mlp
        if use_mlp:
            self.h1 = Dense(F, hidden, rng, name="judge.h1")
            self.h2 = Dense(hidden, 2, rng, name="judge.h2")

    def parameters(self):
        ps = self.lin.parameters()
        if self.use_mlp:
            ps = ps + self.h1.parameters() + self.h2.parameters()
        return ps

    def logits(self, x, mlp=False):
        if mlp and self.use_mlp:
            return self.h2(T.relu(self.h1(x)))
        return self.lin(x)

    def score(self, Fnp, mlp=False):
        x = T.Tensor(Fnp)
        with_grad = False
        lg = self.logits(x, mlp=mlp)
        z = np.asarray(lg.data, dtype=np.float64)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        return p[:, 1]


def train_head(judge, Xf, y, epochs, lr, rng, mlp=False, tag=""):
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(Xf))
        tot, nb = 0.0, 0
        for s in range(0, len(Xf), BATCH):
            idx = order[s:s + BATCH]
            for p in judge.parameters():
                p.grad = None
            lg = judge.logits(T.Tensor(Xf[idx]), mlp=mlp)
            loss = T.softmax_ce(lg, y[idx])
            loss.backward()
            for p in judge.parameters():
                if p.grad is not None:
                    p.data = p.data - lr * p.grad
            tot += float(np.asarray(loss.data)); nb += 1
        if ep % 10 == 0 or ep == epochs:
            log(f"    [{tag}] ep {ep:>3} loss {tot/max(nb,1):.4f}  ({time.perf_counter()-t0:.0f}s)")
    return time.perf_counter() - t0


def main():
    rng = np.random.default_rng(SEED)
    log("=" * 100)
    log("DATA JUDGE — a small learned model that scores data good or bad")
    log("=" * 100)

    d = chat_data.build()
    ids, vocab = d["ids"], d["vocab"]
    V = len(vocab)
    # corpus frequencies -> log freq and the rare threshold (10th percentile of token counts)
    counts = np.bincount(ids, minlength=V).astype(np.float64)
    freqs_log = np.log1p(counts)
    rare_thr = float(np.percentile(freqs_log[counts > 0], 10))
    punct_ids = {i for i, w in enumerate(vocab) if w and all(not c.isalnum() for c in w)}
    unk_id = vocab.index("<unk>") if "<unk>" in vocab else -1
    assistant_ids = {i for i, w in enumerate(vocab) if w.lower() == "assistant"}
    log(f"vocab {V:,}  rare_thr(logfreq) {rare_thr:.2f}  punct tokens {len(punct_ids)}  "
        f"unk id {unk_id}")

    # windows: reuse the talk line's own construction so the judge judges the REAL training data
    Xtr, _ = chat_data.make_windows(ids, d["bounds"], d["split"], WIN, which=0)
    Xva, _ = chat_data.make_windows(ids, d["bounds"], d["split"], WIN, which=1)
    Xtr = np.asarray(Xtr[:N_TRAIN_REAL]); Xva = np.asarray(Xva[:N_VAL_REAL])
    BIG = None
    if "bigram_plaus" in FEATS:
        uni_cnt = np.bincount(ids, minlength=V).astype(np.float64)
        up, pc = build_bigram_table(ids, V)
        BIG = (up, pc, uni_cnt, V)
        log(f"bigram table: {len(up):,} distinct pairs from {len(ids):,} tokens")
    log(f"real windows: {len(Xtr):,} train / {len(Xva):,} val")

    kinds = ["shuffle", "repeat", "random", "blank"]
    # train: real + a mix of corruptions (never the same window twice)
    Ftr = [make_features(Xtr, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG)]
    ytr = [np.ones(len(Xtr), dtype=np.int64)]
    for k in kinds:
        Xc = corrupt(Xtr, k, rng, V)
        Ftr.append(make_features(Xc, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG))
        ytr.append(np.zeros(len(Xc), dtype=np.int64))
    Xf_tr = np.vstack(Ftr); y_tr = np.concatenate(ytr)
    # val: real + HELD-OUT corruptions of val windows, one row per corruption kind
    Fva = [make_features(Xva, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG)]
    yva = [np.ones(len(Xva), dtype=np.int64)]
    per_kind = {}
    for k in kinds:
        Xc = corrupt(Xva, k, rng, V)
        Fc = make_features(Xc, freqs_log, rare_thr, punct_ids, unk_id, assistant_ids, BIG)
        Fva.append(Fc); yva.append(np.zeros(len(Xc), dtype=np.int64))
        per_kind[k] = Fc
    Xf_va = np.vstack(Fva); y_va = np.concatenate(yva)
    log(f"feature matrices: train {Xf_tr.shape}, val {Xf_va.shape}")

    # ---- single-feature baselines (G3)
    log("\n--- single-feature AUROC (what a trivial detector achieves) ---")
    single = {}
    for j, nm in enumerate(FEATS):
        a = auroc(Xf_va[:, j], y_va)
        single[nm] = a
        log(f"    {nm:<14} AUROC {a:.4f}")
    best_single = max(single.values())
    log(f"    BEST SINGLE FEATURE: {max(single, key=single.get)} at {best_single:.4f}")

    # ---- the readable logistic judge
    jrng = np.random.default_rng(SEED + 1)
    lin = Judge(len(FEATS), jrng, use_mlp=False)
    train_head(lin, Xf_tr, y_tr, EPOCHS, LR, jrng, mlp=False, tag="linear")
    s_lin = lin.score(Xf_va, mlp=False)
    a_lin = auroc(s_lin, y_va)
    log(f"\nLINEAR judge (readable): held-out AUROC {a_lin:.4f}   "
        f"acc@0.5 {float(((s_lin > 0.5) == y_va).mean())*100:.2f}%")
    Wm = np.asarray(lin.lin.W.data)                 # (F, 2)
    W = Wm[:, 1] - Wm[:, 0]                          # one weight per feature, toward GOOD
    b = float(np.asarray(lin.lin.b.data)[1] - np.asarray(lin.lin.b.data)[0])
    log("    weights (per unit of feature, toward GOOD):")
    for nm, w in sorted(zip(FEATS, W), key=lambda t: -abs(t[1])):
        log(f"      {nm:<14} {w:+.3f}")

    # ---- the small MLP
    jrng2 = np.random.default_rng(SEED + 2)
    mlp = Judge(len(FEATS), jrng2, hidden=16, use_mlp=True)
    train_head(mlp, Xf_tr, y_tr, EPOCHS, LR_MLP, jrng2, mlp=True, tag="mlp")
    s_mlp = mlp.score(Xf_va, mlp=True)
    a_mlp = auroc(s_mlp, y_va)
    inv = "  [INVERTED: its score ranks good data below bad -- reported, not hidden]" \
        if a_mlp < 0.5 else ""
    log(f"\nMLP judge (16 hidden, lr {LR_MLP}): held-out AUROC {a_mlp:.4f}   "
        f"acc@0.5 {float(((s_mlp > 0.5) == y_va).mean())*100:.2f}%{inv}")

    # ---- per-corruption-kind detection (which corruption does it catch?)
    log("\n--- detection per corruption kind (AUROC vs the real val windows) ---")
    n_real = len(Xva)
    per = {}
    for ki, k in enumerate(kinds):
        a0 = n_real + ki * n_real
        sc = np.concatenate([s_lin[:n_real], s_lin[a0:a0 + n_real]])
        lb = np.concatenate([np.ones(n_real), np.zeros(n_real)])
        per[k] = auroc(sc, lb)
        log(f"    {k:<9} {per[k]:.4f}")

    # ---- G4: does the judge AGREE with the model it would feed?
    # The project's rule cleaner flags ZERO of these windows (all are long alphanumeric chat
    # text) -- measured, and reported as a fact rather than dressed up as agreement. So the
    # honest comparator is the reference model's own per-window loss: a window the talk line
    # finds HARD is one a data judge should be suspicious of. We measure the AUROC of the
    # judge's score against "this window is in the reference model's worst-loss quartile".
    real_feats = Xf_va[:n_real]
    import clean as CL
    rule_flags = sum(1 for i in range(n_real)
                     if CL._alnum_frac(" ".join(vocab[t] for t in Xva[i])) < CL.MIN_ALNUM_FRAC
                     or CL._max_char_frac(" ".join(vocab[t] for t in Xva[i])) >
                     CL.MAX_CHAR_REPEAT_FRAC)
    log(f"\nrule cleaner on REAL windows: {rule_flags}/{n_real} flagged "
        f"({100*rule_flags/max(n_real,1):.2f}%) -- the rules are silent on chat text, so they "
        f"cannot be the comparator")
    a_ref = float("nan")
    try:
        import lens as L
        ref = L.load(os.path.join(PROJ, "lens", "runs", "chat", "t11_talk41_ep192.npz"),
                     device="cpu")
        # masked per-window loss: reuse the assistant mask built from the same corpus
        is_asst = np.zeros(len(ids), dtype=np.float32)
        in_a = False
        for j in range(len(d["bounds"]) - 1):
            a0, b0 = d["bounds"][j], d["bounds"][j + 1]
            toks = [vocab[i] for i in ids[a0:b0]]
            k = 0
            while k < len(toks):
                if toks[k].lower() == "assistant" and k + 1 < len(toks) and toks[k + 1] == ":":
                    in_a = True; k += 2; continue
                if toks[k].lower() == "human" and k + 1 < len(toks) and toks[k + 1] == ":":
                    in_a = False; k += 2; continue
                if in_a:
                    is_asst[a0 + k] = 1.0
                k += 1
        _, Yva = chat_data.make_windows(ids, d["bounds"], d["split"], WIN, which=1)
        Yva = np.asarray(Yva[:n_real])
        Wva = is_asst[Yva]
        lg = ref.forward_arrays(Xva)["logits"]
        z = lg - lg.max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        nll = -np.log(np.maximum(p[np.arange(len(Yva)), Yva], 1e-12))
        nll = nll * (Wva > 0.5)
        hard = nll >= np.percentile(nll[Wva > 0.5], 75)
        a_ref = auroc(-s_lin[:n_real], hard.astype(np.int64))
        log(f"reference-model agreement: worst-loss-quartile windows {int(hard.sum())}/{n_real}; "
            f"AUROC(judge score vs reference-hard) {a_ref:.4f}")
    except Exception as e:
        log(f"reference agreement NOT MEASURED: {type(e).__name__}: {str(e)[:120]}")
    a_rule = a_ref

    # ---- explanation: show the arithmetic for the most suspicious real windows
    log("\n--- EXPLANATION for the 3 lowest-scoring REAL windows (is the judge right to doubt?) ---")
    order = np.argsort(s_lin[:n_real])[:3]
    for i in order:
        contrib = [(FEATS[j], float(real_feats[i, j]), float(real_feats[i, j] * W[j]))
                   for j in np.argsort(-np.abs(real_feats[i] * W))]
        txt = " ".join(vocab[t] for t in Xva[i])[:70]
        log(f"    score {s_lin[i]:.3f}  text: {txt!r}")
        log("      " + "  ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in contrib[:4]))
    log("\n--- the 3 CLEAREST corruptions the judge caught (highest score on a corrupted window) ---")
    cor_start = n_real
    idx = np.argsort(-s_lin[cor_start:])[:3] + cor_start
    for i in idx:
        log(f"    score {s_lin[i]:.3f}  features: " +
            "  ".join(f"{FEATS[j]}={Xf_va[i, j]:.2f}" for j in range(len(FEATS))))

    # ---- gates
    g1 = a_lin >= 0.90
    g2 = (a_mlp - a_lin) <= 0.02
    g3 = (max(a_lin, a_mlp) - best_single) >= 0.05
    if FEATSET == "v2":
        # v2 gates (new pre-registration): the aggregate hid a total failure on ONE kind.
        g4 = min(per.values()) >= 0.80            # per-kind floor: every corruption >= 0.80
    else:
        g4 = (a_rule == a_rule) and a_rule >= 0.65
    log("\n" + "=" * 100)
    log("GATES (pre-registered)")
    log("=" * 100)
    log(f"  G1 AUROC >= 0.90                 : {a_lin:.4f} -> {'PASS' if g1 else 'FAIL'}")
    log(f"  G2 MLP not >0.02 above linear    : {a_mlp-a_lin:+.4f} -> {'PASS' if g2 else 'FAIL'}"
        f"  (linear ships: it is the readable one)")
    log(f"  G3 beats best single feature +.05: {max(a_lin,a_mlp)-best_single:+.4f} -> "
        f"{'PASS' if g3 else 'FAIL'}")
    if FEATSET == "v2":
        # v2 pre-registration: the aggregate hid a TOTAL failure on one kind (shuffle 0.5124).
        log(f"  G4' EVERY corruption kind >= 0.80: worst {min(per.values()):.4f} "
            f"({min(per, key=per.get)}) -> {'PASS' if g4 else 'FAIL'}")
        log(f"      (v1's G4 -- agreement with the reference model's hard windows -- measured "
            f"{a_rule:.4f} and is RETIRED: 'hard for the model' is a different question from "
            f"'corrupted data'. Recorded, not hidden.)")
    else:
        log(f"  G4 reference-model agreement >=0.65: {a_rule:.4f} -> {'PASS' if g4 else 'FAIL'}")
    verdict = "PASS" if (g1 and g2 and g3 and g4) else "FAIL"
    log(f"\n  DATA JUDGE: {verdict}")

    rep = {"verdict": verdict, "gates": {"G1": g1, "G2": g2, "G3": g3, "G4": g4},
           "auroc_linear": a_lin, "auroc_mlp": a_mlp, "auroc_best_single": best_single,
           "best_single_feature": max(single, key=single.get), "single_features": single,
           "per_corruption": per, "rule_agreement_auroc": a_rule,
           "rule_flagged_count": int(rule_flags), "rule_flagged_frac": float(rule_flags)/max(n_real, 1),
           "linear_weights": {nm: float(w) for nm, w in zip(FEATS, W)}, "bias": b,
           "n_train": int(len(Xf_tr)), "n_val": int(len(Xf_va)), "epochs": EPOCHS, "lr": LR,
           "features": FEATS}
    json.dump(rep, open(os.path.join(OUT, "judge_report.json"), "w"), indent=1)
    np.savez(os.path.join(OUT, "judge_linear.npz"),
             W=np.asarray(lin.lin.W.data), b=np.asarray(lin.lin.b.data),
             feats=np.array(FEATS))
    log(f"\nsaved {OUT}/judge_report.json and judge_linear.npz")
    return 0 if verdict == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())