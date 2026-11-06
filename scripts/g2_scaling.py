"""g2_scaling.py — TARGET G2: does readability carry across model sizes?

Rungs differ ONLY in width D. Everything else (concepts, window, data, optimiser, lr, epochs,
audit) is fixed. For each rung and each model kind, 3 seeds.

The audit is done HERE, in the same process that builds the net, so the net's width and K always
match the checkpoint. That is deliberate: the Target F audit bug was a width/K mismatch between
the checkpoint and the net it was loaded into, and this script cannot make that mistake because
there is no separate loader.

Checkpoints are saved with the REAL key names (emb / concept.W / concept.b / answer.W / answer.b).
"""
import json
import os
import sys
import time

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

from glassnet import tiny_data                     # noqa: E402
from glassnet.optim import SGD                     # noqa: E402
from glassnet.tensor import set_backend            # noqa: E402
from run_t2_iter003 import WIN, BATCH, TinyLM3, embedding, _flatten  # noqa: E402
from train_t2_normal_concat import TinyNormal      # noqa: E402

set_backend("numpy")


def _snap(net, kind):
    if kind == "readable":
        return {k: v.data.copy() for k, v in (("emb", net.emb), ("concept.W", net.concept.W),
                                              ("concept.b", net.concept.b),
                                              ("answer.W", net.answer.W), ("answer.b", net.answer.b))}
    return {k: v.data.copy() for k, v in (("emb", net.emb), ("h1.W", net.h1.W), ("h1.b", net.h1.b),
                                          ("h2.W", net.h2.W), ("h2.b", net.h2.b))}


def _restore(net, kind, st):
    if kind == "readable":
        net.emb.data, net.concept.W.data = st["emb"], st["concept.W"]
        net.concept.b.data, net.answer.W.data = st["concept.b"], st["answer.W"]
        net.answer.b.data = st["answer.b"]
    else:
        net.emb.data, net.h1.W.data = st["emb"], st["h1.W"]
        net.h1.b.data, net.h2.W.data = st["h1.b"], st["h2.W"]
        net.h2.b.data = st["h2.b"]


def _ce(logits, y):
    from glassnet.losses import cross_entropy
    return cross_entropy(logits, y)

R = os.path.join(PROJ, "runs", "t3")
CK = os.path.join(R, "ckpt")
os.makedirs(CK, exist_ok=True)

# The concept targets are loaded HERE, not from run_t2_iter003's module global, because
# T3M_IMPORT_ONLY=1 means that global is never populated (CONCEPT_TARGETS is None) -- which
# crashed the first smoke run. These are the 9 pre-registered idea targets, in training order.
TARGETS = json.load(open(os.path.join(R, "meaningful_targets_ext.json")))["qualifying"]

RUNGS = [int(x) for x in os.environ.get("G2_RUNGS", "32,64,128").split(",")]
SEEDS = [int(x) for x in os.environ.get("G2_SEEDS", "0,1,2").split(",")]
EPOCHS = int(os.environ.get("G2_EPOCHS", "60"))
PATIENCE = int(os.environ.get("G2_PATIENCE", "12"))
LR = float(os.environ.get("G2_LR", "1.0"))
K = 9
# TARGET-I.md: the ONE change is the activation sparsity penalty. 0.2 is the project default
# (Targets C/D/E/F/G2 all used it); 0.0 is the Target I arm. lam_w (weight sparsity) is untouched.
LAM_ACT = float(os.environ.get("G2_LAM_ACT", "0.2"))
OUTJSON = os.environ.get("G2_OUT", "g2_scaling.json")

# HARD GATE on the concept set. run_t2_iter003._load_concept_targets defaults to
# T2_CONCEPT_SET="words" (single-word targets). The first G2 sweep ran without this env set, so
# its models trained on the WORD set while the audit compared the 9 MEANINGFUL targets -- different
# sets, columns not corresponding, every concept near chance. That produced a full invalid sweep.
# This assertion makes the mismatch impossible to repeat silently. It sits AFTER K is defined.
import run_t2_iter003 as _M  # noqa: E402
_ct = _M.CONCEPT_TARGETS
if _ct is None or _ct.get("kind") != "meaningful_ext" or len(_ct.get("words", [])) != K:
    raise SystemExit(
        "REFUSING TO RUN: the model would train on the wrong concept set.\n"
        f"  expected kind='meaningful_ext' with {K} words\n"
        f"  got kind={None if _ct is None else _ct.get('kind')!r} "
        f"with {0 if _ct is None else len(_ct.get('words', []))} words\n"
        "  set T2_CONCEPT_SET=meaningful_ext and T2_N_TAUGHT=9 (see run_g2.sh).")
print(f"concept set gate: kind='{_ct['kind']}', {len(_ct['words'])} targets: "
      f"{', '.join(_ct['words'][:3])}...")
AUROC_BAR, CONTRIB_BAR, FIRING_FLOOR = 0.55, 0.25, 0.02

DATA = tiny_data.build(win=WIN, vocab_cap=1500)
Xtr, ytr = np.asarray(DATA["Xtr"]), np.asarray(DATA["ytr"])
Xva, yva = np.asarray(DATA["Xva"]), np.asarray(DATA["yva"])
V = DATA["V"]


def auroc(score, t01):
    o = np.argsort(score)
    r = np.empty(len(score), dtype=float)
    r[o] = np.arange(1, len(score) + 1)
    n1, n0 = int(t01.sum()), int((~t01).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[t01].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def targets_for(X):
    return np.stack([np.isin(X, t["ids"]).any(1) for t in TARGETS], axis=1)


def evaluate(net, kind):
    lg = np.asarray(net.forward(Xva)["logits"].data) if kind == "readable" \
        else np.asarray(net.forward(Xva).data)
    lg = lg - lg.max(1, keepdims=True)
    p = np.exp(lg)
    p /= p.sum(1, keepdims=True)
    ppl = float(np.exp(-np.log(np.maximum(p[np.arange(len(yva)), yva], 1e-12)).mean()))
    acc = float((lg.argmax(1) == yva).mean())
    return acc, ppl


def audit(net):
    """The STANDING METER (Meter-03), copied byte-for-byte in maths from t3_audit_m3.audit_col.

    This is not my own statistic. The first version of this audit used |h * W| as the
    contribution, which is NOT the meter's logit-gain formula, and it reported 0/9 concepts at
    D=32. That number was meaningless. The meter is:

      AUROC   -- on the LENGTH-RESIDUALISED activation: resid = h[:,c] - A @ lstsq(A, h[:,c]),
                 where A = [1, log1p(non-pad tokens)]. This removes the "how long is the window"
                 confound, which otherwise lets a concept score well just by firing on long windows.
      contrib -- logit gain: bk = concept.W[:,c].reshape(WIN, D); wvec = bk @ emb[i]; summed over
                 fired windows containing token i; divided by the number of fired windows.
      firing  -- fraction of windows where h[:,c] > 0.5, with a 2% floor.
      pass    -- a >= 0.55 AND contrib >= 0.25 AND firing >= 0.02.
    """
    out = net.forward(Xva)
    h = np.asarray(out["concepts"].data)
    L = np.log1p(np.asarray([(x != V - 1).sum() for x in Xva]))
    A = np.vstack([np.ones_like(L), L]).T
    tgt = targets_for(Xva)
    rows = []
    for c, t in enumerate(TARGETS):
        ids = [i for i in t["ids"] if 0 <= i < V]
        firing = float((h[:, c] > 0.5).mean())
        contains = np.stack([np.isin(Xva, i).any(1) for i in ids]).any(0) if ids else None
        if not ids or contains is None or contains.sum() == 0 or (~contains).sum() == 0:
            rows.append({"col": c, "name": t["name"], "auroc": float("nan"), "contrib": 0.0,
                         "firing": firing, "pass": False, "flag": "NO-TARGET"})
            continue
        coef, *_ = np.linalg.lstsq(A, h[:, c], rcond=None)
        resid = h[:, c] - A @ coef
        a = auroc(resid, contains)
        fired = h[:, c] > 0.5
        contrib = 0.0
        if fired.any():
            bk = net.concept.W.data[:, c].reshape(WIN, net.D)
            tot = 0.0
            for i in ids:
                wvec = bk @ net.emb.data[i]
                take = fired & np.isin(Xva, i).any(1)
                if take.any():
                    tot += float((wvec[None, :] * (Xva[take] == i)).sum())
            contrib = tot / float(fired.sum())
        ok = (a >= AUROC_BAR) and (contrib >= CONTRIB_BAR) and (firing >= FIRING_FLOOR)
        rows.append({"col": c, "name": t["name"], "auroc": float(a), "contrib": float(contrib),
                     "firing": firing, "pass": bool(ok), "flag": "-" if ok else "BROKEN"})
    mass_named = h * np.asarray(out["logits"].data)[:, [0]]
    cov = 1.0 if net.K == len(TARGETS) else 0.0
    return cov, rows


def save(net, path, kind, D, seed, acc, ppl):
    payload = {"emb": net.emb.data, "meta": json.dumps(
        {"kind": kind, "D": D, "K": K, "seed": seed, "acc": acc, "ppl": ppl, "win": WIN})}
    if kind == "readable":
        payload.update({"concept.W": net.concept.W.data, "concept.b": net.concept.b.data,
                        "answer.W": net.answer.W.data, "answer.b": net.answer.b.data})
    else:
        payload.update({"h1.W": net.h1.W.data, "h1.b": net.h1.b.data,
                        "h2.W": net.h2.W.data, "h2.b": net.h2.b.data})
    np.savez(path, **payload)


def param_count(net, kind):
    if kind == "readable":
        return int(sum(p.data.size for p in [net.emb] + net.concept.parameters()
                       + net.answer.parameters()))
    return int(sum(p.data.size for p in [net.emb] + net.h1.parameters() + net.h2.parameters()))


print("=" * 104)
print("TARGET G2 — DOES READABILITY CARRY ACROSS MODEL SIZES?")
print("=" * 104)
print(f"rungs D={RUNGS} | seeds {SEEDS} | {EPOCHS} max epochs, patience {PATIENCE} | lr {LR} | K={K}")
print(f"bars: AUROC >= {AUROC_BAR}, contribution >= {CONTRIB_BAR}, firing >= {FIRING_FLOOR}, "
      f">= 50% of concepts")
print(f"data: tiny_shakespeare, vocab {V}, window {WIN}, {len(Xtr)} train / {len(Xva)} val windows")

out = []
for D in RUNGS:
    for kind in ["readable", "normal"]:
        for seed in SEEDS:
            t0 = time.time()
            rng = np.random.default_rng(seed)
            net = (TinyLM3(V, K, D, WIN, rng=rng, lam_concept=0.1, lam_act=LAM_ACT)
                   if kind == "readable" else TinyNormal(V, K, D, WIN, rng=rng))
            npc = param_count(net, kind)
            opt = SGD(net.parameters(), LR)
            best = {"ppl": float("inf"), "acc": 0.0, "ep": -1, "state": None}
            bad = 0
            for ep in range(1, EPOCHS + 1):
                idx = rng.permutation(len(Xtr))
                for s in range(0, len(Xtr), BATCH):
                    b = idx[s:s + BATCH]
                    opt.zero_grad()
                    if kind == "readable":
                        loss, _parts = net.loss(Xtr[b], ytr[b])
                    else:
                        loss = _ce(net.forward(Xtr[b]), ytr[b])
                    loss.backward()
                    opt.step()
                acc, ppl = evaluate(net, kind)
                if ppl < best["ppl"] - 1e-6:
                    best = {"ppl": ppl, "acc": acc, "ep": ep, "state": _snap(net, kind)}
                    bad = 0
                else:
                    bad += 1
                if bad >= PATIENCE and ep >= 20:
                    break
            _restore(net, kind, best["state"])
            acc, ppl = evaluate(net, kind)
            cov, rows = (audit(net) if kind == "readable" else (float("nan"), []))
            npass = sum(1 for r in rows if r["pass"])
            path = os.path.join(CK, f"g2_{kind}_D{D}_seed{seed}.npz")
            save(net, path, kind, D, seed, acc, ppl)
            print(f"  D={D:>4} {kind:<9} seed {seed}  params {npc:>8}  acc {acc*100:5.2f}%  "
                  f"ppl {ppl:8.3f}  best_ep {best['ep']:>3}  {time.time()-t0:5.0f}s"
                  + (f"  coverage {cov*100:.4f}%  concepts {npass}/{K}" if kind == "readable" else ""))
            out.append({"D": D, "kind": kind, "seed": seed, "params": npc, "acc": acc, "ppl": ppl,
                        "best_ep": best["ep"], "coverage": cov, "n_pass": npass, "rows": rows,
                        "secs": time.time() - t0})
            json.dump(out, open(os.path.join(R, OUTJSON), "w"), indent=1)

# ------------------------------------------------------------------ summary
print("\n" + "=" * 104)
print("SUMMARY BY RUNG")
print("=" * 104)
print(f"{'D':>5} {'params':>9} {'readable acc':>13} {'normal acc':>11} {'readable ppl':>13} "
      f"{'normal ppl':>11} {'cov%':>8} {'concepts':>9}")
for D in RUNGS:
    r = [x for x in out if x["D"] == D and x["kind"] == "readable"]
    n = [x for x in out if x["D"] == D and x["kind"] == "normal"]
    if not r or not n:
        continue
    print(f"{D:>5} {r[0]['params']:>9} {np.mean([x['acc'] for x in r])*100:>12.2f}% "
          f"{np.mean([x['acc'] for x in n])*100:>10.2f}% {np.mean([x['ppl'] for x in r]):>13.3f} "
          f"{np.mean([x['ppl'] for x in n]):>11.3f} {np.mean([x['coverage'] for x in r])*100:>8.4f} "
          f"{np.mean([x['n_pass'] for x in r]):>9.2f}")

print("\nBARS: G2-COVERAGE must be 100.0000% at every rung; G2-READABLE >= 50% of concepts pass;")
print("      G2-NOWORSE = readable not below normal by more than 2x pooled std on acc AND ppl.")
for D in RUNGS:
    r = [x for x in out if x["D"] == D and x["kind"] == "readable"]
    n = [x for x in out if x["D"] == D and x["kind"] == "normal"]
    if not r or not n:
        continue
    cov_ok = all(abs(x["coverage"] - 1.0) < 1e-9 for x in r)
    rp = np.mean([x["n_pass"] for x in r]) / K
    ra, na = np.mean([x["acc"] for x in r]), np.mean([x["acc"] for x in n])
    rpp, npp = np.mean([x["ppl"] for x in r]), np.mean([x["ppl"] for x in n])
    rs, ns = np.std([x["acc"] for x in r], ddof=1), np.std([x["acc"] for x in n], ddof=1)
    rps, nps = np.std([x["ppl"] for x in r], ddof=1), np.std([x["ppl"] for x in n], ddof=1)
    bar_a = 2 * np.sqrt((rs ** 2 + ns ** 2) / 2)
    bar_p = 2 * np.sqrt((rps ** 2 + nps ** 2) / 2)
    print(f"  D={D:>4}: coverage {'PASS' if cov_ok else 'FAIL'} ({np.mean([x['coverage'] for x in r])*100:.4f}%)"
          f" | readable {'PASS' if rp >= 0.5 else 'FAIL'} ({rp*100:.1f}%)"
          f" | acc delta {(ra-na)*100:+.2f} (bar {bar_a*100:.2f})"
          f" | ppl delta {rpp-npp:+.2f} (bar {bar_p:.2f})")


def _snap(net, kind):
    if kind == "readable":
        return {k: v.data.copy() for k, v in (("emb", net.emb), ("concept.W", net.concept.W),
                                              ("concept.b", net.concept.b),
                                              ("answer.W", net.answer.W), ("answer.b", net.answer.b))}
    return {k: v.data.copy() for k, v in (("emb", net.emb), ("h1.W", net.h1.W), ("h1.b", net.h1.b),
                                          ("h2.W", net.h2.W), ("h2.b", net.h2.b))}


def _restore(net, kind, st):
    if kind == "readable":
        net.emb.data, net.concept.W.data = st["emb"], st["concept.W"]
        net.concept.b.data, net.answer.W.data = st["concept.b"], st["answer.W"]
        net.answer.b.data = st["answer.b"]
    else:
        net.emb.data, net.h1.W.data = st["emb"], st["h1.W"]
        net.h1.b.data, net.h2.W.data = st["h1.b"], st["h2.W"]
        net.h2.b.data = st["h2.b"]


def _ce(logits, y):
    from glassnet.losses import cross_entropy
    return cross_entropy(logits, y)
