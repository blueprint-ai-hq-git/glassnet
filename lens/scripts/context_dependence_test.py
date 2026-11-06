"""context_dependence_test.py — IS THE FIRST-TOKEN ANSWER DRIVEN BY THE CONTEXT AT ALL?

Predictions P75a-f: lens/runs/CONTEXT-DEPENDENCE-PREDICTIONS-P75.txt (written BEFORE this file ran).
Output: lens/runs/CONTEXT-DEPENDENCE.txt (env CD_OUT), written INCREMENTALLY so a crash cannot lose numbers.
Line of work: P73 (lens/runs/DECODE-ATTRACTOR.txt, true token reachable at median rank 8) -> P74
(lens/runs/DECODE-SEARCH.txt, beam search hedges MORE than greedy) -> this: at the FIRST generated token, how much
does the model's distribution move when the prompt is real, shuffled, a different conversation, or empty?

It also decomposes the model's own readout into its two parts and PROVES the decomposition with a residual:
    logits_c = h @ answer.W + answer.b                  (the 36-concept path)
    logits_d = relu(flat(x) @ dmix) @ emb^T             (the tied direct path)
    residual = max |logits_c + logits_d - logits|       (must be ~0, or the split is not a measurement)
CPU only; never trains; writes only its own log.
"""
import json
import os
import sys
import time

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

import chat_data                                              # noqa: E402
from lens import backend as BE                                # noqa: E402
from lens.concepts import Concept                             # noqa: E402
from lens.model import LensNet                                # noqa: E402

W = 32
OUT = os.environ.get("CD_OUT", os.path.join(P, "lens/runs/CONTEXT-DEPENDENCE.txt"))
SPECIALS = {"<|im_start|>", "<|im_end|>"}
HEDGE_FRAGMENTS = ("sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
                   "no idea", "can ' t", "cannot", "not a lot of information")

_DEFAULT = [
    ("SHIP CANDIDATE d256 maskfix full-corpus FINAL best (ep3, held-out 35.26%)",
     "lens/runs/chat/ct_d256_full169_fix3_best.npz"),
    ("d512 full-corpus ep1 best (held-out 33.17% by position)",
     "lens/runs/chat/ct_d512_full169_fix3_best.npz"),
]
_spec = os.environ.get("CKPT_LIST", "").strip()
if _spec:
    CKPTS = []
    for _item in _spec.split(";"):
        if not _item.strip():
            continue
        _lab, _p = _item.rsplit("=", 1)
        _p = _p.strip()
        CKPTS.append((_lab.strip(), _p if os.path.isabs(_p) else os.path.join(P, _p)))
else:
    CKPTS = [(lab, os.path.join(P, rel)) for lab, rel in _DEFAULT]

_lines = []
_logf = open(OUT, "w", buffering=1)


def log(s=""):
    _lines.append(str(s))
    print(s, flush=True)
    _logf.write(str(s) + "\n")
    _logf.flush()


def is_hedge(words):
    s = " ".join(words[:6]).lower()
    return any(f in s for f in HEDGE_FRAGMENTS)


t0 = time.time()
data = chat_data.build()
ids = np.asarray(data["ids"])
bounds = np.asarray(data["bounds"], dtype=np.int64)
split = np.asarray(data["split"])
vocab = data["vocab"]
vmap = {w: i for i, w in enumerate(vocab)}

log("=" * 100)
log("CONTEXT-DEPENDENCE: IS THE FIRST-TOKEN ANSWER DRIVEN BY THE PROMPT?   " + time.strftime("%F %H:%M:%S"))
log("predictions: lens/runs/CONTEXT-DEPENDENCE-PREDICTIONS-P75.txt (P75a-f) — written before this run")
log("follows: DECODE-ATTRACTOR.txt (P73) and DECODE-SEARCH.txt (P74)")
log("=" * 100)

ROW = []
other = []
for j in range(len(bounds) - 1):
    if int(split[j]) != 1:
        continue
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    t = [vocab[i] for i in np.asarray(ids[a0:b0])]
    starts = [k for k in range(len(t) - 1)
              if t[k] == "<|im_start|>" and t[k + 1].lower() == "assistant" and k + 2 >= W]
    if not starts:
        continue
    k = starts[0]
    pre_len = k + 2
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len], dtype=np.int64)
    if len(ROW) < 6:
        ROW.append((j, ctx, t[pre_len:pre_len + 24]))
    elif len(other) < 6:
        other.append((j, ctx))
log(f"prompts: {len(ROW)} real pad-free windows (conv ids {[r[0] for r in ROW]})")
log(f"mismatch partners: {len(other)} other held-out windows (conv ids {[o[0] for o in other]})")


def load(ckpt):
    z = np.load(ckpt, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
    net = LensNet(meta["V"], concepts, d=meta["D"], window=meta["W"], seed=meta["seed"],
                  lam_concept=meta["lam_concept"], lam_act=meta["lam_act"],
                  attention=bool(meta.get("attention", False)), numeric=bool(meta.get("numeric", False)),
                  direct=bool(meta.get("direct", False)), direct_rank=int(meta.get("direct_rank", 16)),
                  direct_mode=str(meta.get("direct_mode", "lowrank")))
    net.device_name = "cpu"
    for p in net.parameters():
        p.data = BE.to_dev(z[p.name].astype(np.float32))
    return net, meta


def step(net, ctx, emb, dm, aW, ab):
    """One forward on a single 32-token window: the next-token logits (forward_arrays returns shape (1, V) for
    one window - MEASURED, lens/scripts/context_dependence_test.py's first launch assumed a "window_input" key
    that forward_arrays does not return), the 36-concept vector h, and the two-part decomposition:
        zc = h @ answer.W + answer.b                 (the concept path)
        zd = relu(flat(x) @ dmix) @ emb^T            (the tied direct path), x = emb[ctx] with attention=False
    The residual |zc + zd - logits| is returned so the split is PROVEN, not assumed (verified 4.04e-06 on this
    ship before the run was launched)."""
    out = net.forward_arrays(np.ascontiguousarray(np.asarray([ctx], dtype=np.int64)))
    lg = np.asarray(out["logits"][0], dtype=np.float64)
    h = np.asarray(out["concepts"][0], dtype=np.float64)
    e = emb[np.asarray(ctx)]                                     # (W, D)
    xf = e.reshape(1, -1).astype(np.float64)                     # (1, W*D)
    zc = h @ aW + ab
    hm = np.maximum(xf @ dm, 0.0)
    zd = hm @ emb.T
    resid = float(np.max(np.abs(zc + zd - lg)))
    return lg, h, zc, zd, resid


def topk(z, k=20):
    return set(np.argsort(-z)[:k].tolist())


def kl(p_logits, q_logits):
    def sm(z):
        m = z.max()
        e = np.exp(z - m)
        return e / e.sum()
    p, q = sm(p_logits), sm(q_logits)
    return float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300))))


def show(z, k=6):
    order = np.argsort(-z)[:k]
    return " ".join(repr(vocab[int(i)]) for i in order)


PANEL = {}
for label, ckpt in CKPTS:
    log("")
    log("=" * 100)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net, meta = load(ckpt)
    aW = np.asarray(net.answer.W.data, dtype=np.float64)
    ab = np.asarray(net.answer.b.data, dtype=np.float64)
    if getattr(net, "dmix", None) is None:
        raise SystemExit("this instrument's readout decomposition assumes the TIED direct path (dmix); this "
                         "checkpoint has none, so the split would be a different model than the one shipped")
    dm = np.asarray(net.dmix.data, dtype=np.float64)
    emb_d = np.asarray(net.emb.data, dtype=np.float64)
    log(f"  meta: D={meta['D']} W={meta['W']} attention={meta.get('attention')} direct={meta.get('direct')} "
        f"mode={meta.get('direct_mode')}  K={len(meta['concepts'])} concepts")
    log(f"  reference: answer.b[<unk>] = {ab[0]:+.4f}, rank {int((ab > ab[0]).sum())} of {len(ab)}")
    rng = np.random.default_rng(20261022)
    ctxs = ["REAL", "SHUFFLE", "MISMATCH", "ALLUNK", "IDS"]
    per_ctx = {c: [] for c in ctxs}
    h_all = []
    resid_all = []
    for n_i, (j, ctx, reply) in enumerate(ROW):
        variants = {
            "REAL": ctx,
            "SHUFFLE": np.asarray(rng.permutation(np.asarray(ctx)), dtype=np.int64),
            "MISMATCH": other[n_i % len(other)][1] if other else ctx,
            "ALLUNK": np.zeros(W, dtype=np.int64),
            "IDS": np.concatenate([np.zeros(24, dtype=np.int64), np.asarray(ctx)[-8:]]),
        }
        got = {}
        for cname, c in variants.items():
            lg, h, zc, zd, resid = step(net, c, emb_d, dm, aW, ab)
            got[cname] = (lg, h, zc, zd)
            h_all.append(h)
            if np.isfinite(resid):
                resid_all.append(resid)
            per_ctx[cname].append(lg)
        log("")
        log(f"  conv {j:>6}  PROMPT(last 12) : {' '.join(vocab[int(i)] for i in ctx[-12:])}")
        log(f"                      TRUE reply : {' '.join(reply[:14])}")
        zr = got["REAL"][0]
        for cname in ctxs:
            z = got[cname][0]
            top1 = vocab[int(np.argmax(z))]
            ov = len(topk(zr) & topk(z)) / 20.0
            h_norm = float(np.abs(got[cname][1] - got["REAL"][1]).mean())
            log(f"    {cname:>8}: top1 {top1!r:>12}  top20-overlap-with-REAL {ov:4.2f}  "
                f"KL(REAL||this) {kl(zr, z):6.3f} nats  mean|dh| {h_norm:.4f}")
        log(f"    REAL top-6: {show(zr)}")
        if got["REAL"][3] is not None:
            zc, zd = got["REAL"][2], got["REAL"][3]
            log(f"    the readout split: |concept part| mean {np.abs(zc).mean():8.4f}  "
                f"|direct part| mean {np.abs(zd).mean():8.4f}  residual max "
                f"{np.max(np.abs(zc + zd - zr)):.2e}")
    # ---- the panel
    log("")
    log("=" * 100)
    log("P75 PANEL")
    log("=" * 100)
    H = np.asarray(h_all)
    log(f"  P75a DECOMPOSITION EXACT? residual max over {len(resid_all)} contexts = "
        f"{max(resid_all) if resid_all else float('nan'):.3e}  (predicted <= 1e-4)")
    for cname in ("SHUFFLE", "MISMATCH", "ALLUNK", "IDS"):
        ovs, kls = [], []
        for n_i in range(len(ROW)):
            zr = per_ctx["REAL"][n_i]
            z = per_ctx[cname][n_i]
            ovs.append(len(topk(zr) & topk(z)) / 20.0)
            kls.append(kl(zr, z))
        log(f"  REAL vs {cname:<8}: mean top20 overlap {np.mean(ovs):4.2f}   mean KL(REAL||{cname}) "
            f"{np.mean(kls):6.3f} nats")
    top1s = [vocab[int(np.argmax(per_ctx["REAL"][i]))] for i in range(len(ROW))]
    allt1 = [vocab[int(np.argmax(per_ctx[c][i]))] for c in ctxs for i in range(len(ROW))]
    log(f"  P75d top-1 token of the REAL contexts: {top1s}")
    log(f"        top-1 tokens over ALL {len(ctxs)}x{len(ROW)} contexts: {allt1}")
    log(f"        'i' is top-1 in {allt1.count('i')}/{len(allt1)}; hedge-family top-1 in "
        f"{sum(1 for t in allt1 if is_hedge([t]))}/{len(allt1)}")
    rng_dim = (H.max(axis=0) - H.min(axis=0))
    log(f"  P75e concept vector h over the {len(H)} contexts: {H.shape[1]} coordinates, mean |h| "
        f"{np.abs(H).mean():.4f}; mean coordinate RANGE {rng_dim.mean():.4f} "
        f"(predicted <= 0.15); coordinates moving < 0.05: {int((rng_dim < 0.05).sum())}/{H.shape[1]} "
        f"(predicted >= half)")
    PANEL[label] = dict(top1s=top1s, allt1=allt1, rng=float(rng_dim.mean()),
                        frozen=int((rng_dim < 0.05).sum()), n=H.shape[1])

log("")
log("THE HONEST LIMIT, stated with the numbers: all of the above is the FIRST generated token. v3's teacher-forced")
log("share on 32-token windows (lens/runs/SHIP-TALK-v3.txt block) already shows the model is not blind to context")
log("over a horizon, so nothing here says 'the model ignores its input' in general; it says what the first-token")
log("distribution does when the prompt changes.")
log("")
log(f"files: this run wrote ONLY {OUT} (predictions CONTEXT-DEPENDENCE-PREDICTIONS-P75.txt, runner log "
    f"CONTEXT-DEPENDENCE-RUN.txt). No checkpoint, no stored run, no instrument was edited.")
log(f"wall {time.time()-t0:.0f}s")
print(f"\nwritten to {OUT}")
