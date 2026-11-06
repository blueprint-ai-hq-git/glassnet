"""direct_scale_test.py — IS THE DIRECT TIED PATH THE LEVER? (predictions P77a-g)

Predictions: lens/runs/DIRECT-SCALE-PREDICTIONS-P77.txt (written BEFORE this file existed).
Output: lens/runs/DIRECT-SCALE.txt (incremental), launched by lens/scripts/run_direct_scale.sh.

WHY: P75 (lens/runs/CONTEXT-DEPENDENCE.txt) PROVED the readout splits into
    z = (h @ answer.W + answer.b)  +  (relu(flat(x) @ dmix) @ emb^T)      residual <= 6.8e-06
and measured that the DIRECT part is 3-4x larger than the CONCEPT part. This instrument applies a scale s to the
direct term AT INFERENCE (s in {1.0 control, 0.5, 0.25, 0.0 = the pre-fix concept-only readout, reached without
retraining) and measures, on the same six real pad-free prompts and the same decoder:
  * the text (6 greedy samples per scale, printed in full),
  * the hedge count, the <unk> share, the longest word run,
  * the prompt-steering share of the first-token top-20 (REAL vs MISMATCH: KL and top-20 overlap),
  * the rank of the TRUE first reply token,
  * the decomposition residual at every step, so the scaling is applied to a PROVEN split, not an assumed one.
CPU only; never trains; never uses the GPU; writes only its own log.
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
OUT = os.environ.get("SCALE_OUT", os.path.join(P, "lens/runs/DIRECT-SCALE.txt"))
SCALES = [1.0, 0.5, 0.25, 0.0]
N_NEW = 24
REP = 1.15                       # the v3 decoder's repetition penalty (P73 measured it is NOT the cause)
HEDGE_FRAGMENTS = ("sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
                   "no idea", "can ' t", "cannot", "not a lot of information")

_DEFAULT = [
    ("SHIP CANDIDATE d256 full169 fix3 FINAL best ep3 (by-position 35.26%, 5.9M params)",
     "lens/runs/chat/ct_d256_full169_fix3_best.npz"),
    ("d512 full169 fix3 FINAL best ep3 (by-position 37.18%, 15.5M params, the new best)",
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

_logf = open(OUT, "w", buffering=1)


def log(s=""):
    print(s, flush=True)
    _logf.write(str(s) + "\n")
    _logf.flush()


def is_hedge(text):
    s = " ".join(text.split()[:6]).lower().replace(" ’ ", "' ")
    return any(f in s for f in HEDGE_FRAGMENTS)


def topk(z, k=20):
    return set(np.argsort(-z)[:k].tolist())


def kl(p_logits, q_logits):
    def sm(z):
        m = z.max()
        e = np.exp(z - m)
        return e / e.sum()
    p, q = sm(p_logits), sm(q_logits)
    return float(np.sum(p * np.log((p + 1e-300) / (q + 1e-300))))


t0 = time.time()
data = chat_data.build()
ids = np.asarray(data["ids"])
bounds = np.asarray(data["bounds"], dtype=np.int64)
split = np.asarray(data["split"])
vocab = data["vocab"]

log("=" * 110)
log("DIRECT-SCALE: IS THE DIRECT TIED PATH THE LEVER?   " + time.strftime("%F %H:%M:%S"))
log("predictions: lens/runs/DIRECT-SCALE-PREDICTIONS-P77.txt (P77a-g) — written before this instrument existed")
log("follows: P75 lens/runs/CONTEXT-DEPENDENCE.txt (|direct| 3-4x |concept|) and P73/P74 (the decoder is not the cause)")
log("=" * 110)

# ---- the same six REAL PAD-FREE prompts the ship decision uses (identical rule to talk_fair_v3_ship.py)
ROW, other = [], []
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
    reply = t[pre_len:pre_len + N_NEW]
    if len(reply) < 16:
        continue
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len], dtype=np.int64)
    if len(ROW) < 6:
        ROW.append((j, ctx, reply))
    elif len(other) < 6:
        other.append((j, ctx, reply))
log(f"prompts: {len(ROW)} real pad-free windows, conv ids {[r[0] for r in ROW]}")
log(f"mismatch partners: {len(other)} other held-out windows, conv ids {[o[0] for o in other]}")
SMOKE = bool(os.environ.get("SCALE_SMOKE"))
if SMOKE:
    log("SMOKE MODE: 2 prompts, scales [1.0, 0.0], output goes to SCALE_OUT (the real record is NOT written)")
    ROW, other = ROW[:2], other[:2]
    SCALES = [1.0, 0.0]
if not SMOKE and (len(ROW) < 6 or len(other) < 6):
    raise SystemExit("not enough pad-free windows: the comparison would be over a smaller set than the ship test")


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


def parts(net, emb, dm, aW, ab, ctx):
    """zc (concept path), zd (tied direct path) and the residual that PROVES logits == zc + zd."""
    out = net.forward_arrays(np.ascontiguousarray(np.asarray([ctx], dtype=np.int64)))
    lg = np.asarray(out["logits"][0], dtype=np.float64)
    h = np.asarray(out["concepts"][0], dtype=np.float64)
    xf = emb[np.asarray(ctx)].reshape(1, -1)
    zc = h @ aW + ab
    zd = (np.maximum(xf @ dm, 0.0) @ emb.T)[0]
    resid = float(np.max(np.abs(zc + zd - lg)))
    return zc, zd, lg, resid


def gen_scaled(net, emb, dm, aW, ab, s, ctx, n_new=N_NEW, rep=REP):
    """Greedy decode with the direct term scaled by s. Returns (tokens, max residual over the loop)."""
    ctx = [int(t) for t in ctx]
    out, used, rmax = [], {}, 0.0
    for _ in range(n_new):
        zc, zd, lg, resid = parts(net, emb, dm, aW, ab, ctx)
        rmax = max(rmax, resid)
        z = zc + s * zd
        for tok, cnt in used.items():
            z[tok] -= np.log(rep) * cnt
        nxt = int(np.argmax(z))
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out, rmax


txt = lambda v: " ".join(vocab[i] for i in v)

PANEL = {}
for label, ckpt in CKPTS:
    log("")
    log("=" * 110)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net, meta = load(ckpt)
    if getattr(net, "dmix", None) is None:
        raise SystemExit("this instrument requires the TIED direct path (dmix); this checkpoint has none, so "
                         "a scaled readout would be a different model from the one trained")
    dm = np.asarray(net.dmix.data, dtype=np.float64)
    emb = np.asarray(net.emb.data, dtype=np.float64)
    aW = np.asarray(net.answer.W.data, dtype=np.float64)
    ab = np.asarray(net.answer.b.data, dtype=np.float64)
    log(f"  meta: D={meta['D']} W={meta['W']} attention={meta.get('attention')} direct={meta.get('direct')} "
        f"mode={meta.get('direct_mode')}  K={len(meta['concepts'])} concepts")
    # magnitudes of the two terms, printed (this is the P77a half that reproduces P75 instead of re-arguing it)
    mc, md = [], []
    for j, ctx, reply in ROW:
        zc, zd, lg, resid = parts(net, emb, dm, aW, ab, ctx)
        mc.append(float(np.abs(zc).mean()))
        md.append(float(np.abs(zd).mean()))
        if resid > 1e-4:
            raise SystemExit(f"ABORT: decomposition residual {resid:.3e} on conv {j}: the split is not proven, "
                             f"so a scaled readout would not be this model. Nothing is written.")
    log(f"  term magnitudes at the first position (mean over 6 prompts): |concept| {np.mean(mc):.4f}   "
        f"|direct| {np.mean(md):.4f}   ratio direct/concept {np.mean(md)/np.mean(mc):.3f}")
    log(f"  answer.b[<unk>] = {ab[0]:+.4f}  (rank {int((ab > ab[0]).sum())} of {len(ab)})")

    rmax_all = 0.0
    for s in SCALES:
        log("")
        log("-" * 110)
        log(f"  SCALE s = {s:4.2f}   (logits = zc + {s:4.2f} * zd)")
        log("-" * 110)
        hedge_n, unk_shares, runs, ranks, ovs, kls, tops = 0, [], [], [], [], [], []
        for n_i, (j, ctx, reply) in enumerate(ROW):
            zc, zd, lg, resid = parts(net, emb, dm, aW, ab, ctx)
            rmax_all = max(rmax_all, resid)
            zr = zc + s * zd
            # the true first reply token, by id (the vocab list is the id order; find it directly)
            tid = None
            for i, w in enumerate(vocab):
                if w == reply[0]:
                    tid = i
                    break
            if tid is not None:
                ranks.append(int((zr > zr[tid]).sum()))
            # mismatch: another held-out conversation's window
            jm, ctxm, _ = other[n_i % len(other)]
            zcm, zdm, _, residm = parts(net, emb, dm, aW, ab, ctxm)
            rmax_all = max(rmax_all, residm)
            zm = zcm + s * zdm
            ovs.append(len(topk(zr) & topk(zm)) / 20.0)
            kls.append(kl(zr, zm))
            tops.append(vocab[int(np.argmax(zr))])
            toks, rmax_gen = gen_scaled(net, emb, dm, aW, ab, s, ctx)
            rmax_all = max(rmax_all, rmax_gen)
            s_txt = txt(toks)
            unk = sum(1 for t in toks if t == 0) / len(toks)
            unk_shares.append(unk)
            best, run = 0, 0
            for t in toks:
                run = run + 1 if t != 0 else 0
                best = max(best, run)
            runs.append(best)
            h = is_hedge(s_txt)
            hedge_n += int(h)
            log(f"    conv {j:>6}  hedged {str(h):>5}  <unk> {100*unk:5.1f}%  run {best:>2}  "
                f"true-token rank {ranks[-1] if tid is not None else -1:>6}  "
                f"top20-overlap-with-MISMATCH {ovs[-1]:4.2f}  KL(REAL||MISMATCH) {kls[-1]:6.3f}  "
                f"top1 {tops[-1]!r}")
            log(f"             PROMPT(last10): {' '.join(vocab[int(i)] for i in ctx[-10:])}")
            log(f"             TRUE   : {' '.join(reply[:14])}")
            log(f"             S={s:4.2f}   : {s_txt}")
        PANEL[(label, s)] = {"hedge": hedge_n, "n": len(ROW), "unk": float(np.mean(unk_shares)), "runs": runs,
                             "median_true_rank": float(np.median(ranks)) if ranks else float("nan"),
                             "mean_overlap_mismatch": float(np.mean(ovs)), "mean_kl_mismatch": float(np.mean(kls)),
                             "top1": tops}
        log("")
        log(f"  SUMMARY s={s:4.2f}: hedge {hedge_n}/{len(ROW)}   mean <unk> {100*np.mean(unk_shares):5.1f}%   "
            f"longest word run min/med {min(runs)}/{int(np.median(runs))}   median true-token rank "
            f"{PANEL[(label, s)]['median_true_rank']:.1f}   mean top20 overlap REAL-vs-MISMATCH "
            f"{np.mean(ovs):4.2f}   mean KL {np.mean(kls):6.3f} nats   top1 tokens {tops}")
    log("")
    log(f"  P77a decomposition residual max over EVERY step of this checkpoint's run: {rmax_all:.3e}  "
        f"(predicted <= 1e-4)")

log("")
log("=" * 110)
log("P77 PANEL (the verdict table; every row is a full block above with its text printed)")
log("=" * 110)
log(f"{'checkpoint':<62} {'s':>5} {'hedge':>8} {'unk%':>6} {'med-rank':>9} {'ovl-mm':>7} {'KL-mm':>7}")
for (label, s), d in PANEL.items():
    log(f"{label[:60]:<62} {s:5.2f} {d['hedge']:>4}/{d['n']} {100*d['unk']:6.1f} "
        f"{d['median_true_rank']:9.1f} {d['mean_overlap_mismatch']:7.2f} {d['mean_kl_mismatch']:7.3f}")
log("")
log(f"wall {time.time()-t0:.1f}s   written to {OUT}")
