"""decode_search_test.py — DOES A SEARCHING DECODER BEAT GREEDY ON THE SHIP CANDIDATE?

Predictions P74a-f: lens/runs/DECODE-SEARCH-PREDICTIONS-P74.txt (written BEFORE this file ran).
Output: lens/runs/DECODE-SEARCH.txt (env SEARCH_OUT); runner log lens/runs/DECODE-SEARCH-RUN.txt.
Context it follows: lens/runs/DECODE-ATTRACTOR.txt (P73) — greedy hedges and the corpus does not, sampling
removes the hedge, and the TRUE reply token sits at pooled median rank 8 of 12,002 under teacher forcing.

WHAT IS MEASURED. The same six real, UNPADDED 32-token prompts (the ones v2/v3/P73 used), the same two
checkpoints, and four decoders over them:
    GREEDY   (rep 1.15 — the control; must reproduce P73/P72 numbers)
    BEAM4    (plain beam search, width 4, rep 1.0)
    BEAM8    (plain beam search, width 8, rep 1.0)
    RERANK32 (32 samples at T=1.0/top-k40/rep 1.0, each scored by its OWN total log-probability; best kept)
Metrics per prompt: the printed text, hedge (the P73 detector), <unk> share, and token-F1 against the TRUE
reply's first 24 words — the mechanical proxy for "did it answer" that I must not replace with my own vibe.
It never trains, never writes a checkpoint, never touches another file, and stays on the CPU.
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
from lens import backend as BE                              # noqa: E402  (named BE: B is used for a row below)
from lens.concepts import Concept                             # noqa: E402
from lens.model import LensNet                                # noqa: E402

W = 32
OUT = os.environ.get("SEARCH_OUT", os.path.join(P, "lens/runs/DECODE-SEARCH.txt"))
HEDGE_FRAGMENTS = ("sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
                   "no idea", "can ' t", "cannot", "not a lot of information")
SPECIALS = {"<|im_start|>", "<|im_end|>"}

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
# WRITTEN INCREMENTALLY (added after the first launch of this instrument died at 1 s with
# "ValueError: windows must be 32 tokens wide, got 33" — a beam that grew its own context past W — and left
# NOTHING but the runner's stdout. A crash must leave the numbers it already measured, so every line is
# flushed to OUT as it is produced.)
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
log("DECODE-SEARCH: DOES A SEARCHING DECODER BEAT GREEDY?   " + time.strftime("%F %H:%M:%S"))
log("predictions: lens/runs/DECODE-SEARCH-PREDICTIONS-P74.txt (P74a-f) — written before this run")
log("follows: lens/runs/DECODE-ATTRACTOR.txt (P73, verdict DECODE_SIDE on both checkpoints)")
log("=" * 100)

ROW = []
for j in range(len(bounds) - 1):
    if int(split[j]) != 1 or len(ROW) >= 6:
        continue
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    t = [vocab[i] for i in np.asarray(ids[a0:b0])]
    starts = [k for k in range(len(t) - 1)
              if t[k] == "<|im_start|>" and t[k + 1].lower() == "assistant" and k + 2 >= W]
    if not starts:
        continue
    k = starts[0]
    pre_len = k + 2
    reply = [w for w in t[pre_len:pre_len + 24]]
    if len(reply) < 16:
        continue
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len], dtype=np.int64)
    ROW.append((j, ctx, reply))
log(f"prompts: {len(ROW)} (conv ids {[r[0] for r in ROW]}) — the same six v2/v3/P73 used, pad-free 32-token "
    f"windows")


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


def logits_of(net, ctx):
    return np.asarray(net.forward_arrays(np.ascontiguousarray(np.asarray([ctx], dtype=np.int64)))["logits"][0],
                      dtype=np.float64)


def apply_rep(z, used, rep):
    if rep != 1.0:
        for tok, cnt in used.items():
            z[tok] -= np.log(rep) * cnt
    return z


def logprob(z):
    m = z.max()
    return z - (m + np.log(np.exp(z - m).sum()))


def gen_greedy(net, ctx, n_new=24, rep=1.15):
    ctx = [int(t) for t in ctx]
    out, used = [], {}
    for _ in range(n_new):
        z = apply_rep(logits_of(net, ctx).copy(), used, rep)
        nxt = int(z.argmax())
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out, None


def gen_beam(net, ctx, n_new=24, width=4, rep=1.0):
    """Plain beam search over the model's own log-probabilities (no length penalty, no repetition penalty at
    rep=1.0). Returns the best sequence by total log-probability."""
    beams = [([int(t) for t in ctx], 0.0, {})]
    for _ in range(n_new):
        cands = []
        for seq, score, used in beams:
            z = apply_rep(logits_of(net, seq[-W:]).copy(), used, rep)
            lp = logprob(z)
            top = np.argpartition(-lp, width)[:width]
            for t in top:
                u2 = dict(used)
                u2[int(t)] = u2.get(int(t), 0) + 1
                cands.append((seq + [int(t)], score + float(lp[int(t)]), u2))
        cands.sort(key=lambda c: -c[1])
        beams = cands[:width]
    best = beams[0]
    return best[0][-n_new:], best[1]


def gen_rerank(net, ctx, n_new=24, n_samples=32, temp=1.0, top_k=40, rep=1.0, seed=0):
    """Draw n_samples sequences, score each by its OWN total log-probability, keep the highest-scoring one."""
    ctx = [int(t) for t in ctx]
    rng = np.random.default_rng(seed)
    best, best_score, scores = None, -np.inf, []
    for s in range(n_samples):
        cur, used, out, sc = list(ctx), {}, [], 0.0
        for _ in range(n_new):
            z = apply_rep(logits_of(net, cur).copy(), used, rep)
            lp = logprob(z)
            zz = lp / max(temp, 1e-6)
            if top_k and top_k < len(zz):
                cut = np.partition(zz, -top_k)[-top_k]
                zz = np.where(zz >= cut, zz, -np.inf)
            m = np.nanmax(zz)
            p = np.exp(zz - m)
            p = p / p.sum()
            nxt = int(rng.choice(len(p), p=p))
            sc += float(lp[nxt])
            out.append(nxt)
            used[nxt] = used.get(nxt, 0) + 1
            cur = (cur + [nxt])[-W:]
        scores.append(sc)
        if sc > best_score:
            best, best_score = out, sc
    return best, best_score, scores


def words_of(gen_ids):
    return [vocab[int(i)] for i in gen_ids if vocab[int(i)] not in SPECIALS]


def f1_vs_true(gen_words, true_words):
    """Token-level F1 against the true reply (first 24 words). A mechanical proxy for topicality, not a
    judgement of coherence."""
    g = [w.lower() for w in gen_words[:24]]
    t = [w.lower() for w in true_words[:24]]
    if not g or not t:
        return 0.0
    from collections import Counter
    cg, ct = Counter(g), Counter(t)
    common = sum(min(cg[w], ct[w]) for w in cg)
    if common == 0:
        return 0.0
    prec, rec = common / len(g), common / len(t)
    return 2 * prec * rec / (prec + rec)


DECODERS = [
    ("GREEDY rep1.15 (control)", lambda net, ctx: gen_greedy(net, ctx, rep=1.15)[0]),
    ("BEAM4 rep1.0", lambda net, ctx: gen_beam(net, ctx, width=4, rep=1.0)[0]),
    ("BEAM8 rep1.0", lambda net, ctx: gen_beam(net, ctx, width=8, rep=1.0)[0]),
    ("RERANK32 T1.0 k40 rep1.0", lambda net, ctx: gen_rerank(net, ctx, n_samples=32, seed=3)[0]),
]

PANEL = {}
for label, ckpt in CKPTS:
    log("")
    log("=" * 100)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net, meta = load(ckpt)
    log(f"  meta: D={meta['D']} W={meta['W']} attention={meta.get('attention')} direct={meta.get('direct')} "
        f"mode={meta.get('direct_mode')}")
    rows = {}
    for dlabel, fn in DECODERS:
        per = []
        for (j, ctx, reply) in ROW:
            g = fn(net, ctx)
            gw = words_of(g)
            per.append({"conv": j, "text": " ".join(vocab[int(i)] for i in g), "words": gw,
                        "unk": sum(1 for i in g if int(i) == 0) / max(len(g), 1),
                        "hedge": is_hedge(gw), "f1": f1_vs_true(gw, reply),
                        "opening": " ".join(gw[:4]), "true": " ".join(reply[:16])})
        rows[dlabel] = per
        log("")
        log(f"  DECODER {dlabel}")
        log(f"    hedges {sum(1 for r in per if r['hedge'])}/{len(per)}   mean <unk> share "
            f"{100*sum(r['unk'] for r in per)/len(per):.1f}%   mean token-F1 vs TRUE reply "
            f"{sum(r['f1'] for r in per)/len(per):.3f}   best F1 {max(r['f1'] for r in per):.3f}   "
            f"distinct openings {len({r['opening'] for r in per})}")
        for r in per:
            log(f"      conv {r['conv']:>6}  F1 {r['f1']:.3f}  <unk> {100*r['unk']:5.1f}%  "
                f"hedge={str(r['hedge']):5}")
            log(f"                  TRUE  | {r['true']}")
            log(f"                  MODEL | {r['text'][:104]}")
    PANEL[label] = rows

log("")
log("=" * 100)
log("P74 VERDICT PANEL — rule and predictions were fixed in lens/runs/DECODE-SEARCH-PREDICTIONS-P74.txt "
    "BEFORE this run")
log("=" * 100)
for label, rows in PANEL.items():
    g = rows["GREEDY rep1.15 (control)"]
    b4 = rows["BEAM4 rep1.0"]
    b8 = rows["BEAM8 rep1.0"]
    rr = rows["RERANK32 T1.0 k40 rep1.0"]
    gh = sum(1 for r in g if r["hedge"])
    gunk = 100 * sum(r["unk"] for r in g) / len(g)
    log("")
    log(f"  {label}")
    log(f"    P74d control: GREEDY hedges {gh}/6 (predicted 4/6 on the d256 ship candidate), mean <unk> "
        f"{gunk:.1f}% (predicted 2.1% there)")
    log(f"    P74a beam4 hedges {sum(1 for r in b4 if r['hedge'])}/6 (predicted >= 3 on the d256 ship)")
    log(f"    P74b  mean token-F1 vs TRUE reply: greedy {sum(r['f1'] for r in g)/len(g):.3f}   "
        f"beam4 {sum(r['f1'] for r in b4)/len(b4):.3f}   beam8 {sum(r['f1'] for r in b8)/len(b8):.3f}   "
        f"rerank32 {sum(r['f1'] for r in rr)/len(rr):.3f}   (predicted: every variant <= 0.15)")
    log(f"         prompts with F1 >= 0.30: greedy {sum(1 for r in g if r['f1'] >= 0.30)}/6   "
        f"beam4 {sum(1 for r in b4 if r['f1'] >= 0.30)}/6   beam8 {sum(1 for r in b8 if r['f1'] >= 0.30)}/6   "
        f"rerank32 {sum(1 for r in rr if r['f1'] >= 0.30)}/6   (predicted: at most 1 prompt overall)")
    log(f"    P74c rerank32 mean <unk> {100*sum(r['unk'] for r in rr)/len(rr):.1f}% vs greedy {gunk:.1f}%;  "
        f"rerank32 mean F1 {sum(r['f1'] for r in rr)/len(rr):.3f} vs greedy "
        f"{sum(r['f1'] for r in g)/len(g):.3f}  (predicted: rerank32 does NOT beat greedy's F1 by > 0.05)")
    best = max((r for dl in rows.values() for r in dl), key=lambda r: r["f1"])
    log(f"    P74e BEST ANY DECODER DID, by the mechanical proxy: conv {best['conv']} F1 {best['f1']:.3f} "
        f"with text | {best['text'][:104]}")
    log(f"    THE HONEST READING: token-F1 is a proxy, not the decider. The decider is whether the printed text "
        f"answers the prompt; read the MODEL lines above against their TRUE lines and grade them as they are.")

log("")
log(f"files: this run wrote ONLY {OUT} (predictions DECODE-SEARCH-PREDICTIONS-P74.txt, runner log "
    f"DECODE-SEARCH-RUN.txt). No checkpoint and no stored run was edited or overwritten.")
log(f"wall {time.time()-t0:.0f}s")

with open(OUT, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nwritten to {OUT}")