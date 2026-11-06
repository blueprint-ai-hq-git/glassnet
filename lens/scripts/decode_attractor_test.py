"""decode_attractor_test.py — IS THE HEDGE IN THE DECODER OR IN THE WEIGHTS?

Predictions: lens/runs/DECODE-ATTRACTOR-PREDICTIONS-P73.txt (P73a-i) were written BEFORE this file ran.
Output:      lens/runs/DECODE-ATTRACTOR.txt (env ATTRACTOR_OUT), run log ...-RUN.txt (the runner).

WHY THIS FILE EXISTS. v3's ship talk test (lens/runs/SHIP-TALK-v3.txt) found that five of six greedy samples
open with a hedge ("i ' m not sure ...", "i don't have a lot of information ...", "i ' m sorry ..."). Two
explanations fit that observation and they imply opposite next steps, so this file measures which one is true:

  (1) DECODE-SIDE: greedy (+ a 1.15 repetition penalty) collapses onto the modal opening of the corpus, while the
      model's own distribution over the true reply token is not degenerate.
  (2) IN THE WEIGHTS: the model cannot reach the true reply at all; the hedge is the only thing it can say.

WHAT IS MEASURED, in three parts:
  PART 1 (no model): the corpus's own assistant openings, counted over every assistant turn in the corpus,
                     split into TRAIN and HELD-OUT, with the SAME hedge detector used on generated text.
  PART 2 (the decoder sweep): the same six real, pad-free 32-token prompts as v2/v3, decoded under six decoder
      configurations that differ ONLY in the decoder (greedy/no-penalty/sampled/top-p/nucleus/second seed), plus
      the KV-path question, which the checkpoint metadata answers by itself for these ships (no attention block).
  PART 3 (the decisive test): the TRUE reply, teacher-forced token by token on the same prompt — the RANK of each
      true token in the model's own distribution. Rank is what separates (1) from (2): a true continuation that
      sits at rank 1-20 was always reachable and the decoder is at fault; a true continuation at rank 1000+ is
      not in the model's reach and no decoder will produce it.

It never trains, never writes a checkpoint, never edits another file, and never touches the GPU (device cpu).
Every number it prints is written to its own log, which is the only file it writes.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

import chat_data                                              # noqa: E402
from lens import backend as B                                 # noqa: E402
from lens.concepts import Concept                             # noqa: E402
from lens.model import LensNet                                # noqa: E402

W = 32
OUT = os.environ.get("ATTRACTOR_OUT", os.path.join(P, "lens/runs/DECODE-ATTRACTOR.txt"))

# ---- THE PRE-DECLARED HEDGE DETECTOR (P73, fixed before the run: same list for corpus text and generated text)
HEDGE_FRAGMENTS = (
    "sorry", "not sure", "don ' t know", "don't know", "don ' t have", "don't have",
    "no idea", "can ' t", "cannot", "not a lot of information",
)
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


def log(s=""):
    _lines.append(str(s))
    print(s, flush=True)


def is_hedge(text_words):
    """The pre-declared detector, on a list of word tokens (specials must already be removed)."""
    s = " ".join(text_words[:6]).lower()
    return any(f in s for f in HEDGE_FRAGMENTS)


def words_of(vocab, ids_row):
    return [vocab[int(i)] for i in ids_row]


# ==================================================================================================
# PART 1 - THE CORPUS'S OWN ASSISTANT OPENINGS (no model, no decode)
# ==================================================================================================
t0 = time.time()
data = chat_data.build()
ids = np.asarray(data["ids"])
bounds = np.asarray(data["bounds"], dtype=np.int64)
split = np.asarray(data["split"])
vocab = data["vocab"]
vmap = {w: i for i, w in enumerate(vocab)}
log("=" * 100)
log("DECODE-ATTRACTOR: IS THE HEDGE IN THE DECODER OR IN THE WEIGHTS?   " +
    time.strftime("%F %H:%M:%S"))
log("predictions: lens/runs/DECODE-ATTRACTOR-PREDICTIONS-P73.txt (P73a-i) — written before this run")
log("=" * 100)
log(f"corpus: {len(ids):,} tokens, {len(bounds)-1:,} conversations, vocab {len(vocab):,}  "
    f"(ids dtype {ids.dtype}, loaded in {time.time()-t0:.0f}s)")
log(f"hedge detector (fixed in the predictions, used for corpus AND for generated text): "
    f"{list(HEDGE_FRAGMENTS)}")

IM = vmap.get("<|im_start|>")
AS = vmap.get("assistant")
if IM is None or AS is None:
    raise SystemExit("the corpus format this instrument assumes (<|im_start|> assistant) is not in the vocab")
n = len(ids)
pos = np.flatnonzero(ids == IM)
pos = pos[pos + 1 < n]
pos = pos[ids[np.minimum(pos + 1, n - 1)] == AS]
conv = np.searchsorted(bounds, pos, side="right") - 1
conv = np.clip(conv, 0, len(bounds) - 2)
sp = split[conv]
start = pos + 2
keep = start + 5 <= n
start, sp = start[keep], sp[keep]
vnp = np.array(vocab, dtype=object)
words4 = vnp[ids[start[:, None] + np.arange(5, dtype=np.int64)[None, :]]]      # (N, 5) object array
ctr = {0: Counter(), 1: Counter()}          # opening key -> count, per split
hedge_turns = {0: 0, 1: 0}
tot_turns = {0: 0, 1: 0}
for row, s_ in zip(words4, sp.tolist()):
    ws = [str(w) for w in row]
    if any(w in SPECIALS for w in ws):
        continue
    tot_turns[s_] += 1
    key = " ".join(ws[:4])
    ctr[s_][key] += 1
    if is_hedge(ws):
        hedge_turns[s_] += 1
log("")
log("PART 1 - THE CORPUS'S OWN ASSISTANT OPENINGS (P73a). Every assistant turn in the corpus, first four word "
    "tokens, counted per split.")
for s_ in (0, 1):
    name = "TRAIN" if s_ == 0 else "HELD-OUT"
    tot = max(tot_turns[s_], 1)
    log("")
    log(f"  {name}: {tot_turns[s_]:,} assistant turns;  hedge-family share {100.0*hedge_turns[s_]/tot:.2f}%")
    for k, c in ctr[s_].most_common(12):
        log(f"      {c:>8,}  {100.0*c/tot:6.2f}%  {k!r}{'   <- HEDGE' if is_hedge(k.split()) else ''}")
    if s_ == 0:
        TOP8 = [k for k, _ in ctr[0].most_common(8)]
log(f"  top-8 TRAIN openings used later as 'modal opening' matchers: {TOP8}")
top1_train = ctr[0].most_common(1)[0][0]
p73a_modal_is_hedge = is_hedge(top1_train.split())
p73a_share = 100.0 * hedge_turns[0] / max(tot_turns[0], 1)
log(f"  P73a verdict: modal TRAIN opening {top1_train!r} is a hedge: {p73a_modal_is_hedge}   "
    f"(predicted True); hedge share in TRAIN {p73a_share:.2f}% (predicted >= 8.0%)  -> "
    f"{'PASS' if (p73a_modal_is_hedge and p73a_share >= 8.0) else 'FAIL'}")


# ==================================================================================================
# THE PROMPTS: the same six real, UNPADDED 32-token windows v2/v3 used (same selection code)
# ==================================================================================================
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
    reply = t[pre_len:pre_len + 24]
    if len(reply) < 16:
        continue
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len], dtype=np.int64)
    ROW.append((j, ctx, reply))
log("")
log(f"prompts: {len(ROW)} real pad-free 32-token windows, the same six v3 used "
    f"(conv ids {[r[0] for r in ROW]})")


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
        p.data = B.to_dev(z[p.name].astype(np.float32))
    return net, meta


def logits_of(net, ctx):
    return np.asarray(net.forward_arrays(np.ascontiguousarray(np.asarray([ctx], dtype=np.int64)))["logits"][0],
                      dtype=np.float64)


def gen(net, ctx, n_new=24, mode="greedy", temp=1.0, top_k=0, top_p=0.0, rep=1.15, seed=0):
    """One decoder, parameterised. mode in {'greedy','sample'} — the ONLY differences between variants are these
    arguments, so a variant that moves the opening moved it because of the decoder, not because of the code."""
    ctx = [int(t) for t in ctx]
    out, used = [], {}
    rng = np.random.default_rng(seed)
    for _ in range(n_new):
        z = logits_of(net, ctx).copy()
        if rep != 1.0:
            for tok, cnt in used.items():
                z[tok] -= np.log(rep) * cnt
        if mode == "greedy":
            nxt = int(z.argmax())
        else:
            zz = z / max(temp, 1e-6)
            if top_k and top_k < len(zz):
                cut = np.partition(zz, -top_k)[-top_k]
                zz = np.where(zz >= cut, zz, -np.inf)
            if 0.0 < top_p < 1.0:
                order = np.argsort(-zz)
                srt = zz[order]
                p = np.exp(srt - srt[0])
                p /= p.sum()
                c = np.cumsum(p)
                cut_i = int(np.searchsorted(c, top_p)) + 1
                mask = np.zeros(len(zz), dtype=bool)
                mask[order[:max(cut_i, 1)]] = True
                zz = np.where(mask, zz, -np.inf)
            m = np.nanmax(zz)
            p = np.exp(zz - m)
            s = p.sum()
            p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out


def word_seq(v):
    return [vocab[int(i)] for i in v if vocab[int(i)] not in SPECIALS]


def summarize(net, ctx, gen_ids):
    ws = word_seq(gen_ids)
    unk = sum(1 for i in gen_ids if int(i) == 0) / max(len(gen_ids), 1)
    opening = " ".join(ws[:4])
    return {"unk_share": unk, "opening": opening, "hedge": is_hedge(ws),
            "modal_opening": opening in set(TOP8), "text": " ".join(vocab[int(i)] for i in gen_ids),
            "words": ws}


VARIANTS = [
    ("A greedy rep1.15   (v3's exact decoder — the CONTROL)", dict(mode="greedy", rep=1.15)),
    ("B greedy rep1.00   (repetition penalty REMOVED)", dict(mode="greedy", rep=1.0)),
    ("C sampled T0.7 k20 rep1.00", dict(mode="sample", temp=0.7, top_k=20, rep=1.0, seed=11)),
    ("D sampled T1.0 k40 rep1.15  (shipping default temp)", dict(mode="sample", temp=1.0, top_k=40, rep=1.15, seed=0)),
    ("E sampled T1.0 p0.90 rep1.00 (nucleus)", dict(mode="sample", temp=1.0, top_p=0.90, rep=1.0, seed=5)),
    ("F sampled T1.0 k40 rep1.15 seed 7 (seed control)", dict(mode="sample", temp=1.0, top_k=40, rep=1.15, seed=7)),
]

PANEL = {}
for label, ckpt in CKPTS:
    log("")
    log("=" * 100)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net, meta = load(ckpt)
    ab = None
    for p in net.parameters():
        if p.name == "answer.b":
            ab = np.asarray(p.data)
    log(f"  meta: D={meta['D']} W={meta['W']} attention={meta.get('attention')} direct={meta.get('direct')} "
        f"mode={meta.get('direct_mode')}  params={sum(p.data.size for p in net.parameters()):,}")

    # ---- P73h: the KV path, settled by the checkpoint itself, not by an opinion
    if not meta.get("attention", False):
        log("  P73h KV-DECODE PATH: NOT APPLICABLE to this ship. The checkpoint has no attention block "
            "(attention=False), so lens/kvcache.py::WindowCache refuses it by construction (it raises "
            "'this model has no attention block, so there is nothing for a cache to hook into'). The KV "
            "candidate fix is therefore NOT available for the tied-direct ships; KV parity remains a property "
            "of the attention ships and stays with the queued run_kv_decode.sh line.")
    else:
        log("  P73h KV-DECODE PATH: this ship HAS an attention block, but a KV arm is NOT run here — KV parity "
            "is owned by the queued lens/scripts/run_kv_decode.sh line (P30a-P30i, lens/runs/KV-DECODE.txt), and "
            "running it here would re-measure someone else's stored number with an instrument it does not belong to.")

    if ab is not None:
        rank_unk = int((ab > ab[0]).sum())
        log(f"  reference (v3's P72c/P71b, re-read here): answer.b[<unk>] = {ab[0]:+.4f}, rank {rank_unk} of "
            f"{len(ab)} (0 = largest)")

    # ---- PART 2: the decoder sweep
    log("")
    log("PART 2 — THE DECODER SWEEP on the six pad-free prompts (P73b-P73e). Only the decoder differs per block.")
    panel_rows = {}
    for vlabel, kw in VARIANTS:
        rows = []
        for (j, ctx, reply) in ROW:
            g = gen(net, ctx, n_new=24, **kw)
            rows.append((j, summarize(net, ctx, g)))
        hedge = sum(1 for _, r in rows if r["hedge"])
        modal = sum(1 for _, r in rows if r["modal_opening"])
        mean_unk = sum(r["unk_share"] for _, r in rows) / max(len(rows), 1)
        distinct = len({r["opening"] for _, r in rows})
        panel_rows[vlabel] = dict(hedge=hedge, modal=modal, mean_unk=mean_unk, distinct=distinct, rows=rows)
        log("")
        log(f"  VARIANT {vlabel}")
        log(f"    hedges {hedge}/{len(rows)}   openings that are a top-8 modal corpus opening {modal}/{len(rows)}   "
            f"mean <unk> share {100.0*mean_unk:.1f}%   distinct openings {distinct}")
        for j, r in rows:
            log(f"      conv {j:>6}  <unk> {100.0*r['unk_share']:5.1f}%  hedge={str(r['hedge']):5}  "
                f"| {r['text'][:104]}")

    VA = panel_rows[[k for k in panel_rows if k.startswith("A ")][0]]
    VB = panel_rows[[k for k in panel_rows if k.startswith("B ")][0]]
    best_sampling = min((v for k, v in panel_rows.items() if k[0] in "CDEF"), key=lambda d: d["hedge"])
    best_label = [k for k, v in panel_rows.items() if k[0] in "CDEF" and v is best_sampling][0]

    log("")
    log("  P73b CONTROL CHECK (A must reproduce v3's decoder on the same six prompts):")
    log(f"    A hedges {VA['hedge']}/6 (predicted >= 4 -> {'PASS' if VA['hedge'] >= 4 else 'FAIL'});  "
        f"A mean <unk> share {100.0*VA['mean_unk']:.1f}% (stored v3 value for this checkpoint: 2.1%; predicted "
        f"within +/-1.0 point -> {'PASS' if abs(100*VA['mean_unk'] - 2.1) <= 1.0 else 'FAIL'})")
    log(f"  P73c PENALTY IS NOT THE CAUSE: B (no penalty) hedges {VB['hedge']}/6 "
        f"(predicted >= 3 -> {'PASS' if VB['hedge'] >= 3 else 'FAIL'})")
    log(f"  P73d SAMPLING ESCAPES: best sampling variant = {best_label!r}: hedges {best_sampling['hedge']}/6 "
        f"(predicted <= 2 -> {'PASS' if best_sampling['hedge'] <= 2 else 'FAIL'}), distinct openings "
        f"{best_sampling['distinct']} (predicted >= 4 -> {'PASS' if best_sampling['distinct'] >= 4 else 'FAIL'})")
    log(f"  P73e NO-FLOOD CHECK: B mean <unk> share {100.0*VB['mean_unk']:.1f}% (predicted <= 15.0% -> "
        f"{'PASS' if 100*VB['mean_unk'] <= 15.0 else 'FAIL'})")

    # ---- PART 3: the decisive test — the rank of the TRUE reply token
    log("")
    log("PART 3 — THE DECISIVE TEST (P73f/P73g): teacher-force the TRUE reply and read the RANK of each true token "
        "in the model's own distribution. A true token at rank 1-20 was always reachable; rank 1000+ means no "
        "decoder can produce it.")
    all_ranks, per_conv = [], {}
    for (j, ctx, reply) in ROW:
        ranks, tops = [], []
        cur = [int(t) for t in ctx]
        reply_words = [w for w in reply if w not in SPECIALS]      # feed the true reply, specials skipped
        for step in range(min(12, len(reply_words))):
            used_word = reply_words[step]
            true_id = vmap.get(used_word)
            if true_id is None:
                break
            z = logits_of(net, cur)
            order = np.argsort(-z)
            r = int(np.flatnonzero(order == true_id)[0])
            ranks.append(r)
            if step < 3:
                tops.append((step, used_word, r, round(float(z[order[0]] - z[true_id]), 3),
                             vocab[int(order[0])], vocab[int(order[1])]))
            cur = (cur + [true_id])[-W:]
        if ranks:
            ranks_arr = np.asarray(ranks)
            per_conv[j] = dict(ranks=ranks, median=float(np.median(ranks_arr)),
                               top1=float((ranks_arr == 0).mean()), top5=float((ranks_arr < 5).mean()),
                               top20=float((ranks_arr < 20).mean()), tops=tops)
            all_ranks += ranks
            log("")
            log(f"  conv {j:>6}  true-token ranks over {len(ranks)} steps: {ranks}")
            log(f"              median rank {np.median(ranks_arr):.0f}   top-1 {100*(ranks_arr==0).mean():.0f}%   "
                f"top-5 {100*(ranks_arr<5).mean():.0f}%   top-20 {100*(ranks_arr<20).mean():.0f}%")
            for step, w, r, gap, a1, a2 in tops:
                log(f"              step {step}: true {w!r} rank {r}  (gap to argmax {gap:+.2f}; "
                    f"argmax {a1!r}, 2nd {a2!r})")
    med = float(np.median(all_ranks)) if all_ranks else float("nan")
    arr = np.asarray(all_ranks)
    log("")
    log(f"  POOLED over {len(all_ranks)} teacher-forced steps, {len(per_conv)} prompts: median rank {med:.0f}   "
        f"top-1 {100*(arr==0).mean():.1f}%   top-5 {100*(arr<5).mean():.1f}%   top-20 {100*(arr<20).mean():.1f}%   "
        f"top-100 {100*(arr<100).mean():.1f}%   top-1000 {100*(arr<1000).mean():.1f}%")

    # ---- the sampling escape count, for the verdict rule
    esc = []
    for (j, ctx, reply) in ROW[:3]:
        hs = 0
        for s in range(8):
            g = gen(net, ctx, n_new=24, mode="sample", temp=0.8, top_k=40, rep=1.0, seed=100 + s)
            if summarize(net, ctx, g)["hedge"]:
                hs += 1
        esc.append(hs)
    log("")
    log(f"  ESCAPE COUNT (independent samples at T=0.8/top-k40/no penalty, 8 per prompt, first 3 prompts): "
        f"{esc} hedges out of 8 each")
    PANEL[label] = dict(A=VA, B=VB, best=best_sampling, best_label=best_label, med=med, per_conv=per_conv,
                        hedges=dict(A=VA["hedge"], B=VB["hedge"], best=best_sampling["hedge"]),
                        escape=esc, skip=not meta.get("attention", False))

# ==================================================================================================
# P73i - THE VERDICT RULE (fixed in the predictions before any of the above ran)
# ==================================================================================================
log("")
log("=" * 100)
log("P73i VERDICT PANEL — the rule was fixed in lens/runs/DECODE-ATTRACTOR-PREDICTIONS-P73.txt BEFORE the run,")
log("     and the values it compares are printed beside it.")
log("=" * 100)
for label, p in PANEL.items():
    best_hedge = p["best"]["hedge"]
    med = p["med"]
    if not np.isfinite(med):
        verdict = "NO_DATA"
    elif best_hedge <= 2 and med <= 50:
        verdict = "DECODE_SIDE"
    elif med >= 200:
        verdict = "IN_WEIGHTS"
    else:
        verdict = "MIXED"
    log("")
    log(f"  {label}")
    log(f"    hedges: greedy(A) {p['A']['hedge']}/6   no-penalty(B) {p['B']['hedge']}/6   "
        f"best sampling {best_hedge}/6 ({p['best_label']})   distinct sampling openings "
        f"{p['best']['distinct']}")
    log(f"    escapes per prompt (8 samples each, first 3 prompts): {p['escape']}")
    log(f"    teacher-forced TRUE-token median rank: {med:.0f}   "
        f"(rule: <=50 supports DECODE_SIDE, >=200 is IN_WEIGHTS)")
    log(f"    modal-opening rate of greedy output: {p['A']['modal']}/6")
    log(f"    -> RULE FIRED: {verdict}")
    if verdict == "MIXED":
        log("       reading: the hedge is escapable by decoding, but the true reply is not reachable from the "
            "model's own distribution at this rank — the next lever is the objective / concept-to-answer "
            "interface, not the decoder and not raw capacity.")
    elif verdict == "DECODE_SIDE":
        log("       reading: the true continuation IS reachable (median rank <= 50) and a decode variant removes "
            "the hedge — the next lever is the decoder (beam / no-repeat-ngram), which is CPU-cheap.")
    elif verdict == "IN_WEIGHTS":
        log("       reading: the true continuation is not in the model's reach at all; no decoder will produce it.")

log("")
log(f"files: this run wrote ONLY {OUT} (predictions are DECODE-ATTRACTOR-PREDICTIONS-P73.txt; runner log is "
    f"DECODE-ATTRACTOR-RUN.txt). No checkpoint, no stored run and no instrument was edited or overwritten.")
log(f"wall {time.time()-t0:.0f}s")

with open(OUT, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nwritten to {OUT}")
