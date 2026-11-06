"""talk_fair_v2_padfree.py — the talk test on UNPADDED, in-distribution prompts. Predictions P71a-f.

WHY v2 EXISTS (measured, not assumed): the stored talk blocks build their "corpus format" prompt as
ids[a0:a0+pre_len] left-padded with id 0 to W=32 whenever the conversation's opening turn is shorter than 32
tokens. All six conversations used are like that, so the model was asked to continue a context of 18-24 leading
<unk> tokens - a shape it never sees in training. This file measures, for the same checkpoints and the same
decoder, the difference between that padded prompt and a REAL 32-token window (a later assistant turn in the
conversation, chosen so no padding is needed).

IT ALSO MEASURES, for the same checkpoint: the pad sensitivity (share of <unk> among 20 generated tokens as the
number of leading pad tokens rises), the <unk> bias in answer.b, and the share of id-0 targets among
assistant-position held-out targets (P71f).

Writes NOTHING over any stored file; output goes to the file named by PADFREE_OUT (default
lens/runs/PADFREE-TALK.txt). CPU only, one model load per checkpoint.
"""
import json
import os
import sys

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
OUT = os.environ.get("PADFREE_OUT", os.path.join(P, "lens/runs/PADFREE-TALK.txt"))
CKPTS = [
    ("LEGACY ep1 (full corpus)", os.path.join(P, "lens/runs/chat/ct_d256_full169_3_ep1.npz")),
    ("MASKFIX ep1 (full corpus)", os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_best.npz")),
    ("LEGACY best ep3 (full corpus, the shipped candidate before this turn)",
     os.path.join(P, "lens/runs/chat/ct_d256_full169_3_best.npz")),
]

_lines = []


def log(s=""):
    _lines.append(str(s))
    print(s, flush=True)


data = chat_data.build()
ids = np.asarray(data["ids"])
bounds = np.asarray(data["bounds"])
split = np.asarray(data["split"])
vocab = data["vocab"]
vmap = {w: i for i, w in enumerate(vocab)}
Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)

# ---- the assistant mask, by position (the fix's definition), over the held-out windows
is_asst = np.zeros(len(ids), dtype=bool)


def mark_conv(j, mark):
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    k, in_asst = 0, False
    while k < len(toks):
        if toks[k] == "<|im_start|>" and k + 1 < len(toks):
            in_asst = toks[k + 1].lower() == "assistant"
            k += 2
            continue
        if toks[k] == "<|im_end|>":
            k += 1
            continue
        if in_asst:
            mark[a0 + k] = True
        k += 1


for j in range(len(bounds) - 1):
    if int(split[j]) == 1:
        mark_conv(j, is_asst)

mask = is_asst[np.asarray(Sva) + W]
unk_share = float((np.asarray(Yva)[mask] == 0).mean()) if mask.any() else float("nan")
log("=" * 100)
log("PAD-FREE TALK TEST v2 (predictions: lens/runs/PADFREE-TALK-PREDICTIONS-P71.txt)  " +
    __import__("time").strftime("%F %H:%M"))
log("=" * 100)
log(f"held-out windows: {len(mask):,}  assistant-by-position: {int(mask.sum()):,}  "
    f"P71f <unk> (id 0) share among those targets: {unk_share*100:.3f}%")

# ---- real, UNPADDED 32-token prompts: the first assistant turn that starts at index >= W in its conversation
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
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len], dtype=np.int64)   # NO PADDING: a real window
    ROW.append((j, a0, pre_len, k, reply, ctx))
log("")
log("REAL UNPADDED PROMPTS USED (P71a evidence: the padded version of the same conversations had pre_len < 32):")
for j, a0, pre_len, k, reply, ctx in ROW:
    log(f"  conv {j}: first '<|im_start|> assistant' at token {pre_len-2} of its own conversation -> prompt is a "
        f"real {len(ctx)}-token window, pad tokens = 0")
log("  (the stored block A used the OPENING assistant turn and left-padded id 0 whenever pre_len < 32; the pad "
    "count per stored conversation is printed at the END of this file from the stored log itself)")


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
    return net


def gen(net, ctx, n_new=24, temp=0.8, greedy=False, seed=0, top_k=40, rep_penalty=1.15):
    ctx = [int(t) for t in ctx]
    out, rng, used = [], np.random.default_rng(seed), {}
    for _ in range(n_new):
        lg = net.forward_arrays(np.ascontiguousarray(np.asarray([ctx], dtype=np.int64)))["logits"][0]
        z = lg.astype(np.float64).copy()
        for tok, cnt in used.items():
            z[tok] -= np.log(rep_penalty) * cnt
        if greedy:
            nxt = int(z.argmax())
        else:
            zz = z / max(temp, 1e-6)
            if top_k and top_k < len(zz):
                cut = np.partition(zz, -top_k)[-top_k]
                zz = np.where(zz >= cut, zz, -np.inf)
            zz = zz - np.nanmax(zz[np.isfinite(zz)])
            p = np.exp(zz - zz.max())
            p = np.where(np.isfinite(p), p, 0.0)
            s = p.sum()
            p = p / s if s > 0 else np.ones(len(p)) / len(p)
            nxt = int(rng.choice(len(p), p=p))
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out


def txt(v):
    return " ".join(vocab[i] for i in v)


for label, ckpt in CKPTS:
    log("")
    log("=" * 100)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net = load(ckpt)
    ab = None
    for p in net.parameters():
        if p.name == "answer.b":
            ab = np.asarray(p.data)
    if ab is not None:
        r0 = int((ab > ab[0]).sum())
        log(f"  P71b answer.b[<unk>] = {ab[0]:+.4f}   rank among {len(ab)} biases (0 = largest): {r0}   "
            f"max {ab.max():+.4f} at id {int(ab.argmax())} ({vocab[int(ab.argmax())]!r})")
    # ---- pad sensitivity on ONE real prompt (P71c)
    if ROW:
        ctx0 = np.asarray(ROW[0][5], dtype=np.int64)      # a real 32-token window, no padding
        log("  P71c PAD SENSITIVITY (one real prompt, greedy 20 tokens; pad = leading id-0 tokens):")
        for pad in (0, 4, 8, 12, 18, 22):
            c = np.concatenate([np.zeros(pad, dtype=np.int64), ctx0[:W - pad]]) if pad else ctx0
            o = gen(net, c, n_new=20, greedy=True)
            log(f"    pad {pad:>2}: {100.0*sum(1 for t in o if t == 0)/len(o):5.1f}% <unk>   {txt(o)[:96]}")
    # ---- the pad-free talk (P71d/P71e)
    log("  REAL-32-TOKEN-WINDOW TALK (pad-free), greedy and top-k:")
    for j, a0, pre_len, k, reply, ctx in ROW:
        g = gen(net, ctx, n_new=24, greedy=True)
        s = gen(net, ctx, n_new=24, temp=0.7, seed=2, top_k=20, rep_penalty=1.3)
        log("")
        log(f"    conv {j}  PROMPT (real, pad-free) {txt(ctx[-14:])}")
        log(f"              TRUE REPLY             {' '.join(reply[:16])}")
        log(f"              MODEL GREEDY           {txt(g)}")
        log(f"              MODEL TOPK20           {txt(s)}")

log("")
log("STORED BLOCK-A PAD COUNTS (from lens/runs/TALK-FAIR-maskfix-ep1.txt, the prompts it actually fed):")
for line in open(os.path.join(P, "lens/runs/TALK-FAIR-maskfix-ep1.txt")):
    if "PROMPT (real, held-out)" in line:
        toks = line.split("PROMPT (real, held-out)")[1].strip().split()
        log(f"  prompt tokens {len(toks):>3} -> leading <unk> pad tokens fed to the model: {max(0, W - len(toks)):>3}"
            f"   [{' '.join(toks[:8])} ...]")

with open(OUT, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nwritten to {OUT}")
