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

v3 = the SHIP DECISION instrument (2026-10-22 19:5x): same measurements, but the checkpoint list comes from the
environment (CKPT_LIST="label=path;label=path") so that this file is never edited twice for a new pair - the v2
list is frozen inside lens/scripts/talk_fair_v2_padfree.py and its output lens/runs/PADFREE-TALK.txt is NOT
touched. Predictions for the v3 run: lens/runs/SHIP-TALK-PREDICTIONS-P72.txt (P72a-h), written BEFORE it.
Run by lens/scripts/run_talk_v3_ship.sh; default output lens/runs/SHIP-TALK-v3.txt.
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
OUT = os.environ.get("PADFREE_OUT", os.path.join(P, "lens/runs/SHIP-TALK-v3.txt"))
# v3: the checkpoint list is ENV-DRIVEN (CKPT_LIST="label=relative/or/abs/path;label2=path2") so this file never
# has to be edited again for a new pair of checkpoints. No CKPT_LIST -> the frozen default below.
_DEFAULT = [
    ("SHIP CANDIDATE: maskfix FULL-corpus d=256 FINAL best (ep3, held-out 35.26%)",
     "lens/runs/chat/ct_d256_full169_fix3_best.npz"),
    ("maskfix FULL-corpus d=256 ep1 (held-out 32.52%, the epoch v2 already reported on)",
     "lens/runs/chat/ct_d256_full169_fix3_ep1.npz"),
]
_spec = os.environ.get("CKPT_LIST", "").strip()
if _spec:
    CKPTS = []
    for _item in _spec.split(";"):
        if not _item.strip():
            continue
        _lab, _p = _item.rsplit("=", 1)   # rsplit: a LABEL may itself contain '=' (it did, first attempt)
        _p = _p.strip()
        CKPTS.append((_lab.strip(), _p if os.path.isabs(_p) else os.path.join(P, _p)))
else:
    CKPTS = [(lab, os.path.join(P, rel)) for lab, rel in _DEFAULT]

# v3 addition: a per-checkpoint SUMMARY the ship decision can be read off without counting <unk>s by eye.
_SUMMARY = []
_F8 = {}

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
# NOTE: the held-out window set is NOT built here any more — it is built per checkpoint below, inside wset(W).

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

# 2026-10-25 self-wake, MEASURED BUG FIX (BUGS.md [BUG-TALK-INSTRUMENT-HARDCODES-W32]): W is no longer one
# literal for the whole file. It is READ PER CHECKPOINT from that checkpoint's own metadata, because a
# window-64 checkpoint refuses 32-wide input BY CONSTRUCTION (lens/model.py line 214) — which is exactly how
# the chained decider died at 11:45:56 on 2026-10-25 and left P95d/e/f/g unmeasured (lens/runs/TIED-W64.txt).
# Everything that depends on W (the held-out window set, the positional assistant mask, the pad-free prompt
# list) is therefore built PER W and cached in _WSET. For a W=32 checkpoint W is still 32, so behaviour on the
# stored ships is unchanged — that is controlled to the digit by P96a in
# lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt.
_WSET = {}


def build_row(W):
    """Real UNPADDED W-token prompts: the first assistant turn starting at index >= W in its own conversation."""
    row = []
    for j in range(len(bounds) - 1):
        if int(split[j]) != 1 or len(row) >= 6:
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
        row.append((j, a0, pre_len, k, reply, ctx))
    return row


def wset(W):
    """The held-out window set, the by-position assistant mask and the pad-free prompts FOR THIS W."""
    if W not in _WSET:
        Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)
        mask = is_asst[np.asarray(Sva) + W]
        unk = float((np.asarray(Yva)[mask] == 0).mean()) if mask.any() else float("nan")
        _WSET[W] = {"W": W, "Xva": Xva, "Yva": Yva, "Sva": Sva, "mask": mask, "unk_share": unk,
                    "ROW": build_row(W)}
        S = _WSET[W]
        log("")
        log(f"WINDOW {W} — held-out windows: {len(mask):,}  assistant-by-position: {int(mask.sum()):,}  "
            f"P71f <unk> (id 0) share among those targets: {unk*100:.3f}%")
        log(f"REAL UNPADDED PROMPTS USED ({len(S['ROW'])} of up to 6 found; the >= {W} constraint is what makes "
            f"them pad-free — if fewer than 6 exist that is printed, never padded):")
        for j, a0, pre_len, k, reply, ctx in S["ROW"]:
            log(f"  conv {j}: first '<|im_start|> assistant' at token {pre_len-2} of its own conversation -> "
                f"prompt is a real {len(ctx)}-token window, pad tokens = 0")
    return _WSET[W]


log("=" * 100)
log("PAD-FREE TALK TEST v3 - THE SHIP DECISION (predictions: lens/runs/SHIP-TALK-PREDICTIONS-P72.txt)  " +
    __import__("time").strftime("%F %H:%M"))
log("=" * 100)
log("(the held-out window set, the assistant mask and the pad-free prompt list are built PER CHECKPOINT, from "
    "that checkpoint's own window — a window-64 ship cannot be measured with 32-wide input)")


def load(ckpt):
    z = np.load(ckpt, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    # ---- THE SENTSTATE GRAFT'S ONE ENTRY POINT (P115e, 2026-10-26)
    # A grafted checkpoint carries six extra tensors. The by-name copy below iterates the NETWORK's parameters,
    # so on an un-grafted net those six are SILENTLY DROPPED and every number printed would belong to a
    # different model (measured 5.742473 of logit difference on the real pre-flight file, no exception:
    # lens/runs/SENTSTATE-DECODE.txt P115f). With GLASSNET_SENTSTATE=1 the net is built by
    # sentstate_wire.load_grafted(), which enables the graft first and REFUSES a partial or absent graft by
    # name. The flag ABSENT path is the code below, unchanged, so every stored arm's numbers keep meaning.
    if os.environ.get("GLASSNET_SENTSTATE", "0").strip() == "1":
        from lens.sentstate_wire import load_grafted
        net = load_grafted(ckpt, device="cpu")
        net.device_name = "cpu"
        log(f"  SENTSTATE GRAFT LOAD: {os.path.basename(ckpt)} built by sentstate_wire.load_grafted() "
            f"({len(net.parameters())} tensors, ds={int(net.ss_ds)})")
        return net
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
    Wn = int(getattr(net, "W", W))        # the CONTEXT TRIM must be this checkpoint's own window
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
        ctx = (ctx + [nxt])[-Wn:]
    return out


def txt(v):
    return " ".join(vocab[i] for i in v)


for label, ckpt in CKPTS:
    log("")
    log("=" * 100)
    log(f"CHECKPOINT {label}: {os.path.basename(ckpt)}")
    net = load(ckpt)
    W = int(getattr(net, "W", W))          # READ FROM THIS CHECKPOINT, not from a file-level constant
    S = wset(W)
    Xva, Yva, Sva, mask, unk_share, ROW = S["Xva"], S["Yva"], S["Sva"], S["mask"], S["unk_share"], S["ROW"]
    log(f"  window read from this checkpoint's own metadata: {W}   "
        f"held-out windows {len(mask):,}   assistant-by-position {int(mask.sum()):,}   "
        f"pad-free prompts {len(ROW)}")
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
    _first8 = {}
    for j, a0, pre_len, k, reply, ctx in ROW:
        g = gen(net, ctx, n_new=24, greedy=True)
        s = gen(net, ctx, n_new=24, temp=0.7, seed=2, top_k=20, rep_penalty=1.3)
        unk = sum(1 for t in g if t == 0) / max(len(g), 1)
        best_run, run = 0, 0
        for t in g:
            run = run + 1 if t != 0 else 0
            best_run = max(best_run, run)
        _SUMMARY.append({"label": label, "conv": j, "unk_share": round(unk, 4),
                         "longest_word_run": best_run, "all_unk": bool(unk == 1.0),
                         "greedy": txt(g), "topk20": txt(s)})
        _first8[j] = list(g[:8])
        log("")
        log(f"    conv {j}  PROMPT (real, pad-free) {txt(ctx[-14:])}")
        log(f"              TRUE REPLY             {' '.join(reply[:16])}")
        log(f"              MODEL GREEDY           {txt(g)}")
        log(f"              MODEL TOPK20           {txt(s)}")
        log(f"              <unk> share (24 greedy) {unk*100:5.1f}%   longest non-<unk> run {best_run}")
    _F8[label] = _first8

log("")
log("STORED BLOCK-A PAD COUNTS (from lens/runs/TALK-FAIR-maskfix-ep1.txt, the prompts it actually fed; that")
log("stored block belongs to the WINDOW-32 twin, so the pad count below is computed against 32 on purpose):")
for line in open(os.path.join(P, "lens/runs/TALK-FAIR-maskfix-ep1.txt")):
    if "PROMPT (real, held-out)" in line:
        toks = line.split("PROMPT (real, held-out)")[1].strip().split()
        log(f"  prompt tokens {len(toks):>3} -> leading <unk> pad tokens fed to the model: {max(0, 32 - len(toks)):>3}"
            f"   [{' '.join(toks[:8])} ...]")

log("")
log("=" * 100)
log("SHIP SUMMARY (v3 addition) - predictions lens/runs/SHIP-TALK-PREDICTIONS-P72.txt (P72a-h)")
log("=" * 100)
for _lab in _F8:
    _rows = [_s for _s in _SUMMARY if _s["label"] == _lab]
    if not _rows:
        continue
    _mean = sum(_r["unk_share"] for _r in _rows) / len(_rows)
    _fluent = sum(1 for _r in _rows if _r["longest_word_run"] >= 4 and not _r["all_unk"])
    log(f"  {_lab}")
    log(f"    mean <unk> share over {len(_rows)} greedy 24-token samples: {_mean*100:5.1f}%   "
        f"samples with a >=4-word run and not all-<unk>: {_fluent}/{len(_rows)}   "
        f"all-<unk> samples: {sum(1 for _r in _rows if _r['all_unk'])}")
    for _r in _rows:
        log(f"      conv {_r['conv']:>6}: <unk> {_r['unk_share']*100:5.1f}%  longest word run {_r['longest_word_run']:>2}"
            f"   {_r['greedy'][:88]}")
if len(_F8) >= 2:
    _labs = list(_F8)
    _diff = 0
    for _j in _F8[_labs[0]]:
        if _j in _F8[_labs[1]] and _F8[_labs[0]][_j] != _F8[_labs[1]][_j]:
            _diff += 1
    log(f"  P72e CROSS-CHECKPOINT DIFF (first 8 greedy tokens): {_diff} of {len(_F8[_labs[0]])} prompts differ "
        f"between '{_labs[0]}' and '{_labs[1]}'")
log("  NOTE (P72f): every accuracy above is a share of a target set in which the printed <unk> share of "
    "assistant-position targets is unguessable by construction; quote the two together.")

with open(OUT, "w") as f:
    f.write("\n".join(_lines) + "\n")
print(f"\nwritten to {OUT}")
