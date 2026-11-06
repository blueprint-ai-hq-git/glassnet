"""talk_fair_test.py — THE TALK TEST'S INSTRUMENT, CHECKED. Then the collapse, measured.

Predictions for this file: lens/runs/TALK-FAIR-PREDICTIONS.txt (P65a-f), written BEFORE this ran.
Output: lens/runs/TALK-FAIR-TEST.txt (this script writes it and prints it).

What it does, in order:
  0. loads the checkpoint WITH ITS OWN direct/tied flags (move 0 is also a bug check on lens.api.load)
  1. REPRODUCES the trainer's teacher-forced masked accuracy (P65e) - if this fails, nothing else here counts
  2. CONTEXT ABLATION: real context vs mismatched context vs shuffled context (P65d)
  3. THE FAIR TALK TEST: the corpus's own ChatML prompt (real held-out dialogue) vs the harness's
     "Human: ... Assistant:" prompt, same decoder, same checkpoint (P65a, P65b)
  4. FREE-RUNNING vs TEACHER-FORCED accuracy by step (P65c), with the true reply printed beside it

CPU only. The GPU arm ct_d256_full169_3 is training while this runs; threads are capped on purpose.
"""
import json
import os
import re
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

import chat_data                                             # noqa: E402
from lens import backend as B                                # noqa: E402
from lens.concepts import Concept                            # noqa: E402
from lens.model import LensNet                               # noqa: E402

# 2026-10-22 18:4x self-wake: CKPT / OUT / the logged number are ENV-OVERRIDABLE so a SECOND checkpoint can be
# measured without overwriting the stored 18:04 TALK-FAIR-TEST.txt. Defaults are unchanged, so re-running with
# no env vars reproduces the stored file exactly. Pre-change copy: lens/runs/_tmp/talk_fair_test_prefix_*.py
CKPT = os.environ.get("TALK_FAIR_CKPT", os.path.join(P, "lens/runs/chat/ct_d256_6M_6ep_best.npz"))
OUT = os.environ.get("TALK_FAIR_OUT", os.path.join(P, "lens/runs/TALK-FAIR-TEST.txt"))
LOGGED_MASKED = float(os.environ.get("TALK_FAIR_LOGGED", "0.3582"))
LOGGED_EP = os.environ.get("TALK_FAIR_EP", "ep5")
PRED_FILE = os.environ.get("TALK_FAIR_PREDICTIONS", "lens/runs/TALK-FAIR-PREDICTIONS.txt")
# 2026-10-25 self-wake, MEASURED BUG FIX (BUGS.md [BUG-TALK-INSTRUMENT-HARDCODES-W32]): W was the literal 32
# here, so feeding a WINDOW-64 checkpoint made the instrument die by construction
# ("ValueError: windows must be 64 tokens wide, got 32", lens/model.py line 214) and P95d was never measured
# (lens/runs/TIED-W64.txt, 11:46). W is now READ FROM THE CHECKPOINT'S OWN METADATA (cheap: the meta is read
# with allow_pickle=False before the corpus is touched); TALK_FAIR_W overrides it by hand if ever needed. For
# a W=32 checkpoint this is the same 32 as before — controlled to the digit by P96a (i) in
# lens/runs/W64-TALK-REPAIR-PREDICTIONS-P96.txt.
_z0 = np.load(CKPT, allow_pickle=False)
_meta0 = json.loads(str(_z0["__meta__"]))
W = int(os.environ.get("TALK_FAIR_W", _meta0["W"]))

_lines = []


def log(s=""):
    _lines.append(str(s))
    print(s, flush=True)


# corpus assets read directly (mmap for the 1.18 GB token stream): the GPU arm ct_d256_full169_3 is
# running beside us and chat_data.build() would load a SECOND full copy of the ids into RAM.
_DD = os.path.join(P, "data/chat/prepared_mix_ctrl_honest")
vocab = json.load(open(os.path.join(_DD, "vocab_v12000.json")))
bounds = np.load(os.path.join(_DD, "bounds_v12000.npy"))
split = np.load(os.path.join(_DD, "split_v12000.npy"))
ids = np.load(os.path.join(_DD, "ids_v12000.npy"), mmap_mode="r")
bounds_a = np.asarray(bounds)
split_a = np.asarray(split)
log("=" * 100)
log("TALK FAIR TEST — is the decider measuring the model or the harness?  " + __import__("time").strftime("%F %H:%M"))
log("=" * 100)
log(f"corpus: {len(ids):,} tokens  {len(bounds_a)-1:,} conversations  vocab {len(vocab):,}")
log(f"checkpoint: {CKPT}")
log(f"logged honest number for it: masked held-out {LOGGED_MASKED*100:.2f}% ({LOGGED_EP}, legacy by-token-id)")
log(f"predictions for this run: {PRED_FILE}")

Xva, Yva, Sva = chat_data.make_windows(ids, bounds_a, split_a, W, which=1, return_index=True)
log(f"held-out windows: {len(Xva):,}  (dtype {Xva.dtype})")

# --- the assistant mask, SAME WALKER AS THE TRAINER (train_chat.py, --assistant-only block),
#     restricted to the held-out conversations so it stays cheap.
is_asst = np.zeros(len(ids), dtype=bool)


def walk(j, mark, offset=0):
    """The trainer's own walker, over one conversation, marking assistant positions.

    `mark` is written at index (a0 + k - offset) when the position is assistant text.
    """
    a0, b0 = int(bounds_a[j]), int(bounds_a[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    k = 0
    in_asst = False
    while k < len(toks):
        if toks[k].lower() == "<|im_start|>" and k + 1 < len(toks):
            in_asst = toks[k + 1].lower() == "assistant"
            k += 2
            continue
        if toks[k].lower() == "<|im_end|>":
            k += 1
            continue
        if in_asst:
            p = a0 + k - offset
            if 0 <= p < len(mark):
                mark[p] = True
        k += 1


for j in range(len(bounds_a) - 1):                      # PASS 1: positional mask, val conversations only
    if split_a[j] == 1:
        walk(j, is_asst)
# PASS 2: the TRAINER'S EXPRESSION. `is_asst[Y]` can only read POSITIONS 0..12,001 (Y holds token ids,
# max 12,001), i.e. the first 12,002 tokens of the corpus - so the legacy mask is that prefix, looked up by
# target token id. Walked separately because PASS 1 covers the held-out conversations and the prefix is not
# in them, which is exactly why the trainer's mask had nothing to do with turns.
legacy_table = np.zeros(len(vocab), dtype=bool)
for j in range(len(bounds_a) - 1):
    if int(bounds_a[j]) >= len(vocab):
        break
    walk(j, legacy_table, offset=0)
mask_legacy = legacy_table[np.asarray(Yva)]             # BUG-MASK-TOKEN-ID: what train_chat.py computed
mask = is_asst[np.asarray(Sva) + W]                     # BY POSITION: the target is really in an assistant turn
log(f"held-out windows whose TARGET is assistant text: {int(mask.sum()):,} of {len(mask):,} "
    f"({mask.mean()*100:.1f}%)  <- THE BY-POSITION MASK (the fix), matches lens/runs/MASK-INDEX-BUG.txt")
log(f"legacy by-token-id mask, for recognition of old logs: {int(mask_legacy.sum()):,} "
    f"({mask_legacy.mean()*100:.1f}%)   overlap {int((mask & mask_legacy).sum()):,}")

if mask.sum() == 0:
    log("REFUSING TO CONTINUE: the assistant mask is EMPTY, so every accuracy below would be a number")
    log("about nothing. That is exactly how my first attempt at this file produced a bogus")
    log("'CONTEXT IS BARELY USED' line at 17:4x. Fix the mask, then re-run.")
    raise SystemExit(1)

z = _z0                                # already loaded above, for W
meta = _meta0
concepts = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
log("")
log(f"checkpoint metadata: d={meta['D']} W={meta['W']} attention={meta.get('attention')} "
    f"direct={meta.get('direct')} direct_mode={meta.get('direct_mode')} rank={meta.get('direct_rank')}")

# THE FLAGS MUST BE PASSED BACK IN. lens.api.load() does NOT pass direct/direct_mode (it passes only
# attention and numeric), so a checkpoint loaded through it silently loses the tied path. Proven below.
# ---- THE SENTSTATE GRAFT (2026-10-26, same rule as the pad-free instrument's P115e branch): a grafted
# checkpoint carries six tensors that this file's by-name loop would drop with only a WARNING, and every
# number below would then belong to a DIFFERENT model. With GLASSNET_SENTSTATE=1 the graft is installed and
# ENABLED before construction, so the six tensors are on the parameter list and are loaded with the rest;
# ds is read from the checkpoint's own sread.W shape. The flag-ABSENT path is this file's code, unchanged.
if os.environ.get("GLASSNET_SENTSTATE", "0").strip() == "1":
    from lens import sentstate_wire as _sw
    _ds = int(np.load(CKPT, allow_pickle=False)["sread.W"].shape[0])
    _sw.install(ds=_ds)
    _sw.enable(ds=_ds)
    log(f"  SENTSTATE GRAFT: graft enabled for construction, ds={_ds} (the six tensors load with the rest)")
net = LensNet(meta["V"], concepts, d=meta["D"], window=meta["W"], seed=meta["seed"],
              lam_concept=meta["lam_concept"], lam_act=meta["lam_act"],
              attention=bool(meta.get("attention", False)), numeric=bool(meta.get("numeric", False)),
              direct=bool(meta.get("direct", False)), direct_rank=int(meta.get("direct_rank", 16)),
              direct_mode=str(meta.get("direct_mode", "lowrank")))
if os.environ.get("GLASSNET_SENTSTATE", "0").strip() == "1":
    _sw.disable()                     # globals back as we found them; this net keeps its own tensors
net.device_name = "cpu"
have = {p.name for p in net.parameters()}
missing = [k for k in z.files if k != "__meta__" and k not in have]
if missing:
    log(f"  WARNING: file has parameters this build does not create: {missing}")
for p in net.parameters():
    if p.name not in z:
        raise KeyError(f"{CKPT} has no parameter {p.name!r}")
    p.data = B.to_dev(z[p.name].astype(np.float32))
log(f"  parameters loaded: {len(have)}")

# --- the api.load BUG CHECK (what every post-hoc tool would have loaded instead)
try:
    import lens.api as api
    net_api = api.load(CKPT, device="cpu")
    api_names = {p.name for p in net_api.parameters()}
    dropped = sorted(have - api_names)
    lg_true = net.forward_arrays(np.ascontiguousarray(Xva[:256]))["logits"]
    lg_api = net_api.forward_arrays(np.ascontiguousarray(Xva[:256]))["logits"]
    md = float(np.abs(lg_true - lg_api).max())
    log("")
    log("  lens.api.load vs the trained model, same checkpoint, same 256 held-out windows:")
    log(f"    parameters the file+metadata need but api.load does not create: {dropped}")
    log(f"    max |logit difference|: {md:.6f}   ({'IDENTICAL' if md == 0.0 else 'DIFFERENT MODEL'})")
except Exception as e:                                     # never let a side check hide the numbers
    log(f"  api.load check raised {type(e).__name__}: {e}")

# ------------------------------------------------------------------ 1. reproduce the trainer (P65e)
def run_acc(X, Y, sel=None, bs=512):
    hit = tot = 0
    for s in range(0, len(X), bs):
        sl = slice(s, min(s + bs, len(X)))
        lg = net.forward_arrays(np.ascontiguousarray(X[sl]))["logits"]
        pred = lg.argmax(axis=1)
        s_ = np.ones(len(pred), bool) if sel is None else sel[sl]
        hit += int((pred[s_] == Y[sl][s_]).sum())
        tot += int(s_.sum())
    return hit / max(tot, 1)

log("")
log("1. REPRODUCTION (P65e): teacher-forced accuracy, BOTH masks, on this checkpoint")
macc = run_acc(Xva, Yva, sel=mask)
macc_legacy = run_acc(Xva, Yva, sel=mask_legacy)
ok = abs(macc_legacy - LOGGED_MASKED) <= 0.002
log(f"   BY POSITION (the fixed assistant-turn column) : {macc*100:.2f}%   ({int(mask.sum()):,} windows)")
log(f"   BY TOKEN ID (LEGACY, what the trainer logged) : {macc_legacy*100:.2f}%   ({int(mask_legacy.sum()):,} windows)")
log(f"   the trainer logged                            : {LOGGED_MASKED*100:.2f}%  <- the LEGACY column")
log(f"   -> {'REPRODUCED' if ok else 'MISMATCH'} : "
    + ("my harness rebuilds the trainer's mask exactly, so the LEGACY column here is the trainer's own number"
       if ok else "MY HARNESS DIFFERS FROM THE TRAINER'S - nothing below may be reported"))

# ------------------------------------------------------------------ 2. context ablation (P65d)
log("")
log("2. CONTEXT ABLATION (P65d): does the model read its own context?")
log(f"   (window READ FROM THE CHECKPOINT: {W} tokens — this line used to say 32 unconditionally)")
rng = np.random.default_rng(0)
Xmis = np.roll(Xva, len(Xva) // 2, axis=0)                 # same targets, windows from other conversations
Xshf = Xva.copy()
for i in range(len(Xshf)):                                 # same window bag, order destroyed
    Xshf[i] = Xshf[i][rng.permutation(W)]
a_real = run_acc(Xva, Yva, sel=mask)
a_mis = run_acc(Xmis, Yva, sel=mask)
a_shf = run_acc(Xshf, Yva, sel=mask)
log(f"   real context      {a_real*100:.2f}%")
log(f"   mismatched context{a_mis*100:.2f}%   (drop {100*(a_real-a_mis):.2f} points)")
log(f"   shuffled tokens   {a_shf*100:.2f}%   (drop {100*(a_real-a_shf):.2f} points)")
log(f"   -> {'CONTEXT IS USED' if (a_real-a_mis) > 0.10 else 'CONTEXT IS BARELY USED (<10 points): the accuracy column has been measuring a marginal-token guesser'}")

# ------------------------------------------------------------------ 3. the fair talk test (P65a,b)
# The harness's decoder, COPIED VERBATIM from lens/scripts/train_chat.py generate() (lines 1002-1041)
# so both formats are decoded by the SAME code. Only the context differs.
def generate_ids(ctx, n_new=45, temp=0.8, greedy=False, seed=0, top_k=40, rep_penalty=1.15):
    ctx = [int(t) for t in ctx]
    out, r = [], np.random.default_rng(seed)
    used = {}
    for _ in range(n_new):
        out_ = net.forward(np.asarray([ctx], dtype=np.int64))   # LensNet.forward returns a DICT
        lg = out_["logits"]
        zz0 = B.to_cpu(lg.data)[0].astype(np.float64)
        for tok, cnt in used.items():
            zz0[tok] -= np.log(rep_penalty) * cnt
        if greedy:
            nxt = int(zz0.argmax())
        else:
            zzz = zz0 / max(temp, 1e-6)
            if top_k and top_k < len(zzz):
                cut = np.partition(zzz, -top_k)[-top_k]
                zzz = np.where(zzz >= cut, zzz, -np.inf)
            zzz = zzz - np.nanmax(zzz[np.isfinite(zzz)])
            p = np.exp(zzz - zzz.max())
            p = np.where(np.isfinite(p), p, 0.0)
            s = p.sum()
            p = p / s if s > 0 else np.ones(len(p)) / len(p)
            nxt = int(r.choice(len(p), p=p))
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-W:]
    return out


def txt(ids_):
    return " ".join(vocab[i] for i in ids_)


# --- real held-out prefixes in the corpus's OWN format: <|im_start|> user ... <|im_start|> assistant
CONVS = []
for j in range(len(bounds_a) - 1):
    if split_a[j] != 1 or len(CONVS) >= 6:
        continue
    a0, b0 = int(bounds_a[j]), int(bounds_a[j + 1])
    t = [vocab[i] for i in np.asarray(ids[a0:b0])]
    try:
        i0 = t.index("<|im_start|>")
        i1 = t.index("<|im_start|>", i0 + 1)
    except ValueError:
        continue
    if i1 + 1 >= len(t) or t[i1 + 1].lower() != "assistant":
        continue
    pre_len = i1 + 2                     # up to and including the "assistant" token
    reply = t[pre_len:pre_len + 30]
    if pre_len < 6 or len(reply) < 20:   # want a short prompt and a real reply to compare against
        continue
    # 2026-10-22 19:0x self-wake, MEASURED BUG FIX (BUGS.md [BUG-PADDED-TALK-PROMPT]): the prompt built below
    # is `ids[a0:a0+pre_len]` LEFT-PADDED with id 0 (=<unk>) to 32 tokens whenever the conversation's opening
    # turn is shorter than the window. Every conversation used in the stored blocks was like that, so the model
    # was asked to continue 18-24 leading <unk> tokens - a context it never sees in training. Measured effect on
    # the mask-fixed checkpoint: 10% <unk> in the reply at 0 pad, 100% at 18 pad
    # (lens/runs/PADFREE-TALK.txt, P71c). From here the corpus-format prompt must be a REAL, unpadded window: a
    # later assistant turn whose start is at least W tokens into its own conversation. Conversations with no such
    # turn are skipped rather than padded.
    if pre_len < W:
        k2 = None
        for k in range(len(t) - 1):
            if t[k] == "<|im_start|>" and t[k + 1].lower() == "assistant" and k + 2 >= W:
                k2 = k
                break
        if k2 is None:
            continue
        i1, pre_len = k2, k2 + 2
        reply = t[pre_len:pre_len + 30]
        if len(reply) < 20:
            continue
    CONVS.append((j, a0, pre_len, reply))

log("")
log("3. THE FAIR TALK TEST (P65a, P65b): same checkpoint, same decoder, only the prompt FORMAT differs")
HARNESS = [
    "Human: hello , how are you ?\n\nAssistant:",
    "Human: what is your name ?\n\nAssistant:",
    "Human: can you help me learn to cook ?\n\nAssistant:",
    "Human: i am feeling sad today .\n\nAssistant:",
    "Human: what is the capital of france ?\n\nAssistant:",
    "Human: tell me about dogs .\n\nAssistant:",
]
TOKRE = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")
vmap = {w: i for i, w in enumerate(vocab)}


def enc_harness(p):
    e = [vmap.get(t, 0) for t in TOKRE.findall(p.lower())]
    if len(e) < W:
        e = [0] * (W - len(e)) + e
    return e[-W:]


rows = []
log("   --- A. CORPUS FORMAT, real held-out dialogue, prompt ids taken straight from the token stream ---")
for (j, a0, pre_len, reply) in CONVS:
    ctx = np.asarray(ids[a0 + pre_len - W:a0 + pre_len]) if pre_len >= W else \
        np.asarray(ids[a0:a0 + pre_len])
    if len(ctx) < W:
        ctx = np.concatenate([np.zeros(W - len(ctx), dtype=ctx.dtype), ctx])
    g = txt(generate_ids(ctx, n_new=45, greedy=True))
    s40 = txt(generate_ids(ctx, n_new=45, temp=0.8, seed=1, top_k=40, rep_penalty=1.15))
    s20 = txt(generate_ids(ctx, n_new=45, temp=0.7, seed=2, top_k=20, rep_penalty=1.3))
    log("")
    log(f"   conv {j}   PROMPT (real, held-out)  {txt(np.asarray(ids[a0 + max(0, pre_len - 26):a0 + pre_len]))}")
    log(f"             TRUE REPLY (held-out)   {' '.join(reply)}")
    log(f"             MODEL GREEDY            {g}")
    log(f"             MODEL TOPK40            {s40}")
    log(f"             MODEL TOPK20            {s20}")
    rows.append({"conv": j, "prompt": txt(np.asarray(ids[a0 + max(0, pre_len - 26):a0 + pre_len])),
                 "true_reply": " ".join(reply), "greedy": g, "topk40": s40, "topk20": s20})

log("")
log("   --- B. THE HARNESS FORMAT, the prompts the recorded talk sections use, same checkpoint ---")
for p in HARNESS:
    ctx = enc_harness(p)
    g = txt(generate_ids(ctx, n_new=45, greedy=True))
    s40 = txt(generate_ids(ctx, n_new=45, temp=0.8, seed=1, top_k=40, rep_penalty=1.15))
    log("")
    log(f"   PROMPT {p.splitlines()[0]!r}   (token ids: {ctx})")
    log(f"          leading id-0 pad tokens in that prompt: {sum(1 for t in ctx if t == 0)} of {W} "
        f"— ABOVE ZERO MEANS THIS BLOCK IS MEASURING THE PADDING (BUG-PADDED-TALK-PROMPT), not the model")
    log(f"          MODEL GREEDY   {g}")
    log(f"          MODEL TOPK40   {s40}")

# ------------------------------------------------------------------ 4. free-running vs teacher-forced (P65c)
log("")
log("4. FREE-RUNNING vs TEACHER-FORCED by step (P65c): can it stay on the rails for 8 tokens?")
sel_idx = np.where(mask)[0]
# windows that have 8 further real assistant tokens inside the same conversation
conv_of = np.searchsorted(bounds_a, Sva, side="right") - 1
keep = []
for i in sel_idx:
    s0 = int(Sva[i])
    if s0 + W + 8 >= int(bounds_a[conv_of[i] + 1]):
        continue
    nxt = np.asarray(ids[s0 + W:s0 + W + 8]).astype(np.int64)
    if not is_asst[s0 + W + 1:s0 + W + 9].all():
        continue
    keep.append(i)
    if len(keep) >= 256:
        break
keep = np.asarray(keep, dtype=np.int64)
log(f"   windows with 8 verifiable real next assistant tokens: {len(keep)} (sampled from the masked held-out set)")
if len(keep):
    Xk = np.asarray(Xva[keep]).astype(np.int64)
    Yk = np.asarray([[int(ids[int(Sva[i]) + W + k]) for k in range(8)] for i in keep])
    fre_ctx = Xk.copy()
    tf_ctx = Xk.copy()
    tf_acc, fr_acc = [], []
    for k in range(8):
        lg_tf = net.forward_arrays(np.ascontiguousarray(tf_ctx))["logits"]
        lg_fr = net.forward_arrays(np.ascontiguousarray(fre_ctx))["logits"]
        tf_p = lg_tf.argmax(axis=1)
        fr_p = lg_fr.argmax(axis=1)
        tf_acc.append(float((tf_p == Yk[:, k]).mean()))
        fr_acc.append(float((fr_p == Yk[:, k]).mean()))
        # teacher-forced: append the REAL token; free-running: append the model's own token
        tf_ctx_next = np.concatenate([tf_ctx[:, 1:], Yk[:, k][:, None]], axis=1)
        fre_ctx = np.concatenate([fre_ctx[:, 1:], fr_p[:, None]], axis=1)
        tf_ctx = tf_ctx_next
    log("   step :     " + "".join(f"{k+1:>7}" for k in range(8)))
    log("   teacher:   " + "".join(f"{a*100:>6.1f}%" for a in tf_acc))
    log("   free-run:  " + "".join(f"{a*100:>6.1f}%" for a in fr_acc))
    log(f"   -> step 8 free-running {fr_acc[-1]*100:.1f}% against teacher-forced {tf_acc[-1]*100:.1f}%")
    # what do the model's own tokens look like?
    log(f"   the model's own first tokens: {txt([int(x) for x in fr_p[:12]])}")
    log(f"   the real tokens             : {txt([int(x) for x in Yk[:12, 0]])}")

log("")
log("=" * 100)
log("VERDICTS (each names the number above it, none of them is a claim about the model talking)")
log("=" * 100)
log(f"  P65e reproduction           : {'PASS' if ok else 'FAIL'}  (LEGACY by-token-id column "
    f"{macc_legacy*100:.2f}% against the logged {LOGGED_MASKED*100:.2f}%; the BY-POSITION column for the same "
    f"windows is {macc*100:.2f}%)")
log(f"  P65d context is used        : {'HELD' if (a_real-a_mis) > 0.10 else 'FAILED'}  "
    f"(mismatched-context drop {100*(a_real-a_mis):.2f} points)")
log(f"  P65c free-running at step 8 : {'HELD' if len(keep) and fr_acc[-1] < 0.10 else 'FAILED'}  "
    f"({fr_acc[-1]*100:.1f}%)" if len(keep) else "  P65c no windows")
log("  P65a/P65b: read the two blocks above - the corpus-format block and the harness-format block are")
log("             the same model, the same decoder and the same checkpoint with ONE variable changed.")

with open(OUT, "w") as f:
    f.write("\n".join(_lines) + "\n")
json.dump({"checkpoint": CKPT, "reproduced_masked": macc, "logs": LOGGED_MASKED,
           "real": a_real, "mismatched": a_mis, "shuffled": a_shf,
           "free_running": fr_acc if len(keep) else None,
           "teacher_forced_by_step": tf_acc if len(keep) else None,
           "corpus_format": rows},
          open(OUT.replace(".txt", ".json"), "w"), indent=1)
print(f"\nwritten to {OUT}")
