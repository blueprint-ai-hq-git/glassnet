"""_inspect_unk_argmax.py — WHY DOES THE MASK-FIXED CHECKPOINT EMIT <unk>? (diagnostic, read-only, CPU)

Observed and unexplained (lens/runs/TALK-FAIR-maskfix-ep1.txt, 18:54): ct_d256_full169_fix3_best.npz (mask fix,
ep1) is BETTER teacher-forced (by-position 32.52%) than ct_d256_full169_3_ep1.npz (legacy mask, ep1; 17.24%),
yet when asked to generate from the same six real held-out ChatML prompts it emits <unk> (id 0) for all 45
tokens, where the legacy model emits real English words. A number that is better beside a behaviour that is
worse is exactly the shape of the verdict-check bug this project keeps hitting, so before either is reported I
need to know what the argmax actually is.

HYPOTHESIS H71 (stated before running): <unk> is the argmax at turn-start prompts with a real margin, i.e. the
fixed model has learned <unk> as its default continuation at those positions; the logits are finite and the
<unk> row of the output matrix is not degenerate. If instead I find NaN/inf logits, or one output row whose norm
explodes, the cause is a training/numeric bug and the 32.52% is not reportable.

What it does: loads both ep1 checkpoints, walks 6 real held-out windows that END at "<|im_start|> assistant"
(turn-start prompts), and prints for each: the target token, the top-5 (token, logit) pairs, the logit of the
<unk> row, and the norm of the answer/output weight row for <unk> against the median row norm. Then the same for
6 mid-reply windows. No training, no writes to any stored result file.
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
DD = os.path.join(P, "data/chat/prepared_mix_ctrl_honest")
data = chat_data.build()
ids, bounds, split = np.asarray(data["ids"]), np.asarray(data["bounds"]), np.asarray(data["split"])
vocab = data["vocab"]
Xva, Yva, Sva = chat_data.make_windows(ids, bounds, split, W, which=1, return_index=True)


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
    return net, z


def tops(net, row):
    lg = net.forward_arrays(np.ascontiguousarray(np.asarray([row], dtype=np.int64)))["logits"][0]
    order = np.argsort(lg)[::-1][:5]
    return lg, [(int(i), vocab[i], float(lg[i])) for i in order]


CKPTS = [("LEGACY ep1", os.path.join(P, "lens/runs/chat/ct_d256_full169_3_ep1.npz")),
         ("MASKFIX ep1", os.path.join(P, "lens/runs/chat/ct_d256_full169_fix3_best.npz"))]

# turn-start windows: the last prompt token is 'assistant' (id 18)
turn_start = [i for i in range(len(Xva)) if int(Xva[i][-1]) == 18][:6]
mid_reply = [i for i in range(len(Xva)) if int(Xva[i][-1]) not in (18, 12000, 12001)][:6]

for label, ckpt in CKPTS:
    print("=" * 90)
    print(f"{label}: {os.path.basename(ckpt)}")
    net, z = load(ckpt)
    names = [p.name for p in net.parameters()]
    print("  parameters:", names)
    for p in net.parameters():
        d = np.asarray(p.data)
        print(f"    {p.name:22} shape {str(d.shape):18} |mean| {np.abs(d).mean():.4g} max {np.abs(d).max():.4g} "
              f"finite={np.isfinite(d).all()}")
    for tag, idxs in (("TURN-START PROMPT (ends at 'assistant')", turn_start),
                      ("MID-REPLY PROMPT", mid_reply)):
        print(f"  --- {tag} ---")
        for i in idxs:
            lg, top = tops(net, Xva[i])
            tgt = int(Yva[i])
            finite = bool(np.isfinite(lg).all())
            print(f"    target {vocab[tgt]!r} (id {tgt})   logit {float(lg[tgt]):+.3f}   "
                  f"unk logit {float(lg[0]):+.3f}   finite={finite}   argmax={vocab[int(lg.argmax())]!r}")
            print("        top5:", ", ".join(f"{t!r}:{v:+.2f}" for _, t, v in top))
