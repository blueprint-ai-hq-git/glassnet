"""chat_eval.py — shared, importable chat evaluation (prompts + the verified decoder + unk counting).

Extracted so verdict scripts can reuse the SAME prompts and the SAME decoder without re-running
diag_chat_unk.py's whole test on import. One definition, used everywhere -- so two evaluations can
never silently disagree because one of them drifted.
"""
import json
import math
import os
import re

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

TOK = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")

PROMPTS = [
    "Human: hello , how are you ?\n\nAssistant:",
    "Human: what is your name ?\n\nAssistant:",
    "Human: can you help me with my homework ?\n\nAssistant:",
    "Human: tell me about yourself .\n\nAssistant:",
    "Human: what is the capital of france ?\n\nAssistant:",
    "Human: i am feeling sad today .\n\nAssistant:",
    "Human: write a short poem about the sea .\n\nAssistant:",
    "Human: explain what a computer is .\n\nAssistant:",
    "Human: thank you for your help .\n\nAssistant:",
    "Human: how do i learn to code ?\n\nAssistant:",
]


def decode(net, vocab, prompt, n_new=40, temp=0.8, top_k=40, rep=1.15, seed=0):
    """The decoder the chat sweep VERIFIED: top-k sampling + repetition penalty.
    Greedy loops; pure sampling drifts. This is a port of the verified implementation, not a new one.
    """
    vmap = {w: i for i, w in enumerate(vocab)}
    enc = [vmap.get(t, 0) for t in TOK.findall(prompt.lower())]
    Wn = net.W
    if len(enc) < Wn:
        enc = [0] * (Wn - len(enc)) + enc
    ctx = enc[-Wn:]
    out, used, r = [], {}, np.random.default_rng(seed)
    for _ in range(n_new):
        lg = net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0]
        z = lg.astype(np.float64)
        for tok, cnt in used.items():
            z[tok] -= math.log(rep) * cnt
        zz = z / max(temp, 1e-6)
        if top_k and top_k < len(zz):
            cut = np.partition(zz, -top_k)[-top_k]
            zz = np.where(zz >= cut, zz, -np.inf)
        zz = zz - np.nanmax(zz[np.isfinite(zz)])
        p = np.exp(zz - zz.max())
        p = np.where(np.isfinite(p), p, 0.0)
        s = p.sum()
        p = p / s if s > 0 else np.ones(len(p)) / len(p)
        nxt = int(r.choice(len(p), p=p))
        out.append(vocab[nxt])
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-Wn:]
    return out


def evaluate(tag, runs_dir=None, prep_dir=None, prompts=None):
    """Load a checkpoint's last epoch + its vocab, generate on the shared prompts, count unks."""
    import lens
    R = runs_dir or os.path.join(PROJ, "lens", "runs", "chat")
    PREP = prep_dir or os.path.join(PROJ, "data", "chat", "prepared")
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        return None
    net = lens.load(os.path.join(R, eps[-1]), device="cpu")
    vfile = os.path.join(PREP, f"vocab_v{getattr(net, 'V', 0)}.json")
    if not os.path.exists(vfile):
        return {"tag": tag, "error": f"no vocab file for V={getattr(net, 'V', None)}"}
    vocab = json.load(open(vfile))
    if len(vocab) != net.V:
        return {"tag": tag, "error": f"V mismatch {net.V} vs {len(vocab)}"}
    log = json.load(open(os.path.join(R, f"{tag}_log.json")))
    ppl = log["history"][-1]["val_ppl"]
    unks, samples = [], []
    for i, pr in enumerate(prompts or PROMPTS):
        words = decode(net, vocab, pr, seed=i)
        unks.append(sum(1 for w in words if w == "<unk>") / len(words))
        if i < 3:
            samples.append(" ".join(words))
    return {"tag": tag, "V": len(vocab), "W": net.W, "ppl": ppl, "bpt": math.log2(ppl),
            "unk": float(np.mean(unks)), "samples": samples,
            "unk_per_prompt": [round(x, 3) for x in unks]}
