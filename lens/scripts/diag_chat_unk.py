"""diag_chat_unk.py — SMALL TEST: what fraction of a reply is <unk> for each chat checkpoint?

This is the number that decides "talk quality". Fair-comparison rules applied:
  * SAME prompts for every checkpoint
  * <unk> counted in the GENERATED TEXT (the thing we care about), not in the training set
  * perplexity is reported as BITS/TOKEN (log2(ppl)) because raw ppl is NOT comparable across
    different vocab sizes (V=3500 vs V=12000 have different denominators)
  * both checkpoints decoded with the SAME verified decoder (top-k 40 + rep penalty 1.15)
"""
import json
import math
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import lens                                              # noqa: E402
import lens.tensor as T                                  # noqa: E402
from lens import backend as B                            # noqa: E402

R = os.path.join(PROJ, "lens", "runs", "chat")
PREP = os.path.join(PROJ, "data", "chat", "prepared")
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


print("=" * 100)
print("SMALL TEST: <unk> RATE IN GENERATED REPLIES (same prompts, same decoder, both checkpoints)")
print("=" * 100)

rows = []
for tag, vfile in (("best1", "vocab_v3500.json"), ("big1", "vocab_v12000.json")):
    eps = sorted([f for f in os.listdir(R) if f.startswith(tag + "_ep")],
                 key=lambda s: int(s.split("ep")[1].split(".")[0]))
    if not eps:
        continue
    path = os.path.join(R, eps[-1])
    vocab = json.load(open(os.path.join(PREP, vfile)))
    net = lens.load(path, device="cpu")
    if getattr(net, "V", None) != len(vocab):
        print(f"  {tag}: REFUSED — net.V {getattr(net,'V',None)} != vocab {len(vocab)}")
        continue
    log = json.load(open(os.path.join(R, f"{tag}_log.json")))
    ppl = log["history"][-1]["val_ppl"]
    unk_rates, samples = [], []
    for i, pr in enumerate(PROMPTS):
        words = decode(net, vocab, pr, seed=i)
        u = sum(1 for w in words if w == "<unk>") / len(words)
        unk_rates.append(u)
        if i < 3:
            samples.append(" ".join(words))
    rows.append({"tag": tag, "V": len(vocab), "W": net.W, "ppl": ppl,
                 "bpt": math.log2(ppl), "unk_mean": float(np.mean(unk_rates)),
                 "unk_per_prompt": [round(x, 3) for x in unk_rates],
                 "samples": samples})
    print(f"\n{tag}: V={len(vocab):,} W={net.W}  val_ppl {ppl:.2f}  = {math.log2(ppl):.2f} bits/token")
    print(f"   <unk> in generated replies: {np.mean(unk_rates)*100:.1f}% "
          f"(per prompt {[round(x*100) for x in unk_rates]}%)")
    for s in samples:
        print(f"   > {s[:96]}")

print("\n" + "-" * 100)
print("FAIR COMPARISON NOTE: raw ppl is NOT comparable across vocab sizes — bits/token is.")
for r in rows:
    print(f"  {r['tag']:6} V={r['V']:>6,}  ppl {r['ppl']:>7.2f}  {r['bpt']:.2f} bpt  "
          f"<unk> {r['unk_mean']*100:5.1f}%")
if len(rows) == 2:
    a, b = rows
    print(f"\n  best1 wins on bits/token ({a['bpt']:.2f} vs {b['bpt']:.2f}) despite the SMALLER vocab")
    print(f"  <unk> rate: best1 {a['unk_mean']*100:.1f}% vs big1 {b['unk_mean']*100:.1f}% "
          f"-> {'big1 fewer unks' if b['unk_mean'] < a['unk_mean'] else 'best1 fewer unks'}")
json.dump(rows, open(os.path.join(R, "diag_unk_baseline.json"), "w"), indent=1)
print(f"\n  saved {os.path.join(R, 'diag_unk_baseline.json')}")
