#!/usr/bin/env python3
"""stop_control.py - does the model know when to STOP?

The user's question: does it stop when it should, or does it keep spilling text out forever?
A next-token model has no built-in "I am done" signal; what it can learn from chat data is the turn
boundary (the marker that means "the human speaks next"). This measures, over N free generations with a
generous budget, whether that boundary ever appears, whether output lands on a sentence end, and how
much of each output is unique tokens (a model that repeats itself never stops in any useful sense).

CPU only, nice 19. Nothing is trained; an existing checkpoint is read.
"""
import json
import os
import re
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)          # so `import lens` finds the package
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")

SHIP = os.environ.get("STOP_TEST_SHIP", f"{ROOT}/lens/runs/chat/sb_S0_d256_noattn_ep96.npz")
CORPUS = os.environ.get("CHAT_DATA_DIR", f"{ROOT}/data/chat/prepared_ext")
N_NEW = int(os.environ.get("STOP_TEST_TOKENS", "150"))
N_SAMPLES = int(os.environ.get("STOP_TEST_SAMPLES", "10"))

import chat_data  # noqa: E402
import lens  # noqa: E402

d = chat_data.build()
vocab = d["vocab"]
vmap = {w: i for i, w in enumerate(vocab)}
net = lens.load(SHIP, device="cpu")
ctx_win = int(getattr(net, "W", 32))
V = len(vocab)

PROMPTS = [
    "human : hello , how are you ? assistant :",
    "human : can you help me with my homework ? assistant :",
    "human : what is the capital of france ? assistant :",
    "human : i am feeling sad today . assistant :",
    "human : tell me a short story assistant :",
]


def tokenize(t):
    return re.findall(r"\w+|[^\w\s]", t.lower())


def generate(prompt, seed=0, temp=0.8, top_k=40, rep_penalty=1.15):
    enc = [vmap.get(x, 0) for x in tokenize(prompt)]
    if len(enc) < ctx_win:
        enc = [0] * (ctx_win - len(enc)) + enc
    ctx, out, used = enc[-ctx_win:], [], {}
    for _ in range(N_NEW):
        z = np.asarray(net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0],
                       dtype=np.float64)
        for tok, cnt in used.items():
            z[tok] -= np.log(rep_penalty) * cnt
        z = z / max(temp, 1e-6)
        if top_k and top_k < len(z):
            cut = np.partition(z, -top_k)[-top_k]
            z = np.where(z >= cut, z, -np.inf)
        m = np.nanmax(z[np.isfinite(z)])
        p = np.exp(z - m)
        p = np.where(np.isfinite(p), p, 0.0)
        p = p / p.sum()
        nxt = int(np.random.default_rng(seed + 1000 * len(out)).choice(V, p=p))
        used[nxt] = used.get(nxt, 0) + 1
        out.append(nxt)
        ctx = (ctx + [nxt])[-ctx_win:]
    return out


TURN_MARKERS = {"human", "assistant"}
END_PUNCT = {".", "!", "?"}
rows = []
for i in range(N_SAMPLES):
    pr = PROMPTS[i % len(PROMPTS)]
    toks = generate(pr, seed=i)
    words = [vocab[t] for t in toks]
    handed_back = [k for k, w in enumerate(words) if w in TURN_MARKERS]
    ends_clean = words[-1] in END_PUNCT if words else False
    unique_ratio = len(set(toks)) / max(len(toks), 1)
    rows.append({"prompt": pr, "len": len(toks), "handed_back_at": handed_back[0] if handed_back else None,
                 "ends_on_punctuation": ends_clean, "unique_token_ratio": unique_ratio,
                 "text": " ".join(words).replace(" '", "'")})
n = len(rows)
back = sum(1 for r in rows if r["handed_back_at"] is not None)
clean = sum(1 for r in rows if r["ends_on_punctuation"])
mean_unique = float(np.mean([r["unique_token_ratio"] for r in rows]))
# If the model DOES emit the boundary, that is where a real chat loop would stop it. Report the answer
# length that implies, so "it learned to stop" comes with a number attached rather than a claim.
stop_lens = [r["handed_back_at"] for r in rows if r["handed_back_at"] is not None]
mean_stop_len = float(np.mean(stop_lens)) if stop_lens else None

print(f"STOP CONTROL - ship {os.path.basename(SHIP)}")
print(f"  generations: {n}   budget per generation: {N_NEW} tokens (nothing forced it to stop earlier)")
print(f"  handed the turn back (said 'human' or 'assistant' again): {back}/{n}")
print(f"  ended on a sentence end (. ! ?): {clean}/{n}")
print(f"  mean share of unique tokens per output: {mean_unique*100:.1f}%  "
      f"(low = repeating itself, so 'stopping' is meaningless)")
if mean_stop_len is not None:
    print(f"  when it DOES hand back, it does so after {mean_stop_len:.1f} tokens on average "
          f"-> that is the answer length a chat loop would use")
else:
    print("  it never handed back, so there is no natural answer length to report")
print()
for r in rows[:5]:
    print(f"  [{r['len']} tok, turn back at {r['handed_back_at']}, ends on punct {r['ends_on_punctuation']}, "
          f"unique {r['unique_token_ratio']*100:.0f}%]")
    print(f"    {r['text'][:240]}")
print()
verdict = ("NO STOP CONTROL: it never hands the turn back and never lands on an ending, "
           "so it only stops because we cut it off")
if back == 0 and clean == 0:
    pass
elif back > 0:
    verdict = f"PARTIAL: it hands the turn back in {back}/{n} generations"
print(f"  VERDICT: {verdict}")

json.dump({"ship": SHIP, "n": n, "budget": N_NEW, "handed_back": back, "ends_clean": clean,
           "mean_unique_ratio": mean_unique, "verdict": verdict, "rows": rows},
          open(f"{ROOT}/lens/runs/stop_control_{os.path.basename(SHIP).replace('.npz','')}.json", "w"),
          indent=1)