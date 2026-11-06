"""talk_format_probe.py — IS THE TALK TEST ASKING IN THE RIGHT FORMAT?

WHY THIS EXISTS (2026-10-22 17:4x self-wake, CPU only, no GPU use):
Rule 4 says the talk test decides everything, and every arm today fails it while the held-out number
climbs to 35.8% (lens/runs/TIED-FULLCORPUS.txt, arm ct_d256_6M_6ep, best ep5 35.82%). A model that
predicts the next assistant token at 35% on held-out real dialogue should not emit
"* * conclusion * * how long long takes your current experiment" -> the harness and the training data
may not speak the same language. This probe reads the CORPUS ITSELF (mmap, no full build) and prints
the exact token format the model is trained on at a turn boundary, plus whether the tokens the talk
test uses ("human", ":", "assistant") even exist in the vocabulary.

PREDICTIONS (written before running, 2026-10-22 17:4x):
 F1 the corpus uses <|im_start|>/<|im_end|> control tokens for turn boundaries, NOT the words
    "human"/"assistant", so the talk test's prompt format is one the model has never seen.
 F2 "human" and "assistant" are either MISSING from the vocab (mapped to id 0 = pad/unk by
    encode()) or rare, so 3-4 of the 10 prompt tokens are garbage ids.
 F3 vocab[0] is a pad/unk token that never appears in training windows, so the 22-token zero padding
    in generate() is also out of distribution.
 A failure of F1 does NOT prove the model can talk; it only removes this explanation.
"""
import json
import os
import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
D = os.path.join(P, "data/chat/prepared_mix_ctrl_honest")

vocab = json.load(open(os.path.join(D, "vocab_v12000.json")))
ids = np.load(os.path.join(D, "ids_v12000.npy"), mmap_mode="r")
bounds = np.load(os.path.join(D, "bounds_v12000.npy"), mmap_mode="r")
print(f"vocab size {len(vocab):,}   tokens {len(ids):,}   conversations {len(bounds)-1:,}")
print(f"vocab[0..5] = {vocab[0:6]}")
print(f"vocab[-6:]  = {vocab[-6:]}")
for w in ["<|im_start|>", "<|im_end|>", "user", "assistant", "human", "system",
          ":", "what", "name", "hello", "?"]:
    print(f"  {w!r:16} -> {(vocab.index(w) if w in vocab else 'MISSING')}")

# how often do the control tokens actually occur?
arr = np.asarray(ids[:5_000_000])
for w in ["<|im_start|>", "<|im_end|>", "user", "assistant", "human"]:
    if w in vocab:
        c = int((arr == vocab.index(w)).sum())
        print(f"  count in first 5M tokens: {w!r:16} {c:,}  ({c/5_000_000*100:.3f}%)")

# what does a real turn boundary look like, token by token?
starts = np.asarray(bounds)
conv = 3
a, b = int(starts[conv]), int(starts[conv + 1])
seg = np.asarray(ids[a:min(b, a + 160)])
print(f"\nconversation {conv}: {b-a} tokens. First 60 tokens decoded:")
print("  " + " ".join(vocab[i] for i in seg[:60]))

# find the control token inside this conversation and show the boundary
if "<|im_start|>" in vocab:
    ci = int(vocab.index("<|im_start|>"))
    pos = np.where(np.asarray(ids[a:min(b, a + 4000)]) == ci)[0]
    print(f"\n  <|im_start|> appears at {len(pos)} positions in the first 4000 tokens of this conversation")
    if len(pos):
        p0 = int(pos[min(1, len(pos) - 1)])
        lo, hi = max(0, p0 - 12), p0 + 20
        print(f"  around position {p0}:")
        print("    " + " ".join(f"{vocab[i]}" for i in np.asarray(ids[a + lo:a + hi])))
