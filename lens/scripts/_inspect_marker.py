"""_inspect_marker.py — READ-ONLY data inspection: what token follows the assistant marker in the corpus?

Question this answers (raised by lens/runs/TALK-FAIR-maskfix-ep1.txt): the mask-fixed model at ep1 emits
<unk> (id 0) for all 45 generated tokens from a real held-out ChatML prompt, while the legacy-mask model at the
same epoch emits real English words. Both are teacher-forced at 32.52% / 17.24% by position. If the token right
after "<|im_start|> assistant" is id 0 in the corpus, then the by-position mask (which DOES cover that position,
because it is inside the assistant turn) trains the model to predict <unk> there - and the legacy by-token-id
mask happened to give that position weight 0. That would explain both observations with one mechanism.

No model, no training, read-only. Prints counts over the first 2000 conversations.
"""
import json
import os

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
DD = os.path.join(P, "data/chat/prepared_mix_ctrl_honest")
vocab = json.load(open(os.path.join(DD, "vocab_v12000.json")))
vmap = {w: i for i, w in enumerate(vocab)}
ids = np.load(os.path.join(DD, "ids_v12000.npy"), mmap_mode="r")
bounds = np.load(os.path.join(DD, "bounds_v12000.npy"))

n_conv = 2000
after_marker = {}
first_content = {}
examples = []
for j in range(n_conv):
    a0, b0 = int(bounds[j]), int(bounds[j + 1])
    toks = [vocab[i] for i in np.asarray(ids[a0:b0])]
    for k in range(len(toks) - 1):
        if toks[k] == "<|im_start|>" and toks[k + 1].lower() == "assistant":
            nxt = toks[k + 2] if k + 2 < len(toks) else "<END>"
            after_marker[nxt] = after_marker.get(nxt, 0) + 1
            if len(examples) < 4:
                seg = toks[k:k + 12]
                segids = [int(x) for x in np.asarray(ids[a0 + k:a0 + k + 12])]
                examples.append((j, seg, segids))
        if toks[k] == "<|im_start|>" and toks[k + 1].lower() == "user":
            nxt = toks[k + 2] if k + 2 < len(toks) else "<END>"
            first_content[nxt] = first_content.get(nxt, 0) + 1

print(f"conversations inspected: {n_conv}")
print("token immediately AFTER '<|im_start|> assistant':")
for t, c in sorted(after_marker.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   {t!r:24} id {vmap.get(t, -1):>6}   {c}")
print("token immediately AFTER '<|im_start|> user':")
for t, c in sorted(first_content.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   {t!r:24} id {vmap.get(t, -1):>6}   {c}")
print("examples (conv, tokens after the marker, their ids):")
for j, seg, segids in examples:
    print(f"   conv {j}: {seg}")
    print(f"             {segids}")
print("vocab id 0 is:", repr(vocab[0]), " id 1:", repr(vocab[1]), " id 2:", repr(vocab[2]))
print("'assistant' id:", vmap.get("assistant"), " '<|im_start|>' id:", vmap.get("<|im_start|>"),
      " '<|im_end|>' id:", vmap.get("<|im_end|>"))
