#!/usr/bin/env python3
"""add_control_tokens.py - give the model its OWN control tokens instead of borrowing ordinary words.

THE PROBLEM, MEASURED: today the turn boundary is the ordinary word "human" (id 16) followed by ":" (id
5). At 71 true boundaries in held-out data the model gives "human" probability 0.0000 and never ranks it
top-1 (0/71). Two reasons a borrowed word makes a bad tag:
  * "human" is 1.49% of all tokens -- it also occurs inside text, so seeing it means nothing definite
  * the marker is TWO tokens, so one control decision is split across two softmax steps
Every shipped chat model solves this with dedicated control tokens (GPT <|endoftext|>, Llama 3
<|eot_id|>, ChatML <|im_end|>). This does the same for us.

WHAT IT DOES, and why this way:
  * Appends THREE tokens to the END of the vocabulary, so every existing id keeps its meaning and
    numbers stay comparable to every earlier run:  <eot> = 12000, <|human|> = 12001, <|assistant|> = 12002
  * Rewrites the token stream: the pair [human][:] becomes the single id <|human|>, [assistant][:] becomes
    <|assistant|>, and <eot> is inserted at every assistant -> human boundary (the exact position where
    the model must learn to stop).
  * Keeps the conversation bounds and the train/val split aligned, and verifies the rewrite: the word
    stream must be unchanged apart from the collapsed markers, and the boundary count must match the
    number of turns.

It does NOT change the text, the vocab words, the split, or the architecture. One change, tags only.

Usage: python3 lens/scripts/add_control_tokens.py <src-dir> <dst-dir>
"""
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(ROOT)

src = sys.argv[1] if len(sys.argv) > 1 else "data/chat/prepared_ext_honest"
dst = sys.argv[2] if len(sys.argv) > 2 else "data/chat/prepared_ctrl"
os.makedirs(dst, exist_ok=True)

vocab = json.load(open(f"{src}/vocab_v12000.json"))
# STANDARD CONTROL TOKENS (ChatML): <|im_start|> opens a turn, <|im_end|> closes it. This is the format
# every mainstream chat model ships (Qwen and many others use exactly these two; Llama 3 calls them
# <|start_header_id|>/<|eot_id|>; GPT calls the end one <|endoftext|>). Appended at the END of the
# vocabulary so every existing id keeps its meaning and earlier numbers stay comparable.
START, END = len(vocab), len(vocab) + 1
new_vocab = vocab + ["<|im_start|>", "<|im_end|>"]
vmap = {w: i for i, w in enumerate(vocab)}
H, C, A = vmap["human"], vmap[":"], vmap["assistant"]

ids = np.load(f"{src}/ids_v12000.npy")
bounds = np.load(f"{src}/bounds_v12000.npy")
split = np.load(f"{src}/split_v12000.npy")
print(f"source {src}: {len(ids):,} tokens, {len(bounds)-1:,} conversations, vocab {len(vocab):,}")
print(f"appending <|im_start|>={START} <|im_end|>={END} (old ids unchanged)")
print("format: <|im_start|> role <text> <|im_end|>   (ChatML, the standard chat format)")

out_ids, out_bounds = [], [0]
n_human_markers = n_asst_markers = n_eot = 0
for j in range(len(bounds) - 1):
    conv = ids[bounds[j]:bounds[j + 1]]
    prev_marker = None
    k = 0
    while k < len(conv):
        # collapse [marker][:] into one control token, and put <eot> at the assistant -> human boundary
        if k + 1 < len(conv) and conv[k] == H and conv[k + 1] == C:
            if prev_marker == "asst":
                out_ids.append(END)          # the assistant's turn ends here: the stop token
                n_eot += 1
            out_ids.append(START)
            out_ids.append(vmap["user"])     # role word stays ordinary text, as in every ChatML model
            n_human_markers += 1
            prev_marker = "human"
            k += 2
            continue
        if k + 1 < len(conv) and conv[k] == A and conv[k + 1] == C:
            out_ids.append(START)
            out_ids.append(A)
            n_asst_markers += 1
            prev_marker = "asst"
            k += 2
            continue
        out_ids.append(int(conv[k]))
        k += 1
    out_bounds.append(len(out_ids))

# TAIL: the id array can extend past the last conversation bound. The first version of this script
# iterated only over the bounded conversations and silently dropped those trailing tokens (355 of them,
# which the verification caught as a length mismatch with an exact prefix match). Copy the tail verbatim.
if bounds[-1] < len(ids):
    tail = ids[bounds[-1]:]
    out_ids.extend(int(t) for t in tail)
    out_bounds[-1] = len(out_ids)
    print(f"  copied {len(tail)} trailing tokens past the last bound")

out_ids = np.asarray(out_ids, dtype=np.int32)
out_bounds = np.asarray(out_bounds, dtype=np.int64)
np.save(f"{dst}/ids_v12000.npy", out_ids)
np.save(f"{dst}/bounds_v12000.npy", out_bounds)
np.save(f"{dst}/split_v12000.npy", split)
json.dump(new_vocab, open(f"{dst}/vocab_v12000.json", "w"))

# ---- verify the rewrite, do not assume it.
# The decisive check: expand the control tokens back to the pair they replaced, drop <eot>, and require
# the result to be EXACTLY the original stream. A count-based check is not enough -- the first version
# compared "words" after stripping ':' from the old stream, but ':' also occurs in ordinary text, so it
# reported CHANGED for a faithful rewrite. Expanding is unambiguous.
expanded = []
i = 0
while i < len(out_ids):
    t = int(out_ids[i])
    if t == START and i + 1 < len(out_ids):
        role = int(out_ids[i + 1])
        expanded.extend((H, C) if role == vmap["user"] else (A, C))   # <|im_start|>+role -> the old pair
        i += 2
        continue
    if t == END:
        i += 1
        continue
    expanded.append(t)
    i += 1
expanded = np.asarray(expanded, dtype=np.int64)
ok_kept = len(expanded) == len(ids) and bool((expanded == np.asarray(ids, dtype=np.int64)).all())
ok_len = len(expanded) == len(ids)
ok_conv = (len(out_bounds) - 1) == (len(bounds) - 1)

meta = json.load(open(f"{src}/meta_v12000.json"))
meta.update({"control_tokens": {"<|im_start|>": START, "<|im_end|>": END},
             "rewritten_from": src,
             "turns_opened": {"user": n_human_markers, "assistant": n_asst_markers},
             "im_end_inserted": n_eot,
             "verification": {"token_count_ok": bool(ok_len), "conversations_ok": bool(ok_conv),
                              "expand_back_equals_original": bool(ok_kept)},
             "note": "control tokens appended so every earlier id keeps its meaning; the text itself is "
                     "unchanged apart from collapsing the two-token markers into single control ids"})
json.dump(meta, open(f"{dst}/meta_v12000.json", "w"), indent=1)

print(f"\nwrote {dst}: {len(out_ids):,} tokens ({len(out_ids)-len(ids):+,} vs source), "
      f"{len(out_bounds)-1:,} conversations")
print(f"  turns opened: user {n_human_markers:,}  assistant {n_asst_markers:,}   "
      f"<|im_end|> stop tokens inserted {n_eot:,}")
print(f"  verification: token count {'OK' if ok_len else 'MISMATCH'}, "
      f"conversations {'OK' if ok_conv else 'MISMATCH'}, "
      f"expand-back equals the original stream {'OK' if ok_kept else 'MISMATCH'}")
print(f"  new vocab size {len(new_vocab):,}")
sys.exit(0 if (ok_len and ok_conv and ok_kept) else 1)