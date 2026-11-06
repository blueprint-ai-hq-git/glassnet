#!/usr/bin/env python3
"""merge_corpora.py - combine several corpora that share one vocabulary into a single training corpus.

Why this is safe and why it is done this way: every corpus here was built with the SAME 12,000-word
vocabulary (the builders reuse one word list precisely so corpora can be mixed and compared), so the ids
are directly compatible. Merging at the id level therefore changes nothing about the text or the
tokenisation -- it only concatenates.

What it does:
  * concatenates ids, rebuilds the bounds (conversation starts) with the right offsets
  * concatenates the per-conversation split flags
  * records, per source corpus, how many tokens and conversations went in, plus the licences carried in
    each source's meta file, so the combined corpus stays auditable
  * refuses to merge corpora whose vocabularies differ (that would silently corrupt every id)

Usage: python3 lens/scripts/merge_corpora.py <out-dir> <in-dir> [<in-dir> ...]
"""
import json
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
os.chdir(ROOT)

out = sys.argv[1]
srcs = sys.argv[2:]
os.makedirs(out, exist_ok=True)
if not srcs:
    raise SystemExit("usage: merge_corpora.py <out-dir> <in-dir> [<in-dir> ...]")

ref_vocab = None
ids_list, split_list, bounds_list = [], [], []
offset = 0
per_src = {}
for s in srcs:
    vocab = json.load(open(f"{s}/vocab_v12000.json"))
    if ref_vocab is None:
        ref_vocab = vocab
    elif vocab != ref_vocab:
        raise SystemExit(f"REFUSING to merge {s}: its vocabulary differs from {srcs[0]}'s. "
                         f"Merging different vocabularies would silently corrupt every token id.")
    ids = np.load(f"{s}/ids_v12000.npy")
    bounds = np.load(f"{s}/bounds_v12000.npy")
    split = np.load(f"{s}/split_v12000.npy")
    n_conv = len(bounds) - 1
    if len(split) != n_conv:
        print(f"  WARNING {s}: split length {len(split)} != conversations {n_conv}; padding with 0 (train)")
        split = np.pad(split, (0, max(0, n_conv - len(split))))[:n_conv]
    ids_list.append(np.asarray(ids, dtype=np.int32))
    split_list.append(np.asarray(split, dtype=np.int8))
    # keep the leading 0 for the first corpus only; every later corpus drops its own leading 0 and is
    # shifted by the number of tokens already placed
    bounds_list.append(np.asarray(bounds, dtype=np.int64) if not bounds_list
                       else np.asarray(bounds[1:], dtype=np.int64) + offset)
    offset += len(ids)
    meta = json.load(open(f"{s}/meta_v12000.json")) if os.path.exists(f"{s}/meta_v12000.json") else {}
    per_src[s] = {"tokens": int(len(ids)), "conversations": int(n_conv),
                  "licences": meta.get("sources", {}), "val_conversations": int(np.sum(split))}
    print(f"  {s}: {len(ids):,} tokens, {n_conv:,} conversations, {int(np.sum(split)):,} held out")

ids = np.concatenate(ids_list)
bounds = np.concatenate(bounds_list)
split = np.concatenate(split_list)
assert len(bounds) == len(split) + 1, f"bounds {len(bounds)} vs split {len(split)} + 1 mismatch"
assert bounds[-1] == len(ids), f"last bound {bounds[-1]} != token count {len(ids)}"

np.save(f"{out}/ids_v12000.npy", ids)
np.save(f"{out}/bounds_v12000.npy", bounds)
np.save(f"{out}/split_v12000.npy", split)
json.dump(ref_vocab, open(f"{out}/vocab_v12000.json", "w"))
json.dump({"merged_from": srcs, "tokens": int(len(ids)), "conversations": int(len(bounds) - 1),
           "per_source_corpus": per_src, "vocab_size": len(ref_vocab),
           "note": "ids are directly compatible because every source corpus uses the same vocabulary; "
                   "merging only concatenates and rebuilds bounds",
           "merged": __import__("time").strftime("%F %H:%M")},
          open(f"{out}/meta_v12000.json", "w"), indent=1)
print(f"\nwrote {out}: {len(ids):,} tokens, {len(bounds)-1:,} conversations, "
      f"{int(split.sum()):,} held out")