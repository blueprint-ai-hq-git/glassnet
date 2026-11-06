"""val_train_leak_check.py - is the held-out split of a corpus actually held out?

WHY THIS EXISTS
---------------
The gate test of 2026-10-21 (PLAN item d2b, lens/runs/GATE-TEST.txt) measured something nobody was looking
for: on data/chat/prepared_clean, 5 of 256 probe windows drawn from the HELD-OUT split also appear in the
training set, matched on window content (the 32 tokens plus the target token), not on indices. The small
corpus built for that test shows 0. A hard rule of this project is that no figure may be quoted from a val
that overlaps train, so the size and the CAUSE of that overlap have to be named before the val numbers are
used again.

WHAT IT MEASURES
----------------
At the CONVERSATION level, exactly: for every conversation the corpus marks as val, is its token sequence
byte-identical to some conversation marked train? That is the likely cause - the sources (hh-rlhf
helpful/harmless) contain duplicated conversations, and a split made on ROW INDEX cannot see a duplicate
that landed on the other side. It also reports the same count for the reverse direction, and the fraction of
val CONVERSATIONS and of val TOKENS affected, so the window-level overlap can be checked against it.

Run:  python3 lens/scripts/val_train_leak_check.py [prepared_clean ...]
"""
import hashlib
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
CH = os.path.join(PROJ, "data", "chat")
VOCAB_CAP = 12000


def check(name):
    d = os.path.join(CH, name)
    need = [f"ids_v{VOCAB_CAP}.npy", f"bounds_v{VOCAB_CAP}.npy", f"split_v{VOCAB_CAP}.npy"]
    if not all(os.path.exists(os.path.join(d, n)) for n in need):
        print(f"\n{name}: SKIPPED - no cached ids/bounds/split in {d}")
        return None
    ids = np.load(os.path.join(d, need[0]))
    bounds = np.load(os.path.join(d, need[1]))
    split = np.load(os.path.join(d, need[2]))
    print(f"\n{name}: {len(ids):,} tokens, {len(bounds):,} conversations "
          f"({int((split == 0).sum()):,} train / {int((split == 1).sum()):,} val)")

    # every conversation's token sequence hashed: exact equality, no sampling
    h = {}
    toks = {"train": 0, "val": 0}
    for j in range(len(bounds) - 1):
        a, b = int(bounds[j]), int(bounds[j + 1])
        key = hashlib.md5(np.ascontiguousarray(ids[a:b]).tobytes()).hexdigest()
        h.setdefault(key, {"train": 0, "val": 0, "len": b - a})
        side = "val" if split[j] == 1 else "train"
        h[key][side] += 1
        toks[side] += b - a
    dup_val = [k for k, v in h.items() if v["val"] and v["train"]]
    dup_train = [k for k, v in h.items() if v["train"] > 1]
    dup_val_only = [k for k, v in h.items() if v["val"] > 1]
    n_val_conv = int((split == 1).sum())
    n_val_tok_dup = sum(h[k]["len"] * h[k]["val"] for k in dup_val)
    print(f"  conversations whose token sequence is identical to at least one conversation on the OTHER side: "
          f"{len(dup_val)}")
    print(f"  val conversations that duplicate a TRAIN conversation: "
          f"{sum(h[k]['val'] for k in dup_val)} of {n_val_conv:,} "
          f"({sum(h[k]['val'] for k in dup_val) / max(n_val_conv, 1) * 100:.2f}%)")
    print(f"  val TOKENS inside such conversations: {n_val_tok_dup:,} of {toks['val']:,} "
          f"({n_val_tok_dup / max(toks['val'], 1) * 100:.2f}%)")
    print(f"  duplicate pairs WITHIN train: {len(dup_train)} sequences repeated; "
          f"WITHIN val: {len(dup_val_only)}")
    res = {"corpus": name, "tokens": int(len(ids)), "conversations": int(len(bounds)),
           "train_conversations": int((split == 0).sum()), "val_conversations": n_val_conv,
           "val_conversations_duplicating_train": int(sum(h[k]["val"] for k in dup_val)),
           "val_conversation_leak_fraction": float(sum(h[k]["val"] for k in dup_val) / max(n_val_conv, 1)),
           "val_tokens_duplicating_train": int(n_val_tok_dup),
           "val_token_leak_fraction": float(n_val_tok_dup / max(toks["val"], 1)),
           "duplicate_sequences_within_train": len(dup_train),
           "duplicate_sequences_within_val": len(dup_val_only)}
    return res


def main():
    names = sys.argv[1:] or ["prepared_clean", "prepared_gate_small"]
    out = []
    for n in names:
        r = check(n)
        if r:
            out.append(r)
    p = os.path.join(PROJ, "lens", "runs", "val_train_leak_check.json")
    json.dump(out, open(p, "w"), indent=1)
    print(f"\nwritten to {p}")


if __name__ == "__main__":
    main()
