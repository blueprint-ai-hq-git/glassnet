#!/usr/bin/env python3
"""leak_exact_check.py - are the 9 remaining val/train window overlaps REAL text duplicates, or
fingerprint collisions?

The repair step reduces every window to a 64-bit fingerprint, so two different windows can in principle
collide, and the repair's verdict cannot tell the difference between a collision and a genuine duplicate.
This checks the EXACT BYTES, which is the only thing that settles it. Prints the actual token ids of any
offending window so the claim can be inspected rather than trusted.

Usage: python3 leak_exact_check.py <corpus-dir> [window]
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "data/chat/prepared_mix_ctrl_honest"
    WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    ids = np.load(f"{d}/ids_v12000.npy", mmap_mode="r")
    bounds = np.load(f"{d}/bounds_v12000.npy")
    split = np.load(f"{d}/split_v12000.npy")
    print(f"corpus {d}: {len(ids):,} tokens, {len(bounds)-1:,} conversations, "
          f"val {int((split == 1).sum()):,}")

    def windows(which):
        for j in np.flatnonzero(split == which):
            a, b = int(bounds[j]), int(bounds[j + 1])
            for k in range(0, b - a - WIN - 1, WIN // 2):
                yield int(j), bytes(np.asarray(ids[a + k:a + k + WIN]).tobytes())

    # exact bytes, not fingerprints: this is the authority
    train = set()
    for _, w in windows(0):
        train.add(w)
    print(f"  distinct train windows kept: {len(train):,}")

    hits = []
    n_val = 0
    for j, w in windows(1):
        n_val += 1
        if w in train:
            hits.append((j, np.frombuffer(w, dtype=np.int32)[:8]))
    print(f"  val windows checked: {n_val:,}")
    print(f"  EXACT byte matches against train: {len(hits):,} "
          f"({len(hits)/max(n_val,1)*100:.4f}%)")
    for j, head in hits[:5]:
        print(f"    conversation {j}: window starts with ids {head.tolist()} ...")

    # if the same window occurs more than once INSIDE val, that is a duplicate conversation, not a leak
    val_seen = {}
    dup_in_val = 0
    for j, w in windows(1):
        if w in val_seen and val_seen[w] != j:
            dup_in_val += 1
        else:
            val_seen[w] = j
    print(f"  val windows that also appear in ANOTHER val conversation: {dup_in_val:,} "
          f"(duplicate text inside the corpus, not a leak)")
    verdict = ("CLEAN by exact bytes" if not hits else
               f"REAL OVERLAP: {len(hits)} windows must be moved before any accuracy measured on this val "
               f"can be called honest")
    print(f"  VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())