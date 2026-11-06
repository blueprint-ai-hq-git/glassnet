#!/usr/bin/env python3
"""repair_split.py — build a copy of a corpus whose VAL is disjoint from TRAIN at the training window
size.

WHY (2026-10-17): the leak audits for these corpora were run on 100-token windows and reported zero
leaks, but the model trains on 32-token windows, where 8% (merged) to 38% (splitfix) of val windows
appear VERBATIM in the training text. A number measured on such a val is partly a memorisation score.
The fix belongs in the DATA, not in a filter applied after the fact.

WHAT IT DOES
For every val conversation, if ANY of its windows occurs verbatim in the train windows, the whole
conversation moves to train (moving whole conversations keeps dialogues intact -- a half-moved dialogue
would leak the other half). Nothing is deleted, and the source corpus is never modified: the repaired
corpus is written to a NEW directory with its own split file.

MEMORY REWRITE (2026-10-19 08:2x). The first version called chat_data.make_windows twice, which
materialises EVERY window of the corpus: 16.9M train windows of 32 int64 tokens is 4.3 GB, plus the val
side, plus np.isin over both. That needs about 10 GB. On a 14 GB machine it competed with another job and
the kernel killed it at 08:17, which is why the combined corpus never got finished overnight.
This version never holds the windows: it walks the corpus conversation by conversation, reduces each window
to a 64-bit fingerprint (a rolling multiply-add over its token ids), keeps only the 8-byte fingerprints of
train (16.9M x 8 bytes = 135 MB), and tests val fingerprints against them with np.isin on sorted uint64.
Same decision, same output, roughly 1.5 GB instead of 10 GB.

Collision risk: a 64-bit fingerprint over 17M windows has a birthday-collision chance of about 1e-5, and a
collision can only move one extra conversation from val to train, which is the safe direction (it removes
data from the measured set rather than adding unseen data to it).

SELF-CHECKS (printed, not assumed)
* the window count derived by streaming must match chat_data.make_windows' count exactly
* after the repair, the val/train overlap must measure 0; if it does not, the script says so

Usage: python3 repair_split.py <src-corpus-dir> <dst-corpus-dir> [window]
"""
import json
import os
import time
import shutil
import sys

import numpy as np

sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/lens/scripts")

import chat_data  # noqa: E402

MULT = np.uint64(1000003)


def fingerprints(W):
    """(n, window) token ids -> (n,) uint64 fingerprints. Vectorised, no hashing library needed.

    Empty input is valid and must not crash: a conversation shorter than the window produces no windows,
    and the first version of this function indexed shape[1] on an empty array and died with IndexError.
    """
    v = np.asarray(W, dtype=np.int64)
    if v.size == 0:
        return np.zeros(0, dtype=np.uint64)
    v = v.reshape(-1, v.shape[-1]).astype(np.uint64)
    h = np.zeros(len(v), dtype=np.uint64)
    for k in range(v.shape[1]):
        h = h * MULT + v[:, k]
    return h


def window_count(bounds, split, which, WIN):
    """Exact number of windows make_windows would produce, without building any of them."""
    n = 0
    for j in np.flatnonzero(split == which):
        seg = int(bounds[j + 1] - bounds[j])
        n += len(range(0, seg - WIN - 1, WIN // 2))
    return n


def stream_fingerprints(ids, bounds, split, which, WIN, conv_batch=512):
    """Yield (conv_index, fingerprints) batches, one conversation group at a time. Bounded memory."""
    idx = np.flatnonzero(split == which)
    for s in range(0, len(idx), conv_batch):
        convs = idx[s:s + conv_batch]
        ws, cs = [], []
        for j in convs:
            a, b = int(bounds[j]), int(bounds[j + 1])
            for k in range(0, b - a - WIN - 1, WIN // 2):
                ws.append(ids[a + k:a + k + WIN])
                cs.append(int(j))
        if ws:
            yield np.asarray(cs, dtype=np.int64), fingerprints(np.stack(ws))


def main():
    src, dst = sys.argv[1], sys.argv[2]
    WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    ids = np.load(f"{src}/ids_v12000.npy")
    bounds = np.load(f"{src}/bounds_v12000.npy")
    split = np.load(f"{src}/split_v12000.npy")

    n_tr = window_count(bounds, split, 0, WIN)
    n_va = window_count(bounds, split, 1, WIN)

    # SELF-CHECK on a tiny synthetic corpus: the streaming derivation must reproduce make_windows EXACTLY,
    # not merely produce a similar count. This is the check that the stride rule still matches, and it costs
    # nothing in memory. (The previous version passed make_windows a max_windows argument that does not
    # exist and died on TypeError before doing any work.)
    s_ids = np.arange(1, 201, dtype=np.int64)
    s_bounds = np.asarray([0, 40, 160, 200], dtype=np.int64)   # val conv is 120 tokens: many windows
    s_split = np.asarray([0, 1, 0], dtype=np.int64)
    Xref, _ = chat_data.make_windows(s_ids, s_bounds, s_split, WIN, which=1)
    mine = [fp for _, fp in stream_fingerprints(s_ids, s_bounds, s_split, 1, WIN)]
    n_mine = sum(len(f) for f in mine)
    assert n_mine == len(Xref), f"streaming produced {n_mine} val windows, make_windows produced {len(Xref)}"
    assert np.array_equal(np.sort(fingerprints(Xref)), np.sort(np.concatenate(mine))), \
        "fingerprints do not match the windows make_windows produces"
    print(f"  self-check passed: streaming reproduces make_windows exactly on a synthetic corpus "
          f"({n_mine} val windows, identical fingerprints)")
    print(f"  corpus window counts to process: train {n_tr:,}  val {n_va:,}")

    # train side: keep only fingerprints, sorted for fast membership tests
    print("  fingerprinting the train windows (streaming, nothing materialised)...")
    tr_fp = np.concatenate([fp for _, fp in stream_fingerprints(ids, bounds, split, 0, WIN)])
    assert len(tr_fp) == n_tr, f"train window count mismatch {len(tr_fp):,} vs {n_tr:,}"
    tr_fp.sort()
    print(f"  train fingerprints: {len(tr_fp):,} ({tr_fp.nbytes/1e6:.0f} MB)")

    print("  testing every val window against them...")
    hit_conv = set()
    seen_va = 0
    for cs, fp in stream_fingerprints(ids, bounds, split, 1, WIN):
        seen_va += len(fp)
        m = np.isin(fp, tr_fp)
        for c in np.unique(cs[m]):
            hit_conv.add(int(c))
    assert seen_va == n_va, f"val window count mismatch {seen_va:,} vs {n_va:,}"
    print(f"  val conversations containing at least one window seen in train: {len(hit_conv):,} "
          f"of {int((split == 1).sum()):,}")

    new_split = split.copy()
    for c in hit_conv:
        new_split[c] = 0

    # ITERATE TO A FIXED POINT. Moving a val conversation into train ADDS its windows to train, and those
    # new train windows can overlap val conversations that were clean a moment ago. One pass therefore
    # cannot reach zero: the first version left exactly 9 windows overlapping, confirmed by exact-byte
    # comparison (not by fingerprints, which cannot tell a duplicate from a collision). Loop until no val
    # conversation overlaps train, and say how many passes it took.
    moved_total = len(hit_conv)
    for it in range(2, 12):
        tr_fp_it = np.concatenate([fp for _, fp in stream_fingerprints(ids, bounds, new_split, 0, WIN)])
        tr_fp_it.sort()
        hit_it = set()
        for cs, fp in stream_fingerprints(ids, bounds, new_split, 1, WIN):
            m = np.isin(fp, tr_fp_it)
            for c in np.unique(cs[m]):
                hit_it.add(int(c))
        print(f"  pass {it}: {len(hit_it):,} val conversations still overlapping train")
        if not hit_it:
            print(f"  fixed point reached after {it-1} pass(es)")
            break
        for c in hit_it:
            new_split[c] = 0
        moved_total += len(hit_it)
    print(f"  conversations moved to train in total: {moved_total:,} "
          f"(of {int((split == 1).sum()):,} originally held out)")

    # A destination holding files from an EARLIER run must never be silently reused. On 2026-10-19 the
    # morning's repair wrote into an existing directory, skipped the ids/bounds because they "existed", and
    # produced a corpus whose ids came from a 13.3M-token partial run while its split came from the new
    # 294M-token source. The mixed corpus then failed validation with no visible cause and hours were spent
    # chasing it. Validate the bytes; move a stale destination aside (never delete); then write.
    os.makedirs(dst, exist_ok=True)
    for name in ("ids_v12000.npy", "bounds_v12000.npy"):
        s_file, d_file = os.path.join(src, name), os.path.join(dst, name)
        if os.path.exists(d_file) and os.path.getsize(d_file) != os.path.getsize(s_file):
            stale = f"{dst}_stale_{time.strftime('%Y%m%d_%H%M')}"
            print(f"  destination {name} does not match the source "
                  f"({os.path.getsize(d_file):,} vs {os.path.getsize(s_file):,} bytes)")
            print(f"  moving the stale destination aside to {stale} (nothing is deleted)")
            os.rename(dst, stale)
            os.makedirs(dst, exist_ok=True)
            break

    for name in os.listdir(src):
        if name in ("split_v12000.npy", "split_repair_report.json"):
            continue
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if not os.path.exists(d):
            try:
                os.link(s, d)          # hard link: no extra disk, source untouched
            except OSError:
                shutil.copy2(s, d)
    np.save(f"{dst}/split_v12000.npy", new_split)
    json.dump({"repaired_from": src, "window": WIN, "moved_convs": len(hit_conv),
               "val_convs_before": int((split == 1).sum()),
               "val_convs_after": int((new_split == 1).sum())},
              open(f"{dst}/split_repair_report.json", "w"), indent=1)

    # verify the repair actually removed the overlap, using the same fingerprint method
    tr_fp2 = np.concatenate([fp for _, fp in stream_fingerprints(ids, bounds, new_split, 0, WIN)])
    tr_fp2.sort()
    still = 0
    va_after = 0
    for _, fp in stream_fingerprints(ids, bounds, new_split, 1, WIN):
        va_after += len(fp)
        still += int(np.isin(fp, tr_fp2).sum())
    print(f"  AFTER REPAIR: val windows {va_after:,}   still appearing in train: {still:,} "
          f"({still/max(va_after,1)*100:.4f}%)")
    print(f"  written: {dst}  (source corpus untouched)")
    print(f"  VERDICT: {'CLEAN' if still == 0 else 'STILL LEAKING -- do not use'}")


if __name__ == "__main__":
    main()