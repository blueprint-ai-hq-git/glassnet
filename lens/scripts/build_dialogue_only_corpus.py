#!/usr/bin/env python3
"""build_dialogue_only_corpus.py — SLICE THE HUMAN-DIALOGUE GROUP OUT OF THE MIX CORPUS, WITHOUT RE-TOKENISING.

PREDICTIONS P110a-h WERE WRITTEN BEFORE THIS FILE EXISTED:
    lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt
This script builds the DATA half of that arm and nothing else. It trains nothing, loads no model, and never
materialises a window array (array slicing only), so it cannot repeat the 8.8-9.3 GB build peak that killed two
arms on 2026-10-24 (lens/runs/COORDINATION.txt).

WHAT IT DOES
  source (READ-ONLY): data/chat/prepared_mix_ctrl_honest  — 294,225,090 tokens, 740,665 conversations, a MERGE of
      data/chat/prepared_ext_honest (156,520 conversations: hh-rlhf, oasst1, ultrachat, orca-math = HUMAN
      DIALOGUE) followed by data/chat/prepared_kn2 (584,145: maths/textbook/code/Gutenberg), per the mix's own
      meta_v12000.json. lens/runs/CORPUS-MIX-AUDIT.txt P105a verified the two groups are contiguous in that
      order by decoding samples on both sides of the boundary.
  destination (NEW DIRECTORY, nothing overwritten): data/chat/prepared_dialogue_only_honest
      ids_v12000.npy, bounds_v12000.npy, split_v12000.npy  = the same arrays truncated after conversation
      156,519 (conversation boundaries and the split array carried over ELEMENT FOR ELEMENT, control tokens
      12000/12001 included, no re-tokenisation, no re-encoding, no vocab rebuild)
      vocab_v12000.json = a BYTE COPY of the mix's own vocab file (sha256 compared)
      meta_v12000.json  = the mix's meta with the counts corrected and the provenance of the slice written in
      split_repair_report.json = a copy, so the file's own record of the val repair travels with the slice
      CONSTRUCTION.txt  = the provenance record, with every control's measured value

  The slice is a SLICE: whatever conversation was train stays train, whatever was val stays val, so a trainer
  pointed at it (CHAT_DATA_DIR) sees the same split, the same mask, the same vocabulary and one different thing
  only — which conversations it may sample from.

USAGE (never inline, never behind an env prefix: the approval gate blocks those)
    bash lens/scripts/run_build_dialogue_only.sh
"""
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = os.path.join(P, "data/chat/prepared_mix_ctrl_honest")
DST = os.path.join(P, "data/chat/prepared_dialogue_only_honest")
EXT_KEY = "data/chat/prepared_ext_honest"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def decode(ids, vocab, a, b, limit=28):
    return " ".join(vocab[int(t)] if int(t) < len(vocab) else f"<id{int(t)}>" for t in np.asarray(ids[a:b])[:limit])


def main():
    t0 = time.time()
    print("=" * 110)
    print(f"BUILD DIALOGUE-ONLY CORPUS  {time.strftime('%F %H:%M:%S')}")
    print("predictions P110a-h: lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt (written before this file existed)")
    print(f"source (READ-ONLY): {SRC}")
    print(f"destination (NEW) : {DST}")
    print("=" * 110)

    if not os.path.isdir(SRC):
        print(f"REFUSING: no source corpus at {SRC}")
        return 2
    if os.path.exists(DST):
        print(f"REFUSING: destination already exists: {DST}")
        print("  (this script never overwrites and never repairs a previous build in place — inspect or remove")
        print("   it BY HAND if a rebuild is really wanted, so a half-written corpus can never be trained on)")
        return 3

    meta = json.load(open(os.path.join(SRC, "meta_v12000.json")))
    per = meta["per_source_corpus"]
    keys = [k for k in per if "ext_honest" in k]
    if not keys:
        print(f"REFUSING: cannot find the ext_honest group in the mix meta: {list(per)}")
        return 2
    n_ext = int(per[keys[0]]["conversations"])
    print(f"group boundary, READ from meta_v12000.json: {keys[0]} = conversations 0..{n_ext - 1} "
          f"({n_ext:,} conversations = HUMAN DIALOGUE)")

    ids = np.load(os.path.join(SRC, "ids_v12000.npy"), mmap_mode="r")
    bounds = np.load(os.path.join(SRC, "bounds_v12000.npy"))
    split = np.load(os.path.join(SRC, "split_v12000.npy"))
    vocab = json.load(open(os.path.join(SRC, "vocab_v12000.json")))
    print(f"  source ids {ids.shape} {ids.dtype}   bounds {bounds.shape} {bounds.dtype}   "
          f"split {split.shape} {split.dtype}   vocab {len(vocab):,} entries")

    # ---- bounds convention: with or without a sentinel ------------------------------------------------------
    n_conv = len(split)
    sentinel = (len(bounds) == n_conv + 1) and (int(bounds[-1]) == len(ids))
    if len(bounds) == n_conv:
        print(f"  bounds has NO sentinel ({len(bounds):,} entries for {n_conv:,} conversations)")
    elif sentinel:
        print(f"  bounds HAS a sentinel ({len(bounds):,} entries for {n_conv:,} conversations, "
              f"bounds[-1] == len(ids) == {len(ids):,})")
    else:
        print(f"REFUSING: bounds is {len(bounds):,} entries for {n_conv:,} conversations and does not end with "
              f"len(ids)={len(ids):,} (bounds[-1]={int(bounds[-1]):,}) — the convention is not one this script "
              f"knows, and guessing would silently shift every conversation")
        return 2
    if n_ext > n_conv:
        print(f"REFUSING: the group's own conversation count ({n_ext:,}) exceeds the array's ({n_conv:,})")
        return 2

    tok_end = int(bounds[n_ext])          # first token of the first non-dialogue conversation = the cut
    n_sub = n_ext                          # conversations in the slice
    print("")
    print("SLICE:")
    print(f"  conversations 0..{n_sub - 1}  ({n_sub:,})")
    print(f"  tokens        [0, {tok_end:,}) = {tok_end:,}   (source total {len(ids):,})")
    print(f"  dropped: conversations {n_sub}..{n_conv - 1} ({n_conv - n_sub:,}, the maths/textbook/code group)")

    # ---- C2 (before writing anything): the slice's own ends must read as dialogue --------------------------
    print("")
    print("--- C2 BOUNDARY CHECK (decoded samples of the slice's own ends, before anything is written)")
    print(f"  conv 0            : {decode(ids, vocab, int(bounds[0]), int(bounds[1]))}")
    print(f"  conv {n_sub - 1:<8}: {decode(ids, vocab, int(bounds[n_sub - 1]), tok_end)}")
    print(f"  FIRST DROPPED conv {n_sub:<8}: {decode(ids, vocab, tok_end, int(bounds[n_sub + 1]))}")
    c2 = (int(bounds[0]) == 0) and (tok_end > int(bounds[n_sub - 1]))
    print(f"  C2 {'PASS' if c2 else 'FAIL'} — the slice starts at token 0 and ends at a conversation boundary")
    if not c2:
        print("REFUSING: the slice is not what the meta describes; nothing was written")
        return 2

    # ---- C3: control tokens survive the slice --------------------------------------------------------------
    print("")
    print("--- C3 CONTROL TOKENS <|im_start|>=12000 / <|im_end|>=12001")
    have = set(int(v) for v in np.unique(np.asarray(ids[:tok_end])))
    c3_tokens = [12000 in have, 12001 in have]
    print(f"  distinct ids inside the slice: {len(have):,}   contains 12000: {12000 in have}   "
          f"contains 12001: {12001 in have}")
    print(f"  C3a {'PASS' if all(c3_tokens) else 'FAIL'} — both control tokens occur inside the slice")

    # ---- write --------------------------------------------------------------------------------------------
    os.makedirs(DST)
    print("")
    print(f"WRITING {DST}")
    ids_sub = np.asarray(ids[:tok_end])                      # one slice copy: 156 MB at int32, not 1.17 GB
    np.save(os.path.join(DST, "ids_v12000.npy"), ids_sub)
    bounds_sub = bounds[:n_sub + 1] if sentinel else bounds[:n_sub]
    np.save(os.path.join(DST, "bounds_v12000.npy"), np.asarray(bounds_sub, dtype=np.int64))
    split_sub = split[:n_sub]
    np.save(os.path.join(DST, "split_v12000.npy"), np.asarray(split_sub, dtype=split.dtype))
    shutil.copyfile(os.path.join(SRC, "vocab_v12000.json"), os.path.join(DST, "vocab_v12000.json"))
    srep = os.path.join(SRC, "split_repair_report.json")
    if os.path.exists(srep):
        shutil.copyfile(srep, os.path.join(DST, "split_repair_report.json"))
    print(f"  ids    {ids_sub.shape} {ids_sub.dtype}   ({ids_sub.nbytes:,} B)")
    print(f"  bounds {bounds_sub.shape} {np.asarray(bounds_sub).dtype}")
    print(f"  split  {split_sub.shape} {split_sub.dtype}   val convs in slice: {int((split_sub == 1).sum()):,} "
          f"/ total {len(split_sub):,}")

    m2 = dict(meta)
    m2["conversations"] = int(n_sub)
    m2["tokens"] = int(tok_end)
    m2["n_conv"] = int(n_sub)
    m2["dialogue_only"] = {
        "built": time.strftime("%F %H:%M:%S"),
        "built_by": "lens/scripts/build_dialogue_only_corpus.py",
        "predictions": "lens/runs/DIALOGUE-ONLY-PREDICTIONS-P110.txt",
        "sliced_from": SRC,
        "group": keys[0],
        "conversations": int(n_sub),
        "tokens": int(tok_end),
        "how": "array slice, NO re-tokenisation: ids[0:bounds[n_ext]], bounds[0:n_ext(+1)], split[0:n_ext]; the "
               "mix's own vocab file is copied byte for byte and the mix's split is carried over element for "
               "element, so train/val and every token id keep their meaning",
        "dropped": f"conversations {n_sub}..{n_conv - 1} of the mix (the kn2 maths/textbook/code group)",
    }
    m2["DO_NOT_USE_PER_SOURCE_CORPUS_FOR_GROUP_SPLIT"] = (
        "per_source_corpus below is the PROVENANCE of what this slice contains. THIS CORPUS HOLDS ONE GROUP "
        "ONLY (ext_honest / human dialogue); it has no kn2 conversations, so any instrument that reads the "
        "group boundary from this file will report an empty second group. Point such instruments at "
        "data/chat/prepared_mix_ctrl_honest instead.")
    json.dump(m2, open(os.path.join(DST, "meta_v12000.json"), "w"), indent=1, sort_keys=True)

    # ---- read back and re-check (the project's rule: a written file is not a verified file) -----------------
    print("")
    print("--- READ BACK (nothing below is trusted until it has been read from disk again)")
    r_ids = np.load(os.path.join(DST, "ids_v12000.npy"), mmap_mode="r")
    r_bounds = np.load(os.path.join(DST, "bounds_v12000.npy"))
    r_split = np.load(os.path.join(DST, "split_v12000.npy"))
    r_vocab = json.load(open(os.path.join(DST, "vocab_v12000.json")))
    c1 = (len(r_ids) == tok_end) and (len(r_split) == n_sub)
    print(f"  C1 tokens {len(r_ids):,} (expected {tok_end:,})   conversations {len(r_split):,} "
          f"(expected {n_sub:,}) -> {'PASS' if c1 else 'FAIL'}")
    c3b = sha256(os.path.join(SRC, "vocab_v12000.json")) == sha256(os.path.join(DST, "vocab_v12000.json")) \
        and len(r_vocab) == len(vocab)
    print(f"  C3b vocab byte-identical to the mix's own file (sha256) and {len(r_vocab):,} entries -> "
          f"{'PASS' if c3b else 'FAIL'}")
    c4 = bool(np.array_equal(r_split, np.asarray(split[:n_sub])))
    print(f"  C4 split carried over element for element (vs the source on 0..{n_sub - 1}) -> "
          f"{'PASS' if c4 else 'FAIL'}")
    c5 = bool(((r_split == 1).sum()) <= ((split == 1).sum())) and bool(
        np.array_equal(r_split, np.asarray(split[:n_sub])))
    print(f"  C5 val conversations in slice {int((r_split == 1).sum()):,} of the source's "
          f"{int((split == 1).sum()):,} -> {'PASS' if c5 else 'FAIL'}")
    ok = all([c1, c2, c3b, c4, c5]) and all(c3_tokens)

    print("")
    print("--- P110a's INPUT, MEASURED (the val column's population, from the slice's own arrays)")
    n_val = int((r_split == 1).sum())
    print(f"  val conversations in the slice: {n_val:,}  ({n_val / max(len(r_split), 1) * 100:.2f}% of "
          f"{len(r_split):,})")
    print(f"  the dialogue column of lens/runs/DIALOGUE-VS-MIXED.txt was n = 15,728 assistant-by-position targets "
          f"over 4,501 val conversations of the mix; the arm's own val_asst_n is predicted in [15,600, 15,728]")
    last_val = int(split[n_sub - 1])
    last_msg = ("VAL -> the arm's val set is expected a little under 15,728" if last_val == 1
                else "TRAIN -> length only")
    print("  the slice DROPS conversation %d (the mix's own window builder also drops the last "
          "conversation of whatever corpus it is pointed at); that conversation's split flag is %d (%s)"
          % (n_sub - 1, last_val, last_msg))

    print("")
    print(f"CONSTRUCTION {'COMPLETE' if ok else 'FAILED A CONTROL'}  ({time.time() - t0:.1f}s)")
    print(f"  destination: {DST}")
    if not ok:
        print("  A CONTROL FAILED. This directory must NOT be trained on; report it, do not repair it silently.")
        return 1
    print(f"  next: bash lens/scripts/run_dialogue_only_d256_6ep.sh  (queued; started by the watchdog when no "
          f"trainer is alive)")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    sys.exit(main())
