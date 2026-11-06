"""build_gate_corpus.py - a SMALL corpus that CAN be memorised, for the gap-gate wiring test (PLAN d2b).

WHY IT EXISTS
-------------
LEARN-NOT-MEMORISE.md item 2: a memorisation gate must be shown to FIRE on a corpus where memorisation is
possible, or it is decoration. The arithmetic harness proved that for the gate's own rule
(lens/runs/GAP-GATE-DETECTOR4.txt), but the SHIPPED trainer (lens/scripts/train_chat.py) has never carried
the gate at all, and a gate that is not wired in fires nowhere. This builds the corpus the wired gate is
tested on: real licensed chat text, taken from data/chat/prepared_clean, but SMALL enough that the model can
fit the training side while the held-out side cannot be fitted.

WHAT IT DOES
------------
Reads prepared_clean's cached token stream and split, keeps that split's OWN train/val labels (so the two
sides stay disjoint - a val token that also appeared in train would make every number measured on it a lie),
and writes a new cache directory in exactly the format scripts/chat_data.py expects, so the trainer can be
pointed at it with CHAT_DATA_DIR and nothing in the data path changes.

NOTHING IS DELETED AND NOTHING IS OVERWRITTEN: the output goes to a NEW directory
(data/chat/prepared_gate_small) and the script refuses to run if that directory already has a cache in it.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
SRC = os.path.join(PROJ, "data", "chat", "prepared_clean")
DST = os.path.join(PROJ, "data", "chat", "prepared_gate_small")
VOCAB_CAP = 12000
WINDOW = 32
TRAIN_WINDOW_BUDGET = 1200      # the memorisable side: small enough to fit
VAL_WINDOW_BUDGET = 360         # the probe side: enough targets that the probe loss is not pure noise
SEED = 20261021


def main():
    os.makedirs(DST, exist_ok=True)
    names = [f"ids_v{VOCAB_CAP}.npy", f"bounds_v{VOCAB_CAP}.npy", f"split_v{VOCAB_CAP}.npy",
             f"vocab_v{VOCAB_CAP}.json", f"meta_v{VOCAB_CAP}.json"]
    existing = [n for n in names if os.path.exists(os.path.join(DST, n))]
    if existing:
        raise SystemExit(f"REFUSING: {DST} already holds a cache ({existing}). Delete it by hand if it is "
                         f"genuinely meant to be rebuilt; this script never overwrites one.")

    ids = np.load(os.path.join(SRC, f"ids_v{VOCAB_CAP}.npy"))
    bounds = np.load(os.path.join(SRC, f"bounds_v{VOCAB_CAP}.npy"))
    split = np.load(os.path.join(SRC, f"split_v{VOCAB_CAP}.npy"))
    vocab = json.load(open(os.path.join(SRC, f"vocab_v{VOCAB_CAP}.json")))
    print(f"source {SRC}: {len(ids):,} tokens, {len(bounds):,} conversations, vocab {len(vocab):,}")

    stride = WINDOW // 2

    def windows_of(j):
        return max(0, (int(bounds[j + 1] - bounds[j]) - WINDOW - 1) // stride + 1)

    rng = np.random.default_rng(SEED)
    train_convs = [j for j in range(len(bounds) - 1) if split[j] == 0]
    val_convs = [j for j in range(len(bounds) - 1) if split[j] == 1]
    # deterministic, spread across the source rather than a prefix (a prefix would take one data source only)
    train_pick, val_pick = [], []
    for pool, budget, out in ((rng.permutation(train_convs), TRAIN_WINDOW_BUDGET, train_pick),
                              (rng.permutation(val_convs), VAL_WINDOW_BUDGET, val_pick)):
        got = 0
        for j in pool:
            out.append(int(j))
            got += windows_of(int(j))
            if got >= budget:
                break
        print(f"  picked {len(out)} conversations, {got} windows of {WINDOW} (stride {stride}) for budget "
              f"{budget}")

    keep = sorted(train_pick + val_pick)
    new_ids, new_bounds, new_split = [], [], []
    for j in keep:
        a, b = int(bounds[j]), int(bounds[j + 1])
        if b - a < WINDOW + 2:
            continue
        new_bounds.append(len(new_ids))
        new_ids.extend(ids[a:b].tolist())
        new_split.append(int(split[j]))
    new_ids = np.asarray(new_ids, dtype=np.int64)
    new_bounds = np.asarray(new_bounds, dtype=np.int64)
    new_split = np.asarray(new_split, dtype=np.int8)

    np.save(os.path.join(DST, f"ids_v{VOCAB_CAP}.npy"), new_ids)
    np.save(os.path.join(DST, f"bounds_v{VOCAB_CAP}.npy"), new_bounds)
    np.save(os.path.join(DST, f"split_v{VOCAB_CAP}.npy"), new_split)
    json.dump(vocab, open(os.path.join(DST, f"vocab_v{VOCAB_CAP}.json"), "w"))
    n_win = {"train": 0, "val": 0}
    for i in range(len(new_bounds) - 1):
        w = max(0, (int(new_bounds[i + 1] - new_bounds[i]) - WINDOW - 1) // stride + 1)
        n_win["train" if new_split[i] == 0 else "val"] += w
    meta = {"n_conv": int(len(new_bounds)), "val_frac": float(new_split.mean()),
            "seed": SEED, "window": WINDOW, "stride": stride, "windows": n_win,
            "source": SRC, "built_by": "lens/scripts/build_gate_corpus.py",
            "why": "a corpus small enough that the model can memorise the train side while the held-out side "
                   "cannot be fitted - the corpus the wired gap gate must FIRE on (PLAN item d2b, "
                   "LEARN-NOT-MEMORISE.md item 2)",
            "licence": "derived from data/chat/prepared_clean (Anthropic/hh-rlhf, MIT) - see DATA_LICENSES.md"}
    json.dump(meta, open(os.path.join(DST, f"meta_v{VOCAB_CAP}.json"), "w"), indent=1)
    print(f"wrote {DST}: {len(new_ids):,} tokens, {len(new_bounds)} conversations "
          f"({int(new_split.sum())} val), windows {n_win['train']} train / {n_win['val']} val")


if __name__ == "__main__":
    main()
