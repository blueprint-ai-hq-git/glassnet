#!/bin/bash
# DIAGNOSTIC: is the MERGED corpus's val split usable?
# Symptom: on the merged corpus, train loss falls normally (0.69) while val ppl EXPLODES (880,707 --
# confidently wrong, worse than the 12,000-word uniform baseline) and val accuracy stays at chance.
# The same eval code scores 63.85% on the splitfix corpus, so the eval is not the suspect: the val
# DATA is. This measures the two corpora the same way and prints what actually differs.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import json
import numpy as np

for name in ("prepared_ext", "prepared_ext_big", "prepared_splitfix", "prepared_clean"):
    d = f"data/chat/{name}"
    try:
        ids = np.load(f"{d}/ids_v12000.npy")
        b = np.load(f"{d}/bounds_v12000.npy")
        sp = np.load(f"{d}/split_v12000.npy")
        vocab = json.load(open(f"{d}/vocab_v12000.json"))
    except Exception as e:
        print(f"  {name}: unreadable ({type(e).__name__} {e})"); continue
    vmap = {w: i for i, w in enumerate(vocab)}
    unk = vmap.get("<unk>", 0)
    stats = {}
    for tag, want in (("train", 0), ("val", 1)):
        toks, unks, convs = 0, 0, 0
        for j in range(len(b) - 1):
            if sp[j] != want:
                continue
            t = ids[b[j]:b[j + 1]]
            toks += len(t); unks += int((t == unk).sum()); convs += 1
        stats[tag] = (convs, toks, unks / max(toks, 1))
    vt, vk, vu = stats["val"]
    tr, tk, tu = stats["train"]
    print(f"  {name:<20} train {tr:>6} convs ({tk:>10,} tok, unk {tu*100:5.2f}%)   "
          f"val {vt:>5} convs ({vk:>9,} tok, unk {vu*100:5.2f}%)   vocab {len(vocab)}")
    # what does a val conversation actually look like?
    for j in range(len(b) - 1):
        if sp[j] == 1:
            t = ids[b[j]:b[j + 1]]
            print(f"      first val conv ({len(t)} tok): "
                  f"{' '.join(vocab[i] for i in t[:26])}")
            break
    for j in range(len(b) - 1):
        if sp[j] == 0 and (b[j + 1] - b[j]) > 40:
            t = ids[b[j]:b[j + 1]]
            print(f"      first long train conv ({len(t)} tok): "
                  f"{' '.join(vocab[i] for i in t[:26])}")
            break
PY