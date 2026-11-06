"""migrate_ckpt_vocab.py — put each existing VL checkpoint's OWN vocabulary inside it.

WHY THIS IS NEEDED
The checkpoints store the vocab SIZES (V, QV) but not the WORDS. A loader therefore had to rebuild
the words from data and nothing checked the rebuild. Loading the 14-class vqa2 model against the
12-word Fashion-MNIST vocabulary raised IndexError while drawing a frame -- and if the sizes had
happened to agree, it would have printed the WRONG CLASS NAME with no error.

THE FIX (in lens/vision.py): save_vision now stores the vocabularies and check_vocab_match raises on
any disagreement. These older files predate that, so they are re-saved here with the vocabulary their
own training run used -- which is recoverable because each run's data module defines it.

SAFETY: nothing is overwritten. Each is written as `<name>_vocabfix.npz` and the SIZE is verified
against the weights before writing, so a wrong pairing is caught rather than baked in.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

V = os.path.join(PROJ, "lens", "runs", "vl")

print("=" * 100)
print("MIGRATING VL CHECKPOINTS — vocabularies moved inside the file")
print("=" * 100)

# each checkpoint and the data module that produced its answer vocabulary
TARGETS = [
    ("cam_base_ep6.npz", "vqa1", "vl_data"),
    ("vqa2_small_ep6.npz", "vqa2", "vl_data2"),
]

for name, task, modname in TARGETS:
    path = os.path.join(V, name)
    if not os.path.exists(path):
        print(f"  {name}: missing, skipped")
        continue
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["__meta__"]))
    v_expected = meta["V"]
    print()
    print(f"  {name}   kind={meta['kind']}  V={v_expected}  QV={meta['QV']}  "
          f"answer.W={z['answer.W'].shape}")

    if "avocab" in meta:
        print("    already migrated")
        continue

    # rebuild the vocabulary exactly as that training run did
    import vl_data
    if modname == "vl_data2":
        import vl_data2
        base = vl_data.prepare_vqa("fashion", max_train=None, seed=0)
        d = vl_data2.prepare_vqa2(base, max_train=200, seed=0)
    else:
        d = vl_data.prepare_vqa("fashion", max_train=200, seed=0)
    avocab, qvocab = list(d["avocab"]), list(d["qvocab"])
    print(f"    rebuilt from {modname}: avocab {len(avocab)}  qvocab {len(qvocab)}")
    print(f"      avocab = {avocab}")

    # VERIFY BEFORE WRITING -- a wrong pairing must be caught, not saved
    ok = (len(avocab) == v_expected and len(qvocab) == meta["QV"]
          and z["answer.W"].shape[1] == len(avocab))
    if not ok:
        print(f"    REFUSING: avocab {len(avocab)} vs V {v_expected}, "
              f"qvocab {len(qvocab)} vs QV {meta['QV']}, "
              f"answer.W {z['answer.W'].shape[1]} -- this pairing is wrong")
        continue

    meta["avocab"], meta["qvocab"] = avocab, qvocab
    out = path.replace(".npz", "_vocabfix.npz")
    np.savez(out, **{k: z[k] for k in z.files if k != "__meta__"},
             __meta__=np.asarray(json.dumps(meta)))
    print(f"    wrote {os.path.basename(out)}  ({os.path.getsize(out)/1024:.0f} KB)")

    # read it back and prove the vocab round-trips
    z2 = np.load(out, allow_pickle=True)
    m2 = json.loads(str(z2["__meta__"]))
    same = (m2["avocab"] == avocab and m2["qvocab"] == qvocab
            and z2["answer.W"].shape == z["answer.W"].shape
            and np.array_equal(z2["answer.W"], z["answer.W"]))
    print(f"    read back: avocab match={m2['avocab'] == avocab}  "
          f"weights identical={np.array_equal(z2['answer.W'], z['answer.W'])}  "
          f"-> {'OK' if same else 'MISMATCH'}")

print()
print("=" * 100)
print("  The originals are untouched. The *_vocabfix.npz files are the ones to load from now.")
print("=" * 100)
