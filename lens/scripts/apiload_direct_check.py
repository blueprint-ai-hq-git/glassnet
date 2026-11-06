"""apiload_direct_check.py — VERIFY the apiload fix (BUGS.md, lens/api.py APILOAD-DIRECT FIX).

Predictions, written BEFORE running (P68a-c):
 P68a before the fix api.load() rebuilt a tied-path checkpoint WITHOUT the direct path: 'dmix' missing and
      max |logit difference| 24.964367 against the trained model (already measured, TALK-FAIR-TEST.txt §0).
 P68b after the fix the parameter sets match and the logits are BIT-IDENTICAL (max |dlogit| exactly 0.0) on
      the same 256 held-out windows, on CPU.
 P68c a checkpoint WITHOUT the direct path is unaffected by the fix (same logits as the pre-fix loader):
      checked by loading a pre-direct ship and confirming it has no direct parameters at all.
Nothing here claims the model talks.
"""
import json
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, P)
sys.path.insert(0, P + "/lens/scripts")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import lens.api as api                                      # noqa: E402
from lens import backend as B                               # noqa: E402
from lens.concepts import Concept                           # noqa: E402
from lens.model import LensNet                              # noqa: E402

W = 32
DD = P + "/data/chat/prepared_mix_ctrl_honest"
vocab = json.load(open(DD + "/vocab_v12000.json"))
ids = np.load(DD + "/ids_v12000.npy", mmap_mode="r")
CKPT = P + "/lens/runs/chat/ct_d256_6M_6ep_best.npz"
PRE = P + "/lens/runs/chat/cl_d256_clean_24_ep24.npz"        # pre-direct ship, used by the readability gate

# 256 real held-out windows straight out of the corpus (val conversations), not fabricated input
split = np.load(DD + "/split_v12000.npy").astype(np.int8)
bounds = np.load(DD + "/bounds_v12000.npy")
vj = int(np.where(split == 1)[0][0])
a0 = int(bounds[vj])
X = np.asarray(ids[a0:a0 + 256 * 2]).reshape(256, 2)
X = np.asarray([[int(ids[a0 + i * W + k]) for k in range(W)] for i in range(256)], dtype=np.int32)


def trained_model(path):
    """The model the trainer has in memory: same metadata, flags passed in."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["__meta__"]))
    cs = [Concept(c["name"], c["ids"], support=c["support"]) for c in meta["concepts"]]
    n = LensNet(meta["V"], cs, d=meta["D"], window=meta["W"], seed=meta["seed"],
                lam_concept=meta["lam_concept"], lam_act=meta["lam_act"],
                attention=bool(meta.get("attention", False)), numeric=bool(meta.get("numeric", False)),
                direct=bool(meta.get("direct", False)), direct_rank=int(meta.get("direct_rank", 16)),
                direct_mode=str(meta.get("direct_mode", "lowrank")))
    n.device_name = "cpu"
    for p in n.parameters():
        p.data = B.to_dev(z[p.name].astype(np.float32))
    return n, meta


print("=" * 92)
print("APILOAD DIRECT FLAG CHECK — lens/api.py APILOAD-DIRECT FIX (self-wake 2026-10-22 18:1x)")
print("=" * 92)
for path, label in ((CKPT, "tied-path checkpoint"), (PRE, "pre-direct ship")):
    if not os.path.exists(path):
        print(f"{label}: {path} MISSING, skipped")
        continue
    tru, meta = trained_model(path)
    api_net = api.load(path, device="cpu")
    need = {p.name for p in tru.parameters()}
    got = {p.name for p in api_net.parameters()}
    lg_t = tru.forward_arrays(np.ascontiguousarray(X))["logits"]
    lg_a = api_net.forward_arrays(np.ascontiguousarray(X))["logits"]
    md = float(np.abs(lg_t - lg_a).max())
    print(f"\n{label}: {os.path.basename(path)}")
    print(f"  metadata: direct={meta.get('direct')} mode={meta.get('direct_mode')} rank={meta.get('direct_rank')}")
    print(f"  parameters the trainer's model needs: {sorted(need)}")
    print(f"  parameters api.load created         : {sorted(got)}")
    print(f"  missing after api.load              : {sorted(need - got)}")
    print(f"  max |logit difference| on 256 real held-out windows: {md:.6f}  "
          f"({'IDENTICAL' if md == 0.0 else 'DIFFERENT'})")
print("")
print("P68a: the pre-fix numbers (dmix missing, 24.964367) are on the record in lens/runs/TALK-FAIR-TEST.txt.")
print("P68b/P68c: read the two blocks above; each is one checkpoint and one 256-window forward pass.")
