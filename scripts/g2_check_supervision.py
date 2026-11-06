"""g2_check_supervision.py — is the concept loss actually being applied?

G2's D=64 rung scored 0.33/9 concepts. Target E, same width, same data, same concept set, same
lam_concept=0.1, scored 7/9. Same configuration cannot give both. Prime suspect: the readable
model's loss() applies the concept loss only when the module global CONCEPT_TARGETS is not None,
and G2 ran with T3M_IMPORT_ONLY=1, which leaves it None.

If that is the cause, the entire G2 sweep is void and must be re-run.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

MODE = os.environ.get("CHK_MODE", "import_only")
if MODE == "import_only":
    os.environ["T3M_IMPORT_ONLY"] = "1"
os.environ["T2_CONCEPT_SET"] = "meaningful_ext"
os.environ["T2_N_TAUGHT"] = "9"
os.environ["T2_LAM_CONCEPT"] = "0.1"

from glassnet import tiny_data                      # noqa: E402
from glassnet.tensor import set_backend             # noqa: E402
import run_t2_iter003 as M                          # noqa: E402

set_backend("numpy")

print("=" * 92)
print(f"CONCEPT SUPERVISION CHECK — mode={MODE} (T3M_IMPORT_ONLY={os.environ.get('T3M_IMPORT_ONLY')})")
print("=" * 92)

D = tiny_data.build(win=M.WIN, vocab_cap=1500)
X, y = np.asarray(D["Xtr"][:16]), np.asarray(D["ytr"][:16])

net = M.TinyLM3(D["V"], 9, 64, M.WIN, rng=np.random.default_rng(0), lam_concept=0.1)
print(f"CONCEPT_TARGETS           : {'None  <-- CONCEPT LOSS DISABLED' if M.CONCEPT_TARGETS is None else f'{len(M.CONCEPT_TARGETS)} targets'}")
print(f"net.lam_concept           : {net.lam_concept}")

loss, parts = net.loss(X, y)
print(f"loss() parts keys         : {sorted(parts)}")
aux_key = [k for k in parts if k not in ("ce", "sp_act", "sp_w")]
print(f"concept term present      : {bool(aux_key)}  {aux_key}")

# the decisive test: does zeroing the concept term change the loss?
h = np.asarray(net.forward(X)["concepts"].data)
tgt = np.stack([np.isin(X, t["ids"]).any(1) for t in
                __import__("json").load(open(os.path.join(PROJ, "runs", "t3",
                                                          "meaningful_targets_ext.json")))
                ["qualifying"]], axis=1).astype(np.float32)
eps = 1e-7
manual_aux = float(-(tgt * np.log(h + eps) + (1 - tgt) * np.log(1 - h + eps)).mean())
print(f"manual concept BCE (mean) : {manual_aux:.6f}")

if aux_key:
    print(f"\nVERDICT: the concept loss IS applied (term {aux_key[0]} = {parts[aux_key[0]]:.6f})")
else:
    print(f"\nVERDICT: the concept loss is NOT applied. The concept layer is trained only by the")
    print(f"         LM gradient and the sparsity penalty, so its columns never align with their")
    print(f"         names. ANY audit of such a model will report near-chance AUROC -- which is")
    print(f"         exactly the 0/9 and 0.33/9 G2 produced. THE G2 SWEEP IS VOID.")
