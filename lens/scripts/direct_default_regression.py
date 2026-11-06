#!/usr/bin/env python3
"""direct_default_regression.py - did the parallel turn's DIRECT PATH change the DEFAULT model?

WHY THIS EXISTS: at 2026-10-21 15:35, while the 16-arm opt-vs-SGD comparison was mid-flight, lens/model.py,
lens/api.py and lens/scripts/train_chat.py were changed (a new position-aware direct path, --direct /
--direct-rank, default OFF). The remaining 9 arms + the control run THAT code, and the whole point of that
comparison is that only the step rule differs. So the default path must be shown unchanged, not assumed.

WHAT IT CHECKS, against numbers already on the record (no new claims):
  1. the loaded ships build the SAME parameter count the training log stored: "model: 3,811,534 params
     (d=256, K=36)" (lens/runs/OPT-VS-SGD.txt);
  2. net.direct is OFF by default;
  3. the stored KV-DECODE numbers REPRODUCE exactly, which exercises the default forward, the attention
     block, the concept layer and the answer layer end to end:
       d=256  max |cached - full| 2.86102294921875e-06   (lens/runs/KV-DECODE.txt, step 10)
       d=512  max |cached - full| 1.4781951904296875e-05 (lens/runs/KV-DECODE.txt)
     Same prompts, same code path as lens/scripts/kv_decode_test.py (logit_walk is imported from it).

Run:  nice -n 19 .venv-gpu/bin/python3 -u lens/scripts/direct_default_regression.py
"""
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

import json                     # noqa: E402

import numpy as np              # noqa: E402

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens                                                          # noqa: E402
from lens.scripts.kv_decode_test import logit_walk, load_vocab, PROMPTS, format_prompt  # noqa: E402

STORED = {"cl_d256_attn_12_ep12.npz": ("lens/runs/chat/cl_d256_attn_12_ep12.npz",
                                       "data/chat/prepared_mix_ctrl_honest", 2.86102294921875e-06, 10),
          "sb_B1_d512_attn_ep96.npz": ("lens/runs/chat/sb_B1_d512_attn_ep96.npz",
                                       "data/chat/prepared_ext_honest", 1.4781951904296875e-05, None)}
# The parameter-count comparison must use a NO-ATTENTION ship: "model: 3,811,534 params (d=256, K=36)" was
# printed by a training run whose ship has no attention block, while the two ships above have one
# (+4*D*D = +262,144 entries at d=256). My first version of this check compared the wrong ship and
# reported FAIL; that was a defect in the CHECK, and it is kept on the record rather than deleted.
NOATTN = ("lens/runs/chat/cl_d256_clean_24_ep24.npz", "data/chat/prepared_mix_ctrl_honest", 3811534)

print("=" * 100)
print("DID THE DIRECT PATH CHANGE THE DEFAULT MODEL? (regression against numbers already on the record)")
print("=" * 100)
ok = True
for name, (ship, corpus, stored_diff, stored_step) in STORED.items():
    net = lens.load(os.path.join(ROOT, ship), device="cpu")
    vocab, vfile = load_vocab(net, corpus)
    vmap = {w: i for i, w in enumerate(vocab)}
    nparams = int(sum(int(np.asarray(p.data).size) for p in net.parameters()))
    direct = bool(getattr(net, "direct", False))
    import re
    ctx = format_prompt(PROMPTS[3], vocab)
    enc = [vmap.get(w, 0) for w in re.findall(r"\w+|[^\w\s]", ctx.lower())]
    if len(enc) < net.W:
        enc = [0] * (net.W - len(enc)) + enc
    worst, step = logit_walk(net, enc[-net.W:], fold=True)
    print(f"\n{name}")
    print(f"  params {nparams:,}   direct={direct}   (this ship HAS attention: +4*D*D = {4*net.D*net.D:,} "
          f"over a no-attention ship)")
    print(f"  max |cached - full| logits over 40 steps: {worst!r} (step {step})")
    print(f"  stored value in lens/runs/KV-DECODE.txt:  {stored_diff!r}"
          + (f" (step {stored_step})" if stored_step is not None else ""))
    same = (worst == stored_diff)
    print(f"  EXACTLY the stored float: {same}")
    ok = ok and same and not direct

# the parameter count, on the NO-ATTENTION ship the stored number belongs to
ship, corpus, stored_params = NOATTN
net = lens.load(os.path.join(ROOT, ship), device="cpu")
nparams = int(sum(int(np.asarray(p.data).size) for p in net.parameters()))
print(f"\n{os.path.basename(ship)}  (NO attention)")
print(f"  params {nparams:,}   stored training-log count {stored_params:,}   equal: "
      f"{nparams == stored_params}   direct={bool(getattr(net, 'direct', False))}")
ok = ok and nparams == stored_params and not bool(getattr(net, "direct", False))
print("\n" + "=" * 100)
print(f"DEFAULT PATH UNCHANGED: {'PASS' if ok else 'FAIL'}  (params equal, direct OFF, stored KV-DECODE "
      f"floats reproduce exactly)")
print("=" * 100)
sys.exit(0 if ok else 1)
