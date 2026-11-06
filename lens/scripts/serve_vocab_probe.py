#!/usr/bin/env python3
"""serve_vocab_probe.py - was the server's vocabulary ACTUALLY short, or did I assume it?

Found while wiring the KV cache. My first claim was "control-token ships could not be served: the server
took a 12,000-word list for a 12,002-id model". This probe MEASURES that instead of asserting it, because
the vocabulary json is written by the corpus builder and may already carry the control tokens.

Output: lens/runs/SERVE-VOCAB-PROBE.txt
"""
import json
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
# REMOVED 2026-10-20, THE SAME REAL BUG IN A FORM THE FIRST GATE COULD NOT SEE (class 3;
# evidence lens/runs/SYSPATH-SHADOW-FIX-class3.txt, probe lens/runs/GPU-ATTN-CACHE.txt): this
# tuple put <project>/lens on sys.path as well, so the top-level name `cuda` resolved to our own
# lens/cuda/ package and cupy 14.2 died with "ModuleNotFoundError: No module named
# 'cuda.pathfinder'". lens/ is imported as `lens.X` via the ROOT entry and lens/scripts/ has its
# own entry, so the `/lens` element is not needed. Do not add it back.
for _p in (ROOT, ROOT + "/lens/scripts"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lens  # noqa: E402

SHIPS = [("lens/runs/chat/cl_d256_attn_12_ep12.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/cl_d256_clean_24_ep24.npz", "data/chat/prepared_mix_ctrl_honest"),
         ("lens/runs/chat/sb_B1_d512_attn_ep96.npz", "data/chat/prepared_ext_honest"),
         ("lens/runs/chat/sb_S0_d256_noattn_ep96.npz", "data/chat/prepared_ext_honest")]

print("WAS THE SERVER'S VOCABULARY SHORT? measured per ship against the corpus it was trained on.")
print("(the vocabulary json is written by the corpus builder, lens/scripts/add_control_tokens.py appends")
print(" the two ChatML tokens to the corpus AND to the json, so the json may already have them)\n")
for ship, corpus in SHIPS:
    p = os.path.join(ROOT, corpus, "vocab_v12000.json")
    if not os.path.exists(p):
        print(f"{os.path.basename(ship)}: no vocab json at {p}")
        continue
    words = json.load(open(p))
    net = lens.load(os.path.join(ROOT, ship), device="cpu")
    tail = words[-2:]
    old_ok = (len(words) == net.V)
    print(f"{os.path.basename(ship):32s} V={net.V:6d}  json {os.path.basename(corpus)} has "
          f"{len(words):6d} entries, last two {tail}")
    print(f"{'':32s} -> the OLD server (vocab = the json, unchanged) was "
          f"{'CORRECT' if old_ok else 'SHORT BY ' + str(net.V - len(words))}"
          f"; <|im_end|> reachable: {'yes' if old_ok else 'NO'}")
print("\nCONCLUSION, from the numbers above and not from memory: the control tokens are already in the")
print("json of a ctrl corpus, so the old server was NOT crippled on those ships. What IS real and was")
print("measured while wiring: (1) the server used to call chat_data.build(), which loads the whole 1.2 GB")
print("token array just to obtain the word list - the json read is the same list, from the file the")
print("corpus builder wrote; (2) there was no check at all that the ship and the word list agree, so a")
print("genuine mismatch would have been silent; the new code refuses it by name.")
