#!/usr/bin/env python3
"""stop_control_v2.py - the stop gate, measured through the shared decoder (lens/decode.py).

The old gate (stop_control.py) is kept unchanged so its 0/10 result stays comparable. This one measures
the SAME three things through the decoder that the product actually uses, so the product's behaviour and
the measurement cannot drift apart:
  * does the turn get handed back (R1: the model emitted the next speaker's marker)
  * does the output end on a sentence end
  * how much of each output is unique tokens
and additionally: WHICH rule stopped it, and whether the stop was forced by a rule or came from the model.
A forced stop is labelled forced. Nothing here claims the model learned to stop.

Usage: SHIP=... python3 lens/scripts/stop_control_v2.py
"""
import json
import os
import sys

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
# REMOVED 2026-10-20, REAL BUG (evidence lens/runs/SERVE-GPU-SYSPATH-BUG.txt): this entry put
# this project's lens/ directory on sys.path, which made the top-level name `cuda` resolve to our
# own lens/cuda/ package. That is a REGULAR package, so it beats cuda-python's PEP 420 namespace
# package even when appended AFTER it, and cupy 14.2 then dies with "ModuleNotFoundError: No
# module named 'cuda.pathfinder'" -- no GPU import worked in any process that did this, e.g.
# `lens/serve.py --device gpu` could not start. It is not needed: lens/ is imported as `lens.X`
# via the ROOT entry above, and lens/scripts/ has its own entry. Do not add it back.
sys.path.insert(0, f"{ROOT}/lens/scripts")
os.chdir(ROOT)

import lens  # noqa: E402
import chat_data  # noqa: E402
from lens.decode import generate, TURN_MARKERS, ENDERS, format_prompt  # noqa: E402

SHIP = os.environ.get("SHIP", "lens/runs/chat/te_T1_turnend_d256_24_ep24.npz")
N = int(os.environ.get("N", "10"))
BUDGET = int(os.environ.get("BUDGET", "150"))
RULE = os.environ.get("RULE_STOP", "1") == "1"

PROMPTS = [
    "human : hello , how are you ? assistant :",
    "human : can you help me with my homework ? assistant :",
    "human : what is the capital of france ? assistant :",
    "human : i am feeling sad today . assistant :",
    "human : tell me a short story assistant :",
    "human : what should i eat for dinner ? assistant :",
    "human : explain gravity in one line assistant :",
    "human : give me a list of three colours assistant :",
    "human : who wrote hamlet ? assistant :",
    "human : good morning assistant :",
]

net = lens.load(SHIP, device="cpu")
d = chat_data.build()
vocab, vmap = d["vocab"], {w: i for i, w in enumerate(d["vocab"])}

rows = []
for k, pr in enumerate(PROMPTS[:N]):
    # the prompt must match the corpus the model trained on: standard chat format when the vocabulary
    # carries the control tokens, the older marker format otherwise
    user_text = pr.split("human :", 1)[-1].split("assistant :")[0].strip() if "human :" in pr else pr
    prompt = format_prompt(user_text, vocab)
    g = generate(net, vocab, vmap, prompt, max_new=BUDGET, seed=1000 + k, rule_stop=RULE)
    toks = g["text"].split()
    uniq = len(set(toks)) / max(len(toks), 1)
    handed = any(w in TURN_MARKERS for w in toks)
    ended = bool(toks) and toks[-1] in ENDERS
    rows.append({"prompt": prompt, "len": g["tokens"], "reason": g["stop_reason"], "forced": g["forced"],
                 "handed_back": handed, "ended_on_ender": ended, "unique_ratio": uniq,
                 "text": g["text"][:220]})

handed_n = sum(r["handed_back"] for r in rows)
ended_n = sum(r["ended_on_ender"] for r in rows)
mean_u = sum(r["unique_ratio"] for r in rows) / len(rows)
from collections import Counter
reasons = Counter(r["reason"] for r in rows)
forced_n = sum(r["forced"] for r in rows)

print(f"STOP CONTROL v2 - ship {os.path.basename(SHIP)}")
print(f"  decoder: lens/decode.py, rule_stop={RULE}, budget {BUDGET} tokens per generation")
print(f"  handed the turn back (model emitted a marker): {handed_n}/{len(rows)}")
print(f"  ended on a sentence end (. ! ?):               {ended_n}/{len(rows)}")
print(f"  mean share of unique tokens:                   {mean_u*100:.1f}%")
print(f"  stopped by: {dict(reasons)}")
print(f"  stops that were rule-forced (not learned):     {forced_n}/{len(rows)}")
print("  HONEST READING: a learned stop does NOT work yet (measured: P('human')=0.0000 at 71 true")
print("  boundaries, top-1 0/71). Every stop above that is not R1 is a transparent rule, labelled")
print("  forced, and must not be reported as the model knowing when to stop.")
print("\n  sample generations:")
for r in rows[:4]:
    print(f"   [{r['reason']}] {r['text'][:150]}")
json.dump({"ship": SHIP, "rule_stop": RULE, "n": len(rows), "handed_back": handed_n,
           "ended_on_ender": ended_n, "mean_unique_ratio": mean_u, "reasons": dict(reasons),
           "forced": forced_n, "rows": rows},
          open(f"lens/runs/stop_control_v2_{os.path.basename(SHIP).replace('.npz','')}.json", "w"),
          indent=1)