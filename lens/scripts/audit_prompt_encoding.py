#!/usr/bin/env python3
"""audit_prompt_encoding.py -- is the all-<unk> talk output the PROMPT or the MODEL?

Predictions were written BEFORE this file was run: docs/history/PROMPT-ENCODING-PREDICTIONS.md.

Two arms, same weights, same decoder, ONE change: how the 32-slot context is filled.
  L (control, the live code): left-pad with id 0 ("<unk>")  -- train_chat.py:1483-1485 / variants.py:376-378
  R (pad-free): repeat the prompt ids cyclically so every slot is a real prompt token

CPU only (the GPU is owned by the concept-wall A/B). Read-only on the checkpoints.
USAGE: python3 lens/scripts/audit_prompt_encoding.py > lens/runs/PROMPT-ENCODING-AUDIT.txt 2>&1
"""
import json
import os
import re
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants  # noqa: E402
import lens  # noqa: E402  (for lens.load through variants)

VOCAB = os.path.join(PROJ, "data", "chat", "prepared_mix_ctrl_honest", "vocab_v12000.json")
CONCEPTS = os.path.join(PROJ, "lens", "runs", "chat", "concept_spec_wide.json")
CKPTS = [
    ("d256_best_s2", os.path.join(PROJ, "lens", "runs", "chat", "gpu_best_s2_best.npz"), 24.25),
    ("d512_best_s0", os.path.join(PROJ, "lens", "runs", "chat", "gpu_d512_s0_best.npz"), 25.77),
]
PROMPTS = [
    "Human: hello , how are you ?\n\nAssistant:",
    "Human: what is your name ?\n\nAssistant:",
    "Human: can you help me learn to cook ?\n\nAssistant:",
    "Human: i am feeling sad today .\n\nAssistant:",
    "Human: what is the capital of france ?\n\nAssistant:",
    "Human: tell me about dogs .\n\nAssistant:",
]
N_NEW = 24
REP = 1.15          # train_chat.py's greedy call uses rep_penalty 1.15; identical in both arms
TOK = re.compile(r"[A-Za-z']+|[0-9]+|[^\sA-Za-z0-9]")

vocab = json.load(open(VOCAB))
vmap = {w: i for i, w in enumerate(vocab)}
names = vocab


def encode(t):
    return [vmap.get(x, 0) for x in TOK.findall(t.lower())]


def fill_leftpad(enc, W):
    return ([0] * (W - len(enc)) + enc)[-W:] if len(enc) < W else enc[-W:]


def fill_repeat(enc, W):
    """Pad-free: tile the prompt so no padding id is ever in the context."""
    if not enc:
        raise ValueError("empty prompt")
    reps = (W // len(enc)) + 1
    return (enc * reps)[-W:]


def greedy(net, ctx, n_new, rep):
    out, used = [], {}
    for _ in range(n_new):
        lg = net.forward_arrays(np.asarray([ctx], dtype=np.int64))["logits"][0]
        z = lg.astype(np.float64)
        for tok, cnt in used.items():
            z[tok] -= np.log(rep) * cnt
        nxt = int(z.argmax())
        out.append(nxt)
        used[nxt] = used.get(nxt, 0) + 1
        ctx = (ctx + [nxt])[-len(ctx):]
    return out


print("=" * 100)
print("PROMPT ENCODING AUDIT  --  predictions: docs/history/PROMPT-ENCODING-PREDICTIONS.md (written before this run)")
print(f"vocab {VOCAB}  ({len(names)} words, id 0 = {names[0]!r})")
print(f"decoder: greedy, rep_penalty {REP}, {N_NEW} new tokens; arms differ ONLY in context fill")
print("=" * 100)

# ---- A. context composition (P1) -------------------------------------------------------------
print("\n-- A. what the 32-slot context actually contains --")
for i, p in enumerate(PROMPTS):
    enc = encode(p)
    print(f"  prompt {i}: {p.splitlines()[0]!r}")
    print(f"    ids  {enc}")
    print(f"    words {[names[t] for t in enc]}")
    print(f"    <unk> already in the prompt ids: {sum(1 for t in enc if t == 0)}/{len(enc)}")

results = {}
for tag, path, reported in CKPTS:
    if not os.path.exists(path):
        print(f"\n!! MISSING checkpoint {path} -- skipping {tag} (recorded, not estimated)")
        continue
    v = variants.Variants(talk=path, chat_vocab=vocab, device="cpu")
    net, _ = v.models["talk"]
    W = int(getattr(net, "W", 0))
    print(f"\n{'=' * 100}\n-- {tag}: {os.path.basename(path)}  W={W}  (today's run reported held-out {reported:.2f}%)")
    arm = {}
    for arm_name, fill in (("L_leftpad", fill_leftpad), ("R_padfree", fill_repeat)):
        unk = tot = 0
        lines = []
        for i, p in enumerate(PROMPTS):
            enc = encode(p)
            ctx = fill(enc, W)
            pads = sum(1 for t in ctx if t == 0)
            gen = greedy(net, list(ctx), N_NEW, REP)
            txt = " ".join(names[t] for t in gen)
            u = sum(1 for t in gen if t == 0)
            unk += u
            tot += len(gen)
            lines.append((i, pads, u, txt))
        arm[arm_name] = (unk / tot * 100.0, lines)
        print(f"  [{arm_name}]  <unk> share over {tot} generated tokens: {unk / tot * 100.0:.1f}%")
        for i, pads, u, txt in lines:
            print(f"    p{i} pads={pads:2d}  <unk> {u:2d}/{N_NEW}  {txt[:104]}")
    results[tag] = arm

print("\n" + "=" * 100)
print("-- C. verdict inputs (numbers only; the reading is in AUTOPILOT-LOG.md) --")
for tag, arm in results.items():
    l = arm["L_leftpad"][0]
    r = arm["R_padfree"][0]
    print(f"  {tag:14s}  leftpad {l:5.1f}%   padfree {r:5.1f}%   delta {r - l:+.1f} pp")
print("  P1 expected 20-24 pad slots per prompt; P2 main: padfree <= 15% and leftpad >= 60%;")
print("  P2 falsifier: padfree still >= 60%  -> flood is the model, not the padding.")
print("=" * 100)
