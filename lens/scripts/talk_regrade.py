#!/usr/bin/env python3
"""talk_regrade.py -- re-grade already-trained checkpoints through the FIXED talk decoder.

Why: until 2026-11-06 the talk decoder left-padded short prompts with id 0 = "<unk>" (20-23 of 32
context slots), so the "all-<unk>" replies that were graded as talk failures were measuring the
padding. docs/history/PROMPT-ENCODING.md has the numbers (95.1/96.5% -> 26.4/21.5%). This tool
re-reads the same checkpoints, read-only, with the fixed context and reports <unk> share and text.

Predictions (written before the run, in docs/history/TALK-REGRADE-PREDICTIONS.md):
  P1  every checkpoint prints <unk> <= 40% (greedy)
  P2  0/6 prompts get a correct answer on every checkpoint (content verdict stays FAIL)
  P3  the sampled decoder (top-k 80) has a lower <unk> share than greedy on every checkpoint

CPU only. Nothing is trained, nothing is overwritten.
"""
import json
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))
import variants  # noqa: E402

VOCAB = os.path.join(PROJ, "data", "chat", "prepared_mix_ctrl_honest", "vocab_v12000.json")
PROMPTS = [
    "Human: hello , how are you ?\n\nAssistant:",
    "Human: what is your name ?\n\nAssistant:",
    "Human: can you help me learn to cook ?\n\nAssistant:",
    "Human: i am feeling sad today .\n\nAssistant:",
    "Human: what is the capital of france ?\n\nAssistant:",
    "Human: tell me about dogs .\n\nAssistant:",
]
# checkpoint, the run's own reported held-out by-position accuracy, what it was graded as today
CKPTS = [
    ("ca_attn_w32_s0_best", 8.21, "attention arm, mean of 2 seeds"),
    ("ca_w64_s0_best", 9.45, "window-64 arm, seed 0"),
    ("ca_w64_s1_best", 9.45, "window-64 arm, seed 1"),
    ("gpu_best_s2_best", 24.25, "d256 2M-window arm, seed 2"),
    ("gpu_d512_s0_best", 25.77, "d512 arm, seed 0"),
]
N_NEW = 24
vocab = json.load(open(VOCAB))

print("=" * 100)
print("TALK REGRADE through the pad-free decoder -- predictions: docs/history/TALK-REGRADE-PREDICTIONS.md")
print("decoders: greedy (rep 1.15) and sampled top-k 80 / temp 0.9 / rep 1.6 ; 24 new tokens ; 6 prompts")
print("=" * 100)

summary = []
for name, acc, note in CKPTS:
    path = os.path.join(PROJ, "lens", "runs", "chat", name + ".npz")
    if not os.path.exists(path):
        print(f"\n!! missing {path} -- skipped and recorded, not estimated")
        continue
    v = variants.Variants(talk=path, chat_vocab=vocab, device="cpu")
    print(f"\n{'-' * 100}\n-- {name}  (run reported held-out {acc:.2f}%; {note})")
    row = {"ckpt": name, "reported": acc}
    for arm, kw in (("greedy", dict(greedy=True)),
                    ("topk80", dict(greedy=False, top_k=80, temp=0.9, rep_penalty=1.6))):
        unk = tot = 0
        texts = []
        for i, p in enumerate(PROMPTS):
            r = v.talk_sentence(p, n_new=N_NEW, seed=0, **kw)
            rep = str(r["reply"] if isinstance(r, dict) else r)
            unk += rep.count("<unk>")
            tot += len(rep.split())
            texts.append(rep[:96])
            print(f"   [{arm}] p{i} {rep[:96]}")
        share = unk / max(tot, 1) * 100.0
        row[arm] = share
        print(f"   [{arm}] <unk> share over {tot} tokens: {share:.1f}%")
    summary.append(row)

print("\n" + "=" * 100)
print("SUMMARY  (numbers only; verdicts and what they do NOT mean are in the history file)")
print(f"  {'checkpoint':22s} {'reported':>9s} {'greedy':>8s} {'topk80':>8s}")
for r in summary:
    print(f"  {r['ckpt']:22s} {r['reported']:8.2f}% {r['greedy']:7.1f}% {r['topk80']:7.1f}%")
print("  P1 bar: greedy <= 40% on every checkpoint. P3 bar: topk80 < greedy on every checkpoint.")
print("=" * 100)
