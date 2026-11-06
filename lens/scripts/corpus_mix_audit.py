#!/usr/bin/env python3
"""CORPUS MIX AUDIT — WHAT IS THE MODEL ACTUALLY TRAINED TO PREDICT?

WHY THIS EXISTS (2026-10-25 15:5x self-wake). The corpus directory is
data/chat/prepared_mix_ctrl_honest and its own meta_v12000.json says it was MERGED from two prepared
corpora, with these token and conversation counts (read from that file, not guessed):

    prepared_ext_honest   38,829,582 tokens   156,520 conversations
        hh-rlhf (MIT), oasst1 (Apache-2.0), ultrachat_200k (MIT), orca-math (MIT)
    prepared_kn2         255,227,730 tokens   584,145 conversations
        metamath, cosmopedia auto_math_text, hendrycks_math, OpenR1-Math, cosmopedia
        textbooks (khanacademy/openstax/stanford), gutenberg_english, Magicoder-OSS-Instruct,
        CodeFeedback, OpenCodeInstruct

So 87% of the corpus TOKENS are maths problems/solutions, textbook prose, instruction-code and
19th-century books, and 13% are human dialogue. The training log prints ONE mask number for the whole
mix ("assistant-only (BY POSITION): 12,159,747/17,122,634 training targets are assistant text (71.0%)"
[lens/runs/TIED-D256-6EP.txt]) but NOTHING in this project has ever printed what share of the ASSISTANT
TARGETS — the only tokens the loss actually trains on, since human targets get weight 0 — comes from
each side of that mix. If the speaker is mostly being trained on maths and code, "it does not answer a
chat prompt" has a second candidate cause, and the owner's shape/scale decision needs the number.

WHAT IT DOES, AND WHAT IT DELIBERATELY DOES NOT DO: reads the corpus arrays (mem-mapped, chunked), walks
each conversation's turn structure from the control tokens (<|im_start|> = id 12000, "user" = 351,
"assistant" = 18), and counts ASSISTANT-TURN tokens per source group, plus the distribution of the FIRST
token of every assistant reply (the reply-opener histogram). NO model, NO GPU, NO training, NO corpus
cache (it never calls the trainer's loader, which is what makes the trainer's RAM peak ~9 GB).

PREDICTIONS, WRITTEN BEFORE THE FIRST RUN OF THIS FILE (P105a-d)
  P105a the two groups are contiguous in the merged arrays (ext first, then kn2), so conversations
        0..156,519 are dialogue and 156,520..740,664 are maths/textbooks/code/books. Checked in-run by
        printing decoded samples from both sides of the boundary and refusing (exit 3) if the boundary
        sample on the kn2 side contains no digits and no code-ish punctuation.
  P105b ASSISTANT-TURN token share, kn2 side at or above 80% of all assistant-turn tokens (the mix is
        87% of all tokens by its own metadata, so I expect the assistant targets to be dominated by it).
  P105c THE REPLY-OPENER HISTOGRAM is FLAT-ish, not concentrated: the top single opener under 12%, the
        top 5 together under 40%. If the top few openers instead carry most of the mass, then "it always
        says the same generic thing" is partly a property of the DATA, not only of the model.
  P105d the dialogue group's openers are conversational (i/you/the/yes/no/that/...), the kn2 group's
        openers are maths/code-ish (digits, backslash, $, #, {, step, solution, ```), and the two
        histograms differ visibly. Falsifier: the two top-10 lists are substantially the same.
Usage: python3 lens/scripts/corpus_mix_audit.py > lens/runs/CORPUS-MIX-AUDIT.txt   (or run the runner)
"""
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "data", "chat", "prepared_mix_ctrl_honest")
VOCAB = json.load(open(os.path.join(D, "vocab_v12000.json"), "r"))
META = json.load(open(os.path.join(D, "meta_v12000.json"), "r"))
BOUNDS = np.load(os.path.join(D, "bounds_v12000.npy"), mmap_mode="r")
IDS = np.load(os.path.join(D, "ids_v12000.npy"), mmap_mode="r")

IM_START = 12000
IM_END = 12001
N_EXT_CONV = META["per_source_corpus"]["data/chat/prepared_ext_honest"]["conversations"]
N_KN2_CONV = META["per_source_corpus"]["data/chat/prepared_kn2"]["conversations"]
N_CONV = len(BOUNDS)
CHUNK = 4_000_000


def detok(ids, limit=48):
    return " ".join(VOCAB[t] if 0 <= t < len(VOCAB) else f"<{t}>" for t in list(ids)[:limit])


def walk(conv_ids):
    """Return (assistant_tokens, total_tokens, openers) for one conversation.

    A turn starts at <|im_start|> whose NEXT token is 351 (user) or 18 (assistant). Everything after
    that role token up to the next <|im_start|> is that turn's text. The first token after the
    'assistant' role token is that reply's OPENER.
    """
    a = 0
    t = 0
    openers = []
    in_asst = False
    n = len(conv_ids)
    i = 0
    while i < n:
        tok = int(conv_ids[i])
        t += 1
        if tok == IM_START:
            nxt = int(conv_ids[i + 1]) if i + 1 < n else -1
            if nxt == 18:
                in_asst = True
                if i + 2 < n:
                    openers.append(int(conv_ids[i + 2]))
                i += 2
                t += 1
                if i + 1 < n and int(conv_ids[i + 1]) == IM_END:
                    in_asst = False
                continue
            if nxt == 351:
                in_asst = False
                i += 2
                t += 1
                continue
        if tok == IM_END:
            in_asst = False
        if in_asst:
            a += 1
        i += 1
    return a, t, openers


def main():
    print("=" * 100)
    print("CORPUS MIX AUDIT — the assistant-turn targets by source group, and the reply-opener histogram")
    print("=" * 100)
    print(f"  file:            {os.path.join(D, 'ids_v12000.npy')}  ({IDS.dtype}, {IDS.shape[0]:,} tokens)")
    print(f"  authored tokens per meta_v12000.json: {META['tokens']:,}  conversations: {META['conversations']:,}")
    print(f"  control tokens:  <|im_start|>={IM_START}  <|im_end|>={IM_END}   roles: user={VOCAB.index('user')}"
          f"  assistant={VOCAB.index('assistant')}")
    print(f"  group split used (from meta_v12000.json's own per-source conversation counts): "
          f"ext_honest 0..{N_EXT_CONV - 1:,} ({N_EXT_CONV:,}), kn2 {N_EXT_CONV:,}..{N_CONV - 1:,} ({N_KN2_CONV:,})")

    # ---- P105a: the boundary check, before any number is computed
    b_ext = BOUNDS[N_EXT_CONV - 1]
    b_kn2 = BOUNDS[N_EXT_CONV]
    s_ext = detok(IDS[BOUNDS[N_EXT_CONV - 3]:b_ext + 1], 60)
    s_kn2 = detok(IDS[b_kn2:b_kn2 + 500], 60)
    s_0 = detok(IDS[BOUNDS[0]:BOUNDS[0] + 500], 60)
    print("\n--- P105a BOUNDARY CHECK (decoded samples, before any count)")
    print(f"  conv 0 (ext group)            : {s_0}")
    print(f"  conv {N_EXT_CONV - 1} (last ext): {s_ext}")
    print(f"  conv {N_EXT_CONV} (first kn2)  : {s_kn2}")
    kn2_looks_like_math_or_code = any(c.isdigit() for c in s_kn2) or any(
        c in s_kn2 for c in "\\$#{}()=+*/_`")
    print(f"  kn2 boundary sample contains a digit or code-ish punctuation: {kn2_looks_like_math_or_code}")
    if not kn2_looks_like_math_or_code:
        print("  P105a REFUSED (exit 3): the merged arrays do not look like ext-then-kn2 in this order, so "
              "the group totals below would be wrong. NOTHING is computed.")
        return 3
    print("  P105a PASS — the groups are contiguous in this order; the counts below are attributable.")

    # ---- the pass
    per_group = {0: {"a": 0, "t": 0, "openers": Counter()},
                 1: {"a": 0, "t": 0, "openers": Counter()}}
    global_openers = Counter()
    for c in range(N_CONV):
        start = int(BOUNDS[c])
        end = int(BOUNDS[c + 1]) if c + 1 < N_CONV else IDS.shape[0]
        ids = np.asarray(IDS[start:end])
        a, t, op = walk(ids)
        g = 0 if c < N_EXT_CONV else 1
        per_group[g]["a"] += a
        per_group[g]["t"] += t
        per_group[g]["openers"].update(op)
        global_openers.update(op)
        if (c + 1) % 100_000 == 0:
            print(f"    ... {c + 1:,} of {N_CONV:,} conversations scanned", flush=True)

    tot_a = per_group[0]["a"] + per_group[1]["a"]
    tot_t = per_group[0]["t"] + per_group[1]["t"]
    print("\n" + "=" * 100)
    print("1. THE ASSISTANT-TURN TARGETS BY SOURCE GROUP (the only tokens the loss trains on, human weight 0)")
    print("=" * 100)
    for g, name in ((0, "ext_honest (human dialogue: hh-rlhf, oasst1, ultrachat, orca-math)"),
                    (1, "kn2 (metamath, cosmopedia textbooks/math, hendrycks, openr1, gutenberg, magicoder, "
                        "codefeedback, opencode)")):
        d = per_group[g]
        fa = 100.0 * d["a"] / tot_a if tot_a else 0.0
        ft = 100.0 * d["t"] / tot_t if tot_t else 0.0
        print(f"  {name}")
        print(f"     assistant-turn tokens {d['a']:>13,}   = {fa:6.2f}% of all assistant-turn tokens")
        print(f"     all tokens           {d['t']:>13,}   = {ft:6.2f}% of all tokens in these arrays")
        print(f"     assistant share WITHIN the group: {100.0 * d['a'] / d['t'] if d['t'] else 0:6.2f}%")
    print(f"  TOTAL assistant-turn tokens {tot_a:,} of {tot_t:,} tokens ({100.0 * tot_a / tot_t:.2f}%)")
    print("  P105b (kn2 >= 80% of assistant-turn tokens): "
          f"measured {100.0 * per_group[1]['a'] / tot_a:.2f}%  -> "
          f"{'PASS' if 100.0 * per_group[1]['a'] / tot_a >= 80.0 else 'FAIL'}")

    print("\n" + "=" * 100)
    print("2. THE REPLY-OPENER HISTOGRAM (first token of every assistant reply, whole corpus)")
    print("=" * 100)
    tot_op = sum(global_openers.values())
    print(f"  replies seen: {tot_op:,}   distinct openers: {len(global_openers):,}")
    top = global_openers.most_common(20)
    cum = 0.0
    for tok, n in top:
        sh = 100.0 * n / tot_op
        cum += sh
        print(f"    {VOCAB[tok] if 0 <= tok < len(VOCAB) else tok!r:>12}  {n:>10,}  {sh:6.2f}%   cum {cum:6.2f}%")
    print(f"  top-1 share {100.0 * top[0][1] / tot_op:.2f}%   top-5 {100.0 * sum(n for _, n in top[:5]) / tot_op:.2f}%"
          f"   -> P105c (top-1 < 12% and top-5 < 40%): "
          f"{'PASS' if (100.0 * top[0][1] / tot_op < 12.0 and 100.0 * sum(n for _, n in top[:5]) / tot_op < 40.0) else 'FAIL'}")

    print("\n--- 2b. the same histogram inside each group (P105d)")
    lists = {}
    for g, name in ((0, "ext_honest dialogue"), (1, "kn2 maths/textbooks/code/books")):
        c = per_group[g]["openers"]
        s = sum(c.values())
        sel = c.most_common(10)
        lists[g] = [VOCAB[t] if 0 <= t < len(VOCAB) else str(t) for t, _ in sel]
        print(f"  {name} ({s:,} replies):")
        for tok, n in sel:
            print(f"    {VOCAB[tok] if 0 <= tok < len(VOCAB) else tok!r:>12}  {n:>10,}  {100.0 * n / s:6.2f}%")
    overlap = set(lists[0]) & set(lists[1])
    print(f"  top-10 overlap between the two groups: {len(overlap)} of 10 -> {sorted(overlap)}")
    print(f"  P105d (the two lists differ; falsifier was a substantial overlap): "
          f"{'PASS' if len(overlap) <= 5 else 'FAIL'}")

    print("\n" + "=" * 100)
    print("READ THIS WITH: data/chat/prepared_mix_ctrl_honest/meta_v12000.json (the source split),")
    print("lens/runs/TIED-D256-6EP.txt (the trainer's own single 71.0% mask line), and")
    print("lens/runs/DECISION-SHAPE-SCALE.md (the shape/scale decision this number feeds).")
    print("THIS AUDIT SAYS NOTHING ABOUT MODEL QUALITY. It counts the corpus, nothing else.")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
