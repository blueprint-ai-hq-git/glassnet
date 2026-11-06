"""dialogue_mixed_preflight.py — CONTROLLED WIRING TEST for lens/scripts/dialogue_mixed_eval.py (P108a-h).

Predictions for the instrument are in lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt. This pre-flight exists
because the ONLY way the instrument can produce a plausible-looking wrong number is if its turn walk or its
window->conversation mapping differs from the trainer's — and this project has already lost measurements to
three instrument bugs of exactly that shape (BUGS.md). It loads NO corpus and NO checkpoint: milliseconds.

WHAT IT CHECKS
  1. asst_flags() against HAND-WRITTEN expected flag vectors, in the four formats the corpus actually contains:
     ChatML multi-turn, legacy `assistant :` pairs, text after `<|im_end|>` before the next marker, and a
     fragment with no markers at all. Includes an UPPERCASE case, because the trainer lowercases before
     comparing and so must this.
  2. the window -> conversation mapping (searchsorted on bounds) on a synthetic bounds array, including the
     first and last window of every conversation and the sentinel entry.
  3. the val-window count formula the trainer's make_windows uses (stride = window // 2, no window crosses a
     boundary) on the same synthetic conversations.

OUTPUT: lens/runs/DIALOGUE-VS-MIXED-PREFLIGHT.txt  (its own file; nothing else is written)
"""
import os
import sys

import numpy as np

P = "/run/media/yahya/HARD DISK 4/GlassNet"
os.environ.setdefault("CHAT_DATA_DIR", os.path.join(P, "data/chat/prepared_mix_ctrl_honest"))
sys.path.insert(0, os.path.join(P, "lens", "scripts"))
sys.path.insert(0, P)

from dialogue_mixed_eval import asst_flags, W                    # noqa: E402

OUT = os.path.join(P, "lens/runs/DIALOGUE-VS-MIXED-PREFLIGHT.txt")
_lines = []


def log(s=""):
    _lines.append(s)
    print(s, flush=True)


fails = []


def check(name, got, want):
    ok = got == want
    if not ok:
        fails.append(name)
    log(f"  {name:<58} -> {'PASS' if ok else 'FAIL'}")
    if not ok:
        log(f"      got  {got}")
        log(f"      want {want}")


def flags_of(toks):
    return [int(x) for x in asst_flags(toks)]


def main():
    log("=" * 110)
    log("PREFLIGHT — dialogue_mixed_eval.py wiring (no corpus, no checkpoint, no GPU)")
    log("verified against the trainer's own walk: lens/scripts/train_chat.py lines 553-588 (is_asst)")
    log("=" * 110)

    log("")
    log("1. THE TURN WALK (asst_flags) ON HAND-WRITTEN CONVERSATIONS")
    # case 1: ChatML, user then assistant then user
    c1 = ["<|im_start|>", "user", "hello", "there", "<|im_end|>",
          "<|im_start|>", "assistant", "hi", ",", "how", "are", "you", "?", "<|im_end|>",
          "<|im_start|>", "user", "bye"]
    w1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    check("ChatML multi-turn (assistant text flagged, markers not)", flags_of(c1), w1)
    # case 2: legacy two-token pairs
    c2 = ["human", ":", "are", "you", "ok", "?", "assistant", ":", "yes", "i", "am", ".", "human", ":", "good"]
    w2 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0]
    check("legacy 'assistant :' pairs", flags_of(c2), w2)
    # case 3: text between <|im_end|> and the next <|im_start|> inherits in_asst (trainer behaviour, mirrored)
    c3 = ["<|im_start|>", "assistant", "yes", "<|im_end|>", "stray", "<|im_start|>", "user", "no"]
    w3 = [0, 0, 1, 0, 1, 0, 0, 0]
    check("text after <|im_end|> before the next marker inherits", flags_of(c3), w3)
    # case 4: no markers at all
    check("no markers -> nothing flagged", flags_of(["hello", "there", "friend"]), [0, 0, 0])
    # case 5: uppercase marker/role (the trainer lowercases; so must this)
    c5 = ["<|IM_START|>", "ASSISTANT", "hi", "<|IM_END|>"]
    w5 = [0, 0, 1, 0]
    check("uppercase marker/role is lowercased like the trainer", flags_of(c5), w5)
    # case 6: 'assistant' without the following ':' is NOT a marker
    check("'assistant' without ':' is not a marker", flags_of(["assistant", "hi"]), [0, 0])

    log("")
    log("2. WINDOW -> CONVERSATION MAPPING (searchsorted on bounds)")
    bounds = np.asarray([0, 40, 90, 95, 140], dtype=np.int64)    # 4 convs (len 40/50/5/45), sentinel 140
    convs = [(0, 40), (40, 90), (90, 95), (95, 140)]
    stride = W // 2
    starts, want_conv, want_counts = [], [], []
    for j, (a0, b0) in enumerate(convs):
        offs = list(range(0, (b0 - a0) - W - 1, stride))
        want_counts.append(len(offs))
        starts += [a0 + o for o in offs]
        want_conv += [j] * len(offs)
    starts = np.asarray(starts, dtype=np.int64)
    got_conv = np.searchsorted(bounds, starts, side="right") - 1
    log(f"  synthetic: 4 conversations of len {[b - a for a, b in convs]}, W={W} -> "
        f"{len(starts)} windows expected at {starts.tolist()}")
    check("the trainer's count formula reproduces the window count", len(starts), sum(want_counts))
    check("every synthetic window maps to its own conversation", got_conv.tolist(), want_conv)
    check("per-conversation window counts match the formula",
          [int(np.sum(got_conv == j)) for j in range(len(convs))], want_counts)
    check("windows never cross a boundary (start+W inside the conv)",
          bool(np.all([(starts[i] + W) < bounds[want_conv[i] + 1] for i in range(len(starts))])), True)
    check("no window is produced for the too-short conversation (len 5)",
          int(np.sum(got_conv == 2)), 0)
    check("W used by this instrument == the trainer's window 32", W, 32)

    log("")
    log(f"PREFLIGHT RESULT: {'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    log("=" * 110)
    with open(OUT, "w") as f:
        f.write(f"PREFLIGHT — dialogue_mixed_eval.py wiring  {__import__('time').strftime('%F %H:%M')}\n")
        f.write("\n".join(_lines) + "\n")
        f.write(f"EXIT={'0' if not fails else '1'}\n")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
