# PROMPT-ENCODING: is today's talk output garbage because of the prompt, or the model?

Written 2026-11-06 ~16:55 IST by the chain wake turn (label quick7), BEFORE
`lens/scripts/audit_prompt_encoding.py` was run in any form.

## Why this is the question

Every talk block printed today (lens/runs/GPU-QUEUE-RUNS.txt, 2026-11-05 16:10 and 2026-11-06 16:0x-16:25)
shows GREEDY output that is almost entirely `<unk>`. `docs/history/BUGS.md` already recorded the same
symptom once ([BUG-PADDED-TALK-PROMPT], 2026-10-22): the talk decoder builds its context as

    enc = [vmap.get(x, 0) for x in toks]
    if len(enc) < W: enc = [0] * (W - len(enc)) + enc      # id 0 == "<unk>"

so a short prompt is LEFT-PADDED with the `<unk>` id. Both copies of that code are still live:
`lens/scripts/train_chat.py:1483-1485` and `lens/scripts/variants.py:376-378`. The 2026-10-22 entry says it
was "found and fixed", but these two files still contain the padding, so either the fix was applied only to
the grading instrument or it never landed. That is a claim to check, not to assume.

The 6 prompts used by the trainer are ~10 tokens ("Human: hello , how are you ?\n\nAssistant:" -> 10 ids
after the regex drops whitespace), so 22 of 32 context slots are id 0 in the current path.

## Design (same weights, same decoder, ONE change: how the context is filled)

Checkpoints (the two best that today's runs saved, untouched, read-only):
  lens/runs/chat/gpu_best_s2_best.npz      d 256, held-out by-position 24.25%   (2026-11-05 16:10)
  lens/runs/chat/gpu_d512_s0_best.npz      d 512, held-out by-position 25.77%   (2026-11-06 16:25)

Arm L (current / control) : left-pad with id 0, exactly the live code.
Arm R (pad-free)          : the prompt ids are repeated cyclically to fill the window, so every context
                            slot is a real prompt token (no padding id at all).
Decoder identical in both arms: greedy, repetition penalty 1.15, 24 new tokens, same 6 prompts as
`train_chat.py:1512-1518`. Metric: share of `<unk>` among the 24 generated tokens, and the text itself.

## Predictions (P1-P4) with the number that falsifies each

P1  CONTEXT COMPOSITION. In arm L, 22 of every 32 context slots (68.75%) are id 0 for all 6 prompts,
    because every prompt is 10 ids. If any prompt measures outside 20-24 pad slots, my reading of the
    tokenizer or the window is wrong and no delta below may be quoted.

P2  MAIN. Arm R cuts the mean `<unk>` share over the 6 prompts to <= 15%, against arm L at >= 60%.
    FALSIFIER: if arm R is still >= 60% `<unk>`, the flood is the model (the `<unk>` output bias, which
    `BUGS.md:1000` measured at +5.85, the largest of all 12,002 biases), padding is exonerated, and the
    fix has to be in the head/loss, not the decoder.

P3  NOT A WIN ON ITS OWN. Even if P2 holds, the pad-free replies are predicted NOT to be answers to the
    questions (content is weak - STATUS.md "Coherent conversation 15%"). Accuracy is not claimed here;
    this experiment measures only whether the measurement itself was broken.

P4  SAME DIRECTION ON BOTH CHECKPOINTS. The `<unk>` drop holds on d256 AND d512. A split result (one
    model fixed, one not) means the effect is checkpoint-specific and gets reported as such, not
    averaged.

## What will NOT be claimed

* No accuracy number, no talk pass, no ship. A lower `<unk>` share is not a better model; it means the
  questions were being asked of a context the model never saw in training.
* Nothing is retrained in this experiment. The checkpoints are opened read-only.
