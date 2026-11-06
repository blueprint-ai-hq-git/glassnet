# TALK-REGRADE: were today's "talk 0/6" verdicts real, or the padding?

Written 2026-11-06 ~17:00 IST, BEFORE `lens/scripts/run_talk_regrade.sh` was launched.

## Why

`docs/history/PROMPT-ENCODING.md` (measured minutes earlier, same turn) shows the talk decoder was
left-padding the 6 project prompts with id 0 = `<unk>`: 20-23 of 32 context slots. On the two best
checkpoints the same weights then emitted `<unk>` for 95.1% / 96.5% of reply tokens, against
26.4% / 21.5% with a pad-free context. Every "talk 0/6" reading printed today came from that decoder,
so those verdicts are unmeasured rather than negative.

## Design

Five checkpoints already on disk, opened read-only, nothing retrained, CPU only (cores 2,3; the
concept-wall A/B keeps the GPU and cores 0,1):

| checkpoint | run's own held-out (by position) |
|---|---:|
| `ca_attn_w32_s0_best` | 8.21% (attention arm, mean of 2 seeds) |
| `ca_w64_s0_best`, `ca_w64_s1_best` | 9.45% (window-64 arm, mean of 2 seeds) |
| `gpu_best_s2_best` | 24.25% (d256, 2M windows) |
| `gpu_d512_s0_best` | 25.77% (d512) |

Two decoders per checkpoint, same 6 prompts as `train_chat.py`, 24 new tokens:
greedy (rep 1.15) and sampled top-k 80 / temp 0.9 / rep 1.6 (`TALK_DEFAULT_DECODER`,
`lens/scripts/variants.py:513`, tuned 2026-10-09).

## Predictions

P1  `<unk>` share <= 40% (greedy) on EVERY checkpoint. Falsifier: any checkpoint still >= 40% means
    the padding was not the whole story and the `<unk>` bias dominates for that model.
P2  CONTENT DOES NOT IMPROVE. 0 of 6 prompts get a correct answer on every checkpoint ("what is the
    capital of france" must produce "paris"): the talk verdict stays FAIL, this time measured rather
    than assumed. Falsifier: any checkpoint answers any prompt correctly -> a talk result that was
    previously hidden by the decoder, and it gets reported as such.
P3  Sampled top-k 80 gives a LOWER `<unk>` share than greedy on every checkpoint.

## What will NOT be claimed

No ship, no "it talks", no accuracy change (these checkpoints and their val numbers are untouched).
Low `<unk>` means the words are real words; it does not mean the reply is an answer.
