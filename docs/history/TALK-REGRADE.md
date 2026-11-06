# TALK-REGRADE: the old "0/6" is replaced by a real measurement — the answer is still no

Run 2026-11-06 16:47:20-16:47:41 IST. Predictions written first:
`docs/history/TALK-REGRADE-PREDICTIONS.md`. Raw output: `lens/runs/TALK-REGRADE.txt`. Script:
`lens/scripts/run_talk_regrade.sh` -> `lens/scripts/talk_regrade.py`. CPU only (cores 2,3), no
training, checkpoints opened read-only. Launched through the chain (label `talk_regrade`, rc=0).

## Numbers

`<unk>` share over 144 generated tokens per arm (6 project prompts x 24 tokens).

| checkpoint | run's own held-out | greedy | top-k 80 | was graded today as |
|---|---:|---:|---:|---|
| `ca_attn_w32_s0_best` | 8.21% | 16.7% | 18.8% | talk 0/6 |
| `ca_w64_s0_best` | 9.45% | 7.6% | 7.6% | talk 0/6 |
| `ca_w64_s1_best` | 9.45% | 19.4% | 15.3% | talk 0/6 |
| `gpu_best_s2_best` | 24.25% | 23.6% | 15.3% | talk 0/6 |
| `gpu_d512_s0_best` | 25.77% | 11.8% | 13.2% | talk 0/6 |

Content: `grep -ci paris lens/runs/TALK-REGRADE.txt` = **0**. The "capital of france" prompt got
no answer from any of the 5 checkpoints under either decoder; the "hello" prompt got
": , a a to , you is in <unk> : about a you and <unk> ..." (attn arm) and
"numbers and : : 10 : - : 1 x - 1 + ] * both" (w64 arm). Every checkpoint emits real words now,
and none of them answers.

## Prediction verdicts (binary, as pre-registered)

* **P1 PASS.** Every checkpoint is at or below 40% greedy `<unk>`: 16.7 / 7.6 / 19.4 / 23.6 / 11.8.
  The `<unk>` flood was the padding, and it is gone from all five.
* **P2 HELD.** 0/6 prompts answered on every checkpoint; the content verdict stays FAIL. This is now a
  measured FAIL instead of one produced by a broken context.
* **P3 FAIL.** Sampled top-k 80 was predicted LOWER than greedy on every checkpoint. It was lower on
  2 of 5 (`ca_w64_s1` 15.3 vs 19.4, `gpu_best_s2` 15.3 vs 23.6), equal on 1 (7.6/7.6), and HIGHER on 2
  (`ca_attn_w32` 18.8 vs 16.7, `gpu_d512` 13.2 vs 11.8). Recorded as a miss.

## What this changes

1. The `<unk>` part of every "0/6 talk" verdict printed before 2026-11-06 is retracted as a model
   property: those replies were produced from a context that was 20-23/32 padding tokens.
2. The talk verdict itself does NOT change. Five checkpoints, two decoders, 6 prompts, zero correct
   answers. The blocker is content — the models emit code/maths-shaped text ("def <unk> ( n ) :",
   "the total of the :") because that is what the 294M-token mixture is mostly made of.
3. 7.6-23.6% `<unk>` remains. That is the model's `<unk>` output bias (`BUGS.md:1000`: +5.85, the
   largest of 12,002 biases), not the decoder, and it is a separate item.

## What is NOT claimed

No talk result, no ship, no accuracy change. `<unk>` 7.6% means the words are real words, not that
the reply is an answer.
