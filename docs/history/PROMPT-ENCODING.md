# PROMPT-ENCODING: the talk output was mostly measuring the padding

Measured 2026-11-06 ~16:57 IST. Predictions were written first:
`docs/history/PROMPT-ENCODING-PREDICTIONS.md`. Raw output: `lens/runs/PROMPT-ENCODING-AUDIT.txt`.
Script: `lens/scripts/audit_prompt_encoding.py`. GPU was NOT used (the concept-wall A/B held it);
both checkpoints were opened read-only and nothing was retrained.

## The bug, in one paragraph

The talk decoder built its 32-slot context from the prompt and, when the prompt was shorter than the
window, LEFT-PADDED it with token id 0 — which in this vocabulary is `<unk>`. The six prompts the
trainer prints are only 9-12 ids long ("Human: hello , how are you ?" -> 10 ids), so 20-23 of the 32
context slots were `<unk>`. The model then answered in `<unk>`: 95.1% of generated tokens on the
d256 checkpoint and 96.5% on d512. Same weights, same decoder, context filled by repeating the prompt
ids instead of padding: 26.4% and 21.5%. The code was live in two places
(`lens/scripts/train_chat.py:1482-1485` and `lens/scripts/variants.py:376-378`) even though
`docs/history/BUGS.md` records the same symptom as found-and-fixed on 2026-10-22 — the fix had not
landed in either copy.

## The numbers

| Checkpoint (untouched) | run's reported held-out | left-pad `<unk>` | pad-free `<unk>` | delta |
|---|---:|---:|---:|---:|
| `lens/runs/chat/gpu_best_s2_best.npz` (d256) | 24.25% | 95.1% of 144 tokens | 26.4% | -68.8 pp |
| `lens/runs/chat/gpu_d512_s0_best.npz` (d512) | 25.77% | 96.5% of 144 tokens | 21.5% | -75.0 pp |

Per-prompt pad count (32 slots, 6 prompts, both checkpoints identical): 22, 23, 20, 22, 21, 23.
Prompt ids themselves contain zero `<unk>` (all 9-12 words are in the 12,002-word vocab), so the
`<unk>` in the replies was produced by the padding, not by out-of-vocabulary prompt words.

## Prediction verdicts (binary, as written before the run)

* **P1 PASS.** 20-24 pad slots expected; measured 20-23, prompt length 9-12 ids.
* **P2 FAIL as worded.** The left-pad side passed its clause (>= 60%: measured 95.1/96.5), but the
  pad-free side did NOT reach its bar of <= 15% (measured 26.4% and 21.5%). The direction is large and
  consistent, but the stated threshold was missed and is recorded as a miss rather than re-worded
  after the fact.
* **P3 HELD.** The pad-free replies are still not answers. Examples from the log: "2+2" -> "2 = 3 , 2 +
  4 ) is 7 + 2 = 4 miles"; "hello" -> "\` \` \` python def <unk> ( n ) :". Content is weak, exactly as
  STATUS.md says ("Coherent conversation 15%").
* **P4 PASS.** Same direction on both checkpoints, -68.8 and -75.0 pp.

## What was changed

Both live copies now fill the window by repeating the prompt ids (no padding id is ever placed in the
context), with the measurement cited in the comment so it is not undone by accident. Verified after
the change through the live `variants.talk_sentence` path on the d512 checkpoint: 3 probes, greedy,
24 new tokens each -> 2.8% `<unk>` over 72 tokens ("Hello" -> "\` \` \` python def <unk> ( n ) : ...",
"What is 2+2?" -> "2 = 3 , 2 + 4 ) is 7 ..."). That 2.8% is a 3-probe spot check, not the 6-prompt gate.

## What this does NOT mean

* It is not an accuracy improvement and not a talking model. Only the instrument changed.
* Every earlier "0/6 talk" verdict that was produced by this decoder was measured through a
  `<unk>`-dominated context and should be regarded as unmeasured, not as a property of those models.
  Re-grading those checkpoints with the fixed path is the next queued item.
* 26.4/21.5% `<unk>` is still not 0%. The remaining `<unk>` is the model choosing the `<unk>` output
  bias (`BUGS.md:1000` measured it at +5.85, the largest of 12,002), and that is a model/head issue —
  P2's falsifier territory — not the decoder.
