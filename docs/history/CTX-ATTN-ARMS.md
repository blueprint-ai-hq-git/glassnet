# CTX-ATTN: attention and window-64 against a matched 20k-window reference

Run: 2026-11-04 14:28-14:53. Re-verified 2026-11-06 16:1x by reading the stored run
records directly (`lens/runs/chat/*_log.json`), not from any summary.

Question: at 20,000 windows, held-out accuracy was pinned near 11% regardless of width
(32/64/128/256) and passes (3/12/30). The only lever that had ever raised it was data volume.
Two architectural suspects were therefore tested, one change per arm, against the same
reference.

## Configuration (identical in every row except the named change)

d=256, 3 epochs, batch=256, lr=0.4, lr_decay=0.5, lam_concept=0.3, method=ours, opt_lr=1.0,
max_windows=20000, assistant_only, train_turn_end, direct (tied), gate_tol=0.05,
corpus `data/chat/prepared_mix_ctrl_honest`. Every field was compared programmatically
between the reference and the arms; the only differences are `attention` and `window`.

Metric: `val_asst_acc` (BY POSITION), held out, at the end of epoch 3.

## Results

| arm | changed | seed 0 | seed 1 | mean | vs reference |
|---|---|---:|---:|---:|---:|
| reference | nothing (no attention, window 32) | 11.96% | 10.08% | 11.02% | - |
| `attn_w32` | `--attention` | 9.52% | 6.89% | 8.21% | -2.81 |
| `w64` | `--window 64` | 9.20% | 9.70% | 9.45% | -1.57 |

Per-epoch, from the same records:

- reference s0: 9.68 / 11.72 / **11.96**; s1: 8.40 / 9.67 / **10.08**
- `attn_w32` s0: 6.16 / 6.58 / **9.52**; s1: 6.89 / 6.89 / **6.89**
- `w64` s0: 7.86 / 5.80 / **9.20**; s1: 7.29 / 9.55 / **9.70**

## Verdict

Both levers lost on both seeds, against their own matched seed of the reference:

- attention: -2.44 (seed 0) and -3.19 (seed 1), mean -2.81 points
- window 64: -2.76 (seed 0) and -0.38 (seed 1), mean -1.57 points

Neither arm reached the reference on a single seed, so this is not a noise-level call that
needs more seeds to settle: the prediction written before the run ("if disorder is the wall,
attention beats 11.02%") is falsified, and so is "if the wall is tokens per example,
window 64 beats 11.02%". Doubling tokens per example (640k -> 1.28M at the same 20k windows)
did not move held-out accuracy up, which points at distinct-example count rather than token
count as the binding constraint - consistent with the earlier finding that only data volume
ever raised this number.

Caveat, stated plainly: 3 epochs, 20k windows, one metric, 2 seeds. This closes these two
levers at this scale. It does not prove attention is useless at larger scale or with
positional information - the block tested here is a bare single-head attention with no
positional signal.

## Provenance

Source run records (local, gitignored):

- reference: `lens/runs/chat/lc_w20000_s0_log.json`, `lens/runs/chat/lc_w20000_s1_log.json`
- attention: `lens/runs/chat/ca_attn_w32_s0_log.json`, `lens/runs/chat/ca_attn_w32_s1_log.json`
- window 64: `lens/runs/chat/ca_w64_s0_log.json`, `lens/runs/chat/ca_w64_s1_log.json`
- launcher: `lens/scripts/run_ctx_attn.sh`; console log `lens/runs/CTX-ATTN.txt`

A byte-identical copy of the four arms' 57 artifacts plus a sha256 manifest is kept at
`lens/runs/archive/ctx_attn_2026-11-04/` (461 MB, local only, never committed).
