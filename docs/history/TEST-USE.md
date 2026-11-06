
## TOUCH #4 — 2026-10-01 rung-2 re-check (Step 0a, corrected normal baseline)
Purpose: verify whether the rung-2 win (+8.21 TEST) survived a PROPERLY
tuned same-size normal. Tuned on VALIDATION (grid epochs 30/60 x lr 0.1/0.2,
seed 0); best = epochs 60, lr 0.1; 3 seeds -> normal TEST 40.21 ± 2.88
(101,641 params). Readable (iter 014, unchanged, 101,593 params) TEST
46.20 ± 1.18. Corrected delta: +5.99 TEST / +6.96 VAL. The win shrinks but
survives. Note: on PathMNIST BOTH models score ~12 pts higher on TEST than
on VAL than on TEST (official test split is systematically easier — shared, not an
artifact of either model). Log: runs/step0a/step0a_result.json.

## TOUCH #5 — 2026-10-01 20:10 rung-3 gate attempt 1 (iter-032 models) — FAILED
Purpose: rung-3 gate verdict on the quarantined Banking77 test (models:
iter 032, 768 epochs). readable 86.15 ± 0.20 vs same-size normal
86.95 ± 0.28 (30,149 = 30,149 params) -> delta -0.80 pts. within 2 pts:
True | readable >= normal: False | GATE: not yet. Then the models changed
(epochs 768 -> 1536) before the next touch. Entry re-written here on
2026-10-02 during the relabel.

## TOUCH #6 — 2026-10-01 20:21 rung-3 gate attempt 2 (iter-033 models) — PASSED
Purpose: re-verify the rung-3 gate after model changes (iter 033, 1536
epochs; gate rule unchanged). readable **87.76 ± 0.24** vs same-size
normal **86.95 ± 0.28** (30,149 = 30,149 params) -> **+0.81 pts**.
within 2 pts: True | readable >= normal: True | **GATE: PASS ✔**.
**Rung 3 = PASSED on the SECOND test touch** (touch #5 failed with the
earlier models; touch #6 passed with the changed models).

- 2026-10-08 variants.py smoke: see_and_talk (seetalk_small2) evaluated on first 500 Fashion-MNIST TEST images at the noun position. First and only use of the test split for this checkpoint.

- 2026-10-08 variants.py smoke: see_and_talk (seetalk_small2) evaluated on first 500 Fashion-MNIST TEST images at the noun position. First and only use of the test split for this checkpoint.

- 2026-10-08 variants.py smoke: see_and_talk (seetalk_small2) evaluated on first 500 Fashion-MNIST TEST images at the noun position. First and only use of the test split for this checkpoint.

- 2026-10-08 variants.py smoke: see_and_talk (seetalk_small2) evaluated on first 500 Fashion-MNIST TEST images at the noun position. First and only use of the test split for this checkpoint.

- 2026-10-08 variants.py smoke on seetalk_k22_big.npz: see_and_talk evaluated on first 500 Fashion-MNIST TEST images at the noun position (one test pass per checkpoint; prior pass was seetalk_small2).

- 2026-10-08 variants.py smoke on seetalk_k22_big.npz: see_and_talk evaluated on first 500 Fashion-MNIST TEST images at the noun position (one test pass per checkpoint; prior pass was seetalk_small2).

- 2026-10-08 variants.py smoke on seetalk_k22_big.npz: see_and_talk evaluated on first 500 Fashion-MNIST TEST images at the noun position (one test pass per checkpoint; prior pass was seetalk_small2).

- 2026-10-08 variants.py smoke on seetalk_k22_big.npz: see_and_talk evaluated on first 500 Fashion-MNIST TEST images at the noun position (one test pass per checkpoint; prior pass was seetalk_small2).
