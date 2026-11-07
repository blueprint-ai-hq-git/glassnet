# Public number audit — front page

Run 2026-11-06 by the autonomous worker. Question: is every number on the public front page traceable to a
run, and is the comparison it states the comparison that was actually run?

Method: take each published figure, find the line that produced it, print that line, and print the metric /
arm / epoch it belonged to. No number below is estimated.

**Where evidence lives.** Raw per-run logs are written to `lens/runs/` in the working tree, which is
`.gitignore`d (they are large and machine-local), so they are not in this repository. Every number below
therefore gives the raw log path *and* the tracked file that summarises it. If a tracked file is the only
place a number appears, that is stated rather than implied.

---

## 1. Gradient check — PASS (re-run live this turn)

Published claim: "9/9 PASS, worst absolute error 7.5e-10, worst relative 1.8e-07" (`STATUS.md`, `README.md`).

Re-run: `nice -n 19 .venv-gpu/bin/python3 tests/test_gradcheck.py`, exit 0. Printed rows:

| check | max abs err | max rel err | verdict |
|---|---:|---:|---|
| add (with broadcast) | 2.558e-10 | 1.880e-09 | PASS |
| mul | 8.283e-11 | 1.544e-08 | PASS |
| matmul | 4.761e-10 | 1.815e-07 | PASS |
| relu | 8.048e-11 | 5.750e-10 | PASS |
| sigmoid | 1.320e-11 | 4.619e-10 | PASS |
| Dense->relu->Dense->cross_entropy | 1.205e-10 | 7.263e-09 | PASS |
| sigmoid-net on XOR (mse) | 4.147e-11 | 5.793e-09 | PASS |
| Dense->bce_logits | 1.023e-10 | 2.626e-09 | PASS |
| l1_sum | 7.484e-10 | 3.742e-10 | PASS |

9 rows, 9 PASS. Worst absolute is 7.484e-10 (`l1_sum`), worst relative 1.815e-07 (`matmul`) — exactly the
published figures. Claim left unchanged.

## 2. Optimiser vs SGD — FAILED, retracted and corrected

Previously published: "**23.46% vs 19.14%**, 3 seeds, 20k windows, everything else identical, beats SGD by
4.3 points" (`STATUS.md`, `README.md` twice, `docs/RESULTS.md`, and the chart `optimiser_vs_sgd.png`).

Traced, one number at a time:

| published | where it came from | what it actually is |
|---|---|---|
| 23.44 | `lens/runs/OPT-VS-SGD.txt:1302` | arm `relgrad05` SEED 1, **epoch 11**, `MASKED ... acc 23.44%` |
| 23.47 | `lens/runs/OPT-VS-SGD.txt:1303` | same arm, **epoch 12**, `MASKED ... acc 23.47%` |
| 23.46 | `lens/runs/OPT-VS-SGD.txt:2076` | same arm, **SEED 2, epoch 9**, `MASKED ... acc 23.46%` |
| 18.93 | `lens/runs/LR-AGREE-SWEEP.txt:1316` | METHOD standard, plain `VAL ... acc 18.93%` |
| 19.35 | `lens/runs/LR-PUSH.txt:2918` | METHOD standard, plain `VAL ... acc 19.35%` |
| 19.14 | `lens/runs/LAMCONCEPT.txt:2590` | **METHOD OURS**, plain `VAL ... acc 19.14%` |

So the "three seeds" of our optimiser are three *epochs* taken from two seeds of one arm; the "plain SGD"
side is a *different metric* (plain VAL accuracy over all tokens, not assistant-masked) from three unrelated
runs, and one of those three is our own method standing in as an SGD number.

The two metrics are not close enough to subtract. From single epochs in the same run:

- `lens/runs/OPT-VS-SGD.txt:1449` — `VAL acc 14.27%` and `MASKED acc 27.46%` (same epoch, gap 13.19 pp)
- `lens/runs/OPT-VS-SGD.txt:1459` — `VAL acc 15.05%` and `MASKED acc 28.97%` (same epoch, gap 13.92 pp)

Masking alone moves a number about 13 points, so the published "+4.3 points" measured the metric switch.

## 3. The matched sweep that does exist

`lens/runs/OPT-VS-SGD-TABLE.txt` (collector `lens/scripts/opt_vs_sgd_table.py`), 16 arms, one corpus and
recipe, assistant-masked held-out accuracy, 12 epochs. Summarised in the tracked `docs/history/STATE.md`
(lines 2721-2724, whose arm labels were corrected to the starting settings on 2026-11-06; values unchanged)
and now in `docs/RESULTS.md`.

| arm (starting setting, from `OPT-VS-SGD.txt:5,159`) | best masked per seed | mean | seed spread |
|---|---|---:|---:|
| our relgrad rule `--opt-lr 0.5` | 29.32 / 29.03 / 27.43 | **28.59%** | 1.89 |
| SGD `--lr 1.0` | 27.99 / 27.68 / 27.76 | 27.81% | 0.31 |
| SGD `--lr 0.4` | 28.52 / 17.30 / 28.46 | 24.76% | 11.22 |

At matched epoch 12 (seed 2 of our arm stopped at epoch 2, so n=2 there): our rule 29.28 / 28.97 = 29.12,
SGD `--lr 1.0` 27.05 / 27.15 / 27.24 = 27.15.

The table's own pre-registered bars (`lens/runs/OPT-VS-SGD-PREDICTIONS.txt`):

- P42d best relgrad 29.28 vs best SGD 28.46 → **-0.81**, bar "within 1.00" → PASS
- P42e relgrad learning-rate spread 14.05 vs SGD 1.25 → **FAIL**
- P42f the two SGD settings only 1.25 apart, bar 2.00 → **FAIL**

Verdict: **level to slightly ahead** (+0.78 on best-epoch mean, +1.98 at matched epoch), passing the
"not more than one point worse" bar and failing the stability bar. It is not a 4.3-point win, and the public
page now says so.

## 4. Dead citations — FAILED, fixed

`README.md`'s "Verifying the numbers yourself" table pointed at four files in `docs/history/` that do not
exist: `ARCH-OURS-VERDICT.txt`, `GATE-B-VERDICT.txt`, `ARITHMETIC.txt`, `ARITH_DIAG.txt`. The real files are
`lens/runs/` copies of the same names (working tree, not committed). The table now distinguishes the tracked
summary from the raw log instead of naming a path that has nothing in it.

## 5. Claims checked and found correct (unchanged)

| Claim | Raw log | Tracked summary |
|---|---|---|
| 37.37% d512 ext6 | `lens/runs/TIED-D512-EXT6.txt:98` `ASST(by-position) ... acc 37.37% (n=58191)`, `:181` "best ep6 37.37%; LEGACY 28.57%" | `docs/history/STATE.md:2770`, `docs/RESULTS.md` |
| attention -5.41 points | `lens/runs/GATE-B-VERDICT.txt:5-8` "attention 18.69 (ep3) / no-attention 24.10 (ep3), 5.41 points WORSE, ppl 135.1 vs 123.3" | `docs/RESULTS.md` |
| architecture -1.73 points | `lens/runs/ARCH-OURS-VERDICT.txt:4-7` "PLAIN 5.53 / OURS 3.87; PLAIN 5.87 / OURS 4.07" (means 5.70 vs 3.97) | `docs/RESULTS.md` |

Python line count: `README.md` said "about 91,000". Measured this run —
`find . -name '*.py' -not -path './.venv*' -not -path './.git/*' -not -path './data/*' | xargs wc -l`
→ **104,724** lines. Corrected to "about 105,000".

## 6. Not audited this turn

Stated rather than silently passed: 35.26% (d256 full corpus), 24.20% (2M windows), 17.64% (batch 4096),
KV cache 9.7x, parameter counts 7,081,122 / 5,908,686, arithmetic 0.0% / digit-0 100%, per-layer ms/token.
None were touched by this turn; each still carries its own source file. They are the next audit's list.
