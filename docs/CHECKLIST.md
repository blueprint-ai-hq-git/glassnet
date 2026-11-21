# GLASSNET CHECKLIST — 2026-11-21 10:50 (rebuilt; previous version is in git history, nothing deleted)

**How to read it:** `[x]` done and verified, with the number and the file that produced it · `[~]` partly done, with an honest
percent · `[ ]` not done, with the reason. A box only gets `[x]` when a measurement backs it. "Written" is not "done". Percent is
progress on the item, never a guess at the outcome.

---

## YOUR QUESTION FIRST: "is the make-it-train-times-better part going great?"

**No - and the measurements say exactly where it stands.** The machinery is proven; the learning is not.

| what | measured | source |
|---|---|---|
| own block, forward | hand reference **4.441e-16**, KV-cache equivalence **1.192e-07**, LayerNorm by hand **0.000e+00** | `lens/glass_block.py` |
| own block, backward | worst relative **8.815e-07**, input gradient **2.650e-09** | `lens/glass_grad.py` |
| own trainer on real text | held-out **12.32%** vs a **4.08%** matched constant (**+8.24**) | `lens/runs/chain_glssr32.out` |
| factored embedding | **1.84x fewer parameters** (454,336 vs 836,352), accuracy delta **-0.10** (inside noise) | `lens/runs/GLASSLM-3SEED*.txt` |
| **can it generalise a rule?** | **NO**: held-out **36.72%** vs train **53.96%**, a **17.2-point gap**, 3 seeds | `lens/runs/MEMG-LONG-r8.txt` |
| does more data fix it? | **barely**: 600 -> 5000 pairs gave **+1.45** (seed 0 **38.17%**) | `lens/runs/MEMG-DATA-r9.txt` |
| does more capacity fix it? | **no**: capacity allocation measured **+0.27 +- 0.31** (bar +0.5, NOT MET); the small-table shape was **unstable**, **-0.94 +- 1.67** | `lens/runs/ALLOC-3SEED-r6.txt` |

**Conclusion:** the model **fits its training pairs instead of learning the rule**, and neither more data nor more capacity moves it.
That is the honest state, and it is exactly what your new ask targets.

---

## PHASE 1 — THE ENGINE  (100%)

- `[x] 100%` **Own autograd engine** — `glassnet/tensor.py`, own Tensor with backward. No PyTorch, no autograd library.
- `[x] 100%` **Dense layer, ReLU, sigmoid, MSE + cross-entropy, plain SGD with a manual learning rate** — `layers.py`, `losses.py`, `optim.py`.
- `[x] 100%` **Every gradient checked numerically before being trusted** — `tests/test_gradcheck.py`: **GATE 1 PASS**, worst error **1.023e-10** (dense->bce_logits 1.023e-10, l1_sum 7.484e-10).
- `[x] 100%` **XOR learned to 100%** — `gates/phase1_xor.py`: **GATE 2: PASS**, real weights printed.
- `[x] 100%` **Fashion-MNIST baseline recorded** — `runs/phase1/baseline_summary.json`.

## PHASE 2 — READABLE ARCHITECTURE  (image line closed at 81.95%)

- `[x] 100%` **Named concept neurons** — 12 per-region concepts, human names, documented pixel computation.
- `[x] 100%` **Sparse rule layer** — 32 single-concept rules (L1 + pruning); each rule reads 2-3 concepts.
- `[x] 100%` **Answer = weighted sum of rules** — **81.95% +- 0.53** at **10,166 parameters**, **1 of 12 concepts mislabelled** (named, not hidden).
- `[x] 100%` **`explain()` is causal** — removing the top-2 named rules removes **346.7%** of the logit margin vs **10.8%** for two random rules.
- `[x] 100%` **Metric flaw found and fixed** — the concept *share* ratio overstated the named layer by up to **60x** under cancellation. `explain()` now returns `concept_signed`, `reconstruct`, `cancellation`.
- `[~] 60%` **Ablation + audit gates** — scripts exist, full 3-seed run not re-executed.
- `[ ] 0%` **Chat-model concepts carry the decision** — **NOT MET**: median **18.00%** (d512) / **8.89%** (d1024); `explain()` exact on **0 of 200** windows.
- `[ ] 0%` **Concept layer answers by cancellation** — 17 of 36 concepts supply margin, 19 oppose; top-8 supply **68.7%** against an 80% bar. Decorrelation was measured and **closed** (saturates at 0.5).

## PHASE 3 — AUTO-PILOT TRAINER  (~85%)

- `[x] 100%` **`glassnet/autopilot.py`** — 5 entry points, picks its own settings, no typed hyperparameters.
- `[x] 100%` **Runs unattended through script files** — every arm gets its own systemd unit, log and checkpoint.
- `[~] 85%` **Optimiser automation CLOSED with cause** — hand-set momentum lr 0.05 = **12.51%**; gain rule **8.90 / 5.89**; per-element **0.26 / 0.00**; per-tensor **0.03**; momentum 0.50 **diverged**. Hand-set 0.05 ships.
- `[~] 70%` **Phase 3 gate** (`phase3_autopilot.py`, `phase3_controls.py`) — exist, not re-run in full.

## PHASE 4 — DATA JUDGE  (written, gate output exists)

- `[x] 100%` **Judge written** — `glassnet/judge.py`, 15 functions; public API `train_judge()` / `judge()`.
- `[x] 100%` **Gate output exists** — `runs/phase4/summary.json`.
- `[ ] 0%` **Judge accuracy vs a control measured this session** — not re-measured; no number quoted until it is.

## PHASE 5 — THE SMALL LIBRARY  (100% code, one run owed)

- `[x] 100%` **`glassnet/api.py`** — one `GlassNet` class: `train`, `accuracy`, `explain`, `train_judge`, `judge`, `save`, `load`.
- `[x] 100%` **12-line example** — `examples/example_20_lines.py`; **`lens/api.py`** 7 functions.
- `[x] 100%` **Library tests pass** — `tests/test_library.py`: **7/7 passed**.
- `[ ] 0%` **The example executed end-to-end** — it trains Fashion-MNIST; deferred while trainers held the cores.

## MEMORY BANK  (CLOSED by its own falsifier)

- `[x] 100%` **Readable window bank built and certified** — gradient **4.334e-10**, batched == loop at **1e-16**, no slot collapse (cosine **-0.081**).
- `[x] 100%` **Root cause found** — the read had **no position/age term**: a 16-slot window landed on a wrong token **95%** of the time.
- `[x] 100%` **Age key certified** — **1.713e-09** worst error; zeroed key is a bit-exact no-op (**0.000e+00**); synthetic delay-8 recall **20.61% -> 85.61%**.
- `[x] 100%` **On real text it FAILED** — trainer **23.02 vs 22.88 (+0.14)**, independent **20.56 vs 20.62 (-0.06)**.
- `[x] 100%` **The line is CLOSED**: bank K=8 **16.57%** vs K=0 **16.08%**, delta **+0.49** against a **+1.0** bar, falsifier fired at <= +0.5. **Do not reopen with another seed.**
- `[ ] 0%` **Memory gate on real text** — best k=16 at **2.254%** against a **10%** gate.
- `[ ] 0%` **Unlimited context** — effective context **2 tokens** (L=1 9.27% -> L=2 16.73%, flat to L=128 at 16.40%).

## YOUR OWN BLOCK, TRAINER AND CACHE  (built 2026-11-20)

- `[x] 100%` **Own LayerNorm / attention / MLP / residuals with our own KV cache**, byte-accounted — **544 parameters, every one named**.
- `[x] 100%` **Own backward pass**, every named gradient certified against finite differences.
- `[x] 100%` **Own trainer `lens/glass_lm.py`** — owned embedding, position table, tied output, own momentum SGD: **836,352 parameters in 24 named tensors**.
- `[x] 100%` **Factored embedding `E = A @ B`** — **46% fewer parameters** at no accuracy cost (delta **-0.10**, inside noise).
- `[x] 100%` **Capacity allocation MEASURED, no reliable win** — published, then corrected in public when 3 seeds killed the one-seed signal.
- `[x] 100%` **Memorise-vs-generalise switch WORKS** — facts **11.08% -> 100.00%**, rule accuracy **+0.00**, collisions **0 of 600**, all 3 seeds.
- `[ ] 0%` **Generalisation FAILED and reported as failed** — 36.72% held-out vs 53.96% train.

## PHASE 6 — YOUR NEW ASK: INNER THINKING, SELF-CLEANING, COMMON SENSE  (starts today)

Built **inside** the network, one mechanism at a time, each with its own written gate before it runs:

- `[ ] 0%` **6.1 Self-assessment ("thinking about its own answer")** — beside each answer the model outputs a score for whether
  that answer is right. **Gate:** self-score ranks correct above wrong at **AUROC >= 0.70** on held-out, reported **with** a firing
  floor (AUROC measures ranking, not detection).
- `[ ] 0%` **6.2 Self-cleaning ("cleaning itself while it trains")** — inject **20%** label noise; the model downweights the
  examples it judges unreliable. **Gate:** recover **>= 50%** of the gap between training-on-noise and training-on-clean.
- `[ ] 0%` **6.3 Common sense (invariant checks)** — a second pass re-checks the answer against invariants it learned.
  **Gate:** rejecting inconsistent answers beats accepting all, held-out, 3 seeds.
- `[ ] 0%` **6.4 Learns quicker** — steps to a fixed held-out threshold, with vs without the inner loop. **Gate:** a real
  reduction, 3 seeds, paired.
- **Falsifier for the phase:** if 6.1 cannot rank its own correctness above **AUROC 0.60**, the inner loop has nothing to stand on
  and **Phase 6 closes** rather than being patched.

## EXPORT, SERVING, TRANSPARENCY

- `[x] 100%` **`.safetensors` export + Hub publish, verified from the downloaded bytes** — accuracy reproduces from the Hub file alone.
- `[~] 50%` **`.gguf` export** — `lens/export.py` exists; **never verified against a standard container**.
- `[ ] 0%` **Local server** · `[ ] 0%` **live browser inspector** · `[ ] 0%` **standard tags / thinking block**.
- `[ ] 0%` **Per-word state** · `[ ] 0%` **10B-on-disk execution**.

## INFRASTRUCTURE AND SAFETY

- `[x] 100%` **Stop/start loses nothing** — resume verified; refuses on missing tensors; recomputes lr from the absolute step.
- `[x] 100%` **Boot autostart, game-guarded** — two real bugs fixed (enabled after boot; the space in "the project disk" split `ExecStart`).
- `[x] 100%` **Freeze watchdog** — every ~14 s, includes GPU responsiveness; two of my own bugs fixed.
- `[x] 100%` **Dead-flag guard** `tools/check_env_wiring.py`; verifier fixed; **privacy gate caught my own name in 5 public files and 3 new ones**.
- `[x] 100%` **Code published** — 19 runner scripts + 5 modules, sanitised, **77 .py and 19 .sh tracked**, PII gate HITS: none.
- `[x] 100%` **Off-drive ext4 safety copy** — **2.1 GB, 3,605 files verified**, git bundle included.
- `[ ] 0%` **GPU freeze root cause UNKNOWN** — not load: gaming showed **31-39% util, 44-50 W of 125 W, 45 C**, CPU **70% idle**.
- `[ ] 0%` **NTFS damage on the project disk** — **89% full**, repeating MFT corruption per boot. Needs his password.
- `[ ] 0%` **Off-drive backup behind** — `sdb` not mounted.

---

## THE HONEST QUEUE, IN ORDER

1. **6.1 self-assessment head** — the foundation every other common-sense idea needs; without it there is no inner signal to act on.
2. **6.2 self-cleaning** on top of 6.1, measured against injected label noise.
3. **Generalisation: architectural, not data** — 5000 pairs gave +1.45, so the next attempt is a **pre-registered** structural
   change (a loop over output digits / scratchpad), not more data and not a wider model.
4. **The honest benchmark you asked for** — how good is it *actually*, against the **4.08%** matched constant, the **12.32%**
   own-trainer line and the **81.95%** readable image model.
5. Owed but not started: `.gguf` verified, local server, live inspector, thinking block.
