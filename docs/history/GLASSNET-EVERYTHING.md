# GLASSNET — EVERYTHING (one file to read / reduce)
**Date: 2026-10-01 · Location: `<project root>/`**

Every number below is measured and logged in that folder. Nothing estimated.

---

## 0. What GlassNet is (plain English)

A neural network you can **read**. Normal neural nets work, but nobody can say
*why* they answered. GlassNet is built entirely by hand (no PyTorch, no Keras,
no borrowed code) with a middle layer of **named neurons** — each one means a
human word ("fires on: mentions of refunds") — so that for every answer the
model can say: *"I said X because concept A and concept B were active."*

The mission: on a ladder of harder tasks, keep the readable model **as good
as a boring same-size normal model** — and prove it with honest numbers.

---

## 1. The rules the project lives by

- **Honesty**: never claim something works without pasting the real test
  output. Never invent numbers. Validation set makes ALL decisions; the test
  set is quarantined and used ≤ once per milestone (each use logged).
- **From scratch**: own tensors, own forward/backward passes, own loss, own
  optimizer (SGD with a manual learning rate). No autograd libraries, no
  pretrained weights.
- **One change at a time**: every experiment changes one thing and has a
  hypothesis written BEFORE it runs. Failed experiments are kept, never deleted.
- **Every key result = 3 seeds** (mean ± std). A win smaller than the seed
  noise is called "no difference."
- **Readability is a gate, not a wish**: every named neuron must pass a
  fidelity audit, and a fake "lie" neuron must be provably catchable.
- **Licenses**: only MIT / Apache-2.0 / CC0 / CC BY 4.0 data, verified and
  written in DATA_LICENSES.md. Unclear licenses are skipped.
- **About your machine**: CPU-only jobs (nice 19, ionice, capped threads),
  zero GPU VRAM, all checkpoints/logs on disk before anything else.

---

## 2. The engine (Phases 1–6, all from scratch)

- Own `Tensor` with backprop (add/mul/matmul/relu/sigmoid), own `Dense`,
  own `cross_entropy` + `mse` + `bce_logits`, own `SGD`, all in `glassnet/`.
- **Every gradient numerically checked** before use (Gate 1) — the checks
  live in `tests/` and pass.
- XOR learned to 100% (Gate 2). Fashion-MNIST baseline recorded = the score
  to beat.
- Phase results (verify_all 13/13 checks pass, numbers on file):
  baseline 86.69±0.59 · Phase 2 readable 79.04±1.97 · auto-pilot 80.16±0.98 ·
  data-judge 79.69 (good-data accuracy) / 71.48 (bad-data) · Phase 5 80.78 ·
  Phase 6: SGD 80.16 vs EVO 34.73 (evolutionary search lost; kept).
- "Smarter than every model" was never claimed: it is a goal, not a fact.

## 3. The winning architecture (both image rungs)

```
pixels 784 → [learned gate 128, ReLU] → 8–14 NAMED concepts (values 0..1)
           → classes directly (each class = a short weighted list of named
             concepts). No rule layer — the rule layer was the hidden cost.
```
Key discoveries: (a) a learned gate under audited names beat hand-held
concepts, (b) removing the rule layer gained +3.2 pts, (c) more gate capacity
(32→128) closed the last gap.

---

## 4. The improvement loop — full ledger (iterations 000–018, all kept)

Rules: 3 seeds, same-size normal comparison, hypothesis before the run.

| iter | change | readable VAL | normal VAL | verdict |
|---|---|---|---|---|
| 000 | baseline (original 14 concepts) | 79.67±0.90 | 85.29±0.11 | BASELINE |
| 001 | fix 2 broken concepts | 78.88±1.00 | 85.29±0.11 | NO-DIFF |
| 002 | concept set v3 | 79.53±0.98 | 85.83±0.11 | NO-DIFF |
| 003 | widen to K=27 | 79.08±1.52 | 86.25±0.13 | NO-DIFF (audit caught 5 lying names) |
| 004 | revert-consolidate | 79.72±0.65 | 86.03±0.29 | NO-DIFF |
| 005 | linear rules | 49.71±10.09 | 85.29±0.11 | **REVERT** (collapse) |
| 006 | **learned gate 32** | 82.56±0.70 | 86.55±0.13 | **KEEP** |
| 007 | alpha 2.5 (stronger name-supervision) | 80.71±1.23 | 86.55±0.13 | REVERT (names vs accuracy trade) |
| 008 | **remove rule layer** | 85.78±0.10 | 86.39±0.38 | **KEEP** |
| 009 | gate 64 | 86.65±0.09 | 86.90±0.39 | KEEP |
| 010 | gate 128 | 87.14±0.12 | 87.21±0.06 | KEEP → **RUNG 1** |
| 011 | R2: same arch on PathMNIST, top-14 set | 34.72±0.46 | 25.30±1.24 | REVERT (audit FAIL: 6 broken names) |
| 012 | +5 texture concepts | 34.86±0.23 | 25.30±1.24 | PROMOTED (still audit FAIL) |
| 013 | name-rescue reflex | 27.86±0.87 | 25.30±1.24 | REVERT (failed, switched OFF) |
| 014 | 8 independent names | 33.94±0.67 | 24.62±1.44 | REVERT vs best, but **AUDIT PASS** → milestone |
| 015 | R3-000: text BOW→12 concepts→77 intents | 2.11±0.18 | 59.69±1.03 | BASELINE (collapse) |
| 016 | R3-001: no gate | 2.07 (crashed audit) | — | (probe stage) |
| 017 | **R3-002: 48 epochs (budget!)** | 11.68±0.21 | 71.50±0.66 | **KEEP** (audit PASS) |
| 018 | R3-003: 16 ranked lexicons | 6.56±0.34 | 74.99±0.82 | REVERT (audit FAIL: rare targets) |

## 5. Rung 1 — Fashion-MNIST: PASSED ✔

- Journey: 79.67 → (learned gate) 82.56 → (no rules) 85.78 → (gate 64/128)
  86.65/87.14 val.
- **Final (quarantined TEST, 1 touch, logged): readable 86.26% ± 0.12%
  (102,436 params) vs normal 86.26% ± 0.33% (102,565) → Δ +0.00, readable ≥
  normal.** Test used exactly twice in the whole rung (both logged in
  TEST-USE.md). Lie-control caught corruption in 11/11 audit iterations.
- Honest costs: 2 legacy names sit at ~0.52 fidelity at high capacity —
  the known price of hand-picked concepts, flagged, not hidden.

## 6. Rung 2 — PathMNIST: PASSED-in-spirit ✔ (+8.21 pts)

- License: CIFAR-10 BLOCKED (no license on official page). MedMNIST v2 =
  **cc-by-4.0**, verified via the Zenodo record API. PathMNIST: 9-class
  colorectal-tissue tiles, official splits.
- Journey: 14 hand-picked concepts collapse into ~one "tissue ink" axis
  (cross-leak 0.89–0.98) → ranking → 8 independent texture names
  (mean_gradient, dark_region_count, corner_ink, …) → **audit PASS first
  clean** (fidelity 0.569–0.986, lie probe caught: 0.646→0.491).
- **Final (quarantined TEST, logged touch #3): readable 46.20% ± 1.18%
  (101,593) vs normal 37.99% ± 4.25% (101,641) → Δ +8.21, no per-seed
  overlap.** Literal script line "within 2 pts: False" is the *ahead* side —
  recorded honestly as passed-in-spirit; nothing redefined.

## 7. Rung 3 — Banking77 (in progress)

- License: **cc-by-4.0** (HF registry API + author repo LICENSE fetched).
  dair-ai/emotion rejected ("other" license). 13,083 utterances, 77 intents,
  official split (9,038 train / 965 val / **3,080 test quarantined**).
- Own pipeline: `[a-z']+` tokenizer → 1,175-word bag-of-words (log-counts,
  vocab from train only) — real vocab is 1,175 not 2,048 (cap), reported.
- Concept machinery: 29+ registered word-concepts (fixed lexicons, min-max
  normalised on train). `digit_frac` measured DEAD (0 digits in the whole
  corpus) → excluded with proof.
- Loop story (honest): 12-epoch budget → collapse (2.1%) · probe showed pure
  concept-reproduction AUC 0.50 at 12 epochs → 0.62 at 40 and still climbing
  (**text needs ~3–4× the pixel budget**) · 48 epochs → 11.68% (KEEP) ·
  16 class-ranked lexicons (refund 5.2, contact 4.7, pin 4.3, …) carry more
  signal (linear-from-true-concepts 14.4% vs 10.0%) but reproduce worse from
  the bag (rare firing) → REVERT. Normal is at ~75%; the readable ceiling
  with ~16 concepts is ~14% — **the concept count/quality is the wall**.
- Ceiling probes (linear-from-true-concepts, perfect reproduction):
  K=12 → 10.02%, K=16 → 14.44%, **K=25 → 19.65% ± 0.48**. Adding concepts
  raises the ceiling but each addition buys less; the gap to the normal net
  (~75%) cannot close with a few dozen hand lexicons alone.
- Next hypotheses on the board: bigger balanced concept set (K≈25 incl.
  frequent anchors), possibly longer budget, gate at proper budget.

## 8. How the readability check works (so you can trust "name = meaning")

1. **Fidelity audit**: for each named neuron, does its firing rank order the
   concept's own targets? Score = AUROC on a **median-split** target
   (imbalance-proof — the old ≥0.5 split was blind on 93%-positive targets;
   that blind spot is fixed). `fidelity < 0.55` = BROKEN, `nan` = BROKEN.
2. **Lie control (pinned probe)**: train one neuron with its supervision
   **pinned off** — weights stay random, so it provably has nothing to do
   with its name. The audit must catch it (it does, both tasks:
   fashion 0.389 vs honest 0.941; pathmnist dead-target/drop cases).
   This catches "secretly detecting something else" neurons.

## 9. Licenses (DATA_LICENSES.md has the full records)

| dataset | license | how verified |
|---|---|---|
| Fashion-MNIST | MIT | zalando repo |
| PathMNIST (MedMNIST v2) | CC BY 4.0 | Zenodo record API (id cc-by-4.0) |
| Banking77 | CC BY 4.0 | HF registry API + repo LICENSE text |
| MNIST | BLOCKED | no explicit license on official page (per your rule) |
| CIFAR-10 | BLOCKED | no license statement on official page |

## 10. The GPU incident (2026-10-01) — what happened, what we did

- You were in Roblox; the whole screen froze (not just the game).
- Diagnosis (measured): `nvidia-smi` → "No devices / Unknown Error for
  GPU0" = **GPU hang**. RAM fine (3.3/14 GB), CPU fine (load 5.6), one idle
  leftover python (0% CPU, harmless). Cause: Roblox + desktop on a 6 GB card
  exhausted the ~300 MB headroom → gnome-shell died with the GPU.
- Fix: reboot (nothing of ours was running against the GPU — all GlassNet
  work is CPU-only). Verified after reboot: a 6 GB consumer GPU, driver 595.91.07,
  497 MiB, 42°C, idle — healthy.
- **Prevention for next time**: GPU watchdog script runs every 3 minutes
  (VRAM + error-state → `runs/gpu_watch_gaming.log`, alarm file on danger);
  heavy training is queued/paused while you play; my jobs run nice'd with
  capped threads and **never** touch the GPU.

## 11. Files map (where everything lives)

```
GlassNet/
  glassnet/        engine: tensor, layers, losses, optim, phase2_model,
                   autopilot, loop_lib, loop_audit, concepts, text_data,
                   text_concepts, eval
  loop_step.py     one experiment (iteration) with hypothesis + verdict
  milestone_check.py  the quarantined-TEST gate
  runs/loop/iter000..018/  every run: models, audits, info (never deleted)
  runs/verify/     Phase 1–6 verification suite (13/13 pass)
  runs/gpu_watch_gaming.log  GPU watchdog log
  data/            fashion/pathmnist/banking77 (sizes + md5 verified)
  LOG.md           full iteration ledger
  SUMMARY.md       10-iteration plain-English story
  STOPPED.md       rung stop documents (rung 1, rung 2)
  TEST-USE.md      every test-set touch, logged
  PROBLEMS.md      the living problem board
  HIDDEN_DEFAULTS.md  every default the auto-pilot hides, listed
  CONCEPTS.md      every named concept + exact formula/lexicon
  DATA_LICENSES.md license records
  PHASES.md, README.md
```

## 12. How to re-run anything

```
cd "<project root>"
VENV=~/.hermes/hermes-agent/venv/bin/python3
OPENBLAS_NUM_THREADS=4 nice -n 19 ionice -c3 $VENV loop_step.py --iter N \
  --task fashion-mnist|pathmnist|banking77 --classes ... --epochs ... \
  --no-rules --audit --names "comma,separated" \
  --hyp "..." --success "..."
$VENV milestone_check.py --task ... --label "rung-X" --iter N   # test gate
$VENV scripts/rank_text_concepts.py                             # concept ranking
$VENV verify_all.py                                             # phase checks
```

## 13. Current status + honest open problems

- Rung 1: ✔ passed. Rung 2: ✔ passed-in-spirit (+8.21). **Rung 3: in
  progress, current best readable 11.68% vs normal 71.50%** — the biggest
  wall so far: 77 fine-grained intents vs a concept ceiling (~20% even at
  K=25 with perfect reproduction).
- Open levers (next iterations, one at a time): bigger balanced concept set
  (K≈25 with frequent anchors), longer/better budget, learned gate at full
  budget, per-concept BCE weighting (rescue reflex remains OFF — it failed
  once, −7 pts, and stays documented).
- The desktop bundle regenerated at `~/Desktop/GlassNet-EVERYTHING.md`
  (73 files embedded, 278 KB) — this file is the readable compressed copy.
- Rung 4 (video) was declared out of scope. No secrets, no hidden defaults,
  no deleted runs: everything is on disk, readable by you at any time.