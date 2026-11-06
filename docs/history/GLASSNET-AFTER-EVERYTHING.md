# GLASSNET — WHAT HAPPENED AFTER THE OLD FILE
**Written 2026-10-01 · Everything done since the previous "everything" file
(which covered Phases 1–6 and the engine). Every number is logged on disk in
`<project root>/` — nothing estimated.**

---

## 1. Timeline (what happened, in order)

1. You approved the **autonomous improvement loop** — one change per
   experiment, hypothesis written before every run, 3 seeds, test set locked.
2. **Rung 1 (Fashion-MNIST)** — 11 experiments (iters 000–010) → **PASSED**.
3. **Rung 2 (PathMNIST)** — license hunt (CIFAR-10 blocked), 4 experiments
   (iters 011–014), deep re-engineering of the honesty checks → **PASSED-in-
   spirit, +8.21 pts**.
4. **Rung 3 (Banking77, text)** started — license verified, own text
   pipeline built, 4 iterations + 5 probes so far (iters 015–018) → **in
   progress, wall identified**.
5. **GPU incident** — your screen froze in Roblox, GPU hung, we rebooted,
   I installed a watchdog so it can't sneak up again.

---

## 2. The mission rules (the loop)

- Validation set makes every decision; **test set quarantined**, used ≤ once
  per milestone, every use logged in TEST-USE.md.
- Every idea vs the current best, **3 seeds**, mean ± std. Win smaller than
  noise = "no difference".
- **Same-size normal comparison** every time (both parameter counts reported).
- Every named neuron audited (does it really mean its label?) + a fake
  "lie" neuron that proves the checker catches liars.
- Failed experiments kept forever (runs/loop/iterNNN/), nothing deleted.

---

## 3. Rung 1 — Fashion-MNIST: PASSED ✔ (iterations 000–010)

What we tried (one change each):
- 000 baseline: readable 79.67 vs normal 85.29 → the gap.
- 001–004: fixed broken names, changed/expanded concept sets → ~79.7
  (no-diff). Audits caught 5 *lying* concepts (removed).
- 005 linear rules: 49.71 — **collapse, reverted**.
- 006 **learned gate (32 units)** under audited names: 82.56 — **KEEP**.
- 007 stronger name-supervision (alpha 2.5): 80.71 — reverted (names vs
  accuracy trade-off).
- 008 **remove the rule layer entirely** (classes read named concepts
  directly): 85.78 — **KEEP** (+3.2).
- 009–010 gate 64 → 128: 86.65 → 87.14 val.

**Final (locked TEST, touch #2): readable 86.26% ± 0.12% (102,436 params)
vs normal 86.26% ± 0.33% (102,565) → Δ +0.00 → RUNG PASSED.**
Lie control caught corruption in 11/11 audits. 2 legacy names ~0.52 fidelity
flagged honestly.

Key discovery: **the rule layer was the hidden cost all along** — removing
it made the model BOTH more accurate (+3.2) and more readable.

---

## 4. Rung 2 — PathMNIST: PASSED-in-spirit, Δ +8.21 (iters 011–014)

Licenses: CIFAR-10 BLOCKED (no license on official page). MedMNIST v2 =
**cc-by-4.0 verified via Zenodo API**. PathMNIST = 9-class colorectal
tissue tiles (texture task).
- 011 same arch + top-separation concepts: 34.72 vs 25.30 (+9.4!) but
  audit FAIL (6 broken names, imbalanced targets fooling the metric).
- 012 +5 texture concepts: 34.86 — promoted, still audit-FAIL.
- 013 name-rescue reflex: 27.86 — **reverted, switched OFF** (failed).
- 014 8 independent names: 33.94 — **AUDIT PASS (first clean)** → milestone.

**The audit was re-engineered (3 failures → final tool):**
1. Old metric was blind on 93%-positive targets (a garbage neuron scored
   0.996) → fixed: **median-split** binarisation (balanced by design),
   nan = broken.
2. Lie control upgraded to a **pinned probe**: suspect neuron's supervision
   off entirely (weights stay random) → provably not connected to its name.
   Catches on both tasks.

**Final (locked TEST, touch #3): readable 46.20% ± 1.18% (101,593) vs
normal 37.99% ± 4.25% (101,641) → Δ +8.21, no per-seed overlap.**
Recorded honestly as passed-in-spirit (script's "within 2 pts: False" is the
AHEAD side; the beat condition is met decisively).

---

## 5. Rung 3 — Banking77 (text): in progress (iters 015–018 + probes)

License: **cc-by-4.0** (HF registry + author repo LICENSE fetched).
13,083 utterances, **77 intents**, official split: 9,038 train / 965 val /
**3,080 test quarantined**.

Own text pipeline (no sklearn): `[a-z']+` tokenizer → 1,175-word
bag-of-words (log-counts, vocab from TRAIN only; 1,175 < the 2,048 cap —
reported honestly). 29+ named word-concepts registered (fixed lexicons,
min-max normalised on train). `digit_frac` measured dead (0 digits in the
entire corpus — proven, excluded).

Ledger:
- 015 (12 epochs): readable **2.11** vs normal 59.69 — collapse.
- Probe: pure concept-reproduction AUC 0.50 @12 epochs → 0.62 @40, still
  climbing → **text needs ~3–4× the pixel training budget**.
- 017 (**48 epochs**): **11.68 ± 0.21** vs 71.50 — KEEP (audit PASS).
- Probe: true-concept ceiling (linear readout): K=12 → 10.0%.
- 018 (16 class-ranked lexicons: refund 5.2, contact 4.7, pin 4.3…):
  **6.56** — REVERT; rare-firing concepts reproduce poorly from the bag
  (audit FAIL 12/16).
- Ceiling probes: K=16 → 14.44%, **K=25 → 19.65% ± 0.48**.

**The wall, stated plainly:** 77 fine-grained intents can't be covered by a
few dozen hand lexicons — even perfect concepts cap the readable net at
~20% while the normal net sits at ~75%. Next levers (one at a time):
K=25 at 48 epochs, then the learned gate at full budget (never tried with
enough budget on text — it won both image rungs).

---

## 6. The GPU incident + what changed

- You were in Roblox → whole screen froze. Measured: `nvidia-smi` =
  "No devices / Unknown Error" = **GPU hang** (Roblox + desktop blew past
  the ~300 MB headroom on 6 GB). RAM/CPU fine. **Nothing of GlassNet's
  touched it** (all our jobs are CPU-only, zero VRAM).
- Fix: reboot. Verified after: GTX 1660 SUPER healthy (driver 595.91.07).
- **Prevention:** `scripts/gpu_watch.sh` runs every 3 min (VRAM + error
  state → runs/gpu_watch_gaming.log, alarm on danger); heavy training is
  paused while you play; my jobs run nice'd, capped threads, never GPU.
- Everything was already saved to disk — the reboot lost nothing.

---

## 7. Files map (project folder)

```
GlassNet/
  glassnet/            engine + text pipeline modules
  loop_step.py         one experiment (hypothesis + verdict)
  milestone_check.py   the locked-TEST gate
  runs/loop/iter000..018/  every run kept (models, audits, info)
  LOG.md               full iteration ledger
  STOPPED.md           rung stop documents
  TEST-USE.md          every test touch (3 total)
  PROBLEMS.md          problem board · HIDDEN_DEFAULTS.md · CONCEPTS.md
  DATA_LICENSES.md     license records · PHASES.md · README.md
  GLASSNET-EVERYTHING.md  the full project file (all-in-one, this one is the delta)
  runs/gpu_watch_gaming.log
```

Re-run any experiment:
```
cd "<project root>"
OPENBLAS_NUM_THREADS=4 nice -n 19 ionice -c3 \
  ~/.hermes/hermes-agent/venv/bin/python3 loop_step.py \
  --iter N --task banking77 --classes 77 --input-dim 1175 --epochs 48 \
  --no-rules --audit --names "refund_family,contact_family,..." \
  --hyp "..." --success "..."
```

---

## 8. Current status (honest)

- Rung 1 ✔ · Rung 2 ✔ (+8.21) · **Rung 3 in progress**: best readable
  11.68% vs normal 71.50%; wall = concept coverage on 77 intents.
- No hidden defaults (HIDDEN_DEFAULTS.md), no deleted runs, no unreported
  numbers. Every claim above has its log file on disk.