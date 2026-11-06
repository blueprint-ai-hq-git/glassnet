"""correct_f.py — fix the record on Target F's concept claim.

Two separate defects were found in my own F evaluation:

  (1) LOADER: f_train saves short keys (Wc/Wa/bc/ba); the audit loader matches by parameter
      NAME, so only the embedding loaded and the concept/answer layers stayed random.
  (2) CHECKPOINT SELECTION: f_train keeps the snapshot at the best LM perplexity. On generated
      data the LM loss is best at epoch 1-2 and never improves, so the SAVED student is an
      epoch-1 model -- audited before its concepts had any chance to train.

The long diagnostic (same corpus, same settings, no early stopping) shows the concepts DO train
on generated data and stay alive: mean AUROC 0.67 -> 0.75 over 30 epochs, max 1.0000.

So the "concepts died" claim is WITHDRAWN. The LM collapse claim stands.
"""
import glob
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")

print("=" * 92)
print("CORRECTING TARGET F — checking the mechanism before writing anything")
print("=" * 92)
rows = []
for f in sorted(glob.glob(os.path.join(R, "models_f_f_*.json"))):
    rows += json.load(open(f))
print(f"{'model':<10} {'rule':<7} {'seed':>4} {'best_epoch':>11} {'stopped_at':>11} "
      f"{'val_ppl':>10}")
for r in rows:
    print(f"{r['model']:<10} {r['rule']:<7} {r['seed']:>4} {r['best_epoch']:>11} "
          f"{r['stopped_at']:>11} {r['val_ppl']:>10.2f}")
be = [r["best_epoch"] for r in rows]
print(f"\nbest_epoch values: {sorted(set(be))}  -> "
      f"{'ALL EARLY: the saved students are untrained' if max(be) <= 3 else 'some later'}")

diag = json.load(open(os.path.join(R, "f_diag_generated.json")))
print(f"\nconcept AUROC on the SAME generated corpus, no early stopping, 30 epochs:")
print(f"  epoch 1 : {diag['rows'][0]['auroc_mean']:.4f}")
print(f"  epoch 6 : {diag['rows'][5]['auroc_mean']:.4f}")
print(f"  epoch 30: {diag['rows'][-1]['auroc_mean']:.4f}  "
      f"(max over concepts {diag['rows'][-1]['auroc_max']:.4f})")
print("  => the concepts TRAIN FINE on generated data. They were never dead.")

# ---------------------------------------------------------------- patch the results file
p = os.path.join(PROJ, "TARGET-F-RESULTS.md")
txt = open(p).read()
banner = """# TARGET F — RESULTS

> **CORRECTION, 2026-10-05 21:50 — read this first.** Section 3's claim that the concepts "die"
> under self-training is **WITHDRAWN**. It was produced by two defects in my own evaluation:
> **(1)** the audit loader matched checkpoints by parameter name while `f_train` saved short keys,
> so only the embedding loaded and the concept layer stayed **random**; and **(2)** `f_train`
> keeps the snapshot at the **best LM perplexity**, which on generated data is **epoch 1–2**, so
> the audited student was an untrained model. A controlled diagnostic on the same corpus with the
> same settings (no early stopping, 30 epochs) shows the concepts **train normally and stay
> alive**: mean AUROC **0.67 → 0.75**, max 1.0000.
> **What stands:** the LM collapse (§2) — the best student on generated data reaches ~1800 ppl
> against 76 on real data, and the real-data control reproduces the teacher. **What is withdrawn:**
> the concept-death claim, and the "coverage 100% while every concept is dead" framing built on it.
> The underlying lesson survives in a different form: **a best-LM checkpoint can be an untrained
> model**, and any audit must state which epoch it measured.
"""
assert txt.startswith("# TARGET F — RESULTS")
txt = txt.replace("# TARGET F — RESULTS", banner, 1)
open(p, "w").write(txt)
print("\npatched TARGET-F-RESULTS.md with the correction banner")

LOG = """
---

## 2026-10-05 21:50 — TARGET F CORRECTION (my own evaluation was wrong twice)

**Two defects in my F evaluation, both mine:**

1. **Loader mismatch.** `f_train.py` saves short keys (`emb/Wc/bc/Wa/ba`); `M3.load_ckpt` matches
   by parameter NAME (`emb/concept.W/concept.b/answer.W/answer.b`). So only the embedding loaded
   and the **concept and answer layers stayed RANDOM**. Proven: concept col0 mean **0.5036**
   (broken load) vs **0.0453** (correct load) -- `allclose` False.
2. **Checkpoint selection.** `f_train` keeps the snapshot at the **best LM perplexity**. On
   generated data the LM loss is best at **epoch 1-2** and never improves, so the SAVED student is
   an **epoch-1 model** -- audited before its concepts had any chance to train. Verified: every
   F student's `best_epoch` is 1-3.

**Controlled diagnostic on the SAME generated corpus, same settings, no early stopping:**
mean concept AUROC **0.6719 (ep1) -> 0.7119 (ep6) -> 0.7533 (ep30)**, max 1.0000. **The concepts
train normally on generated data. They were never dead.**

**WITHDRAWN:** the "all nine concepts died" claim, and the "coverage 100.0000% while every
concept is dead" framing built on top of it.
**STANDS:** the LM collapse. The best student on generated data is ~1800 ppl against 76 on real
data, and the real-data control reproduces the teacher (30.29% / 76.27). That result never
depended on the audit.

**Lesson recorded:** a best-LM checkpoint can be an untrained model, and any audit must state
which epoch it measured. Also: a checkpoint's key names are part of its interface -- loading must
be verified, not assumed.

**Artifacts:** `runs/t3/f_diag_generated.json`, `f_diag_real.json`, corrected `runs/t3/f_audit.json`.
"""

STATUS = """# GLASSNET — STATUS

Updated 2026-10-05 21:50. **Record: A = NOT MET, A2 = NOT MET, A3 = MET, C = PARTIAL, D = 100% readable / BEHIND, E = 100% readable / PARITY, F = LM collapse CONFIRMED (concept claim WITHDRAWN).**

1. **CORRECTION I OWE YOU:** my Target F claim that "all nine concepts died" is **WITHDRAWN**. It came from two bugs in my own evaluation, not from the model.
2. **Bug 1 — loader:** my trainer saves keys `Wc/Wa`, the audit loader matches by parameter *name* (`concept.W/answer.W`), so **only the embedding loaded and the concept layer stayed random**. Proven: concept col0 mean 0.5036 (broken) vs 0.0453 (correct).
3. **Bug 2 — checkpoint selection:** the trainer keeps the snapshot at the **best LM perplexity**, which on generated data is **epoch 1–2**, so I audited an untrained model. Every F student's `best_epoch` is 1–3.
4. **What a controlled diagnostic shows** (same corpus, same settings, no early stopping): concepts **train normally and stay alive** — mean AUROC 0.67 → 0.75 over 30 epochs, max 1.0000.
5. **What still stands — the LM collapse.** The best student on generated data is **~1800 ppl** against **76** on real data, and the real-data control reproduces the teacher (30.29% / 76.27). That never depended on the audit.
6. **So your self-improvement goal still FAILED** — the student is far worse than its teacher — but the *reason* is the data, not broken concepts.
7. **Lesson recorded:** a best-LM checkpoint can be an untrained model, and **any audit must state which epoch it measured**. A checkpoint's key names are part of its interface — loading must be verified, not assumed.
8. **Library testing in progress:** the Phase-5 library (`GlassNet/train/explain/accuracy/judge/save/load`) has a 7-scenario suite running now (gate1, construct, train+explain+report, judge, save/load, the ≤20-line gate, experimental optimizer).
9. **Still open:** G2 (readability across sizes) untested · G4 (our own fine-tuning and from-scratch methods) untouched · the **human R4 score** still needs you.
10. **Next:** finish the library suite, fix the F trainer's checkpoint rule, and pre-register the verifiable-task design so it can run while you're at school.

Detail: `TARGET-F-RESULTS.md` (correction banner at the top) · `LIT-CHECK-SELFIMPROVE.md` · `TARGET-E-RESULTS.md` · `LOG.md`.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
