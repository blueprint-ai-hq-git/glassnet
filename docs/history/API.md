# GlassNet — API quickstart (one readable vision-language stack)

Everything below is local, our own engine, our own checkpoints. Nothing phones home; nothing
needs a GPU to run inference (CPU is fine — one image, milliseconds).

## 1. The three-model interface

```python
import os, sys
sys.path.insert(0, "<project root>")
sys.path.insert(0, "<project root>/lens/scripts")
from variants import build_default

v = build_default()                 # loads what exists: ['see', 'see_talk', 'talk']

# SEE — a 28x28 float array in [0,1], flattened (784,)
r = v.see(img)                      # {"answer": "class_9", "confidence": 0.85, ...}

# TALK — one next word
w = v.talk([0] * 16)                # {"next_id": ..., "word": "...", "confidence": ...}

# TALK — a sentence (verified decoder: top-k 40 + repetition penalty)
s = v.talk_sentence("Human: hello , how are you ?\n\nAssistant:", n_new=40, seed=0)
# s["reply"] -> "can you <unk> ..." (the vocab-3500 arm; many words are <unk> — stated honestly)

# SEE + TALK — the fused model: a caption word driven ONLY through named concepts
prefix = [pad, pad, bos, this, is, a]           # token ids from the checkpoint's own vocab
c = v.see_and_talk(img, prefix)                 # {"word": "sandal", "confidence": 0.76,
                                                #  "fired_concepts": [("is_dark", 0.78), ...]}

# EXPLAIN — the exact arithmetic behind that word (residual 0.0 vs the model)
e = v.explain(img, prefix)                      # concept -> contribution table for the chosen word

# REFUSAL (tested, enforced): a build without `talk` RAISES on v.talk(...) instead of guessing.
```

Checkpoints: `lens/runs/seetalk/seetalk_k22_big.npz` (ship, K=22, 4/4 gates),
`lens/runs/image/lad_image_cls_ep24.npz` (see), `lens/runs/chat/best1_ep19.npz` (talk).
Vocab words travel INSIDE each checkpoint; `talk` additionally requires
`data/chat/prepared/vocab_v3500.json` with EXACT width match (mismatch = refuse, not guess).

## 2. The customer-data cleaner (optional ON/OFF)

```python
from clean import load_any, clean
items = load_any("your_data.jsonl")        # jsonl / json / txt / directory — auto field pick:
                                           # text, content, problem, question, code, solution
kept, report = clean(items, enabled=True)  # enabled=False = exact pass-through
print(report["dropped"])                   # counts PER RULE, plus examples per rule
```

Rules (thresholds frozen in `clean_rules_frozen.json`, all stated, all tunable via `rules=[...]`):
drop_empty · drop_too_short (<20) · drop_too_long (>20000) · drop_low_signal (alnum<0.35) ·
drop_repetition (single char >0.35) · drop_duplicate (exact) · strip_boiler (opt-in).
Proven: magicoder 75,197→99.95%, codeparrot-apps 5,000→99.22%, self-test counts exact.

## 3. The teacher loop (less data, optional ON/OFF)

```
python lens/scripts/teacher_loop.py --teacher-on on     # argmax teacher, fraction --frac 0.1
python lens/scripts/teacher_soft.py --teacher <npz> --tau 2.0 [--noise 0.3]
```
MEASURED RESULT (4 experiments, 3 seeds): 10% of the data + teacher output lands within
8.6-9.8 pts of the 100%-data ceiling (t1 PASS 4/4). The teacher does NOT beat 10% REAL labels
(t2 FAIL 4/4) — on deterministic-label tasks ground truth dominates (teacher mass 0.743 vs 1.000,
measured). Distillation pays where labels are ambiguous or expensive; boundary documented in
LOG.md so nobody re-tries this blind.

## 4. Ship gate — run this after ANY change

```
bash lens/scripts/run_selfcheck.sh        # 28 checks, ~2 s, exit 0 or it is broken
```
Cleaner counts · every checkpoint: load + bit-identical round-trip + explain identity + determinism
· tampered concept formula must be REFUSED · all three variants answer · both refusals enforce.

## 5. Demo videos (real model output, labeled honestly)

- `lens/runs/video/cam_labeled.mp4`   — see-only camera, 85.0% on 120 frames, "MODEL EYES 28x28"
- `lens/runs/video/caption_cam_k22.mp4` — fused model writing captions word-by-word, 38/48 = 79.2%

## 6. What is NOT true (kept here so nobody oversells it)

- It is not open-vocabulary: captions come from a 12-30 word vocab; "sharper eyes"/"more data"/
  "more concepts beyond K=22" were each MEASURED and changed nothing on this task family —
  the 78-80% band is the task (10 classes in 784 px), not the architecture.
- "Readable concepts improve LEARNING" is not supported (real ≈ dummy random projections at
  every width). What the concept layer demonstrably owns: exact explainability, audited firing
  (AUROC), refusal behaviour, and transparency at ZERO accuracy cost.
- Face capabilities: blocked on licence (all checked datasets fail the rule) — owner decision
  pending (skip / own photos / named dataset).
