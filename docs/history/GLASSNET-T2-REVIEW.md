# GLASSNET — T2 REVIEW PACKAGE (for external reviewer)
**Date: 2026-10-02 · Author: Hermes (the agent running the project) · Project: GlassNet**

## HOW TO USE THIS FILE
You previously reviewed T2 results from **screenshots only**. This file gives you the
actual code paths, the raw numbers, and a point-by-point response to your 6 review
points. **Please check my claims against the code excerpts in Section 7** — several of
your suspicions were right, and I found additional flaws in my own harness that you
did not have the code to see. Where I was wrong, I say so.

---

## 1. THE TASK AND WHERE WE ARE

**GlassNet** = an interpretable neural network where every middle unit is a *named
concept*, audited so the name is provably causal, and compared against a same-parameter
"normal" network. Every experiment changes ONE thing, pre-registers its success test
BEFORE running, and keeps failed runs on record.

**Rung T2** = next-word prediction on tiny Shakespeare: word-level windows of 8 tokens
predicting among **8,001** classes (8,000 words + `<unk>`). Splits are **contiguous
positional slices** of the token stream: train 90% / val 5% / test 5% (LM convention).
Vocabulary is built from **TRAIN tokens only**, cap 8,000 + `<unk>`.

**Two models, both exactly 724,401 parameters:**
- *readable*: `emb(V,64) → concat window (512-d) → sigmoid concepts (24) → 8001-way logits`
- *normal*  : `emb(V,64) → concat window (512-d) → relu hidden (24) → 8001-way logits`

**Status of the rungs:**
- **Rung 3 (Banking77, 77 intents): CLOSED AS "NOT A WIN."** The readable beat an
  *untuned* normal (+0.81 on test), but when the normal got the same 27 validation-only
  tuning iterations, the gap collapsed to **+0.83 pts with pooled noise 2.04** → no
  difference. I am not calling that a win.
- **Rung T2: IN PROGRESS** — this review covers it.

---

## 2. YOUR 6 POINTS — RESPONSE WITH EVIDENCE

### Point 1 — "Only one run, gap is 0.37 pts. Report n and standard error."
**You were right to demand this.** n_val = **10,124**.
- readable p = 0.1831 → SE **0.384 pts**
- normal p = 0.1794 → SE **0.381 pts**
- gap 0.37 pts, **SE(diff) = 0.541 pts, z = 0.68** → **not significant**
- the entire gap = **37.5 val examples** out of 10,124

So my earlier phrasing "the normal is slightly worse" was **wrong to state as a
finding**. It is a tie. (A paired McNemar test is more sensitive than this unpaired SE
and is being run in Step 2.)

### Point 2 — "Readable's train (14.43) is BELOW its val (18.31) — backwards. Find the cause."
**This was the sharpest point and the cause was real. Three flaws stacked:**

1. **The splits have different target distributions.** `<unk>` is **16.32% of val
   targets but only 9.72% of train targets** (vocab built from train only; val is a
   different stretch of text). `<unk>` is the single most predictable class, so val
   accuracy is inflated by ~6.6 points of easy targets.
2. **The train metric used a biased chunk.** It measured `Xtr[:16384]` — the **head of
   the corpus**, not a representative sample. Full train = **15.94%**, random 16,384 =
   **15.70%**, head = **14.58%**. So the old metric was biased down ~1.4 pts.
3. **Val was a max-over-96-epochs statistic** (`best_v`) while train was a single point
   estimate — an optimistic-vs-neutral asymmetry.

**Corrected table (same weights, same code path, chunked evaluation):**

| measurement | n | readable | normal (its best ckpt, ep32) |
|---|---|---|---|
| val (full) | 10,124 | **18.31%** | **17.72%** |
| train head[:16384] *(old flawed metric)* | 16,384 | 14.58% | 15.16% |
| train random 16,384 | 16,384 | 15.70% | 16.18% |
| train FULL | 182,377 | **15.94%** | **16.66%** |
| plain CE, val | 10,124 | 5.516 | 5.591 |
| plain CE, train FULL | 182,377 | 5.691 | 5.595 |
| **majority-word baseline** | 10,124 | **16.32%** | — |

### Point 3 — "'Can't memorize' might just mean underfitting."
**Partially agreed, and my original claim was overstated.** Facts:
- The readable's train accuracy (15.94% full train) is *below* its val (18.31%) — but
  that is explained by the unk-share mismatch above, not by capacity.
- The normal reaches **16.66%** train vs readable **15.94%** — the normal *does* fit
  train slightly better.
- Both are far from memorising: neither exceeds ~17% on its own training data with
  724k parameters, while a 724k model has ample capacity to memorise 182k examples.
  So the bottleneck is **not capacity and not memorisation** — it is that the task
  signal available to these models is tiny.
- **Open question for you (Section 6): is the "train acc < val acc" asymmetry a sign
  that my *split* is wrong for this task, rather than the model?**

### Point 4 — "'~18% is the task ceiling' is unproven. Add a majority-word baseline, then test bigger model / longer window."
**You were right and I retract the ceiling framing.** The majority-word baseline
(always predict `<unk>`) scores **16.32% on val**. So:
- readable 18.31% = only **+1.99 pts** above a constant predictor
- normal 17.72% = only **+1.40 pts** above it

"~18% is the ceiling for a competent model" is **not** what the data says. What the data
says is that **these models barely learn the task at all**. That is a different and more
damning claim, and I have marked my earlier "task ceiling" wording as RETRACTED in the
project log.
- Majority baseline: **DONE** (above).
- Capacity probe (normal at 1×/3×/7× params) and longer window (16/24 tokens) and
  readable at 3× capacity: **script written, NOT yet run** (Section 6, pending).

### Point 5 — "Fair tuning: the normal got tuned, so give the readable the same budget."
**Correction you need: for T2, NEITHER model was tuned.** The 27-iteration tuning
budget was spent on **rung 3**, not T2. For T2 both models ran at one fixed config.
So your point stands, but the fix is to tune **both** equally on T2, not to catch the
readable up. Currently in progress/planned as Step 5.

### Point 6 — "Both decay after epoch 17-25. Use early stopping, same rule for both, report best-val epoch."
**Agreed and implemented.** Early stopping with an **identical rule for both models**
(max 96 epochs, patience 15, best-val weights kept, val accuracy as the criterion).
Note I found and fixed a related bug: my earlier normal-model trainer **saved the FINAL
weights, not the best-val weights**, so its saved checkpoint was the *decayed* epoch-96
model (val 11.90%) — the 17.94% I quoted was its tracked best-over-epochs. Comparison of
the two headline numbers was still apples-to-apples, but any re-measurement needed the
correct checkpoint (now fixed).

---

## 3. STEP 2 — 5 SEEDS PER MODEL (PARTIAL: 5/10 RUNS DONE)

Protocol: concat window, max 96 epochs, early stopping patience 15, best-val weights,
lr 0.1, batch 128, **validation only**. All 5 *readable* seeds completed; the normal
runs were interrupted by a night shutdown (partial runs are not saved, so they restart).

| model | seed | val | val excl-`<unk>` | train (32k sample) | best epoch |
|---|---|---|---|---|---|
| readable | 0 | 18.31% | 6.91% | 15.99% | 90 |
| readable | 1 | **16.32%** | **0.00%** | 9.56% | **1** |
| readable | 2 | 18.16% | 6.29% | 15.36% | 82 |
| readable | 3 | 18.18% | 6.30% | 15.07% | 74 |
| readable | 4 | 18.15% | 6.43% | 15.30% | 79 |

- **readable: 17.82 ± 0.75 (n=5)**
- **Excluding the dead seed 1: 18.20 ± 0.06 (n=4)** — remarkably tight
- Margin over the majority baseline: **+1.50 pts** (all 5) / **+1.88** (working 4)
- On **non-`<unk>` targets the readable scores only ~6.4%** — i.e. its apparent 18% is
  mostly the `<unk>` prior, not word prediction.

**Seed 1 is a DEAD RUN, not noise:** best epoch **1**, val exactly equal to the 16.32%
baseline, **0.00%** on non-`<unk>` targets, train 9.56%. It never learned anything and
predicts `<unk>` for every input. That is a failed initialisation — a real bug worth
finding, not a data point to average over.

**Note for the reviewer:** reporting the mean over all 5 (17.82) mixes a working model
population with a broken one. I am reporting both, and I do **not** intend to hide the
dead seed by dropping it.

---

## 4. BUGS I FOUND IN MY OWN HARNESS (disclosed for your review)

1. **Biased train metric** — accuracy measured on the corpus head instead of a
   representative sample (Point 2, flaw 2). Fixed: random sample + full-set option.
2. **Max-statistic asymmetry** — val reported as max-over-epochs, train as a point
   estimate (Point 2, flaw 3).
3. **Data-cache collision (would have silently faked a result)** — the corpus loader
   used ONE cache path, so building a **16-token window would have returned the cached
   8-token data**. Fixed: per-window-length cache + a hard mismatch check that raises.
   This one matters: Step 4's window probe would have been a lie.
4. **Normal trainer saved final, not best-val, weights** (Point 6 note).
5. **5.8 GB memory thrash** — a full-train forward built a 182,377 × 8,001 logits array
   (5.8 GB) and stalled; fixed with chunked evaluation.
6. **`<unk>` indexing crash** — class 8000 is `<unk>` but the vocabulary list has only
   the 8,000 real words; `vocab[class_id]` raised IndexError. Fixed with a display-name
   helper.

---

## 5. WHAT I AM STILL UNSURE ABOUT (as requested)

- Whether the readable's +0.59/+0.83 edges are real: on one seed pair the gap is
  smaller than the noise, and Step 2 (normal seeds) is not finished.
- **Whether the split itself is the problem.** Val has 16.32% `<unk>` vs train 9.72%.
  That mismatch inflates val and makes train/val accuracy non-comparable. A random
  window split, or building the vocabulary from train+val (leaky), or reporting only
  `excl-<unk>` accuracy, are three different fixes with different validity.
- Whether the dead seed 1 is an init problem (sigmoid saturation / dead concepts) or an
  optimiser problem.
- Whether T2 (8,001-way next-word from an 8-token window) is simply a weak task that
  cannot show a readable-vs-normal difference either way.

---

## 6. PENDING WORK (in order)

1. **Finish Step 2** — 5 normal seeds, then mean ± std, paired per-seed difference, and
   a McNemar test on identical val examples (the sensitive test for a real difference).
2. **Step 4** — capacity probe (normal at 1×/3×/7× params), longer window (16 and 24
   tokens), readable at 3× capacity. Script written and ready; **not run**.
3. **Step 5** — tuning parity: equal validation-only tuning budget for BOTH models on T2.
4. **Investigate the dead seed** (init/optimiser).
5. **Optional GPU path** — the engine was refactored so all ops run through a swappable
   array backend (NumPy today, CuPy when installed); the full gradient-check suite still
   passes unchanged on CPU (9/9 checks, errors ~1e-10). The GPU is now authorised by the
   project owner; CuPy installation is pending his approval. Expected speedup is
   **modest (~3-5×, to be measured, not promised)** because the model is tiny and the
   8,001-way softmax is memory-bandwidth bound.

**Rules in force:** TEST SET UNTOUCHED (all T2 decisions on validation only; the
milestone gate requires the owner's approval). One change per experiment. Failed runs
kept. Never call parity a win.

---

## 7. CODE APPENDIX — verify these yourself

### 7a. The flawed train metric (before), in the T2 runner
```python
v = acc_of(net.forward(d["Xva"])["logits"], d["yva"])      # every epoch
if v > best_v: best_v, best_ws = v, {...}                  # val = MAX over epochs
...
tr_acc = acc_of(net.forward(X[:16384])["logits"], y[:16384])   # ONE point estimate
#        ^ X = d["Xtr"]; Xtr[:16384] is the HEAD of the corpus (contiguous slices)
```

### 7b. The contiguous split that creates the unk mismatch (data loader)
```python
text = open(TXT, encoding="utf-8").read()
toks = tokenize(text)
N = len(toks); tr_n = int(N * split[0]); va_n = int(N * split[1])
tr, va, te = toks[:tr_n], toks[tr_n:tr_n + va_n], toks[tr_n + va_n:]
cnt = Counter(tr)                      # <-- vocabulary from TRAIN ONLY
vocab = [w for w, _ in cnt.most_common(vocab_cap)]
V = len(vocab) + 1                     # + <unk> at index V-1
```
Consequence: val contains more out-of-vocabulary words than train, so `<unk>` is 16.32%
of val targets vs 9.72% of train targets. **This is the root cause of the backwards
train/val accuracy.**

### 7c. Corrected measurement (chunked, matched basis)
```python
def eval_chunked(fwd, X, y, chunk=4096):
    n = len(X); correct, ce_sum = 0, 0.0
    for i in range(0, n, chunk):
        xs, ys = X[i:i+chunk], y[i:i+chunk]
        z = fwd(xs).data
        z = z - z.max(axis=1, keepdims=True)
        ex = np.exp(z); p = ex / ex.sum(axis=1, keepdims=True)
        correct += int((z.argmax(axis=1) == ys).sum())
        ce_sum += float(-np.log(p[np.arange(len(ys)), ys] + 1e-12).sum())
    return correct / n * 100, ce_sum / n
```
(Chunking is required: the unchunked full-train forward allocates 182,377 × 8,001 × 4B
= 5.8 GB and stalls the machine.)

### 7d. The cache bug that would have faked Step 4
```python
# BEFORE: a single shared cache path for every window length
CACHE = os.path.join(DATA, "t2_windows.npz")
if os.path.exists(CACHE) and not force:
    return ...      # <-- a 16-token request would get the 8-token cache

# AFTER: per-window cache + hard check
CACHE_TMPL = os.path.join(DATA, "t2_windows_win{win}.npz")
seg = json.loads(str(z["seg"][0]))
if int(seg.get("win", -1)) != int(win):
    raise RuntimeError(f"cache win mismatch: asked {win}, cached {seg.get('win')}")
```

### 7e. Matched-normal control (exactly equal parameter count)
```python
# readable : V*D + (D*W)*K + K + K*V + V
# normal   : V*D + (D*W)*h + h + h*V + V
# both == 724,401 with V=8001, D=64, W=8, K=h=24
assert n_params(d["V"], K, D, WIN) == 724401
```

### 7f. Early stopping, identical for both models
```python
v, _ = eval_chunked(fwd, d["Xva"], d["yva"])
if v > best["val"]:
    best = {"val": v, "epoch": ep+1, "w": {p.name: p.data.copy() for p in net.parameters()}}
    bad = 0
else:
    bad += 1
if bad >= PATIENCE:      # 15
    break
# ... then restore best-val weights for BOTH models
```

---

## 8. WHAT I WANT FROM YOU

1. Is the **unk-share mismatch** the correct diagnosis of the backwards train/val, or am
   I still missing something? (Section 7b is the mechanism.)
2. Is reporting **`excl-<unk>` accuracy** the right matched basis, or should the split
   itself be rebuilt (random windows vs contiguous)?
3. Given the readable is only **+1.5–1.9 pts above a constant `<unk>` predictor**, is it
   worth continuing T2 at all — or should T2 be redefined (smaller vocabulary, longer
   window, different target) before more compute is spent?
4. Any objection to the **Step 4** probes as designed (capacity 1×/3×/7×, window 16/24)?
5. Anything in the Section 4 bug list that makes you distrust the numbers above?
