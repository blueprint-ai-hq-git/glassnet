"""build_review07.py — assemble GLASSNET-REVIEW-07.md with the REAL code embedded.

Every code block is sliced from the file on disk (no retyping), and the file list,
line counts and hashes are generated, so the reviewer can verify that what is shown
is what actually ran.
"""
import hashlib
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "GLASSNET-REVIEW-07.md")


def read(path):
    with open(os.path.join(PROJ, path)) as f:
        return f.read().splitlines()


def block(path, start_marker, end_marker=None, start_off=0):
    """Slice lines from the first line containing start_marker up to (not
    including) the first later line containing end_marker."""
    lines = read(path)
    s = next(i for i, L in enumerate(lines) if start_marker in L) + start_off
    if end_marker is None:
        e = len(lines)
    else:
        e = next(i for i in range(s + 1, len(lines)) if end_marker in lines[i])
    return "\n".join(lines[s:e]).rstrip()


def whole(path):
    return "\n".join(read(path)).rstrip()


def sha(path):
    with open(os.path.join(PROJ, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def nlines(path):
    return len(read(path))


parts = []
A = parts.append

A("""# GLASSNET — REVIEW 07

**Date:** 2026-10-04 (evening) · **Machine:** idle, GPU 0% / 187 MiB / 39 °C
**Status:** Target A **NOT MET** — and the reason is a defect in the audit meter
itself. One decision is needed at the end.
**Test set:** untouched for this task. **Bars:** unchanged all day.
**Every number below comes from a log on disk** — paths are given so they can be checked.

---

## 0. Two-minute summary

The owner approved an auxiliary concept loss (a switch, default OFF) to make the
concept layer actually *trained* to detect things, instead of being named after the
fact. It was implemented, gradient-checked, and run (3 seeds, validation only).

**It raised the audit pass fraction from 30.2% to 71.9% — above the 50% bar — but the
lie probe was MISSED at every setting, where it had been CAUGHT at baseline. Under
the standing rule that is NOT MET.**

Investigating the missed probe found something that matters more than the number:
**the audit meter can be passed by a neuron that has effectively stopped firing.**
A nearly dead column fires on a sliver of windows, the namer then describes exactly
those windows, and the resulting AUROC is ~1.0 *by construction*. Measured three
independent ways:

1. On the frozen lie-probe column: it fires on **1.15%** of validation windows and its
   name's support is **0.0115 — the same number**.
2. Across all passing rows: the name's support tracks the column's own firing rate
   with correlation **0.24 at baseline vs 0.68–0.82 under supervision**.
3. Projecting a firing floor onto the existing checkpoints: **supervised columns fall
   from 93.8% to 31.2% at a 2% floor**, i.e. below the unsupervised 47.9%.

**So supervision's apparent win was mostly dead neurons.** The one genuine,
non-circular gain is the *unsupervised* half: 30.2% → 50.0% (47.9% under a floor),
with broader support (median 16.4% vs 7.6%). Real, modest, and not enough for the bar.

The margin half of Target A passes throughout: the readable is no worse than the
matched normal on accuracy and perplexity at all three settings.

**Nothing is running. No rule has been changed. The decision is at the end (section 9).**

---

## 1. What came before (so this file stands alone)

Earlier the same day (see REVIEW 06 for the long version): the audit was stuck at
~31% against the fixed 50% bar. Four single-change ideas were tried and **all four
were NOT CONFIRMED**:

| idea | result |
|---|---|
| position-aware naming (fixed a real namer bug) | 22.9% → 30.2% (K=32) |
| support-gate sweep 5/10/20% | all worse (16.7 / 19.8 / 19.8%) |
| activation sparsity lam_act 0.5 / 1.0 | worse (14.6% / 5.2%); lam_act=1.0 also fails the margin half |
| weight sparsity lam_w 2.0 / 5.0 | best 31.2%; 5.0 worse |

Diagnosis recorded then: *nothing in the loss encouraged a concept to detect
anything* — concepts were arbitrary linear features with post-hoc names. That is
exactly what the auxiliary loss was approved to fix.

---

## 2. What the owner approved (verbatim, and how each condition was met)

> "APPROVED: add an auxiliary concept loss as a switch (default OFF, old model kept
> intact). Conditions:
> 1. Concept targets must be derived automatically from TRAIN text only ... pre-registered in a file BEFORE any run. No val/test text in targets.
> 2. Pre-register lam_concept grid (0.1, 0.5, 1.0) and use the frozen SELECTION-RULE.md (validation perplexity, tie-break accuracy). 3 seeds.
> 3. Audit bars unchanged: AUROC 0.55, contribution 0.25, 50% pass, lie probe caught. Because supervised concepts make AUROC easier, ALSO report contribution alone and the pass fraction on concepts that were not directly supervised as a check on circularity.
> 4. Always report the unsupervised baseline and the position-aware naming fix, before and after, labeled.
> 5. Accuracy/perplexity must stay no worse than the matched normal (Target A's second half). If supervision hurts it, report it plainly.
> 6. One change at a time, predictions written first, test set untouched, same stop conditions. If Target A is met, do NOT claim a win; go to Target B first."

| condition | how it was met |
|---|---|
| 1 | `scripts/t3_concept_targets.py` reads the **TRAIN** split only; froze the top 16 non-stopword TRAIN tokens (`And` 7.05% → `What` 1.49%) with per-column `pos_weight` in `CONCEPT-TARGETS.md` before any run |
| 2 | grid {0.1, 0.5, 1.0} × lr {0.5, 1.0}, frozen joint rule applied by `t3_lamc_pick.py`, 3 seeds |
| 3 | bars untouched; **supervised vs unsupervised split reported** (section 4.2) plus contribution medians for both |
| 4 | baseline (lam_concept=0) and the position-aware namer reported, labeled, before and after |
| 5 | margin table in section 4.3 — passes everywhere |
| 6 | one change; predictions pre-registered in `LOG.md` before the run; test set untouched |

---

## 3. What actually ran

* **One change:** `lam_concept` (auxiliary loss weight). Everything else at baseline:
  K=32, lam_act 0.2, lam_w 1.0, position-aware namer, support 1%.
* **Loss:** `loss = ce + lam_act*mean(h) + lam_w*mean(|W|) + lam_concept * balanced_BCE(z_supervised, y)`
  where `z` is the concept pre-activation, `y` is "this window contains target word k",
  and the BCE is class-balanced with `pos_weight = (1-p)/p` from that column's TRAIN
  base rate. `lam_concept = 0` (default) never builds the term — the old model is
  bit-for-bit unchanged.
* **Only columns 0..15 receive the auxiliary gradient.** Columns 16..31 stay
  unsupervised — that is the owner-required circularity control, and section 6.4
  proves the gradient on them is *exactly* zero.
* **Budget identical for both models**; 3 seeds at the selected lr; validation only.

### 3.1 Before trusting it: the gradient check (and a lesson)

The project rule is "check every gradient with a numerical gradient test before
trusting it". Doing that caught **two bugs**, one in the code and one in the check:

* **Code bug:** `ysup = np.isin(Xb, ids)` returns an array shaped like the *window*
  (N,W) — "which tokens are targets" — not the (N,C) target matrix. It crashed with
  `shapes (6,8) (6,16)`. Fixed with a broadcast comparison plus a shape assertion.
* **Check bug (worth knowing):** the engine's Tensor constructor casts to float32, so
  the loss *value* is float32 even with float64 inputs. Central-differencing at
  eps=1e-6 therefore measures float32 quantisation (~1.19e-7), and reported a bogus
  relative error of 1.0. Swept the step: 1e-6/1e-5 fail, 1e-4/1e-3/1e-2 pass. The
  analytic gradient matches the hand-derived `(s-t)*w/size` to **1.4e-17**.

Final: all checks pass in both switch positions, including *exactly zero* gradient on
the unsupervised columns (0.0e+00).

---

## 4. Results (all from logs)

### 4.1 Accuracy / perplexity vs the matched normal (owner condition 5)

```
lam_concept  lr    readable acc      readable ppl      normal acc       normal ppl     dacc     dppl
0.1          1.0   30.73±0.15        63.67±2.71        30.24±0.11       68.50±3.78    +0.49    +4.83
0.5          1.0   30.66±0.09        62.95±1.90        30.26±0.11       68.45±3.96    +0.40    +5.51
1.0          0.5   30.70±0.10        62.99±3.06        30.42±0.16       67.42±5.27    +0.28    +4.44

margin clause (readable no worse than normal on BOTH):
0.1  +0.49 vs 2*sd 0.372   +4.83 vs 9.309   OK
0.5  +0.40 vs 2*sd 0.283   +5.51 vs 8.792   OK
1.0  +0.28 vs 2*sd 0.383   +4.44 vs 12.188  OK
```

### 4.2 The audit, with the circularity split (owner condition 3)

```
lam_concept  lr    pass/K    frac    SUP pass   UNSUP pass   sup c-med   uns c-med   lie
0.1          1.0   19.7/32   61.5%     93.8%      29.2%      +11.197     +0.570     MISSED
0.5          1.0   22.0/32   68.8%     93.8%      43.8%      +14.718     +0.529     MISSED
1.0          0.5   23.0/32   71.9%     93.8%      50.0%      +16.017     +0.567     MISSED
0 (baseline) 1.0    9.7/32   30.2%       —          —        +0.318       —         CAUGHT

AUROC clause alone: SUP 45/45 at every setting; UNSUP 17/48, 22/48, 25/48
```

**Verdict: TARGET A = NOT MET.** The pass fraction clears 50%, the lie probe does
not. Under "never change a gate's pass rule after seeing the result", the probe
failure decides it. The 71.9% is carried by columns trained to detect exactly the
word the audit then asks about — the circularity condition 3 exists to expose.

---

## 5. THE FINDING — the meter can be passed by a dead neuron

### 5.1 The frozen lie-probe column, measured

The probe freezes concept column 2 (gradients zeroed; weights stay at init). The
freeze was verified in **both** training loops, so its weights genuinely never move.

```
BASELINE (lam_concept=0)   frozen col 2, name 'And_I_The_DUKE_To_What' (6 words)
  auroc 0.5059  contrib +0.0261  flag BROKEN   -> CAUGHT
  firing: mean 0.4545  frac>0.5 0.3000  frac>0.9 0.0000  frac<0.1 0.0000
  emb norm 84.734

SUPERVISED (lam_concept=1.0, lr=0.5)  frozen col 2, name 'The_gods_people_o'' (4 words)
  auroc 1.0000  contrib +8.1065  flag -        -> MISSED
  firing: mean 0.0188  frac>0.5 0.0115  frac>0.9 0.0095  frac<0.1 0.9852
  emb norm 121.608        pos_frac 0.0115
```

The auxiliary loss inflates the embedding norm (84.7 → 121.6), which drives the
frozen column's fixed random projection into saturation: **98.5% dead**, firing on
117 of 10,124 windows. The namer then picks a word-set present in exactly those
windows — `pos_frac` is *identical* to `frac>0.5` — so AUROC is 1.0 by construction.

### 5.2 What a stricter meter would do (measured on the existing checkpoints)

No rule changed; only the recorded audit values re-applied with an added firing floor,
columns matched by index.

```
                                        pass/K    frac     SUP     UNSUP
lc=1.0  lr=0.5  floor 0% (today)        23.0/32   71.9%   93.8%    50.0%
                floor 2%                12.7/32   39.6%   31.2%    47.9%
                floor 5%                 8.3/32   26.0%    6.2%    45.8%
                floor 10%                6.7/32   20.8%    0.0%    41.7%

lc=0.5  lr=1.0  floor 0%                22.0/32   68.8%   93.8%    43.8%
                floor 2%                12.0/32   37.5%   31.2%    43.8%
lc=0.1  lr=1.0  floor 0%                19.7/32   61.5%   93.8%    29.2%
                floor 2%                 9.3/32   29.2%   31.2%    27.1%

baseline lc=0   floor 0%                 9.3/32   29.2%
                floor 2%                 9.3/32   29.2%   (barely moves)
```

**The supervised columns' 93.8% is almost entirely dead neurons** — at a 2% floor
they fall to 31.2%, *below* the unsupervised 47.9%; at 5% they nearly vanish (6.2%)
while the unsupervised hold at 45.8%. The baseline barely moves, confirming its
passes were broad and genuine.

Support among passing rows corroborates it: baseline median **29.0%** (0/28 below 2%);
supervised median **7.6%** (3/45 below 2%); unsupervised median **16.4%** (0/24).

### 5.3 General confirmation — the name tracks the column's own firings

All passing rows, audit values from the recorded JSONs and firing rates from the
saved checkpoints (no retraining):

```
case                  n pass   corr(pos_frac, firing)   med|diff|   max|diff|
baseline lc=0            28              0.242            0.1262      0.3391
lc=0.1                   59              0.823            0.0578      0.2366
lc=0.5                   66              0.742            0.0652      0.2953
lc=1.0                   69              0.685            0.0593      0.4045

sparsest passing columns (pos_frac < 5%):
baseline lc=0    1 row   name support 0.0499   firing 0.1703   <- name NARROWER than firing
lc=0.1          11 rows  name support 0.0300   firing 0.0087
lc=0.5          12 rows  name support 0.0319   firing 0.0115
lc=1.0          16 rows  name support 0.0328   firing 0.0140
```

At baseline a name describes something much broader than the column's firing
(0.24). Under supervision the name tracks the firing pattern (0.68–0.82) and the
sparse passing columns fire on ~1% of windows. So the meter's AUROC on those rows is
largely a rare-firing coincidence rather than evidence of detection — the same hole
the probe fell through, operating across the whole supervised result.

### 5.4 Why this is not just "the probe was unfair"

Three readings are possible, and the measurements separate them:

* *"The probe is invalid because the frozen column reads a supervised representation."*
  Partly true — but that is exactly why the probe exists: it must catch a column that
  passes without detecting anything. It failed to.
* *"The meter is fine and the model is genuinely readable."* Refuted: the supervised
  half collapses from 93.8% to 31.2% under a 2% firing floor, i.e. below the
  unsupervised half.
* *"The meter has a hole."* Supported by all three measurements. A neuron firing on
  ~1% of windows can be named to match those windows and score AUROC ~1.0.

The honest consequence either way: **even with a firing floor the pass fraction is
39.6%, still under the 50% bar.** Tightening the meter does not manufacture a win; it
stops a false one. And the *unsupervised* rise (30.2% → 50.0%, 47.9% under a floor)
is real, non-circular, and worth keeping.

---

## 6. The code (as it actually is on disk)

Every block below is sliced from the file at the given path; hashes and line counts
are in section 10 so the reader can confirm the text matches what ran.

### 6.1 The autograd engine — `glassnet/tensor.py`
""")

A("```python\n" + whole("glassnet/tensor.py") + "\n```\n")
A(f"*{nlines('glassnet/tensor.py')} lines, sha256[:16] `{sha('glassnet/tensor.py')}`*\n")

A("### 6.2 Losses — `glassnet/losses.py` (full)\n")
A("```python\n" + whole("glassnet/losses.py") + "\n```\n")
A(f"*{nlines('glassnet/losses.py')} lines, sha256[:16] `{sha('glassnet/losses.py')}`*\n")

A("""### 6.3 The model, the auxiliary loss, the namers and the meter — `scripts/run_t2_iter003.py`

**Targets loader + the auxiliary switch (lines ~44-86):**
""")
A("```python\n" + block("scripts/run_t2_iter003.py", "LAM_A = float(", "def _mean(x):") + "\n```\n")

A("**`TinyLM3`: the readable model, `freeze()`, `forward()`, and the loss with the switch:**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "class TinyLM3:", "def _flatten(E):") + "\n```\n")

A("**The namers — `auto_name_t2c` (the original) and `auto_name_t2c_pa` (position-aware fix):**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "def auto_name_t2c(net, d, Xtr", "def _auroc(") + "\n```\n")

A("**The audit meter itself — `_auroc` and `audit_t2c` (this is where the hole lives):**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "def _auroc(", "def main():") + "\n```\n")

A("""### 6.4 The audit driver and the lie probe — `scripts/t3_audit.py`

**Backend/checkpoint handling, `to_cpu`, `eval_val`, and `train_readable` (note the freeze at lines 140-143):**
""")
A("```python\n" + block("scripts/t3_audit.py", "def log(msg):", "def audit_one(") + "\n```\n")

A("**`main()` — the audit loop, the **lie probe** (frozen column 2), and the circularity split:**\n")
A("```python\n" + block("scripts/t3_audit.py", "def main():") + "\n```\n")
A(f"*{nlines('scripts/t3_audit.py')} lines, sha256[:16] `{sha('scripts/t3_audit.py')}`*\n")

A("### 6.5 The gradient check — `scripts/t3_aux_gradcheck.py` (full)\n")
A("```python\n" + whole("scripts/t3_aux_gradcheck.py") + "\n```\n")

A("### 6.6 The lie-probe diagnostic — `scripts/diag_lie_probe.py` (full)\n")
A("```python\n" + whole("scripts/diag_lie_probe.py") + "\n```\n")

A("### 6.7 The firing-floor projection — `scripts/t3_meter_floor.py` (full)\n")
A("```python\n" + whole("scripts/t3_meter_floor.py") + "\n```\n")

A("### 6.8 The name-tracks-firing confirmation — `scripts/t3_name_matches_firing.py` (full)\n")
A("```python\n" + whole("scripts/t3_name_matches_firing.py") + "\n```\n")

A("### 6.9 The frozen selection rule, applied — `scripts/t3_lamc_pick.py` (full)\n")
A("```python\n" + whole("scripts/t3_lamc_pick.py") + "\n```\n")

A("### 6.10 The run driver — `scripts/t3_lamc_resume.sh` (full)\n")
A("```bash\n" + whole("scripts/t3_lamc_resume.sh") + "\n```\n")

A("""---

## 7. My own mistakes today (four, all logged, all fixed)

Listed because a review file that only shows clean work is not honest.

| # | mistake | how it showed | fix |
|---|---|---|---|
| 1 | **gradient CHECK was invalid** — central-differenced a float32-valued loss at eps=1e-6, measuring quantisation | bogus relative error 1.0 while the gradient was exact | eps=1e-2, 1e-2 tolerance; documented |
| 2 | **wrong interpreter** — pointed the sweep at a venv without cupy | every block died instantly; script still printed "done" | correct venv + preflight `import cupy` + rc checks that abort |
| 3 | **picker crashed** — read `lam_concept` from a record that never had the field | selection stayed empty; confirm + audit silently skipped | record now carries it; picker reads the filename tag; resume aborts if selection is short |
| 4 | **bad import** — `to_cpu` is defined in `t3_audit.py`, not `tensor.py` | ImportError in the diagnostic | use `A.to_cpu` |

The shared lesson of 2 and 3: **a script that can fail silently must not be trusted.**
Both produced a "done" line after doing nothing. Any phase whose output feeds a later
phase must now assert that its output exists and is non-empty.

Also fixed earlier and relevant here: an **audit filename collision** (a new knob not
in the artifact name → files overwritten) and a **lie-probe hole** (`nan < 0.55` is
False, so a NO-NAME frozen neuron scored as MISSED; the test is now
`caught = flag != "-"`, which is strictly stricter).

---

## 8. What was NOT done

* **No bar was changed.** AUROC 0.55, contribution 0.25, 50% pass, lie probe caught —
  identical all day, including after seeing the results.
* **The test set was not touched** for this task.
* **No rule was changed.** The meter is exactly as it was; the firing floor in
  section 5.2 is a *measurement on existing checkpoints*, not an applied change.
* **Target B was not started.** The owner's rule: if A is met, go to B before
  claiming anything. A is not met, so B is untouched.
* **Target F (scaling ladder) was not started** — the owner's order is A, B, F, D, E.

---

## 9. The decision needed

The audit meter, as it stands, lets a nearly dead column pass. Closing that is a
**rule change**, so it is the owner's call, not mine.

**A. Add a minimum firing rate to the meter** (e.g. a column must fire on ≥2% of
validation windows for its AUROC to count). Makes the meter *stricter*; closes the
hole the probe fell through; would require re-auditing earlier results; lowers every
pass fraction. Honest consequence: the pass fraction becomes **39.6%** — still under
the bar, so this does not create a win, it removes a false one.

**B. Keep the meter as it is** and accept the lie-probe failure as the verdict
(Target A NOT MET), accepting that a dead column can pass.

**C. Fix the lie probe only** (e.g. give the frozen column a live bias so it cannot
saturate). Narrower — but it leaves the hole open for real columns, and section 5.3
shows real columns are already using it.

**D. Something else the owner prefers.**

### What I would recommend, with the reason

**A**, because the three measurements show the hole is not confined to the probe: it
is inflating the headline number itself. A floor makes the meter measure detection
instead of a rare-firing coincidence, and it costs nothing in honesty — it lowers the
score. If A is chosen, the whole supervised line should be re-audited under the floor
before any further claim is made.

---

## 10. Files, sizes and hashes (so the text above can be verified)

""")

FILES = [
    "glassnet/tensor.py", "glassnet/losses.py",
    "scripts/run_t2_iter003.py", "scripts/t3_audit.py",
    "scripts/t3_aux_gradcheck.py", "scripts/diag_lie_probe.py",
    "scripts/t3_meter_floor.py", "scripts/t3_name_matches_firing.py",
    "scripts/t3_lamc_pick.py", "scripts/t3_lamc_resume.sh",
    "scripts/t3_lamc_sweep.sh", "scripts/t3_concept_targets.py",
    "CONCEPT-TARGETS.md", "SELECTION-RULE.md", "LOG.md", "STATUS.md", "STOPPED.md",
]
A("| file | lines | sha256[:16] |")
A("|---|---|---|")
for f in FILES:
    p = os.path.join(PROJ, f)
    if os.path.exists(p):
        A(f"| `{f}` | {nlines(f)} | `{sha(f)}` |")
    else:
        A(f"| `{f}` | (missing) | |")

A("""
### Where the numbers live

| number | file |
|---|---|
| audit tables, circularity split | `runs/t3/audit.log`, `runs/t3/audit_K32_lr*_pa_lc*.json` |
| accuracy / perplexity per seed | `runs/t3/models_K32_lc*_lr*_sel.json` |
| frozen rule application | `runs/t3/lamc_selection.json` |
| sweep + resume progress, all rc values | `runs/t3/lamc_sweep.log`, `runs/t3/lamc_resume.log` |
| lie-probe measurement | `runs/t3/diag_lie.log` |
| floor projection | re-runnable: `scripts/t3_meter_floor.py` |
| name-tracks-firing | re-runnable: `scripts/t3_name_matches_firing.py` |
| pre-registration, every result, all four mistakes | `LOG.md` |
| the decision options | `STOPPED.md` stop #3 |

---

*Prepared by the working session. Every figure here was produced by a script on this
machine and is reproducible from the files listed above. Where a claim is a
hypothesis rather than a measurement, it is labelled as one.*
""")

with open(OUT, "w") as f:
    f.write("\n".join(parts) + "\n")

print(f"wrote {OUT}")
print(f"  {nlines('GLASSNET-REVIEW-07.md')} lines")
print(f"  sha256[:16] {sha('GLASSNET-REVIEW-07.md')}")
