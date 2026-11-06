"""build_review08.py — assemble GLASSNET-REVIEW-08.md with the REAL code embedded.

Every code block is sliced from the file on disk (no retyping); file list, line
counts and hashes are generated so the reviewer can verify that what is shown is
what actually ran.
"""
import hashlib
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
OUT = os.path.join(PROJ, "GLASSNET-REVIEW-08.md")


def read(path):
    with open(os.path.join(PROJ, path)) as f:
        return f.read().splitlines()


def whole(path):
    return "\n".join(read(path)).rstrip()


def block(path, start_marker, end_marker=None):
    lines = read(path)
    s = next(i for i, L in enumerate(lines) if start_marker in L)
    e = (next(i for i in range(s + 1, len(lines)) if end_marker in lines[i])
         if end_marker else len(lines))
    return "\n".join(lines[s:e]).rstrip()


def sha(path):
    with open(os.path.join(PROJ, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def nl(path):
    return len(read(path))


parts = []
A = parts.append

A("""# GLASSNET — REVIEW 08

**Date:** 2026-10-04 (evening) · **Machine:** idle, GPU 0% / 188 MiB / 39 °C
**Status:** Target A **NOT MET** — the audit half now CLEARS the 50% bar on a genuine
non-circular gain, but the **lie probe fails**, and the reason is a circularity in the
audit's design. One decision is needed at the end.
**Test set:** untouched. **Bars:** unchanged all day.
**Every number comes from a log on disk** — paths given so it can be checked.

---

## 0. Two-minute summary

Two owner-approved changes were made and verified today, then applied to every earlier
result. Both worked. Together they showed that **the project's central measurement does
not measure what it claims to**.

1. **Meter-02 — a firing-rate floor.** A nearly dead concept column could be named to
   match its own few firings and score AUROC ~1.0 by construction. The meter now also
   requires the column to fire on ≥2% of validation windows. Verified as a pure
   re-score: with the floor off, it reproduces every recorded pass count **and every
   per-column flag** exactly.
2. **A LIVE lie probe** — the probe column now trains normally until it fires at 2%,
   and only then are its gradients frozen, so it cannot pass merely by being dead.

**What they found:**

* The floor removed **31–32 points** of false passes from the supervised runs (71.9% →
  39.6%) and caught dead neurons in the rungs (Banking77: 3/16 → **0/16**, with
   essentially all 16 neurons firing on **0.0000** of validation windows).
* **The live lie probe is MISSED.** A column frozen after **one epoch** stays live
  (fires 8.5%) and passes with **AUROC 0.80**. Under supervision the column still dies,
  so the floor catches it — reported as INCONCLUSIVE, not a pass.
* **Why:** the namer derives each name **from that column's own firings**, and the
  answer layer learns to use **whatever columns exist**. So any live column gets a name
  that matches its firings and a real contribution. **The meter asks "does the name
  describe the firings?" — which the namer guarantees — not "was this concept meant to
  detect anything?"**
* Then the approved cluster-target run **cleared the 50% bar for the first time
  (54.2%)**, with the honest non-circular half rising 29.2% → **45.8%**, and the floor
  no longer biting at all. Real progress — measured by a meter a frozen column also
  passes. Under the standing rule, **NOT MET**.

Also fixed: an OOM in the re-audit, root-caused to **reference cycles in the autograd
graph** (a full-TRAIN forward cost 2.7 GB; chunked + explicit GC → 348 MB).

---

## 1. Meter-02 — the firing-rate floor

**The hole it closes.** The lie probe froze concept column 2; its weights never moved
(freeze verified in both training loops), yet its AUROC went 0.5057 → 1.0000. Measured
cause:

```
BASELINE   frozen col 2  auroc 0.5059  firing mean 0.4545  frac>0.5 0.3000  emb norm  84.7
SUPERVISED frozen col 2  auroc 1.0000  firing mean 0.0188  frac>0.5 0.0115  emb norm 121.6
                         pos_frac 0.0115  ==  frac>0.5 0.0115   (identical)
```

The auxiliary loss inflates the embedding norm, driving the frozen column's fixed
random projection into saturation: **98.5% dead**, firing on 117 of 10,124 windows. The
namer then picks a word-set present in *exactly* those windows, so AUROC is 1.0 **by
construction**.

**The rule** (`METER-RULE-02.md`, pre-registered before any re-audit): a column passes
only if length-residualised AUROC ≥ 0.55 **AND** contribution ≥ 0.25 **AND** firing
rate ≥ 2% of validation windows (activation > 0.5). 1% and 5% reported as sensitivity.

**The honest cost, recorded in advance:** 8 of the 16 single-word targets have TRAIN
support below 2% (`What` 1.49%, `As` 1.51%, `more` 1.55%, `I'll` 1.66%, `they` 1.74%,
`My` 1.76%, `For` 1.83%, `good` 1.98%). A concept that is a *perfect, precise* detector
of `What` — firing on exactly those windows, AUROC 1.0 — is **excluded** by this floor.

---

## 2. The re-audit — every earlier result, old vs new

`min_firing = 0` reproduces every recorded Meter-01 count and flag exactly (23/32,
22/32, 19.7/32, 9.3/32 all MATCH), so this is a pure re-score of the same checkpoints.

### T2 concept audits

```
result                        Meter-01          2% (rule)        change
baseline (unsupervised)        9.3/32 = 29.2%    9.3/32 = 29.2%   none
supervised lc=0.1             19.7/32 = 61.5%    9.3/32 = 29.2%   -32.3 pts
supervised lc=0.5             22.0/32 = 68.8%   12.0/32 = 37.5%   -31.3 pts
supervised lc=1.0             23.0/32 = 71.9%   12.7/32 = 39.6%   -32.3 pts
K=8                            0.7/8  =  8.8%    0.3/8  =  3.8%   -5.0 pts
K=16                           2.3/16 = 14.4%    2.3/16 = 14.4%   none
K=24                           7.0/24 = 29.2%    6.7/24 = 27.9%   -1.3 pts
K=32                           9.7/32 = 30.3%    9.7/32 = 30.3%   none
```

### Rungs 1 / 2 / 3

```
Fashion-MNIST (rung 1)  13/14 = 92.9%  ->  12/14 = 85.7%  border_ink (fid 0.769, fires 0.0004)
PathMNIST    (rung 2)    8/8  = 100%   ->   8/8  = 100%   none (recorded values reproduced EXACTLY)
Banking77    (rung 3)    3/16 = 18.8%  ->   0/16 =  0.0%  3 neurons, all firing 0.0000
```

Banking77 across all three surviving seeds — the finding is robust, not seed luck:

```
seed  pass(old)  pass(2%)  neurons firing on 0.0000 of validation windows
  0     3/16       0/16    all 16   (fidelity up to 0.623)
  1     3/16       1/16    14 of 16 (fidelity up to 0.789)
  2     4/16       0/16    all 16   (fidelity up to 0.783)
```

**Claims this changes (flagged, not buried):**

* **"T2 supervised: 71.9%, above the bar" is OVERTURNED** — 39.6% under the rule.
* **Rung 3's verdict (FAILED) stands, but its description was wrong** — essentially all
  16 neurons are dead, so its per-neuron fidelity numbers measured nothing. "12 of 16
  broken" was right for the wrong reason.
* **Fashion-MNIST is NOT like-for-like**: its recorded audit used the held-out TEST
  split while this re-score is on validation. Not presented as comparable; no test touch
  was made.
* **PathMNIST reproduces exactly** (all 8 fidelity values identical), so "no change"
  there is a real result.

---

## 3. The live lie probe — MISSED, and why it matters more than the numbers

```
SUPERVISED (lc=1.0): never reached 2% firing while unfrozen -> INCONCLUSIVE
                     at audit: auroc 0.634, contrib +16.51, firing 0.0115
                     flag BROKEN -> caught_by_floor_only = True
BASELINE  (lc=0):    reached 4.32% at epoch 1 -> frozen, STAYED LIVE
                     at audit: firing 0.1552, auroc 0.5591, contrib +0.4449
                     flag "-" -> NOT CAUGHT (MISSED)
CLUSTERS  (lc=0.5):  frozen at epoch 1, live at audit (firing 0.0850)
                     auroc 0.8000, contrib +15.52 -> NOT CAUGHT (MISSED)
```

Both pre-registered predictions about the probe were **NOT CONFIRMED**.

**The mechanism.** The namer derives a concept's name **from that column's own
firings**; the answer layer learns to use **whatever columns exist**. So any live column
acquires a name that matches its firings (AUROC clears 0.55 by construction) and a real
contribution (clears 0.25). The meter therefore answers *"does the name describe the
firings?"* — which the namer guarantees — not *"was this concept meant to detect
anything?"*. **A live lie probe essentially cannot fail under this design.** The floor
removed one class of false passes (dead columns); it does not touch this one.

---

## 4. Cluster targets (owner condition 4) — the bar is met, honestly

Method pre-registered in `CONCEPT-TARGETS-02.md` **before** computing anything:
TRAIN-only; non-STOP words, length ≥ 2, document frequency ≥ 1%; top 100; pairwise
window co-occurrence; PPMI with negatives clipped; rows L1-normalised; k-means k=16,
fixed seed 0, 100 iterations, k-means++; target = "window contains ≥1 member"; clusters
ordered by TRAIN support. numpy only — no sklearn, nothing downloaded.

```
run                          pass/K    frac     SUP     UNSUP
v1 single words, lc=1.0     12.7/32   39.6%   31.2%    47.9%
unsupervised baseline lc=0   9.3/32   29.2%      —        —
v2 clusters, lc=0.1         16.3/32   51.0%   62.5%    39.6%
v2 clusters, lc=0.5         17.3/32   54.2%   62.5%    45.8%   <- bar MET
v2 clusters, lc=1.0         16.7/32   52.1%   62.5%    41.7%
```

* **The floor no longer bites**: 0% and 2% give IDENTICAL counts. Cluster concepts fire
  broadly, so they pass on merit rather than being rescued — against v1 where the
  supervised columns collapsed 93.8% → 31.2% under the same floor.
* **The unsupervised (non-circular) half rose to 45.8%** from the baseline's 29.2% —
  the highest honest number so far.
* Margin clause (readable no worse than the matched normal on accuracy AND perplexity)
  **passes** at all three settings.
* Honest caveat: only 37 words met the frozen 1% df threshold, so **9/16 clusters are
  multi-member and 7 are singletons**.
* **Predictions:** #1 CONFIRMED (13/16 clear the floor vs 8/16). #2 NOT CONFIRMED
  (supervised 62.5%, not ≥90%). **#3 NOT CONFIRMED — I predicted the unsupervised half
  would land at 25–40% and it reached 45.8%, i.e. I was wrong in the favourable
  direction.** #4 CONFIRMED (bar not met on the unsupervised half).

---

## 5. The engine bug found on the way — reference cycles in the autograd graph

The re-audit died twice with no traceback. The kernel log, not guesswork:

```
Memory cgroup out of memory: Killed process (python3) anon-rss:4179860kB
oom_memcg=.../hermes-worker-proc_0676add15298.scope
```

Measured in a FRESH process per mode (the allocator keeps freed memory resident, so
measuring both in one process is meaningless):

```
full-set forward (old namer):      peak 2758 MB   (+2712 MB)
chunked 20k at a time:             grows ~277 MB PER CHUNK, never released
chunked + gc.collect() per chunk:  69 -> 73 -> 79 -> 83 -> 89 MB   (flat)
```

**Cause:** every op's `_backward` closure closes over its own output, so
`out → _backward → out` is a **reference cycle** that refcounting cannot free and the
cyclic collector reclaims only lazily — far too rarely for the memory each cycle holds.
Chunking alone did nothing (2556 vs 2712 MB); the explicit collection is the fix.

**Why training never noticed:** batch=128 arrays are small enough for the GC to keep
pace. A 182,377-window forward is not. **Latent bug in the engine, not a namer quirk.**

Fixed memory-only, numbers verified unchanged (same 32 names; the regression still
reproduces every recorded count and flag; gradient check ALL PASSED). **The proper fix
— have each op's backward take the output gradient as an argument instead of closing
over `out` — is logged as open**, because it touches every op in the verified engine and
the owner's rule is one change at a time.

---

## 6. The code (as it actually is on disk)

### 6.1 The meter with the firing floor — `scripts/run_t2_iter003.py`

**The meter (`_auroc` + `audit_t2c`). This is where the hole was, and the floor now lives:**
""")
A("```python\n" + block("scripts/run_t2_iter003.py", "def _auroc(", "def main():") + "\n```\n")

A("**The namer memory fix (`_fired_matrix`) — chunked + explicit GC:**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "def _fired_matrix(", "def auto_name_t2c(") + "\n```\n")

A("**The loss with the switch, supporting BOTH single-word and cluster targets:**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "class TinyLM3:", "def _flatten(E):") + "\n```\n")

A("**Target loading for both sets:**\n")
A("```python\n" + block("scripts/run_t2_iter003.py", "def _load_concept_targets():", "CONCEPT_TARGETS = ") + "\n```\n")

A("### 6.2 The live lie probe and the firing-rate measurement — `scripts/t3_audit.py`\n")
A("```python\n" + block("scripts/t3_audit.py", "def firing_rate(", "def audit_one(") + "\n```\n")
A("```python\n" + block("scripts/t3_audit.py", "    # ---- lie probe: a LIVE column", "    # ---- summary ----") + "\n```\n")

A("### 6.3 The cluster derivation (frozen method) — `scripts/t3_concept_clusters.py` (full)\n")
A("```python\n" + whole("scripts/t3_concept_clusters.py") + "\n```\n")

A("### 6.4 The implementation check — `scripts/t3_meter02_check.py` (full)\n")
A("```python\n" + whole("scripts/t3_meter02_check.py") + "\n```\n")

A("### 6.5 The memory measurement — `scripts/mem_namer.py` (full)\n")
A("```python\n" + whole("scripts/mem_namer.py") + "\n```\n")

A("### 6.6 The Banking77 three-seed robustness check — `scripts/t3_banking77_seeds.py` (full)\n")
A("```python\n" + whole("scripts/t3_banking77_seeds.py") + "\n```\n")

A("""---

## 7. My own mistakes today (all logged, all fixed)

| # | mistake | how it showed | fix |
|---|---|---|---|
| 1 | gradient CHECK invalid (float32 loss differenced at eps=1e-6) | bogus rel err 1.0 on an exact gradient | eps=1e-2; documented |
| 2 | wrong interpreter (venv without cupy) | every block died; script still printed "done" | correct venv + preflight + rc guards |
| 3 | picker read a field the record never had | selection empty; confirm+audit silently skipped | read from filename tag; abort if short |
| 4 | `to_cpu` imported from the wrong module | ImportError | use `A.to_cpu` |
| 5 | namer full-set forward leaked (autograd cycles) | OOM kill at 4.18 GB, twice | chunked + `gc.collect()`; measured |
| 6 | rung name parser swept up TWO tables (28 vs 14) | loader broadcast error | stop at the next heading + K guard |
| 7 | audit/checkpoint names omitted the concept set | would load v1 checkpoints AND overwrite v1 audits | concept set in both names; verified |
| 8 | report printed UNSUP before SUP against its header | misleading table | fixed and re-run |
| 9 | a patch joined two lines and broke syntax | SyntaxError | linter caught it; fixed same turn |

Lessons 2, 3, 5, 6, 7 share one shape: **a step failed and the surrounding script
carried on as if it had succeeded.** Every phase whose output feeds a later phase now
asserts that its output exists and is non-empty.

---

## 8. What was NOT done

* **No bar was changed.** AUROC 0.55, contribution 0.25, 50% pass, lie probe caught —
  identical all day, including after seeing results.
* **The test set was not touched.**
* **No model was retrained or re-tuned to satisfy the new meter** — the re-audit is a
  re-score, proven by the floor=0 reproduction.
* **Target B was not started** (owner's rule: it comes after A; A is not met).
* **The engine's reference cycles were not refactored** — logged as open, one change at
  a time.

---

## 9. The decision needed

The pass fraction now clears the bar, and the gain is real, but the meter cannot tell a
trained concept from a frozen one. Closing that is a **design change**, so it is the
owner's call.

**A. Audit supervised columns against their PRE-REGISTERED target** (the frozen word or
cluster they were trained on) instead of a name the namer derived from the column's own
firings. A frozen, never-supervised probe is then tested against a target it was never
trained on, so it can genuinely fail. Composes directly with the cluster work.

**B. Fix the names before training** — pre-register every concept's name and train the
layer to detect them, so the namer can no longer define a name from the firings.

**C. Leave the meter as it is**, accept that a live frozen column passes, and treat the
50% crossing as provisional.

**D. Something else the owner prefers.**

### What I would recommend

**A**, because it is the minimal change that makes the lie probe meaningful, it uses the
pre-registered targets that already exist, and it keeps the unsupervised columns as the
circularity control. It also means the 45.8% non-circular figure can be compared against
a supervised figure that is no longer inflated by construction.

---

## 10. Files, sizes and hashes

""")

FILES = [
    "scripts/run_t2_iter003.py", "scripts/t3_audit.py",
    "scripts/t3_concept_clusters.py", "scripts/t3_meter02_check.py",
    "scripts/mem_namer.py", "scripts/t3_banking77_seeds.py",
    "scripts/t3_reaudit_t2.py", "scripts/t3_reaudit_rungs.py",
    "scripts/t3_lamc_pick.py", "scripts/t3_clu_pick.py",
    "scripts/t3_lamc_sweep.sh", "scripts/t3_clu_sweep.sh",
    "scripts/live_probe_run.sh", "scripts/reaudit_all.sh",
    "METER-RULE-02.md", "REAUDIT-METER-02.md", "CONCEPT-TARGETS-02.md",
    "CONCEPT-TARGETS.md", "SELECTION-RULE.md", "LOG.md", "STATUS.md",
]
A("| file | lines | sha256[:16] |")
A("|---|---|---|")
for f in FILES:
    if os.path.exists(os.path.join(PROJ, f)):
        A(f"| `{f}` | {nl(f)} | `{sha(f)}` |")
    else:
        A(f"| `{f}` | (missing) | |")

A("""
### Where the numbers live

| number | file |
|---|---|
| Meter-02 re-scores, all T2 targets | `runs/t3/reaudit_t2.json`, `runs/t3/reaudit_all.log` |
| rung re-scores (1/2/3) | `runs/t3/reaudit_rungs.json` |
| live lie probe (supervised + baseline) | `runs/t3/live_probe.log`, `runs/t3/audit_K32_lr*_pa_*_ff002.json` |
| cluster sweep (tuning, selection, confirm, audit) | `runs/t3/clu_sweep.log`, `runs/t3/clu_selection.json` |
| cluster definitions + supports | `runs/t3/concept_clusters.json` |
| memory measurements | `scripts/mem_namer.sh` (re-runnable) |
| pre-registration + every result + all nine mistakes | `LOG.md` |

---

*Prepared by the working session. Every figure was produced by a script on this machine
and is reproducible from the files listed above. Hypotheses are labelled as such.*
""")

with open(OUT, "w") as f:
    f.write("\n".join(parts) + "\n")
print(f"wrote {OUT}")
print(f"  {nl('GLASSNET-REVIEW-08.md')} lines, sha256[:16] {sha('GLASSNET-REVIEW-08.md')}")
