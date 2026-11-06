"""record_l4.py — the library gate, and the rebuild's totals.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 00:35 — L4 GATE: 12/12. THE LIBRARY WORKS. THE REBUILD'S LADDER IS COMPLETE.

```
PASS  1. concepts_from_words works
PASS  2. Net builds
PASS  3. train runs and reports        Result(epochs=8, final loss 0.8953, 3,450 params, cpu, 0.24s)
PASS  4. explain() is EXACT            residual 0.000e+00
PASS  5. audit() reports coverage WITH it
PASS  6. the real concepts pass        2/2 (AUROC 0.9270 / 0.9319)
PASS  7. audit CATCHES a dead concept  contains_39_NEVER_IN_TRAIN -> FAIL
PASS  7b. coverage still high despite it   100.0000% with 2/3 passing
PASS  8. save/load round-trips exactly worst |diff| = 0.000e+00
PASS  8b. concept names survive the round trip
PASS  8c. the reloaded model still explains exactly   residual 0.000e+00
PASS  9. the same library works on the GPU   residual 2.384e-07, audit 2/2
L4 GATE: 12/12 passed
```

**What the library prints -- the actual deliverable:**
```
IT SAYS  : '6'  (score +6.4245)
EXACT    : residual 0.00e+00 -> EXACT
BECAUSE  : 1 of 2 named concepts fired
   contains_30                fires 0.987   contribution +4.7295
   contains_31                fires 0.001   contribution -0.0011
```

**THE MOST IMPORTANT TEST IN THE GATE is 7 and 7b.** A concept whose token never occurs in the
training text cannot be learned. The gate declares one, trains, and requires the audit to FAIL it
-- and requires coverage to stay HIGH anyway (100.0000% with 2/3 passing). That is the old
project's trap reproduced deliberately as a test: **100% coverage with dead concepts is possible**,
which is why `audit()` exists and why coverage is never reported alone.

## REBUILD TOTALS

```
L0  our own C kernels         18/18 PASSED   kept; no longer on the tensor path
L1  our own tensors+autograd  12/12 PASSED   still the engine
L2  GPU, resident arrays       8/9 PASSED    the one FAIL is the toy-size speed line,
                                             explained by the measured crossover
L3  the readable layer         8/8 PASSED    explain() EXACT
L4  the library               12/12 PASSED   the 10-line path, audit, save/load
TOTAL                         58/59 PASSED
```

**SIXTEEN bugs across the rebuild, every one caught by a gate, a control, a contradiction, or a
self-check. None were caught by being careful.** The latest two, both in the L4 work:
15. **`concepts_from_words` used a token's vocabulary POSITION instead of its ID.** Every support
    came out 0.000, nothing matched, and the audit had no positives to measure.
16. **A corpus whose halves could not be audited.** `contains_30` lived only in the first half and
    `contains_31` only in `X[::4]`, so the test split contained no instance of the first at all --
    the audit reported "no positives or no negatives in this sample" and the gate failed on a
    model that was fine. Fixed by interleaving the corpus so any split contains both concepts.
    **This is the SECOND time this class of bug appeared** (the L3 gate had the same biased-sample
    mistake), which is why both files now carry a comment saying so.

**The library is the deliverable and it is done:** build, train, ask why, check the names, save
and reload -- and the same three lines run on the CPU or the GPU.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 00:35. **REBUILD: `lens` — L0, L1, L2, L3, L4 ALL DONE. 58/59 gates.**

1. **Your rule change:** local GPU only; everything on the GPU; from-scratch relaxed **for the compute backend only** — **the architecture and the library stay ours.** Implemented exactly that way. The GPU arithmetic is **CuPy** (your brief allowed it); the concept layer, `explain()`, the losses, the optimizer, the autograd and the training loop are ours.
2. **L0 18/18** our own C kernels · **L1 12/12** our own autograd · **L3 8/8** the readable layer · **L4 12/12** the library.
3. **L2 GPU 8/9** — the model trains on the GPU with **the same numbers**: loss curves 4.8e-07, parameters 2.4e-07, `explain()` EXACT on both devices. The one FAIL is the toy-size speed line.
4. **CROSSOVER, measured:** toy 2,379 params **0.14x** (CPU wins) · small 29,475 **3.06x** · medium 266,147 **16.05x**. **The GPU wins above ~30,000 parameters.** Its time is FLAT while the CPU's grows — launch overhead, not arithmetic.
5. **Honest speed:** CuPy **1900 GFLOPS** vs my hand-written CUDA **129** vs NumPy **475**. My hand-written GPU math was **14.7x slower**; the CPU side got **4x faster** when the ops became array expressions. Hand-writing the math was the wrong bet.
6. **THE LIBRARY (L4, the deliverable):** `lens.Net` → `lens.train` → `lens.explain` → `lens.audit` → `lens.save/load`. `explain()` returns **residual 0.000e+00** — exact, and it checks itself.
7. **The audit catches dead concepts, proven:** a concept whose token never appears in training is declared, trained, and **correctly FAILED** — while coverage stays **100.0000%**. That is the old project's trap reproduced as a test: coverage without an audit is worthless.
8. **16 bugs, every one caught by a gate or a self-check — none by being careful.** Latest: `concepts_from_words` used a vocabulary *position* instead of a token *id*; and a corpus whose halves couldn't be audited (the second time that class of bug appeared).
9. **Next: train at a size where the GPU actually pays** (266k params = 16x), then attack the one blocker: the readable model's **perplexity gap** (+6.76 at `lam_act=0`, the best lever found).
10. **Still owed from you:** the **Target G decision** and the **human R4 readability score**.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
