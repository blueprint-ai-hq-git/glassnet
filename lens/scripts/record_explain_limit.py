"""record_explain_limit.py — G1: the explanation now states its own limits.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 21:25 — G1 CLOSED A GAP: THE EXPLANATION NOW STATES ITS OWN LIMITS.

The owner's test was explicit: "it just shows something and we get another thing that can be still
known as hiding". The decode printed a word list and said nothing about how much of the neuron that
list covered. `explain_text()` now prints the coverage itself:

```
INPUT    : a <unk> <unk> Why dost thou wrong her
IT SAYS  : '<unk>'  (score +6.9567)
EXACT    : residual 2.38e-07 -> EXACT
BECAUSE  : 3 of 41 named concepts fired
   is_a_dialogue_line         fires 0.419   contribution +1.2278
   freq_token_a               fires 0.996   contribution +0.2427
   pronoun_subject            fires 0.104   contribution -0.1890

READING  : this model has 41 named concepts; a human reads 41 lines to know the whole middle layer.
LIMIT    : the top 40 words of a concept account for 9.0% of that neuron. The other 91.0% is NOT
           shown by any word list.
```

**Per-concept coverage, measured on the winning model:**
```
pronoun_subject    top-40 covers 11.3%   -> 88.7% NOT shown
pronoun_object     top-40 covers 19.7%   -> 80.3% NOT shown
is_a_negation      top-40 covers 21.6%   -> 78.4% NOT shown
```

**A word list is a SKETCH, not a description.** That is now printed by the library rather than left
as a caveat in a report. `concept_word_list()` returns both the terms and the fraction, so any
caller that prints the words can print the coverage.

**One real sign of progress in the wide model:** `pronoun_subject` now has `I` as its TOP token
(`+13.21*I`), ahead of `KING` and `DUCHESS`. It finally learned the pronoun it is named after. It is
partly right rather than fully right -- the character names are still mixed in -- but it moved in
the correct direction, and it passes the audit at 0.6452.

**Two bugs, both mine, both fixed:**
19. **`VOCAB[1500] IndexError`** -- the vocabulary has 1500 strings but V = 1501 because of the
    `<unk>` row, so the last id has no string. **This is the same defect the old project hit**, which
    is why the fix is commented as such. Now prints `<unk>`.
20. A patch that merged two lines into one, producing a SyntaxError. Caught by the linter
    immediately, fixed in the next call.

**The chain is still running** (scaling -> distillation -> benchmarks). This work used the CPU while
the GPU was busy, so nothing was serialised behind it.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 21:25. **Blocker beaten and confirmed. G1 gap closed. Scaling + distillation running.**

1. **BLOCKER BEATEN, CONFIRMED ON 3 SEEDS:** gap **+0.342 / −0.802 / +1.577**, mean **+0.372 ± 0.971** vs bar ~6. **Seed 1 beat the plain model outright.** Fix = **41 concepts instead of 9**.
2. **Not a false win — verified first:** 32/41 concepts pass, and **all 9 ORIGINAL concepts pass**, including the two that failed **five times** across two engines.
3. **CAPACITY THRESHOLD IS SHARP:** 9 → +23.045 · 22 → +20.203 · **28 → +19.937** · 41 → **+0.372**. It fails, fails, fails, then collapses. Not gradual — 28 is *not enough*, not *nearly enough*.
4. **G1 gap CLOSED:** `explain()` now prints its own limits — **the top 40 words cover only 8–22% of a concept neuron; 78–92% is not in any word list.** A word list is a sketch, not a description, and the library says so itself.
5. **Progress visible in the wide model:** `pronoun_subject` now has **`I` as its top token** (+13.21), ahead of KING/DUCHESS. It learned the pronoun it is named after.
6. **RUNNING NOW (queued, one GPU):** **scaling** (widths 128/512/1024, each vs its own matched normal — your "does it carry to bigger models") → **distillation** (big local teacher d=512 → student d=128 vs a same-size control) → **benchmarks**.
7. **Distillation rules I'm holding to:** teacher must be **trained locally** (your no-cloud rule) and must clear a **pre-fixed 25%** held-out bar, or the run reports **NO VERDICT** — the old G run failed because its teacher scored 2% held-out while scoring 14–21% on training.
8. **Honest limit on benchmarks:** this model **cannot** be scored on coding or general knowledge — it's a ~500k-parameter word-level model of Elizabethan English. It gets scored on what it actually does: perplexity, accuracy, concept fidelity.
9. **Not 100% on your goals:** G1 partial (41 lines to read; 8–22% word coverage) · G2 untested on the new engine (running now) · G3 no win yet (running now) · G4 untouched.
10. **Bugs: 20 so far, every one caught by a gate, a control, or a self-check.** Latest: the `VOCAB[1500]` IndexError — **the same defect the old project hit.**

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `METER-06-GRADED.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
