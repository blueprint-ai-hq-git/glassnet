"""record_g2_void.py — record that the first G2 sweep was VOID and why, plus the two gates added.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 15:35 — THE FIRST G2 SWEEP IS VOID. RECORDED BEFORE ANY RE-RUN.

**Result of the first G2 sweep (D=32/64/128, 3 seeds):** coverage 100.0000% at every rung, but
"concepts passing" was 0.00 / 0.33 / 0.00 of 9. **That result is void and is not a finding.**

**Why it is void -- two defects, both mine, both caught before the numbers reached the owner:**

**Defect 1 -- the audit used the wrong statistic.** My first audit used `|h * W|` as the
contribution and the raw (non-residualised) activation for AUROC. The standing meter is a
**length-residualised AUROC** (`resid = h[:,c] - A @ lstsq(A, h[:,c])`, `A = [1, log1p(non-pad)]`)
and a **logit-gain** contribution (`bk = concept.W[:,c].reshape(WIN,D)`, `wvec = bk @ emb[i]`,
summed over fired windows containing token i, divided by fired count). Completely different
numbers. Caught by running a **known-answer control**: the rewritten audit reproduces Target E's
recorded figures to four decimals (`pronoun_subject 0.5125/0.0769`, `is_archaic_address
0.7485/0.4147`, **R1 7/9 = recorded 7/9**).

**Defect 2 -- the model trained on the WRONG CONCEPT SET.** `run_t2_iter003._load_concept_targets`
defaults to `T2_CONCEPT_SET="words"` (single-word targets). The G2 runner never set it, so the
models trained on the **word** set while the audit compared the **9 meaningful targets**. Different
sets, columns not corresponding, so every concept scored near chance. **This is why D=64 gave
0.33/9 when Target E -- same width, same data, same concepts, same lam_concept -- gave 7/9.** That
contradiction is what exposed it; the run was killed rather than reported.

**Two hard gates added so neither can recur silently:**
1. `g2_scaling.py` **refuses to start** unless `CONCEPT_TARGETS['kind'] == 'meaningful_ext'` with
   exactly 9 words. Verified BOTH ways: without the env it prints "REFUSING TO RUN" and exits;
   with `T2_CONCEPT_SET=meaningful_ext T2_N_TAUGHT=9` it proceeds.
2. `run_g2.sh` runs the **audit control first** and aborts if it does not reproduce Target E's 7/9.

**Corrected sweep launched 2026-10-06 15:35** (proc_633bc676b89c): D=32/64/128, 3 seeds, 60 epochs,
with `T2_CONCEPT_SET=meaningful_ext T2_N_TAUGHT=9 T2_LAM_CONCEPT=0.1` now explicit.

**Prediction 5 from TARGET-G2.md stands unchanged** and is what this sweep tests: the 9-concept
bottleneck should hurt MORE as the model grows, so G2-NOWORSE should fail at the large rungs.
Predictions are not revised after a void run.

**Note on my own diagnosis:** while chasing defect 2 I misread `CONCEPT_TARGETS` as "7 targets"
when it is a dict with 7 keys, and my first CS2 session pairing produced 25,710-minute sessions
from a burst-of-PIDs structure I had not read carefully. Both were caught by checks, not by care.
The pattern is consistent: every wrong number today was caught by a control, a contradiction, or a
self-check -- never by being careful.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

p = os.path.join(PROJ, "STATUS.md")
txt = open(p).read()
old = "7. **TARGET G2 is pre-registered and RUNNING**"
new = ("7. **TARGET G2: the first sweep was VOID and is recorded as void, not as a result.** Two "
       "defects of mine: the audit used the wrong statistic (fixed, verified against Target E's "
       "7/9), and the model trained on the WORD concept set because my runner never set "
       "`T2_CONCEPT_SET` while the audit compared the 9 meaningful targets. That mismatch is why "
       "D=64 gave 0.33/9 when Target E gave 7/9. Both are now hard gates: the script refuses to "
       "start on the wrong set (verified both ways), and the runner aborts unless the audit "
       "reproduces a known result. **Corrected sweep running** (D=32/64/128, 3 seeds). "
       "**Was:** TARGET G2 is pre-registered and RUNNING")
assert old in txt, "STATUS anchor not found"
txt = txt.replace(old, new, 1)
open(p, "w").write(txt)
print("STATUS.md updated")
