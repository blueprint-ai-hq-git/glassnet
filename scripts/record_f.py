"""record_f.py — log Target F, refresh STATUS.md, patch the coverage line in the results file.
"""
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

# --- patch the results file with the CORRECTED, confirmed coverage -----------------------
p = os.path.join(PROJ, "TARGET-F-RESULTS.md")
txt = open(p).read()
old = """**And the point that matters most:** because K = 9 and all nine columns are taught, readability
coverage is **100.00% by construction** — the model is, structurally, fully readable."""
new = """**And the point that matters most — now measured, not assumed.** Because K = 9 and all nine
columns are taught, readability coverage is **100.0000%** — confirmed by measurement across all
three seeds and both rules (`min 100.0000%` over all 10,124 windows). **The model is,
structurally, perfectly readable.**"""
assert old in txt, "results-file anchor not found -- not patching blindly"
txt = txt.replace(old, new)
old2 = """This is precisely the failure mode the owner's reviewer flagged in caveat 4 of the R4 result:"""
new2 = """```
rule 'all'    seed 0/1/2:  R1 0/9  R5 0/9  coverage 100.0000%
rule 'conf50' seed 0/1/2:  R1 0/9  R5 0/9  coverage 100.0000%
```

**100.0000% readable. Zero of nine concepts mean anything.** Coverage and the audit disagree
completely, and the audit is right.

This is precisely the failure mode the owner's reviewer flagged in caveat 4 of the R4 result:"""
assert old2 in txt
txt = txt.replace(old2, new2, 1)
open(p, "w").write(txt)
print("patched TARGET-F-RESULTS.md with the confirmed coverage")

LOG = """
---

## 2026-10-05 20:50 — TARGET F: self-improvement FAILED. Coverage 100%, concepts 0/9

**Question:** can a student trained only on its teacher's generated data beat that teacher?
**Answer: no — it is ~22x worse.** This is a REPLICATION of the Curse of Recursion
(Shumailov et al. 2023, verified arXiv:2305.17493), not a discovery. Reported as one.

```
                                  acc        ppl
normal    rule=all      (3 seeds)  0.11%    1828
readable  rule=all      (3 seeds)  0.11%    1850
normal    rule=conf50   (3 seeds)  0.11%    4986      <- selection made it WORSE
readable  rule=conf50   (3 seeds)  0.11%    2515
teacher (real data, K=9 normal)   30.19%      81.51
CONTROL (same script, REAL data)  30.29%      76.27      <- the script is not the problem
```

**The control is what makes it believable:** identical script, identical hyperparameters, real
data -> 30.29% / 76.27, matching the teacher. The collapse is a property of the DATA.

**Worse than uniform:** uniform over 1501 tokens is ppl ~1501; `all` sits at ~1828 and `conf50`
up to ~4986. The students learned the generated text's quirks in a way that actively misleads
them on real text.

**Concepts did not survive.** Audited on real validation against their own pre-registered
targets: **R1 0/9, R5 0/9 on all three seeds of both rules** (AUROC ~0.50 = chance, contribution
~0.002 vs a 0.25 bar). Target E, trained on real data, was R1 7/9, R5 6/9.

**THE FINDING WORTH KEEPING:** coverage is **100.0000%** (confirmed by measurement, min over all
10,124 windows) while **every one of the nine concepts is dead.** A coverage number with no audit
behind it would call this model perfectly transparent. This is the owner's R4 caveat 4 one level
up -- "a reader can predict the firing" does not mean "the concept matters" -- and the audit is
the only thing that catches it.

**Predictions scored:** 1 CONFIRMED (all is worse -- and far worse than predicted) - 2 WRONG AND
BACKWARDS (conf50 is worse, 4986 vs 1828) - 3 CONFIRMED (headline is "no improvement") -
4 CONFIRMED (lost concepts -- it lost ALL nine) - 5 CONFIRMED (coverage stays 100%, structural,
and that is exactly why coverage alone is worthless) - 6 WRONG AND BACKWARDS (corpus is more
uniform than real text, not repetitive: entropy 6.96 vs 5.06 nats).

**What G3 needs now: structural, not a parameter.** The literature's positive results have a
stronger teacher, a verifier, or far more data. We have none. Next-word prediction has no
verifier, so the loop has no signal to filter on. A task where correctness is CHECKABLE
(arithmetic or a synthetic rule) gives the loop the signal it needs.

**Process notes:** (1) generator shape bug caught by the smoke test; (2) cupy `.get()` bug
killed all 12 runs in 3s each; (3) audit `T3M_K` defaulted to 32 and printed a coverage of 35%
for a K=9 model -- impossible, so it was caught and re-run with T3M_K=9 (R1/R5 for cols 0-8 were
unaffected); (4) the pre-registered `conf >= 0.5` selection admitted 0 of 288 windows and was
amended to a quantile rule BEFORE the arms ran, recorded in TARGET-F.md.

**Artifacts:** `TARGET-F-RESULTS.md`, `LIT-CHECK-SELFIMPROVE.md`, `runs/t3/f_gen.json`,
`f_corpus_{all,conf50}.npz`, `models_f_f_*.json`, `f_audit.json`, `f_cycle.log`,
`ckpt/student_*_f_seed*.npz`, `selfgen_smoke.json`.
"""

STATUS = """# GLASSNET — STATUS

Updated 2026-10-05 20:50. **Record: A = NOT MET, A2 = NOT MET, A3 = MET, C = PARTIAL, D = 100% readable / BEHIND, E = 100% readable / PARITY, F = FAILED (self-improvement).**

1. **Target F tested your self-improvement goal and it FAILED:** a student trained only on its teacher's generated data is **~22× worse** than the teacher (ppl 1787–4986 vs 81.51; acc 0.11% vs 30.19%).
2. **The control proves the test:** the identical script on **real** data reaches 30.29% / 76.27, matching the teacher. So the collapse is the *data*, not the code.
3. **Selection made it worse, not better** — `conf50` ppl 4986 vs `all` 1828. My prediction was wrong and backwards.
4. **All nine concepts died** (R1 0/9, R5 0/9 on every seed of both rules; AUROC ≈ 0.50 = chance). Target E on real data was 7/9 and 6/9.
5. **THE FINDING WORTH KEEPING: coverage was 100.0000% while every concept was dead.** A coverage number with no audit behind it would call that model perfectly transparent. Your R4 caveat 4, one level up — and the audit is the only thing that catches it.
6. **Honestly labelled: this is a replication, not a discovery** — the Curse of Recursion (Shumailov et al. 2023, verified). We reproduced the degradation direction from scratch on our own models.
7. **Why it failed, and what G3 needs next:** the literature's positive results have a stronger teacher, a verifier, or far more data. We have none, and next-word prediction has **no way to tell a correct sentence from an incorrect one** — so the loop has no signal to filter on.
8. **Next design: a task where correctness is CHECKABLE** (arithmetic or a synthetic rule), pre-registered before any run, so the loop gets the signal the literature says it needs.
9. **Bugs caught:** generator shape bug (smoke test); cupy `.get()` bug (killed 12 runs in 3s); audit `T3M_K` default of 32 printing an impossible 35% coverage; pre-registered selection amended BEFORE the arms ran because it admitted 0 of 288 windows.
10. **Still open:** G2 (readability across model sizes) untested · G4 (our own fine-tuning and from-scratch methods on our own datasets) untouched · the human R4 score still needs you.

Detail: `TARGET-F-RESULTS.md` · `LIT-CHECK-SELFIMPROVE.md` · `TARGET-E-RESULTS.md` · `TARGET-D-RESULTS.md` · `LOG.md` · `QUEUE-AFTER-B.md`.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
n = sum(1 for l in STATUS.splitlines() if re.match(r"^\\d+\\.", l))
print(f"STATUS.md rewritten, {n} numbered lines")
