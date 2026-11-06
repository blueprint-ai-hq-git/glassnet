"""record_g.py — record Target G's result and the G2 pre-registration, then refresh STATUS.md.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-06 14:10 — TARGET G RUN: NO WIN, AND THE REASON IS THE TEACHER

Built the verifiable-task loop end to end (our own code, gradient-checked first): the teacher
trains on a fixed 70 of 100 addition problems, answers all 100 twenty times each, integer
arithmetic verifies the answers, three student arms train, all compared on the 30 held-out
problems.

**Result:**
```
                        held-out exact accuracy
teacher                 2.00%  +/- 1.83     (train-set 14-21%)
student verified        9.33%  +/- 6.41     delta +7.33 (bar 9.43)  FAIL
student unverified      6.67%  +/- 2.36     delta +4.67 (bar 4.22)  FAIL
student real           18.00%  +/- 6.06     delta +16.00
G-WIN FAIL. G-CONTROL FAIL -> G-WIN VOID.
```

**Prediction 1 was WRONG and is recorded as wrong:** I predicted the teacher would reach >=90% on
its training set; it reached **14-21%**. **The teacher never learned the task, so the experiment
could not test the hypothesis** -- every student arm beat it by learning anything at all.

**The control did its job:** `student-unverified` also cleared the bar (+4.67 > 4.22), so the gain
is from clean data, not from verification. Under the frozen rules that VOIDS G-WIN rather than
letting it be reported as a win.

**What is real:** the pipeline works end to end; the verifier kept 4.9-8.0% of samples and found
64-80 of 100 problems the teacher could answer correctly at least once; real data wins clearly
(+16 pts); the gradient check passed, including the embedding scatter path that a smoke test
caught before the real run (`no gradient for 'emb'` -- the reshape node had no backward).

**Pre-registered stop condition applies:** "the teacher cannot learn the task at all ... a result
worth reporting rather than tuning around." Reported, not tuned.

**Decision owed and NOT taken unilaterally:** four options were put to the owner (easier task /
bigger model / both / stop G). **The question timed out with no answer**, so under the standing
rules ("if a command needs approval and times out, log it, skip it, and keep doing other safe
work" and "do not change my instructions on your own") **nothing was changed.** Recommendation
recorded in `TARGET-G-RESULTS.md`: option 1 (easier task, same size), because it changes one thing
and keeps the architecture identical.

## 2026-10-06 14:10 — TARGET G2 PRE-REGISTERED AND RUNNING

`TARGET-G2.md` written BEFORE any run: does readability carry across model sizes? Rungs differ
**only in width** (D = 32/64/128/256/512, ~20x range). Bars frozen: coverage 100.0000% at every
rung; >=50% of concepts pass the meter at every rung; readable no worse than matched normal on
accuracy AND perplexity at every rung. Six predictions written first, including the honest one --
**prediction 5 says G2-NOWORSE will FAIL at the large rungs**, because a 9-concept bottleneck
should hurt more as the model grows.

**Scope stated up front:** this machine can only test up to ~20M parameters, so "big models" here
means up to ~20M and nothing about frontier scale will be claimed.

**Why this is safe to run without a new decision:** G2 is item two of the owner's own four-part
goal (G1 100% readable, G2 carries across sizes, G3 own training methods, G4 own fine-tuning), and
the standing direction is "just build everything ... no stopping". It changes nothing the owner
set and needs no new hardware.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-06 14:10. **Record: A = NOT MET, A2 = NOT MET, A3 = MET, C = PARTIAL, D = 100% readable / BEHIND, E = 100% readable / PARITY, F = LM collapse (concept claim WITHDRAWN, corrected), G = NO WIN (teacher too weak).**

1. **TARGET G ran and FAILED — honestly.** The verifiable-task self-improvement loop (teacher → generate → verify by integer arithmetic → student) is built and works, but **the teacher only reached 14–21% on its own training set** against my predicted ≥90%. With no competent teacher, the comparison tests nothing.
2. **My prediction 1 was WRONG** and is recorded as wrong. Every student arm beat the teacher (verified +7.33, unverified +4.67, real +16.00 pts) simply by learning anything.
3. **The control voided the win, exactly as designed:** `student-unverified` also cleared the bar (+4.67 > 4.22), so any gain here is from clean data, not verification. **G-WIN = VOID.**
4. **What's real from G:** the pipeline works end to end · the verifier kept 4.9–8.0% of samples and found 64–80 of 100 answerable problems · **real data wins clearly (+16 pts)** · the gradient check passed (it caught a missing embedding backward first).
5. **I did NOT tune around it.** My own pre-registered stop condition covers this case, so it is reported as the result.
6. **A decision is owed you and I did NOT take it alone:** easier task / bigger model / both / stop G. My question timed out unanswered, so per your rules I logged it and did nothing. **My recommendation: easier task, same size** — one change, architecture identical.
7. **TARGET G2 is pre-registered and RUNNING** (`TARGET-G2.md`): does readability survive scale? Rungs differ only in width, D=32→512 (~20×). Frozen bars: coverage 100.0000% and ≥50% concepts passing at **every** rung, readable no worse than matched normal at every rung.
8. **My honest prediction for G2, written before running:** the bottleneck **will hurt more as the model grows** — I expect G2-NOWORSE to FAIL at the large rungs. If that's right it's an important negative about this project's central bet.
9. **Scope, stated up front:** this machine can only test to **~20M parameters**. "Big models" here means ~20M. Nothing about frontier scale will be claimed from this.
10. **Still open:** G1 (readability → 100%) · G3 (own training methods — G failed, needs the task fixed) · G4 (own fine-tuning / from-scratch methods) · the **human R4 score** still needs you.

Detail: `TARGET-G-RESULTS.md` · `TARGET-G2.md` · `TARGET-F-RESULTS.md` (correction banner) · `LIT-CHECK-SELFIMPROVE.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
