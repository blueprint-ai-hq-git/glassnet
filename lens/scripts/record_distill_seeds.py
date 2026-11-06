"""record_distill_seeds.py — the single-seed distillation win did NOT survive. The seed rule paid off.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 18:50 — DISTILLATION: THE SINGLE-SEED WIN DID NOT SURVIVE 3 SEEDS.

### THE RESULT

```
seed 0  teacher 29.53%  A 29.78% / ppl 105.357   B 29.65% / ppl 107.353   d_acc +0.125%  d_ppl -1.996
seed 1  teacher 29.80%  A 29.83% / ppl 103.971   B 29.58% / ppl  98.575   d_acc +0.250%  d_ppl +5.396
seed 2  teacher 29.65%  A 29.70% / ppl 103.504   B 29.70% / ppl 105.598   d_acc +0.000%  d_ppl -2.094

A-vs-B accuracy delta:   mean +0.125%   std 0.102%
A-vs-B perplexity delta: mean +0.435    std 3.508

seeds where A beats B on accuracy:   2/3
seeds where A beats B on perplexity: 2/3

VERDICT: NO CONSISTENT DIFFERENCE. The single-seed result did not survive seeds.
```

### THIS IS THE SEED RULE DOING EXACTLY WHAT IT EXISTS FOR

After seed 0 the result looked like a win: **A beat B on BOTH metrics, and A beat its own 4x larger
teacher.** That was reported with the caveat that +0.125% accuracy is inside noise and it was one
seed.

**Two more seeds and it fell apart.** Seed 1 has A *worse* than B on perplexity by +5.396, and the
perplexity delta across seeds has a standard deviation of 3.508 -- **eight times larger than the
mean of +0.435.** The apparent win was noise.

**This is recorded as: distillation gives NO CONSISTENT DIFFERENCE at this scale.** The honest
statement is that the single-seed result was a false win, and the only reason it was not reported as
a success is that the rule required 3 seeds.

**The earlier framing is withdrawn:** "a model we trained ourselves made a smaller model better than
itself" was said after one seed and it does not hold.

### AND A SECOND, SHARPER FINDING: THE SMALL MODEL BEATS THE BIG ONE REGARDLESS

```
                        A beats teacher   B beats teacher
  seed 0  perplexity          yes              yes
  seed 1  perplexity          yes              yes
  seed 2  perplexity          yes             no
  accuracy (all seeds)        yes           mixed
```

**The 216k-parameter student beats the 820k-parameter teacher on all three seeds.** But the CONTROL
student B also beats the teacher on 2 of 3. So this is **not** a distillation effect -- **it is a
capacity effect: the 820k teacher is over-parameterised for this corpus, and a smaller model
generalises better here.**

That is a real and useful finding on its own, and it is a different claim from "distillation works".
It also explains why the distillation signal is undetectable: the teacher is not actually better than
the student, so there is nothing for the teacher to transfer. **A teacher that is not better cannot
teach better** -- which is the same precondition failure that killed the old G experiment, showing up
in a milder form.

### WHAT THIS MEANS FOR THE PROJECT'S HONESTY RULES

Every claim in this project is now backed by the seed rule, and the rule has now caught:
1. **FIX-02's false win** -- language numbers improved because the concepts died.
2. **The single-seed distillation win** -- did not survive two more seeds.
3. **The 41-concept win's size-specificity** -- +0.342 at d=256, +12 to +16 at other widths.

**Three wins that looked real and were not.** None of them was caught by careful reasoning; all
three were caught by a rule that was applied before the result was known to be weak.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 18:50. **The 3-seed rule caught a false win: distillation is NOT a consistent win.**

1. **DISTILLATION — NO CONSISTENT DIFFERENCE.** Single seed 0 looked like a win (A beat B on both metrics AND beat its 4× larger teacher). **Two more seeds and it fell apart:** seed 1 has A *worse* on perplexity by +5.396, and the perplexity delta std (**3.508**) is **8× the mean (+0.435)**. **The apparent win was noise. The earlier framing is withdrawn.**
2. **Second finding: the small model beats the big one regardless.** The 216k student beats the 820k teacher on all 3 seeds — **but so does the control on 2 of 3.** So it is a **capacity effect** (the teacher is over-parameterised for this corpus), not distillation. **A teacher that is not better cannot teach better** — the same precondition failure that killed the old G run, in milder form.
3. **The seed rule has now caught THREE false wins:** FIX-02's (numbers improved because the concepts died), the single-seed distillation win, and the size-specificity of the 41-concept win. **None was caught by careful reasoning — all three by a rule applied before the result was known to be weak.**
4. **OPTIMISATION — the diagnostic found the real defect:** the concept layer takes **1,024 numbers in and puts 9 out (114:1)** where the scaling win used **50:1**. **93% of the model is the embedding table; the concept layer is 0.6% and is the only path from input to output.**
5. **Fix as a grid:** run 1 done (d=128 K=9, ppl 353.28, 5/8) · **big1 (d=384 K=48) running** · then d=384 K=9 (the bad-ratio control) · d=128 K=48 (**21:1, better than the win**).
6. **Character level built:** **432 chars, 60.9M characters, no `<unk>`.** The concept layer goes from **0.6% → 67% of the model**, ratio 114:1 → 29:1. **But the quality argument is NOT made:** in bits per character, word level wins **1.850 vs 5.443** on the data actually seen. Full run queued; **if it doesn't beat 1.850, character level is wrong for this corpus and I'll say so.**
7. **Talking model run 1:** learned the **shape** of dialogue (`Human:` → `Assistant:`), not coherence. **Not a chatbot.** Concepts **5/8 pass** (`is_a_question` 0.9715, `mentions_feeling` 1.0000).
8. **THE CEILING:** 10M = 3.5 hours, 50M = 3.7 days, **GPT-3 = 10,513 years**. "Talk" and "code" reachable; **"better than any other model" is ~5 orders of magnitude of compute away.**
9. **30 bugs, every one caught by a gate, a control, or a self-check.**

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
