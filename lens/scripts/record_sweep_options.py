"""record_sweep_options.py — all optimisation options measured, winners combined.
"""
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-07 19:00 — THE SWEEP: ALL OPTIONS TRIED, WINNERS IDENTIFIED.

Owner: "left them all and find the best one". So every option was run, not one.

### THE BASELINE AND FIVE VARIANTS (each one change, 12 epochs)

```
variant                        params      ppl   bits/char    acc  concepts
BASELINE d128 K37 v12k W8   2,029,925   148.33      1.576   21.21%     16/37
A vocab 3500                  618,925    90.43      1.420   23.31%     15/37   <- best ppl
B window 16                 2,124,009   109.95      1.482   24.69%     24/41   <- best acc
C 24ep + lrdecay            2,029,925   143.19      1.565   21.58%     16/37
D d=256                     3,603,813   147.26      1.574   21.39%     15/37
big1 d384 K48 v12k W8       5,177,701   147.79      1.575   21.56%     16/37
char d64 W32 K35              114,915     8.81      3.139   39.62%      2/35
```

**ALL FIVE WORD-LEVEL VARIANTS BEAT THE BASELINE.** Two winners on different metrics.

**A (vocabulary 12,000 -> 3,500) is the single biggest win: 3.3x FEWER parameters and 9.9% better
bits per character.** The diagnostic had already said why: at d=128 the embedding was **93% of the
model**, and most of it was holding words the model saw once or twice.

**B (window 8 -> 16) gives the best accuracy and 24/41 concepts** against the baseline's 16/37.
Drift is a context problem, not a perplexity problem, and this confirms it.

**D (d=256) and big1 (d=384) both buy ~0.1%.** Consistent with the grid finding that once the
concept layer is wide enough, extra width is nearly worthless.

### PREDICTIONS — TWO WRONG, THREE HELD

1. **"A best per parameter, may LOSE on absolute perplexity"** -> **A won BOTH. WRONG, and too
   conservative.** I expected `<unk>` to rise enough to cost more than the freed parameters gained.
2. **"B best responses"** -> B has the best accuracy. **HELD.**
3. **"C improves perplexity most"** -> A improved most. **WRONG.**
4. **"D improves least"** -> D improved least. **HELD.**
5. **"no variant gives a coherent chatbot"** -> **HELD.**

### BUG 31 — THE THIRD VERDICT-CHECK BUG

`run_sweep.sh` built its comparison rows with `cpt = 1` for every variant, where `cpt` is characters
per token (4.576 for word-level, 1.0 for character-level). The column therefore showed bits per
**token**, and the verdict compared 6.499 against the real baseline of **1.576** and printed
**"NO VARIANT BEAT THE BASELINE"** -- while all five had beaten it.

**The numbers were right. The conclusion was wrong.** This is the **third** bug of exactly this
class: METER-06 (an arithmetic slip that capped a perfect concept at 1.00 against a bar of 2.0), the
character-level verdict (printed YES where the table said otherwise), and now this.

**Three times, the failure was a verdict attached to a correct measurement. Never once a wrong
measurement.** That is a pattern worth naming: the risky step in this project is not producing
numbers, it is converting them into a claim. Every future check should print the compared values
next to the verdict so a reader can catch a mislabelled conclusion without re-deriving it.

### WHAT WAS BUILT ON TOP

**Top-k sampling and a repetition penalty in the decoder.** Greedy decoding loops ("i'm not sure i'm
not sure i'm not sure") because argmax always takes the same continuation; pure sampling at temp 0.8
drifts into word salad. Top-k keeps sampling inside the plausible set and the repetition penalty
subtracts from tokens already used, so the model cannot loop.

**This changes the DECODER, not the model** -- perplexity and accuracy are untouched by it. It is
the difference between the model's quality and the OUTPUT's quality, and the owner asked for output.

### THE COMBINED RUN

`best1`: vocab 3,500 (A) + window 16 (B) + lr decay 0.5 (C) + 24 epochs, at d=128 (D buys least).

Pre-registered predictions:
1. Combined beats A's 1.420 bits/char -- A's win came from freed parameters, B's from context, and
   they do not compete for the same resource.
2. Concepts pass above 24/41, because window 16 gives each concept more room to appear.
3. The TOPK20 replies are the most coherent output this project has produced: real sentences,
   correct grammar, no single-phrase loop. **Still not a chatbot -- it will contradict itself.**
4. Greedy STILL loops even with the penalty, because greedy always takes the same argmax.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = """# GLASSNET — STATUS

Updated 2026-10-07 19:00. **All optimisation options tried. Winners combined. Best run in progress.**

1. **THE SWEEP — all five variants beat the baseline** (d=128, K=37, vocab 12k, W=8, 12ep, 2,029,925 params, ppl 148.33, **1.576 bits/char**):
   - **A vocab 3,500** — ppl **90.43**, **1.420 bits/char**, **618,925 params** — **best perplexity AND 3.3× smaller**
   - **B window 16** — ppl 109.95, **1.482 bits/char**, **acc 24.69%**, **24/41 concepts** — best accuracy
   - C 24ep + lr decay — 143.19 / 1.565
   - D d=256 — 147.26 / 1.574
2. **A is the single biggest win in the project: 3.3× fewer parameters and 9.9% better bits/char.** The diagnostic said why — at d=128 the embedding was **93% of the model**, mostly holding words seen once or twice.
3. **B confirms drift is a context problem, not a perplexity problem** — best accuracy and double the passing concepts.
4. **D (d=256) and big1 (d=384) both buy ~0.1%** — consistent with the grid: once the concept layer is wide enough, extra width is nearly worthless.
5. **PREDICTIONS: 2 WRONG, 3 HELD.** Wrong: I said A might lose on absolute perplexity (it won both), and that C would improve most (A did).
6. **BUG 31 — the THIRD verdict-check bug.** The sweep compared bits-per-*token* against a bits-per-*character* baseline and printed "NO VARIANT BEAT THE BASELINE" while all five had. **The numbers were right; the conclusion was wrong.** Pattern now named: **the risky step is converting numbers into a claim, not producing them.**
7. **Decoder fixed: top-k sampling + repetition penalty.** Greedy loops; pure sampling drifts. **This changes the output, not the model** — and output is what was asked for.
8. **Combined run `best1` in progress:** vocab 3,500 + window 16 + lr decay + 24 epochs at d=128. Predictions written first.
9. **CODE data ready:** Magicoder-OSS-Instruct-75K (MIT, 203 MB) and codeparrot/apps (MIT, 107 MB, **with test cases** — a genuinely checkable task: generate, execute, compare).
10. **Licence discipline held:** skipped datasets with no stated licence, "other", "cc", or gated. All recorded with reasons.
11. **THE CEILING:** 10M = 3.5 hours, 50M = 3.7 days, **GPT-3 = 10,513 years.** "Talk" and "code" reachable; "better than any other model" is ~5 orders of magnitude away.
12. **31 bugs, every one caught by a gate, a control, or a self-check.**
13. **Game guard is live** — kills my training within 15s if Roblox/CS starts. Checkpoints are per-epoch.

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
