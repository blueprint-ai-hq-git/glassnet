"""record_chat_run1.py — the talking model's first real run: it learned the format.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
d = json.load(open(os.path.join(PROJ, "lens", "runs", "chat", "real1_replies.json")))
h = d["history"][-1]
a = d["audit"]

LOG = f"""
---

## 2026-10-07 17:15 — THE TALKING MODEL, RUN 1: IT LEARNED THE SHAPE OF A CONVERSATION.

### THE NUMBERS

```
ep 1  loss 6.8703  concept 0.7192  val ppl    523.65  val acc   9.44%   (21s)
ep 2  loss 6.1590  concept 0.2357  val ppl    400.73  val acc  10.43%   (21s)
ep 3  loss 5.9765  concept 0.2620  val ppl    353.28  val acc  11.08%   (21s)
parameters: 1,652,200     corpus: 13,318,447 tokens     window: 8
```

**Next-token accuracy 11.08% against a 12,000-word vocabulary.** Random is 0.008%, so this is
roughly 1,400x chance. That is real learning, and it is nowhere near a chatbot.

### WHAT IT SAYS

```
PROMPT   Human: what is your name ?
GREEDY   assistant : assistant : assistant : the the the the the the the...
SAMPLED  assistant : you here you : new and do . this of a minutes , ' , a the the okay , of karat
         oatmeal for do ,

PROMPT   Human: can you help me learn to cook ?
SAMPLED  assistant : i fight ' ' try be think . t ' ' which you are , to the the take to it herbert
         other should are you
```

**It learned the FORMAT and nothing else.** Given `Human:`, it emits `Assistant:` and then
word-like English with correct punctuation and spacing. The structure of a conversation is there.
**The content is not.** Greedy decoding collapses to `the` because with 3 epochs the safest
continuation is the most common token; sampling produces real words in implausible order.

**Honest verdict: not a chatbot.** It is a model that has learned the SHAPE of dialogue, which is a
genuine and measurable first step, and I am not going to describe it as more than that.

### THE CONCEPTS ARE WORKING — THIS IS THE REAL RESULT

```
is_a_question      AUROC 0.9715  fires 0.116  PASS
mentions_the_user  AUROC 0.6193  fires 0.297  PASS
mentions_self      AUROC 0.9854  fires 0.144  PASS
mentions_feeling   AUROC 1.0000  fires 0.022  PASS
is_a_request       AUROC 0.9998  fires 0.127  PASS
mentions_food      AUROC 1.0000  fires 0.016  FAIL
mentions_a_number  AUROC 1.0000  fires 0.017  FAIL
mentions_health    AUROC 1.0000  fires 0.008  FAIL
-> 5/8 pass   coverage 100.00%
```

**Three of the failures have AUROC 1.0000** -- perfect fidelity, failing only the 2% firing floor.
A concept that is never active cannot be doing work, so the floor is right, and it is the same floor
that caught the FIX-03 dead concepts. **The floor does not move because a result looks unfair.**

**The conversational concepts reach higher fidelity than the Shakespeare ones ever did**
(`is_a_question` 0.9715, `mentions_self` 0.9854, `mentions_feeling` 1.0000). Conversational
categories are simply cleaner targets than literary ones.

### WHAT RUNS NEXT, WITH PREDICTIONS WRITTEN FIRST

Run 2: d=384 (~4.9M params), 12 epochs. Pre-registered:
1. val perplexity falls well below 100 (from 353). **If it does not, the problem is not training
   time and that gets said plainly.**
2. Replies become locally coherent (real words in real short phrases) but the model still
   contradicts itself over longer spans -- a 5M model does not hold a topic.
3. The concept pass rate stays at or above 5/8, and the three firing-floor failures stay failures
   unless their firing rate genuinely rises.

### BUGS (27)

27. **The scaling run's final report crashed** with `KeyError: 'readable'` because it was launched
    with `BLOCKER_ONLY=readable` while the report expects both arms. Results were already on disk,
    so nothing was lost, but the report is part of the run and a crash there is a defect.
"""

with open(os.path.join(PROJ, "LOG.md"), "a") as f:
    f.write(LOG)
print(f"appended {len(LOG)} chars to LOG.md")

STATUS = f"""# GLASSNET — STATUS

Updated 2026-10-07 17:15. **Talking model run 1 done: it learned the format. Not a chatbot yet.**

1. **Run 1 result:** 1.65M params, 3 epochs, **val ppl 353.28, next-token acc 11.08%** vs 12,000-word vocab (random 0.008% — ~1,400× chance). **It learned the SHAPE of dialogue** (`Human:` → `Assistant:` + word-like English with correct punctuation). **Content is not coherent. Honest verdict: not a chatbot.**
2. **The concepts are working — the real result:** **5/8 pass**, `is_a_question` **0.9715**, `mentions_self` **0.9854**, `mentions_feeling` **1.0000**, `is_a_request` **0.9998**. The 3 failures are `mentions_food/number/health` at **AUROC 1.0000 but firing 0.008–0.017**, under the 2% floor. **The floor does not move because a result looks unfair.**
3. **Conversational concepts reach higher fidelity than the Shakespeare ones ever did** — cleaner targets.
4. **Run 2 queued:** d=384 (~4.9M params), 12 epochs. Predictions written first: ppl falls well below 100 (if not, training time is not the problem and I say so) · replies become locally coherent but not topic-holding · concept pass rate stays ≥5/8.
5. **SCALING — G2 FAILS:** the +0.342 win was size-specific (+15.7/+11.9/+13.2 at d=128/512/1024 vs bar ~6). **Does not carry to bigger models.** But the readable model is **more accurate at every width** (+1.05/+2.53/+2.65, growing), and the cause is visible: **concepts stayed at 41 while the model grew 8×.**
6. **DISTILLATION WORKS — 1 seed:** distilled student (216k) **beat its same-size control on both metrics** and **beat its 4× larger teacher** (ppl 105.36 vs 107.54). Caveats: accuracy margin +0.125% is inside seed noise; **2 more seeds running.**
7. **THE CEILING:** 10M model = 3.5 hours; 50M = 3.7 days; **GPT-3 = 10,513 years** on this GPU. **"Talk" and "code" reachable; "better than any other model" is ~5 orders of magnitude of compute away** — reported per your LIMIT RULE, not papered over.
8. **27 bugs, every one caught by a gate, a control, or a self-check.**

Detail: `REBUILD-PLAN.md` · `TARGET-LEDGER.md` · `LOG.md`.
"""
open(os.path.join(PROJ, "STATUS.md"), "w").write(STATUS)
print("STATUS.md rewritten")
