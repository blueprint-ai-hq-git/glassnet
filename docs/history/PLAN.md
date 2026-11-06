# GLASSNET — THE FULL BUILD, STEP BY STEP

Written 2026-10-08 after the owner's instruction: *"just do all of them and actually do everything
that I told you — make the model vision language can see talk; Alpha on separate version that can
only see and the version that can only talk; cover every purpose of an AI model; fully transparent
and readable; the model should get smart from its own data made by the teacher model; it can get
smart on any dataset we throw at it because of our own cleaning script; less data but smart;
optional on/off; step by step, go small then big then big then big; tell me what my computer can
handle without breaking."*

---

## THE MACHINE, MEASURED (so the plan is grounded)

```
RAM      14.5 GB total, 10.6 GB available
CPU      Ryzen 5 PRO 4650G, 12 threads
GPU      GTX 1660 SUPER, 6144 MiB, 5545 MiB free, no tensor cores
DISK     228.3 GB free; GlassNet uses 8.84 GB (limit 20 GB growth)

params     train mem    fits 5GB VRAM?    100M tokens takes
  2,000,000      184 MB   yes               21 min
  5,000,000      460 MB   yes               53 min
 10,000,000      920 MB   yes              1.8 h
 20,000,000    1,840 MB   yes              3.5 h
 50,000,000    4,600 MB    NO (over the 5 GB project limit)   8.8 h
100,000,000    9,200 MB    NO               17.5 h
500,000,000        ---    NO               ~1 year
```

**THE ENVELOPE: everything up to ~20M parameters fits comfortably. 50M is over the 5 GB rule.
Anything that needs comparison at 500M is impossible here and always will be.**

---

## WHAT ALREADY EXISTS (do not rebuild)

```
lens/tensor.py     from-scratch autograd, gradients numerically verified (rel diff 0.003457)
explain()          EXACTLY the model's arithmetic (residual 4.77e-07)
image front-end    patch embedding, 16 patches == 16 words, shared concept layer
text front-end     window of 16 words, same concept layer
concept layer      NAMED, audited (AUROC + contribution + lie probe + random control)
three front ends   vision-only, text-only, and vision+question->answer (all measured)
virtual camera     live frames through the model, its own labels drawn on
ladder             tiny/small/medium/big, binary gates, real budgets recorded
```

---

## THE STEPS, SMALL TO BIG. ONE CHANGE AT A TIME, EACH WITH A GATE.

### STEP 1 — `clean.py`: the data cleaner, on any dataset, with an on/off switch
Why first: *"it can get smart on any dataset we throw at it because of our own cleaning script"*.
Every later step depends on being able to take an arbitrary dataset and make it usable.
Readable rules, each with a stated reason. Reports exactly what it dropped and why.
**GATE:** on a known dataset the cleaner drops only junk; frozen rule list; a random
half of the data removed must NOT improve the score (that is how a cleaner that secretly
overfits gets caught).

### STEP 2 — ONE interface for three variants (alpha builds)
`lens.variants`: `see()` / `talk()` / `see_and_talk()`, all behind one API, all transparent.
Why: the owner asked for separate see-only and talk-only alpha versions alongside the fused one.
**GATE:** each variant loads, answers, and `explain()`s; the see-only variant cannot be asked to
generate text and says so instead of failing.

### STEP 3 — THE TALK+SEE MODEL (the thing that does not exist yet)
A decoder that writes text *conditioned on the image*, sharing ONE concept layer with the
text front-end. Trained on image+caption pairs (COCO / TextVQA, both CC BY 4.0).
Built TINY first (d=32, 2 epochs, few hundred images).
**GATE:** image changes -> output changes (a real dependence, not text-only); concepts fire;
`explain()` still exact; beats a matched text-only control on caption accuracy.

### STEP 4 — SCALE IT: small -> medium -> big
Same code, growing budget, on THIS machine's envelope (2M / 5M / 10M / 20M params).
**GATE:** at every rung, monotonic improvement. If a rung does not improve, the rung is
mis-sized and that gets fixed before going further — this exact mistake was made twice already.

### STEP 5 — LESS DATA BUT SMARTER: the teacher loop
**Status 2026-10-08: argmax teacher t2 FAIL (+0.06, control bug fixed); soft-label same-pool
teacher t2 FAIL (-0.64); disjoint d128 teacher trained (78.00%, 4/4 gates); decisive disjoint
soft-label run IN FLIGHT.** Mechanism analysis in LOG.md. t1 PASSES every time: 10% data + teacher
gets within 9-10 points of 100% — "less data but smart" is shown even when the teacher does not
beat the same 10% real data.
The model generates its own training set; the cleaner filters it; the filtered set trains the
next model. Optionally ON or OFF, and both paths reported side by side.
**GATE:** student trained on teacher-generated data beats the same student trained on the same
N real samples, on a held-out set, on 3 seeds. A win inside seed noise is called no difference.

### STEP 6 — SHARPER EYES (the video quality complaint)
The current model's eyes are 28x28, so the video is 784 pixels stretched 15x. Two parts:
(a) label the true resolution on the frame — cheap and immediate;
(b) a higher-resolution front-end. NOTE THE TRADEOFF: 16 patches matching 16 words is what makes
the fused concept layer work; a 112x112 image at 7x7 patches is 256 patches, which breaks that
match. So this is a fusion redesign, not a number change, and it is done AFTER step 3 works.

### STEP 7 — FACES (BLOCKED, waiting on the owner)
Every real face dataset checked states no licence, is unresolvable, or is non-commercial, and the
project rule is to skip unclear licences. **The owner must pick one:**
(a) skip faces and build captioning instead; (b) use the owner's own photos; (c) name a specific
face dataset and approve it as an exception. **Nothing is downloaded until this is answered.**

---

## RULES THAT DO NOT MOVE

```
- work only inside this folder; never overwrite an old result
- VRAM under 5 GB; disk growth under 20 GB; a checkpoint every 2 hours at most
- safe licences only (MIT / Apache-2.0 / CC0 / CC-BY-4.0), recorded in DATA_LICENSES.md
- from scratch: no pretrained weights, no outside training code
- no scraping, no paid APIs, nothing leaves this machine
- gates are BINARY and fixed BEFORE the run; never reworded after a result
- 3 seeds for key results; a win smaller than seed noise is "no difference"
- every number comes from a log; failed runs are kept
- small to big, one change at a time, never jump ahead
```

## HONEST SCOREBOARD OF TODAY

14 claims were made that outran their evidence and every one was caught and written down:
a "+11.57 win" that was 55% a different change; a "0/25" that was a token-join bug; two silent
0.00% readings from a counter that could not see the token it counted; two unfair comparisons of
differently-sized runs; a "big" rung smaller than "small"; a timeout handler that crashed instead
of reporting; and a fresh-checkpoint vocabulary bug that would have printed wrong class names
silently. **The models have been more reliable than the instruments built to judge them.**
