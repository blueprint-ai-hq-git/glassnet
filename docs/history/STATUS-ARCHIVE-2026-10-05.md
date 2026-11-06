# GLASSNET — STATUS

**Updated:** 2026-10-04 21:02 IST
**Machine:** idle — nothing running.
**TARGET A3 = MET (all four clauses). Target A and A2 remain NOT MET. Target B not started.**

---

## Headline

**A3 passes all four clauses — and the concepts are TAUGHT, NOT DISCOVERED.**

```
clause                                        requirement    actual          verdict
A3-1 taught concepts pass Meter-03            >= 50%         8/10 = 80.0%    PASS
A3-2 lie probe caught while live              caught & live  caught, live    PASS
A3-3 random-column control < 5%               < 5%           0.0% / 0.0%     PASS
A3-4 readable no worse than matched normal    both clauses   both OK         PASS
hold-out: untaught targets ever pass          reported       0/4  (+0/2 extra)
```

**AUROC is 1.0000 on all 10 taught columns.** The only 2 failures are again the firing
floor (`The`, `him` — firing just under 2%). Changing **only the target set** (keep targets
with TRAIN support ≥ 2.5%) moved the taught pass rate **33.3% → 80.0% with the meter
untouched** — which confirms the A2 diagnosis directly: the constraint was the floor meeting
thin targets, not the architecture.

**Hold-out 0/4 (plus 0/2 extra): teaching works, discovery does not.** Untaught columns fire
freely (27–51%) at chance AUROC (0.4528–0.5770) and pass nothing.

**NOT A PERFORMANCE WIN — stated plainly:** readable is **worse on perplexity by 1.50**
(71.80 vs 70.30) and **slightly worse on accuracy** (30.39% vs 30.56%). Both clear the
pre-registered 2σ clauses, which is the rule, but that is parity at best and marginally
behind on both metrics. **Parity is never a win.** The gap did narrow sharply from A2's
−4.87.

**Tuning:** re-tuned for this target set by the frozen rule — winner `lam_concept=0.1,
lr=1.0` (3-seed mean ppl 71.7980). 9 combos screened on seed 0, top 2 confirmed on seeds
1–2 (a pre-registered budget deviation, recorded).

**Predictions: 7 of 8 confirmed.** One imprecise: I predicted the perplexity gap at 2–6 and
it was 1.50.

**One error of mine, corrected before any run:** my first A3 hold-out rule would have taught
only 6 of the 10 qualifying targets, contradicting "teach all qualifying targets". Amended
with a dated note in `TARGET-A3.md` §3 before training.

Detail: `TARGET-A3.md` (rule), `TARGET-A3-RESULTS.md`, `LOG.md`.

**Awaiting the owner before Target B.**

---

## Superseded headline (earlier today, kept for the record)

**A2 (Option B) NOT MET — fails on the firing floor, not on detection.**

**A2 (Option B: teach every concept with a pre-registered name) is NOT MET — and it fails
on the firing floor, not on detection.**

```
clause                                        requirement    actual      verdict
A2-1 taught concepts pass Meter-03            >= 50%         33.3%       FAIL
A2-2 lie probe caught while live              caught & live  caught,live PASS
A2-3 random-column control near 0%            < 5%           0.0%        PASS
A2-4 readable no worse than matched normal    both clauses   both OK     PASS
hold-out: untaught targets ever pass          reported       0/8         —
```

**The concepts WERE learned.** AUROC is **1.0000 on 22 of the 24 taught columns** (the two
exceptions never occur in validation, so they are unauditable). Contribution is +12 to +15
against a 0.25 bar. **Every one of the 16 failures is the firing floor** — the taught
targets are thin (median 1.89% support), so their columns fire below 2%.

```
median TRAIN support : failing 0.0164 vs passing 0.0246
SENSITIVITY ONLY: at a 1% floor the taught pass would be 19/24 = 79.2%
```

This is exactly the cost recorded in `METER-RULE-02.md` §4 before any of this ran: the
floor that closed the dead-column hole also excludes narrow but genuinely-learned concepts.

**Hold-out 0/8: teaching works, discovery does not.** The untaught columns fire freely
(39–61%), sit at chance AUROC (0.39–0.55) and pass nothing.

**Prediction 2 was WRONG** (I wrote A2-1 PASS 50–75%; it was 33.3%). I did flag the risk
before running, but that came after the pre-registration, so I am not counting it as the
prediction.

**Honest notes:** A2-4 passes but readable is **worse on perplexity by 4.87**; and A2 as
specified **cannot** be met with thin targets — a ≥50% taught clause plus a 2% floor
requires most taught targets above 2% support, and this set has 11/24.

**Verification: 53 claim-vs-artifact checks, 0 mismatches.**

Detail: `TARGET-A2.md` (rule), `TARGET-A2-RESULTS.md`, `LOG.md` 19:20.

---

## Superseded headline (earlier today, kept for the record)

**Option A fixed the lie probe. The other half of Target A collapsed.**

```
                     own target   auroc    contrib   firing   Meter-03   Meter-02 said
supervised lc=1.0    The          0.7633   +0.0166   0.3638   CAUGHT     CAUGHT (floor only)
baseline   lc=0      The          0.5088   +0.0017   0.4462   CAUGHT     MISSED
```

Both caught **live** (firing 36% / 45%, far above the 2% floor) and both fail even against
their **best** of all 16 targets — so the catch is real and not an unlucky pairing.
Auditing against a pre-registered target the column was never trained on is what does it.

**Random-column control: 0.0% free pass rate** (concept layer at random init, never
trained, live in every one of 30 draws). No free passes — every pass is real.

**But the non-circular half is ~0:**

```
WORDS    lc=0.1/0.5/1.0: SUP 31.2%   UNSUP 0.0%   (excess +0.0)
CLUSTERS lc=0.5        : SUP 62.5%   UNSUP 2.1%   (excess +2.1)
bar for the claim: UNSUP excess >= 50%
```

**The 45.8% figure is OVERTURNED.** That was Meter-02's non-circular number — the one I
had been calling the honest gain. Against pre-registered targets those columns were never
trained on, it is **0.0–2.1%**. It was an artifact of the namer deriving names from each
column's own firings.

**What the concept layer actually does: it detects pre-registered targets only where it
was explicitly trained. Supervision works; generalisation to untaught targets does not.**

**Predictions 1–3 were all wrong** (I predicted supervised 85–100%, unsupervised 5–20%,
random 10–30%; actual 31.2/62.5%, 0.0–2.1%, 0.0%).

**Name-swap control: 100% rejected**, decomposed rather than celebrated — most raw
rejections come from the firing floor (a column property that says nothing about
target-matching); restricted to live columns it is still 100%, driven by the contribution
clause.

**Option B (fix names before training) is proposed in `METER-03-RESULTS.md` §9 and NOT
run.** Engine refactor remains separate and later. Full detail: `METER-RULE-03.md`,
`METER-03-RESULTS.md`, `LOG.md` 18:15.

---

## Superseded headline (earlier today, kept for the record)

**Cluster targets cleared the 50% bar for the first time — on a genuine non-circular
gain — but the lie probe still fails, so Target A is NOT MET under the rule.**

```
run                          pass/K    frac     SUP     UNSUP
v1 single words, lc=1.0     12.7/32   39.6%   31.2%    47.9%
unsupervised baseline lc=0   9.3/32   29.2%      —        —
v2 clusters, lc=0.5         17.3/32   54.2%   62.5%    45.8%   <- bar met
```

* **The unsupervised (non-circular) half rose to 45.8% from the baseline's 29.2%** —
  the highest honest number so far.
* **The firing floor no longer bites** (0% and 2% give identical counts), because
  cluster concepts fire broadly. That is a real concept-quality improvement over v1,
  where the supervised columns collapsed 93.8% → 31.2% under the same floor.
* **The lie probe is MISSED again**: a column frozen after ONE epoch stays live
  (fires 8.5%) and passes with AUROC 0.80. So the pass fraction is measured by a meter
  that cannot tell a trained concept from a frozen one.
* Margin clause passes at all three settings.

**My pre-registered prediction about the unsupervised half (25–40%) was WRONG in the
favourable direction — it reached 45.8%.** Recorded plainly.

**The blocker is a design decision, not more tuning:** audit supervised columns against
their **pre-registered cluster target** instead of a name the namer derived from the
column's own firings. That is what would let a never-supervised probe fail. Owner
approval required; nothing is running.

**Also today:** Meter-02 (firing floor) implemented, verified (floor=0 reproduces every
recorded Meter-01 count and flag exactly) and applied to every earlier result —
overturning "T2 supervised 71.9%" to 39.6% and showing rung 3's neurons are essentially
all dead. An OOM in the re-audit was root-caused to reference cycles in the autograd
graph. See `LOG.md`, `REAUDIT-METER-02.md`, `CONCEPT-TARGETS-02.md`.

---

## Superseded headline (earlier today, kept for the record)

**Meter-02 (firing floor + live lie probe) is implemented, verified and applied to
every earlier result. It works — and it found that the deeper problem is a
circularity in the audit design.**

**1. The floor catches dead-column false passes.** With `min_firing=0` the meter
reproduces every recorded Meter-01 count and flag exactly, so this is a pure re-score.
Under the 2% rule:

```
T2 supervised lc=1.0      71.9%  ->  39.6%   (below the 50% bar)
T2 baseline (unsupervised) 29.2%  ->  29.2%   (unchanged)
Fashion-MNIST rung 1       92.9%  ->  85.7%   border_ink: fidelity 0.769, fires 0.0004
PathMNIST    rung 2       100.0%  -> 100.0%   unchanged (recorded values reproduced exactly)
Banking77    rung 3        18.8%  ->   0.0%   3 neurons, all firing 0.0000
```

**So "T2 supervised: 71.9% of concepts pass, above the bar" is OVERTURNED.** Rung 3's
verdict (FAILED) stands, but its description was wrong — essentially all 16 neurons are
dead, so its fidelity numbers measured nothing (14–16 of 16 fire on 0.0000 across all
three seeds).

**2. The LIVE lie probe is MISSED.** A column frozen after a single epoch — before it
could have been trained to detect anything — stays live (fires on 15.5% of windows) and
**passes** (auroc 0.5591, contribution +0.4449). Under supervision the column still
dies, so the floor catches it — INCONCLUSIVE, not a pass.

**Why: the audit is circular by construction.** The namer derives each name FROM that
column's own firings, and the answer layer learns to use whatever columns exist. So any
live column gets a name that matches its firings (AUROC clears 0.55) and a real
contribution (clears 0.25). The meter asks "does the name describe the firings?" — which
the namer guarantees — not "was this concept meant to detect anything?". **A live lie
probe essentially cannot fail under this design.**

**TARGET A = NOT MET.** Fixing the circularity needs a design change (test a concept
against a target it was NOT named from, or fix names before training) — that is the
owner's call, and nothing is running.

**Also fixed today:** an OOM in the re-audit, root-caused to reference cycles in the
autograd graph (full-TRAIN forward cost 2.7 GB; chunked + explicit GC → 348 MB). See
`LOG.md` 15:45. Full detail: `REAUDIT-METER-02.md`, `METER-RULE-02.md`.

---

## Superseded headline (earlier today, kept for the record)

**Target A is still NOT MET — and today's run explains why in a way that changes
what the number means.**

The owner-approved auxiliary concept loss was built, gradient-checked, and run
(3 seeds, K=32, validation only, test set untouched). It produced a **71.9%** pass
fraction — above the 50% bar — **but the lie probe was MISSED at every setting**
(it was CAUGHT at baseline). Under the standing rule that is **NOT MET**.

**The real finding: the audit meter has a hole.** A nearly dead concept column can
be named to match exactly its own few firings and then scores AUROC 1.0. Measured on
the frozen lie-probe column: it fires on **1.15%** of validation windows and its
name's support is **0.0115 — the same number**. The auxiliary loss inflates the
embedding norm (84.7 → 121.6), which saturates the frozen column to death; the namer
then matches it to those few windows.

**This is not just a probe artifact — it inflates the real result.** Applying only a
2% firing-rate floor to the existing checkpoints:

```
                                     pass/K    frac     SUP     UNSUP
lc=1.0  floor 0% (today's meter)    23.0/32   71.9%   93.8%    50.0%
        floor 2%                    12.7/32   39.6%   31.2%    47.9%
baseline lc=0  floor 0%              9.3/32   29.2%
               floor 2%              9.3/32   29.2%  (barely moves)
```

**Supervision's apparent 93.8% on the supervised columns was mostly dead neurons** —
at a 2% floor they fall to 31.2%, *below* the unsupervised 47.9%. So the "win" was
an artifact. Note the honest consequence either way: at a 2% floor the pass fraction
is **39.6%**, still under 50% — tightening the meter does not create a win, it stops
a false one.

**The one genuine, non-circular gain:** the *unsupervised* columns (never trained on
any target word) rose from the baseline's 30.2% to **50.0%**, and they hold at
**47.9%** under a 2% floor. Their passing rows are also broader than the supervised
ones (median support 16.4% vs 7.6%), i.e. they are not dead-column artifacts.

**The other half of Target A stays healthy:** the readable beats the matched normal
on accuracy AND perplexity at all three settings (e.g. +0.28 acc / +4.44 ppl against
2σ thresholds of 0.383 / 12.188).

**Decision needed from the owner (a rule change, so I have not made it):** add a
minimum firing rate to the audit meter, keep it as is, or fix the lie probe instead.
Options are written out in `STOPPED.md` stop #3. **Nothing is running until you
answer.**

---

## Superseded headline (earlier today, kept for the record)

**Target A is NOT MET. Best audit result: 31.2%** of concepts passing against the
fixed 50% bar (K=32, lam_w 2.0, lr 0.5, 3 seeds). **Four single-change ideas tried
today, all four NOT CONFIRMED.** The bars were never moved.

**But the diagnosis is now sharp, and it is a design gap rather than a tuning
problem:** nothing in the loss encourages a concept to detect anything, so the
concepts are arbitrary linear features with post-hoc names. AUROC sits at ~0.55 in
*every* word-frequency band, so no naming or sparsity knob moves it.

**The good news:** the *other* half of Target A is healthy. At lam_w 2.0 the
readable beats the matched normal on accuracy **and** perplexity with a **bigger
margin than the baseline** (+0.39 acc, +4.69 ppl vs +0.29/+3.41). The only blocker
is the concept audit.

**The decision needed:** the remaining lever is auxiliary supervised concept
training (train each neuron to detect something — what the original Phase-2 spec
described). That changes the loss, and the standing rule is "Don't change the
architecture without asking me", so **nothing is running until you answer**.

---

## Steps 1–3 (done)

**Step 1 — joint selection rule:** frozen in `SELECTION-RULE.md` before any run.
lr chosen by lowest validation perplexity, tie-break accuracy (within 1%
relative), then lower lr.

**Step 2 — K sweep {8,16,24,32}:** 3 seeds, matched params, equal budget, frozen
selection, re-audited at every K. **The reviewer's hypothesis is NOT confirmed** —
fewer concepts do not make each one matter more.

| K | params | readable acc / ppl | normal acc / ppl | audit pass |
|---|---|---|---|---|
| 8 | 113,677 | 30.35 ± 0.10 / 74.38 | 30.39 ± 0.08 / 79.45 | **8.3%** |
| 16 | 129,789 | 30.76 ± 0.14 / 64.83 | 30.17 ± 0.09 / 71.82 | **14.6%** |
| 24 | 145,901 | 30.79 ± 0.02 / 64.76 | 30.58 ± 0.11 / 66.66 | **26.4%** |
| 32 | 162,013 | 30.86 ± 0.11 / 62.05 | 30.56 ± 0.07 / 65.46 | **22.9%** |

Fidelity passes everywhere (77–90%); contribution fails everywhere (14–35%).
**K=16 is the only K where both margin clauses pass.** K=8 is the only K where
the normal wins accuracy.

**Step 3 — bypass check:** no bypass exists (single path
`emb → flatten → concept → sigmoid → answer → logits`). But forcing all concepts
to zero scores exactly the majority baseline (29.64%) vs the full model's 30.44%,
so the concepts add only **+0.80 pts**; the answer bias *is* the unigram prior.
Concept activations are low (mean 0.24, 36.7% of neuron-windows dead).

## Step 4 (one change at a time)

**Idea #1 — position-aware naming: NOT CONFIRMED.** The ceiling diagnostic showed
the concepts *are* explainable (66.7% reach 0.25 with their best real word, 96.3%
with six), and the namer had a real bug: its docstring describes a position-aware
score but the code credited every word for weight blocks at positions where it
never appears. Fixing it moved 22.9% → 30.2% at K=32 (predicted 40–75%) and
**flipped the failure mode**: contribution 29% → 62% passing, fidelity 84–90% →
51%.

**Idea #2 — support gate: NOT CONFIRMED.** Sweeping the gate 1% → 20% made things
worse (30.2% → 16.7–19.8%). The clauses move in opposite directions, and a high
gate also leaves most concepts unnameable. Naming-tuning stopped here, as
pre-registered, rather than being iterated to fit.

**Idea #3 — activation sparsity (lam_act): NOT CONFIRMED.**

| lam_act | lr | pass | frac | AUROC≥.55 | contrib≥.25 | nameable rows |
|---|---|---|---|---|---|---|
| **0.2** | 1.0 | 9.7/32 | **30.2%** | 51% | 62% | 86/96 |
| 0.5 | 0.5 | 4.7/32 | 14.6% | 40% | 71% | 45/96 |
| 0.5 | 1.0 | 4.7/32 | 14.6% | 32% | 73% | 45/96 |
| 1.0 | 0.5 | 1.7/32 | 5.2% | 64% | 36% | 14/96 |
| 1.0 | 1.0 | 1.7/32 | 5.2% | 57% | 43% | 14/96 |

Contribution rose as predicted but AUROC fell and **nameable rows collapsed**
(86/96 → 14/96): sparser firing starves the naming. lam_act=1.0 also fails the
accuracy/perplexity half of Target A outright (7.5–7.8 ppl *worse* than the
normal).

**Idea #4 — weight sparsity (lam_w): NOT CONFIRMED.**

| lam_w | lr | pass | frac | AUROC≥.55 | contrib≥.25 | nameable | acc diff | ppl diff |
|---|---|---|---|---|---|---|---|---|
| 1.0 (base) | 1.0 | 9.7/32 | 30.2% | 51% | 62% | 86/96 | +0.29 | +3.41 |
| **2.0** | **0.5** | 10.0/32 | **31.2%** | 58% | 57% | 71/96 | +0.39 | **+4.69** |
| 2.0 | 1.0 | 9.3/32 | 29.2% | 49% | 61% | 72/96 | +0.59 | **+7.80** |
| 5.0 | 0.5 | 3.7/32 | 11.5% | 39% | 51% | 57/96 | +0.22 | +2.60 |
| 5.0 | 1.0 | 5.7/32 | 17.7% | 46% | 61% | 54/96 | +0.42 | +6.04 |

Nameable rows held up (the distinguishing check vs idea #3 passed) and the margins
improved, but the joint pass moved only 30.2% → 31.2%. With AUROC at 58% and
contribution at 57%, the joint is ≈ 0.58 × 0.57 — the clauses are now roughly
independent, so clearing 50% jointly needs each near **71%**.

**Hypothesis tested and REJECTED:** I thought AUROC was structurally weak because
`contains` is nearly always true for common words. Measured: pos_frac max is
**0.379** (no word is in >38% of windows), correlation(pos_frac, AUROC) = −0.183,
and AUROC median is ~0.55 in every band. So the concepts are simply **weakly
discriminative everywhere**, regardless of word frequency. 13 concepts already
pass contribution and fail *only* on AUROC.

---

## What is next

**Nothing runs until you decide.** The sanctioned idea list is exhausted:
(1) stronger penalty on unused concepts → tried as lam_act and lam_w, both NOT
CONFIRMED; (2) forcing the answer to read only the concepts → already true
(verified by the step-3 bypass check); (3) pruning failing concepts → the K sweep
already showed smaller K is worse (8.3% at K=8), so it cannot clear 50%.

The remaining lever is **auxiliary supervised concept training**, which changes the
loss/architecture — hence the stop.

## What needs the owner

1. **The decision** (see `STOPPED.md`): approve the auxiliary concept loss, name a
   different lever, or stop this line of work.
2. **The judgement call from earlier, still open:** the position-aware naming fix
   makes the audit easier to pass. My case: the code never did what its docstring
   says, naming runs on train while the audit verifies on val, and the lie probe
   must still catch a lie. But it is your call whether that counts as fixing a bug
   or moving the goalposts. **The bars are untouched: 0.55 / 0.25 / 50%.**

The test set remains untouched (last touch #6).
