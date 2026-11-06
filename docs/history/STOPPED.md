# STOPPED — rung gate passed

2026-10-01 20:21

MILESTONE (rung-3, iter33, banking77): TEST touch — readable 87.76% ± 0.24% (30149 params) vs same-size normal 86.95% ± 0.28% (30149 params) -> delta +0.81 pts

LABEL (Yahya's relabel, 2026-10-02): **Rung 3 = PASSED on the SECOND test
touch.** Touch #5 (iter-032 models, 20:10) FAILED the gate: readable 86.15 ±
0.20 vs normal 86.95 ± 0.28, delta -0.80 (within 2 pts: True | readable >=
normal: False | GATE: not yet). The models then changed (epochs 768 ->
1536) before touch #6 (iter-033 models, 20:21) PASSED: +0.81 pts. The gate
rule itself never changed (within 2 pts AND readable >= normal, 3 seeds,
quarantined test, every touch logged).

## ⚠ SUPERSEDING VERDICT (2026-10-02): RUNG 3 IS **NOT A WIN**

Yahya ordered the fair comparison: give the NORMAL the same tuning budget
the readable got (27 validation-only iterations over epochs, lr, batch,
hidden at ~30k params). Result, VALIDATION only, test untouched:

- readable (iter 033, 3 seeds): **86.08 ± 0.60**
- tuned normal (best cfg ep=192 lr=0.2 bs=64 h=23, 3 seeds): **85.25 ± 0.82**
- gap **+0.83**, pooled std 1.02, 2x pooled 2.04 -> **gap < seed noise**

**VERDICT: NOT A WIN — "no difference".** The +0.81 test win above was
against an UNTUNED normal, so it is not a valid win. The "passed on the
second test touch" line stands only as a procedural record of the gate
run; the substantive rung-3 verdict is NOT A WIN. Test set NOT re-touched.
Logs: runs/rung3_normal_tune/results.json, confirm_best.json.

Per the mission's stop conditions: a rung's gate passes -> stop and wait for Yahya before the next rung.

---

# STOPPED #2 — 2026-10-04 10:55 — PAUSED FOR AN OWNER DECISION (T2 / Target A)

**Reason: the next step requires an architecture/loss change, and the standing
instruction is "Don't change the architecture without asking me." So this is a
deliberate stop, not a failure.**

## State

- **Target A: NOT MET.** Best audit result: **31.2%** of concepts passing against
  the fixed 50% bar (K=32, lam_w 2.0, lr 0.5, 3 seeds). Bars never changed:
  AUROC 0.55 / contribution 0.25 / 50%.
- Four single-change ideas tried this morning, **all four NOT CONFIRMED**:
  1. position-aware naming (fixed a real bug; 22.9% -> 30.2%, failure mode flipped)
  2. naming support gate (swept 1-20%; made it worse; naming-tuning stopped)
  3. activation sparsity lam_act (30.2% -> 5.2%; nameable rows collapsed 86/96 -> 14/96)
  4. weight sparsity lam_w (best 31.2%; margins improved to +0.39 acc / +4.69 ppl)
- **Margin half of Target A is healthy**: at lam_w 2.0 the readable beats the
  matched normal on accuracy AND perplexity, with a bigger margin than baseline.
  The blocker is purely the concept audit.
- Test set **untouched** (last touch #6).

## Why it stops here

The diagnosis is a **design gap, not a tuning problem**: nothing in the loss
encourages a concept to detect anything, so the concepts are arbitrary linear
features with post-hoc names. AUROC sits at ~0.55 in every word-frequency band, so
no naming or sparsity knob moves it. The sanctioned idea list is exhausted
(see LOG.md 2026-10-04 10:50).

## The decision needed

The remaining lever is **auxiliary supervised concept training** — give the
concept layer its own target so each neuron is trained to detect something (this
is what the original GlassNet Phase-2 spec described: "each neuron has a name and
is trained to match a concept label"). That changes the loss/architecture, so it
needs approval first.

Options to choose from:
- **A. Approve the auxiliary concept loss** (one pre-registered experiment, same
  bars, validation only).
- **B. Approve a different lever** the owner prefers.
- **C. Stop this line of work** and move to another target.

Nothing runs until the owner answers.

---

## STOPPED #3 — 2026-10-04 15:10 — OWNER DECISION NEEDED: THE AUDIT METER HAS A HOLE

**Why stopped.** The approved auxiliary concept loss was implemented, gradient-checked
and run (3 seeds, K=32, validation only, test set untouched). Result:

- pass fraction **71.9%** at lam_concept=1.0 -> clears the 50% bar
- **lie probe MISSED at all three settings** (it was CAUGHT at baseline)
- margin clause (acc/ppl vs matched normal) **PASSES** at all three settings

Under the standing rule ("never change a gate's pass rule after seeing the result"),
**TARGET A = NOT MET**.

**But the reason is a defect in the audit itself, and fixing it is a RULE CHANGE,
which I will not make on my own.**

### The defect

A nearly dead concept column can be given a name that matches exactly its own few
firings, and then scores AUROC 1.0 and passes. Measured on the frozen lie-probe
column: it fires on **1.15%** of validation windows (117 of 10,124) and its name's
support is **0.0115** — identical. The auxiliary loss inflates the embedding norm
(84.7 -> 121.6), which saturates the frozen random column to death; the namer then
matches it to those few windows.

### Why it matters beyond the probe (measured on existing checkpoints)

```
                                     pass/K    frac     SUP     UNSUP
lc=1.0  floor 0% (today's meter)    23.0/32   71.9%   93.8%    50.0%
        floor 2%                    12.7/32   39.6%   31.2%    47.9%
        floor 5%                     8.3/32   26.0%    6.2%    45.8%
baseline lc=0  floor 0%              9.3/32   29.2%
               floor 2%              9.3/32   29.2%  (barely moves)
```

The supervised columns' 93.8% is mostly dead neurons. Requiring only a 2% firing
rate drops them to 31.2% — below the unsupervised 47.9%. **So supervision's apparent
win is an artifact, and the audit as it stands can be passed by a dead neuron.**

Note the honest consequence either way: at a 2% floor the pass fraction is **39.6%**,
still under 50%. Tightening the meter does not create a win; it stops a false one.

### Decision requested (rule change — owner only)

- **A. Add a minimum firing rate to the audit meter** (e.g. the column must fire on
  >= 2% of validation windows for its AUROC to count). This makes the meter
  STRICTER, not weaker, and closes the hole the lie probe fell through. It would
  also require re-auditing the earlier results, and it lowers every pass fraction.
- **B. Keep the meter as it is** and treat the lie probe failure as the verdict
  (Target A NOT MET), accepting that a dead column can pass.
- **C. Fix the LIE PROBE instead** (e.g. give the frozen column a live bias so it
  cannot saturate) — narrower, but it leaves the meter hole open for real columns.
- **D. Something else the owner prefers.**

Nothing runs until the owner answers. No rule has been changed.

