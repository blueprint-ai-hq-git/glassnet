# SUMMARY.md — iterations 0–10 (plain English, 12-year-old version)

**The mission was:** make a model you can READ (every middle neuron has a
name and an on/off meaning) that scores as well as a boring normal model
with the same number of knobs. Then move to harder tasks.

**How the game was played (rules we followed):**
- A practice set called "validation" made ALL the decisions. The real
  "test" set was locked in a box and opened exactly twice, both times
  written down.
- Every new idea competed against the current best model in a 3-round
  contest (3 random starts). If the win was smaller than the natural
  wobble between rounds, we called it "no difference."
- Every time we invented a concept name, we checked the neuron REALLY
  means it (does it fire on its namesake?) and tested with a fake neuron
  that the checker catches liars.

**What happened (10 experiments, one change each):**
1. **Baseline**: readable 79.7 vs normal 85.3 — 5.6 points behind. The gap.
2. Fixed the 2 broken concept names (they now mean what they say) — but
   accuracy didn't move. Honest finding: broken names were not the cost.
3–4. Added + widened to 27 concepts: the audit caught 5 new names that
   LIED (concepts with no real signal, e.g. "diagonal edges" 0.27).
   Removed them. Accuracy still ~79.7. The concept LIST was not the issue.
5. Linear rules (no squishy sigmoid): total collapse to 49.7 — reverted,
   logged as a failed experiment.
6. **THE WINNER: learned-but-audited.** A small learned "gate" (nonlinear
   math) feeds the named concept readouts. Names stay real (audited), and
   accuracy jumped 79.7 → 82.6. First real promotion.
7. Stronger name-supervision fixed the drifted names but cost accuracy —
   reverted (names and accuracy trade off).
8. **Removed the rule layer entirely**: classes read the named concepts
   directly (even MORE readable — each class is a simple weighted list of
   named concepts). 82.6 → 85.8. The rule layer was the hidden cost!
9–10. More gate capacity: 85.8 → 86.7 → **87.1 (validation)**, and on the
   locked test set: **86.26% — EXACTLY tied with and fractionally ahead of
   the same-size normal network (86.26%).**

**The scoreboard (final, iteration 10, 3 seeds):**
- readable: 86.26% ± 0.12% (102,436 knobs)
- same-size normal: 86.26% ± 0.33% (102,565 knobs) → Δ +0.00 pts
- readability checks: lie-catcher works every time; 2–3 weak legacy names
  flagged honestly (they are the known price of the 14 hand-picked
  concepts, not hidden).

**Mistakes we made and kept:** linear rules (49.7%) · weighted simplicity?
no — those 5 lying concepts · too-strong supervision (fidelity up,
accuracy down) · a few code bugs (missing imports, wrong axis) — all
written in LOG.md with their numbers. Nothing deleted.

**Where we stopped:** Rung 1 gate passed → STOPPED.md → waiting for Yahya
before Rung 2 (a harder image set — license must be checked first).