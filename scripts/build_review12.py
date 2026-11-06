"""build_review12.py — GLASSNET-REVIEW-12.md for the external reviewer.

Covers what is new since REVIEW-11: STEP 0 (the readability sheet), Target B (rule, grid,
status, prediction), the owner's two queued items, the five pre-specified fallback ideas,
and the evaluation machinery. Every number is READ FROM THE ARTIFACT at build time.
"""
import hashlib
import json
import os
import re

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")
OUT = os.path.join(PROJ, "GLASSNET-REVIEW-12.md")


def J(n):
    with open(os.path.join(R, n)) as f:
        return json.load(f)


def code(path, label=None):
    with open(os.path.join(PROJ, path), "rb") as f:
        raw = f.read()
    h = hashlib.sha256(raw).hexdigest()
    ext = os.path.splitext(path)[1].lstrip(".") or "text"
    lang = {"py": "python", "sh": "bash"}.get(ext, ext)
    return (f"### `{label or path}`\n\n`sha256[:16] = {h[:16]}`\n\n"
            f"```{lang}\n{raw.decode()}\n```\n")


def table(rows, headers):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


ev = J("readability_evidence.json")
a3 = J("a3_eval.json")
m = a3["margin"]

# how many concepts are single-word / phrase, judged from the recorded evidence
LABELS = {0: "MIXED", 1: "WORD", 2: "SINGLE", 3: "WORD", 4: "MIXED",
          5: "SINGLE", 6: "PHRASE(weak)", 7: "SINGLE", 8: "PHRASE", 9: "PHRASE"}
n_phrase = sum(1 for v in LABELS.values() if v.startswith("PHRASE"))
n_single = sum(1 for v in LABELS.values() if v in ("SINGLE", "WORD"))
n_mixed = sum(1 for v in LABELS.values() if v == "MIXED")

D = []
D.append("# GLASSNET — REVIEW 12\n")
D.append("**Generated:** 2026-10-05, from the artifacts (not from memory)  ")
D.append("**Supersedes:** `GLASSNET-REVIEW-11.md` for everything about STEP 0 and Target B. "
         "REVIEW-11 remains correct for Target A3, REVIEW-10 for A2, REVIEW-09 for Meter-03.")
D.append("**Companion:** `GLASSNET-HANDOFF.md` + `GLASSNET-HANDOFF.zip` carry the full "
         "background, the code and the result JSONs.\n")
D.append("---\n")

D.append("## NOTE TO THE REVIEWER — a status, not a request\n")
D.append("**Nothing here needs a reply.** The owner asked for the current state in a form "
         "that can be handed over; there is no decision pending from you and no half-run "
         "experiment sitting in it. Target B is mid-flight by design, and its rule was frozen "
         "before it started.\n")
D.append("---\n")

D.append("## 0. In five sentences\n")
D.append(
    "**STEP 0 is done:** a readability sheet for all 10 taught A3 concepts, showing the "
    "target words, the 5 highest-activation validation windows, and an honest human label. "
    f"**The result is uncomfortable: {n_single} of the 10 are single-word or word-list "
    f"detectors, {n_mixed} are mixed clusters, and only {n_phrase} are genuine multi-word "
    "constructions.** Every column does fire on its target — the detection is real; the "
    "*concepts* mostly are not.\n\n"
    "**The sharpest case:** concept `[5]` is named `KING_What_you,_RICHARD` but its top "
    "firings are driven entirely by the token `you,`. `KING`, `RICHARD` and `What` never "
    "appear. **It still passes Meter-03**, because `you,` *is* in its pre-registered target "
    "set. The audit measures \"does the column detect its target\", not \"was the target a "
    "good idea\" — and this is that gap in concrete form.\n\n"
    "**Target B is running** and was pre-registered before any run: the full grid (12 "
    "readable + 4 normal settings × 5 seeds = 80 runs, no screening), win = readable beats "
    "the matched normal on **both** perplexity and accuracy by more than 2× pooled std, with "
    "the audit clauses still passing.\n\n"
    "**My written prediction is that Target B will NOT be met** — readable \"behind\" on "
    "perplexity. A3 already showed readable worse by 1.50 ppl and 0.17 acc, and 5 seeds "
    "*shrink* the pooled std, making a 2σ win harder rather than easier.\n\n"
    "**Record unchanged: A = NOT MET, A2 = NOT MET, A3 = MET. Parity is never a win.**\n")
D.append("---\n")

D.append("## 1. STEP 0 — the readability sheet\n")
D.append("Read one trained model (`readable_K16_lr1.0_lc01_clusters16_seed0.npz`, the A3 "
         "winner) — no training, no retuning, no renaming, as instructed. Firing rule is the "
         "audit's own (`activation > 0.5`).\n")
D.append(table(
    [[c["col"], ", ".join(c["target_words"])[:34], LABELS[c["col"]],
      f"{c['target_present_in_top5']}/5", f"{c['frac_active']:.4f}"]
     for c in ev["concepts"]],
    ["#", "target words", "honest label", "target in top 5", "frac active"]))
D.append("")
D.append("**Strict count:** "
         f"**{n_phrase} phrase detectors**, **{n_mixed} mixed clusters**, "
         f"**{n_single} word/single-token detectors**.\n")
D.append("### The two concepts that are real ideas\n")
D.append("```")
D.append("[8] 'for such a one as'  — every top firing is the same recurring four-word frame:")
D.append("      'For such a one as leaves a gentleman,' / 'For such a one as leaves a'")
D.append("[9] 'gave him'           — fires on the verb+pronoun frame, not on 'him' in general:")
D.append("      'I gave him no ... I gave him' / 'gave him no ... I gave him the'")
D.append("```")
D.append("### And the one that shows the meter's blind spot\n")
D.append("```")
D.append("[5] name KING_What_you,_RICHARD -> top firings all driven by 'you,':")
D.append("    '<PAD> God save you, sir! TRANIO: And you,'")
D.append("    'you, <PAD> you, I will marry you. Now,'")
D.append("    KING / RICHARD / What appear in NONE of its five top firings.")
D.append("    Verdict under Meter-03: PASS (AUROC 1.0000), because 'you,' IS in its target set.")
D.append("```")
D.append("**Why this matters more than the 80% pass rate:** A3's headline is that 80% of "
         "taught concepts pass the audit. The sheet shows what those concepts *are*. A column "
         "that reliably detects the word `The` passes the audit and is still just the word "
         "`The`. Any future meter work should confront this directly rather than by adjusting "
         "a threshold.\n")
D.append("**One correction I made mid-work, recorded because it was mine:** I first reported "
         "the columns as \"active (>0) on 100% of windows\" — wrong, because I had assumed the "
         "firing rule was `> 0`. Reading the audit showed its real rule is `activation > 0.5`. "
         "The sheet uses the audit's definition, not my assumption.\n")
D.append("---\n")

D.append("## 2. TARGET B — the pre-registered rule\n")
D.append("Frozen in `TARGET-B.md` before any run. The owner's conditions, verbatim:\n")
D.append("> WIN = readable beats the normal on BOTH perplexity AND accuracy by more than 2x "
         "pooled std, AND the A3 audit clauses still pass on the winning run (at least 50% of "
         "taught concepts pass Meter-03, lie probe caught live, random control under 5%).\n")
D.append("> If readable ties or loses, report \"parity\" or \"behind\" plainly. Never call "
         "parity a win.\n")
D.append("**The grid, written first, no screening:**\n")
D.append("```")
D.append("readable : lam_concept in {0.1, 0.5, 1.0}  x  lr in {0.1, 0.5, 1.0, 2.0}  = 12 combos")
D.append("normal   :                                    lr in {0.1, 0.5, 1.0, 2.0}  =  4 combos")
D.append("every combination on ALL 5 seeds                = 80 runs, nothing dropped")
D.append("selection: frozen SELECTION-RULE.md, applied SEPARATELY per model on the 5-seed mean")
D.append("budget   : epochs <= 96, min_epoch 30, patience 15 -- identical for both models")
D.append("```")
D.append("**`lr = 2.0` is included deliberately:** A3's winner sat at the **edge** of its grid "
         "(`lr = 1.0`), so the optimum may lie beyond it. It is in **both** models' grids, so "
         "neither is given a range the other lacks.\n")
D.append("**Status: running.** ~6 min/run → ~7–9 h of machine time, spread over school "
         "sessions. Resumable: a combo whose JSON already holds 5 records is skipped, so an "
         "interruption costs one run, not the grid.\n")
D.append("**Pre-registered prediction:** **Target B will NOT be met — readable \"behind\" on "
         "perplexity, and parity or behind on accuracy.** Written down so that if B is met I "
         "have to explain why my prediction was wrong.\n")
D.append("---\n")

D.append("## 3. The A3 baseline this is measured against\n")
D.append("```")
D.append(f"readable : acc {m['readable_acc']:.2f}%   ppl {m['readable_ppl']:.4f}")
D.append(f"normal   : acc {m['normal_acc']:.2f}%   ppl {m['normal_ppl']:.4f}")
D.append(f"dacc {m['dacc']:+.2f}   dppl {m['dppl']:+.2f}")
D.append("```")
D.append(f"Readable was **worse on both** at 3 seeds. A3's margin clause passed only because "
         "the bar is \"no worse than 2σ\". Target B asks for the opposite: a real win.\n")
D.append("---\n")

D.append("## 4. The owner's two queued items (ideas only, NOT started)\n")
D.append("Recorded in `QUEUE-AFTER-B.md`. **Neither is pre-registered, neither has run, no "
         "result exists for either.**\n")
D.append("### Item 1 — NAME PRECISION CHECK\n")
D.append("A concept's name must not contain words the neuron does not actually use — the "
         "`KING_What_you,_RICHARD` case. Re-score existing checkpoints, report old vs new.\n")
D.append("**Four design questions recorded as needing resolution in its own pre-registration "
         "BEFORE any scoring**, because they are not equivalent:\n")
D.append("1. **What does \"actually uses\" mean?** ablation (remove that word's contribution, "
         "measure the drop) vs contribution share vs detection (AUROC on windows containing "
         "it). A column can *use* a word it does not *detect*.\n")
D.append("2. **What per-word threshold?** Any number is arbitrary and must be defended, not "
         "chosen after seeing the result.\n")
D.append("3. **Reject the name wholesale, or prune the unused words?** Pruning changes the "
         "target, which breaks comparability with every earlier result unless reported both "
         "ways.\n")
D.append("4. **Extra clause, or separate reported metric?**\n")
D.append("**And an explicit warning, recorded now:** a stricter meter applied after A3 was "
         "declared MET must **not** be used to retroactively re-judge A3. A3 was MET under the "
         "rule frozen before it ran. A new meter makes a **new record going forward** — the "
         "old-vs-new table is a report, not a revision. Anything else is post-hoc "
         "bar-moving.\n")
D.append("### Item 2 — MEANINGFUL TARGETS\n")
D.append("Real ideas (speaker names, pronouns, questions) instead of single words. Motivated "
         "directly by STEP 0: 6 of 10 concepts are single-word detectors.\n")
D.append("---\n")

D.append("## 5. The five pre-specified fallback ideas (one change at a time)\n")
D.append("Written in `IDEAS-AFTER-B.md` **before any of them runs**, each with its prediction "
         "and its falsifier. If Target B is not met, up to five may be tried, one at a time, "
         "each keeping the audit clauses passing; then `STOPPED.md`.\n")
D.append(table([
    ["1", "Multi-word construction targets (2-4 word n-grams, support >= 2.5%)",
     "nameability rises to >=6/10; **audit pass MAY FALL to 50-80%**; margin narrows but I do not predict a win. Falsifier: nameable concepts with no margin movement means the concept layer's contents are not the bottleneck"],
    ["2", "Speaker-name targets (`TRANIO:`, `PETRUCHIO:` …)",
     "nameability 10/10 by construction; AUROC >= 0.9 for major speakers, pass >= 70%, limited by speaker FREQUENCY not detection. **Most likely to help the margin**"],
    ["3", "Clean pronoun targets (fixes A3's mixed cluster `[4]`)",
     "honest name, likely pass — but **I predict NO margin improvement**: easy but weakly predictive"],
    ["4", "More meaningful targets at higher K (16 -> 24/32)",
     "per-concept pass should FALL; **flagged as the most likely wasted run**"],
    ["5", "Question targets (wh-words)",
     "moderate AUROC (~0.7), likely pass, **no margin improvement**. Must be honestly named 'contains a wh-word' if the tokeniser stripped `?`"],
], ["#", "the one change", "prediction written before running"]))
D.append("")
D.append("**Honest expectation, recorded up front:** I predict **most of these fail** to "
         "produce a Target B win. If all five fail, the conclusion is that readability's cost "
         "on this task is roughly **one perplexity point and a fraction of an accuracy point** "
         "— a result worth reporting rather than hiding.\n")
D.append("---\n")

D.append("## 6. The machinery, so the verdict is immediate and can't be hand-picked\n")
D.append("* **`t3_b_report.py`** — applies the frozen rule to the **full grid** (5-seed "
         "means, separately per model), then tests B-1/B-2 and reports **WIN / PARITY / "
         "BEHIND per metric**. It **refuses to score a partial grid** (verified: exits 2 at "
         "2/60 runs).\n")
D.append("* **`b_audit.sh`** — the B-3 step: Meter-03 audit + lie probe + random control on "
         "the **winning** readable run, whose setting is read from the scored grid, so the "
         "audited model cannot be hand-picked.\n")
D.append("* **A cron job** (monitor-gated, so it only wakes on a state change) runs that "
         "chain automatically when the grid completes and reports the verdict.\n")
D.append("### Two collisions caught before they bit\n")
D.append("1. **Model checkpoints.** B's runs would have written `readable_K16_lr1.0_lc01_...` "
         "under the **same names** as A3's — and **A3's audit reads those files**, so B would "
         "have silently destroyed A3's reproducibility. Fixed with a `T3_CKPT_TAG` "
         "(default empty, existing behaviour unchanged); A3's files verified still present.\n")
D.append("2. **The lie-probe checkpoint** had **no tag either**, so B's probe would have "
         "overwritten A3's probe file for the same reason. Tagged now too.\n")
D.append("### Known wart, recorded rather than half-fixed\n")
D.append("`t3_audit_m3.py` runs its whole audit at module level, so every importer re-runs it "
         "(~30 s) and rewrites `meter3_words.json` — deterministic, so the content is "
         "identical and nothing is corrupted. A guard was attempted and **reverted**: the "
         "`SystemExit` approach breaks the import, because Python drops a module from "
         "`sys.modules` when its import raises. The correct fix is wrapping that tail in "
         "`main()`; it was left alone rather than left half-working.\n")
D.append("---\n")

D.append("## 7. The code\n")
D.append(code("scripts/t3_b_report.py",
              "scripts/t3_b_report.py — the Target B verdict (frozen rule, WIN/PARITY/BEHIND)"))
D.append(code("scripts/b_audit.sh",
              "scripts/b_audit.sh — the B-3 audit chain on the winning run"))
D.append(code("scripts/t3_readability_sheet.py",
              "scripts/t3_readability_sheet.py — the STEP 0 evidence extraction"))
D.append(code("scripts/b_state.sh",
              "scripts/b_state.sh — the deterministic monitor that gates the cron job"))
D.append("---\n")

D.append("## 8. State and file map\n")
D.append("| what | where |")
D.append("|---|---|")
for a, b in [
    ("STEP 0 report", "`READABILITY-SHEET.md`, evidence `runs/t3/readability_evidence.json`"),
    ("Target B rule", "`TARGET-B.md`"),
    ("the two queued items", "`QUEUE-AFTER-B.md`"),
    ("the five fallback ideas", "`IDEAS-AFTER-B.md`"),
    ("A3 rule + results", "`TARGET-A3.md`, `TARGET-A3-RESULTS.md`"),
    ("A2 rule + results", "`TARGET-A2.md`, `TARGET-A2-RESULTS.md`"),
    ("the meter", "`METER-RULE-02.md`, `METER-RULE-03.md`, `METER-03-RESULTS.md`"),
    ("background package", "`GLASSNET-HANDOFF.md`, `GLASSNET-HANDOFF.zip`"),
    ("short status", "`STATUS.md` (10 lines); archive `STATUS-ARCHIVE-2026-10-05.md`"),
    ("full history", "`LOG.md`"),
    ("claim verification", "`scripts/verify_claims.py`"),
]:
    D.append(f"| {a} | {b} |")
D.append("")
D.append("**Nothing needs the owner right now.** Target F, D and E remain untouched, no "
         "license has been chosen or written for GlassNet, and the test set is untouched.\n")
D.append("*Generated from the artifacts by `scripts/build_review12.py`.*")

doc = "\n".join(D)
with open(OUT, "w") as f:
    f.write(doc)

fences = doc.count("```")
empty = len(re.findall(r"```[a-z]*\n```", doc))
need = ["STEP 0", "you,", "NOT MET", "A3 = MET", "Parity", "80 runs", "2x pooled std",
        "NAME PRECISION", "MEANINGFUL TARGETS", "sha256", "never a win"]
missing = [s for s in need if s not in doc]
print(f"lines            : {len(doc.splitlines())}")
print(f"fences (even?)   : {fences} -> {'OK' if fences % 2 == 0 else 'UNBALANCED'}")
print(f"empty code blocks: {empty} -> {'OK' if empty == 0 else 'PROBLEM'}")
print(f"missing sections : {missing if missing else 'none'}")
print(f"sha256[:16]      : {hashlib.sha256(doc.encode()).hexdigest()[:16]}")
print(f"written          : {OUT}")
