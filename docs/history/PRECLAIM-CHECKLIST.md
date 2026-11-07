# PRE-CLAIM CHECKLIST (adopted 2026-10-16 from the evaluation-standards research)

Run this BEFORE telling the maintainer any number, and before any release. Each item says WHY and what we
currently do or do not have. Sources: research/eval-standards-and-provenance.md (verified citations).

## The one thing the literature says is genuinely open (our strongest asset)
The 2026-10-16 sweep, searching specifically for a per-concept audit against an INDEPENDENT label with
an ENFORCED DROP, found: the audit half is established (Network Dissection 1704.05796 scores units
against independent labels; concept-existence metrics 2501.19271; robust purity metrics 2301.10367),
and the drop half exists only as thresholded post-hoc labelling, task-utility selection (2205.15480,
2304.06129), expert-driven removal+retrain (CBDebug 2509.21385), or debugging experiments. The UNION
-- independent-label audit + automatic enforced drop, reported before the claim -- is NOT FOUND.
Two independent research passes agree. So this stays our headline, stated with that exact scope.

## Items we must run before claiming a result
1. Mechanically computable concept definitions. [HAVE: every concept is a token-set formula in CONCEPTS.md]
2. Per-concept identifiability measured against a held-out labelled probe (F1/AUROC). [HAVE: the audit's AUROC + firing floor]
3. Per-concept existence/location on inputs (2501.19271). [PARTIAL: vision has it; talk does not]
4. Alignment gate against RIVAL concepts with a threshold declared in advance (1704.05796). [HAVE: the beats-all-rivals test]
5. Declare the DROP RULE in advance and report the audited set separately from the shipped set,
   including the removal+retrain cost. [HAVE for speech (noisy dropped, cost 0.21 pts); must be stated
   in advance for every new line, and the cost always reported]
6. Leakage check per concept (2106.13314, 2301.10367). [DO NOT HAVE as a standing check -- TO ADD]
7. Never report human plausibility as faithfulness; keep claims graded (2004.03685). [STANDING RULE]
8. Randomisation sanity checks of any explanation method (1810.03292). [PARTIAL: the lie-neuron control exists for vision; not standing]
9. Comprehensiveness/sufficiency with a top-k sweep AND a random-score reference row (1911.03429). [DO NOT HAVE -- and our talk-quality meter was built the same way, calibrated by a control, which is the pattern to copy]
10. >=3 SEEDS WITH ERROR BARS beside every number (2003.12206). [NOT SATISFIED TODAY: the small->big
    queue runs seed 0 only. Single-seed margins must be labelled as unproven, and the winning arm
    re-run at seeds 1 and 2 before any release claim.]
11. Frozen pre-registration file per experiment: clauses, predictions, drop rule, bars (2311.18807). [BEING ADDED: the small->big queue's bars are written into the runner before it starts]
12. Model card including training data and the audit verdict table (1810.03993). [TO WRITE for release]
13. Per-corpus provenance record + licence id with a self-reported/verified flag; an untagged dataset
    counts as UNSPECIFIED and is unusable (1803.09010, 2310.16787, HF docs). [HAVE: DATA_LICENSES.md]
14. Claims and limitations section matching the experiments, with a justification for every "no"
    (NeurIPS checklist). [STANDING RULE for the release document]

## What this changes right now
* The small->big queue's numbers will be single-seed: report them as directions, not wins.
* Before release: re-run the winning configuration at seeds 1 and 2, with error bars.
* Add the per-concept leakage check to the standing audit (item 6) -- it is the one missing meter that
  the literature says is where concept layers actually break (2106.13314).
* Our talk-quality meter already follows the pattern the field recommends (bar set by a control, not by
  taste) -- keep building meters that way.
