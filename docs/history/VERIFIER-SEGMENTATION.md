# The verifier could not see an arm that predated the marker convention - fixed, and it found why the 11.04% appendix has no checkpoints

2026-11-07 15:2x IST. `lens/scripts/verify_wall_ab_ckpt.py`.

## What was wrong

The verifier split a log into arms at `----- off seed N` markers. The seed-0 200k run in
`lens/runs/DATASCALE-HONEST.txt` - the arm behind the published **17.43%** - was launched *before* the runner
learned to write that marker, so its block had no marker at all. The tool therefore skipped it silently and
reported only the block appended after it, which made a stale 41-concept run look like the arm in question.

That is the same failure mode this project keeps finding: a tool that answers confidently about the wrong object.
It was also why the 17.43% checkpoint had to be checked by hand last session.

## The fix

1. **Leading unmarked segment.** If the first marker is not the first content line, everything before it is
   emitted as its own arm labelled `UNMARKED`, with the tag taken from the save line inside it. The arm name is
   only used for the wall A/B's pinned expectations, so an unmarked block can still be verified by its tag.
2. **No pinned expectation for non-wall arms.** The expectation table `{"off","on","r64"}` raised
   `KeyError: 'UNMARKED'`. Arms outside that table now report what the checkpoint actually says
   (`direct=<mode>/<bool>`, plus `seed ok` / `seed n/a` when the block carried no marker) instead of inventing a
   verdict.
3. **Accounting.** A verified arm with no pinned expectation was being counted as a problem because the check
   required `meta_ok is True`. A perfectly good checkpoint (sha and arrays identical, params matching) printed as
   a provenance failure. `seed ok` and `seed n/a` are now acceptable; `SEED-DIFF` is not.

## What it says now

    DATASCALE-HONEST.txt   arms verified: 2, problems: 1
    ...s1, s2, ep6, 500k, 2M, nogate, sgd, 20k-s2, WALL-REP (4), WALL-AB (6)   all 0 problems
    total across 11 logs: 20 arms, 1 problem

* **The published 17.43% arm verifies mechanically**: `datascale_honest_s0`, params 3,811,534 as announced,
  best = epoch 2, and `sha256(best) == sha256(ep2)` with every array identical - which is the same conclusion the
  hand check reached last session, now produced by a tool.
* **The one remaining problem is a stale appendix, and it should stay a problem.**
  `datascale_honest_s1` (the 11.04% / 693.45 ppl / 41-concept run on `data/chat/prepared`) announces
  **3,912,509** params, but the checkpoint that its save line points at has **3,811,534** - i.e. it resolves to the
  honest seed-1 run, not to itself. That run has **no checkpoints of its own**, which is exactly consistent with
  how it was already recorded (`docs/history/UNSCHEDULED-DATASCALE-RUN.md`: "recorded as a run, never as
  evidence"). The verifier now proves that mechanically instead of leaving it as prose.

## Why it matters

Provenance checking is only as good as the parser feeding it. An arm invisible to the parser was silently
excluded from every "N arms verified, 0 problems" line printed so far - including lines quoted in yesterday's
write-ups. Those statements were true of the arms the tool could see; the tool could not see one of them.
