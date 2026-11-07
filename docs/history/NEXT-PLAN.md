# TOMORROW'S PLAN (recorded 2026-10-15 night, at the maintainer's instruction)

## His instruction, restated
1. Scan the whole record (the literature the research tracks gathered + every dataset/finding we have)
   and find ALL the problems: theirs AND ours. Solve ours, and where they have a known weakness,
   do it properly in ours instead of copying the weakness.
2. Alternate: move up to the bigger model and make it smart -> come back down to the smaller one and
   make THAT better -> repeat. Never abandon one for the other.
3. When BOTH the big and the small model pass 100% of their tests, THEN release: publish the code,
   upload the models, and push them toward being super smart while staying small.
4. Goal (a goal, not a claim): be better than everyone while being smaller.

## Rule for any "we beat X" statement
A claim of beating anything requires a head-to-head on the SAME data and the SAME metric, with the
compared values printed side by side. Otherwise it is a goal, and it gets labelled a goal.

## Concrete work order (each step ends in a binary gate)
1. Finish the two research tracks that were interrupted tonight (prior-art sweep of concept-bottleneck
   and interpretable-LM work; interpretability-evaluation + licensing/provenance standards). Their raw
   logs are in research/task-0.log and research/task-2.log; re-running costs ~4 minutes each.
2. Write the weakness table: for every prior-art method, what its known limitation is (e.g. concept
   leakage, incompleteness, black-box backbone), and what our answer to it is. Only claim an answer
   where we have a measurement, not an intention.
3. Fix the four recorded harness defects before any new training: matrix_summary restricted to one
   vocabulary family; logger appends history instead of overwriting; cross-corpus resume refusal
   already fixed; window-cap sampling already fixed.
4. Settle the open failure: fresh d512 merged run (NO resume) watched epoch by epoch against the resumed
   M0 that collapsed. The resume is the only remaining difference.
5. Then the up/down alternation: bigger model (d768 attention) -> evaluate -> smaller model (d256/d128)
   -> evaluate, on the SAME vocabulary family and the SAME clean val, so the numbers are comparable.
6. Only after 1-5: build the single self-contained release package (architecture, trainer, audit
   harness, all numbers, all licenses, the failure ledger) and pick the license with the maintainer.

## State as of shutdown
GPU idle (0%, 229 MiB, 15.4 W). No trainers, no matrix, no supervisor, no watch loop. Research tracks
1 and 3 completed and saved under research/. Track 0 completed (result pending delivery). Track 2 was
stopped on the maintainer's instruction; its partial log is saved.
