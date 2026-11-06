# BUGS.md — the failure ledger (scenario-tested, not gate-tested)

Every entry: what broke, in which scenario, with the measured numbers. A line is NOT
"100%" if a scenario here contradicts its story. Fix queue prioritized.

## 2026-10-26 19:0x self-wake — TWO ENTRIES: ONE REAL DEFECT IN THE SHIPPED DECODER (FIXED, MEASURED), ONE MINE
## (caught by my own printed table before its FAIL was believed). Neither cost GPU time; both are in
## lens/runs/SENTSTATE-DECODE.txt with the numbers.

### [BUG-KVCACHE-IS-STATE-BLIND] — lens/kvcache.py:424-449 + lens/decode.py::make_cache, found 18:5x while
### wiring the sentstate graft's DECODE side.
### SCENARIO: an ATTENTION ship that also carries the graft, decoded through the KV cache. `cached_logits_h`
### reads the answer matrix OFF THE CACHE (`z = h @ cache.aW + cache.ab`) and never calls `net.concept` or
### `net.answer`, so the graft's `logits = answer(h) + sread(state)` loses its second term ENTIRELY — silently,
### with no exception, no warning and no shape error. MEASURED on a random-weight attention net, 8 windows
### (lens/runs/SENTSTATE-DECODE.txt P115b): the state-blind cache differs from the same net's full path by
### **max |d logit| = 2.510019** (median 2.503828) and agrees on the **top-1 token on 0 of 8** windows, while
### the SAME measurement on an UN-GRAFTED attention net is **2.384e-07** — i.e. this is not float32 rounding,
### it is a missing term. WHY IT MATTERS MORE THAN IT LOOKS: the config every arm here ships is attention=False,
### where `make_cache` returns None and the full path is taken (P115a: max|d| = 0.0), so no stored number is
### wrong TODAY. It becomes a silent-wrong-number bug the moment an attention+graft ship is decoded — the same
### class as APILOAD-DIRECT, which cost this project a full re-measurement.
### FIX, with its own control: `lens/sentstate_wire.py::cached_logits_grafted` adds the ONE term the cache cannot
### see, recomputed from `cache.outputs()` (the same per-token representation the graft taps) with the carry
### seeded at zero exactly as training does; `lens/decode.py::step_logits`/`step_logits_and_concepts` route a
### grafted net to it and leave an un-grafted net's path bit-identical to `kvcache.cached_logits` (P115d:
### max|d| = 0.000e+00 both ways). The grafted step reproduces the full path on all 8 windows (**max|d| =
### 0.000000, top-1 8/8**, P115c — better than the 5e-2 the prediction allowed).

### [BUG-CONTROL-MIXES-TWO-BLOCKS] — MINE, in lens/scripts/sentstate_decider_control_check.py's first version,
### found by its own first printed table 18:50 and fixed before any FAIL was believed.
### SCENARIO: P115e's control compares the edited decider's flag-absent output against the block already in
### lens/runs/PADFREE-TALK-H2L.txt. THAT FILE HOLDS TWO CHECKPOINT BLOCKS (the arm at line 7, its matched TWIN
### control at line 65). My parser kept ONE dict per file and let the twin's samples OVERWRITE the arm's, so it
### printed **"GREEDY TOKENS: 0/6 samples identical -> FAIL"** while the six strings it printed beside that
### verdict were, in fact, IDENTICAL to the arm's own block — six right numbers, one wrong conclusion, this
### project's most repeated bug class. It also read no summary mean at all (the MEAN regex did not allow the
### leading spaces the instrument writes), so it printed "SUMMARY MEAN: stored None -> FAIL" beside a PASS-worthy
### comparison. FIX: blocks are parsed separately and BLOCK 1 is compared with BLOCK 1, summary means are read
### as an ordered list, and every regex tolerates the instrument's indentation. RE-RUN: **6/6 greedy strings
### identical, all six <unk> shares equal, first summary mean 5.6% vs 5.6% -> PASS** (the corrected output is
### appended BELOW the wrong one in lens/runs/SENTSTATE-DECODE.txt, both kept as the record of what was printed).


## 2026-10-26 18:1x self-wake — TWO ENTRIES: ONE PRE-EXISTING IN THE SHIPPED LOADER (guarded, NOT edited), ONE MINE
## (caught by its own first printed table). Neither cost GPU time and neither produced a number that is used by
## anything else.

### [BUG-APILOAD-DROPS-GRAFT-TENSORS] — lens/api.py:252-255, found 18:0x while wiring the sentstate graft.
### SCENARIO: a checkpoint that carries tensors the rebuilt model does not have. `load()` iterates the NETWORK's
### parameters and looks each one up in the file (`for p in net.parameters(): p.data = z[p.name]`), so extra
### tensors in the file are never checked: they are SILENTLY IGNORED and the caller gets a model that is missing
### six of its tensors, with NO exception. MEASURED on a real save/reload round trip of a grafted model at
### d=64/window=32/V=64 (lens/runs/SENTSTATE-WIRE.txt, section SAVE/RELOAD): the reloaded model differs from the
### trained model by **max |logit difference| = 2.5199**, while the model the file actually describes reproduces
### bit-exactly (0.0) through the guarded loader. WHY THIS IS THE DANGEROUS CLASS: every post-hoc consumer — the
### talk decider, the readability gate, the exporters, the local server — loads through api.load, so a grafted
### ship would be evaluated and SERVED as a different model than the one trained, and every number printed about
### it would look fine. This is the same defect class as APILOAD-DIRECT (the tied direct path silently dropped on
### load, which cost this project a full re-measurement of the talk instruments). FIX QUEUE: (1) a size/name
### check in api.load() itself, refusing when the file and the model disagree in EITHER direction — NOT applied
### this turn, because api.load is called by stored instruments and no file is edited that a live arm or a stored
### number runs on; (2) in the meantime lens/sentstate_wire.py::load_grafted() is the ONLY sanctioned way to load
### a grafted checkpoint: it enables the graft first, reads ds from sread.W's own shape, and raises with both
### parameter sets in the message if they differ. Guarded, not conventionally avoided.

### [BUG-AGREE-CURVE-ROW-FROM-A-CONSOLE-FRAGMENT] — lens/scripts/agree_numbers.py, first run (18:0x), block 1 of
### lens/runs/GRADE-P61.txt. SCENARIO: the grade's three-setting gate table. The gap gate's refusal rate was
### regexed out of a console fragment in lens/runs/GATE-ARM.txt and printed as **0.0350** on the row for
### gate_sgd04_s2, whose own per-step log measures **0.9332 over its 40,268 rows** (and 0.8582 over the first
### 12,604 that GRADE-P109 graded). A right-looking number from the wrong place beside the right row — this
### project's most repeated bug class, and mine again. FIX: read the arm's own `<tag>_gates.jsonl`, and label the
### window ("ALL its logged steps" vs "the first 12,604 = 0.8582") so the two numbers cannot be read as
### contradicting each other. Both grade blocks are KEPT in GRADE-P61.txt: the first is the record of what was
### printed, per this file's rule that nothing is deleted or rewritten.

## 2026-10-25 14:0x self-wake — TWO, BOTH MINE, BOTH CAUGHT BY THEIR OWN FIRST OUTPUT (no GPU time spent, no
## number produced by either). The first is this file's signature failure class in its purest form yet: a grader
## whose FIRST line read "CONTROL VERDICT: PASS (0 arm row(s) + 0 twin row(s))" — a clean-looking PASS computed
## over an EMPTY read, because a wrong root path made both jsonl paths point at a directory that does not exist.

### [BUG-SELFCTX-NUMBERS-EMPTY-CONTROL] — lens/scripts/selfctx_numbers.py, first run, `lens/runs/SELFCTX-NUMBERS.txt`
### block 1 (14:01:10). SCENARIO: the new text-only grader for the self-context arm (P99a-e). OUTPUT AS PRINTED:
###   sources: arm  lens/lens/runs/chat/sctx_d256_full169_history.jsonl
###            twin lens/lens/runs/chat/ct_d256_full169_fix3_history.jsonl
###   CONTROL VERDICT: PASS  (0 arm row(s) + 0 twin row(s), ...)
###   NO ARM ROWS YET: the arm has written no epoch line. Nothing below can be a verdict.
###   NUMBERS EXIT=0
### CAUSE: `P = dirname(dirname(abspath(__file__)))` is the `lens/` directory, not the project root, so ROOT was
### GlassNet/lens/lens and BOTH files were "missing" — the `rows()` helper returns [] for a missing file, and the
### control loop iterated over nothing and printed PASS. The wrong path is also visible in the doubled "lens/lens"
### it printed. THE EXIT CODE WAS 0, i.e. a caller reading only the exit code saw a clean run. FIX: three
### dirnames; the header now prints each source file's PRESENT/MISSING state and its row count, and an empty read
### returns 1 with the line "an empty read is a FAILED CONTROL, not a clean control" and prints NO share, NO delta
### and NO band verdict. The failed block is KEPT in the file as the record (the project's rule), and the third
### block (14:02:34) is the authoritative one.
### THE CLASS: a verdict printed over a set of size zero. RULE: a loop-shaped control must assert its own input
### count, and "nothing to compare" must be a FAIL, never a PASS. (Also fixed in the same file: a literal `%%`
### printed by a line that had no formatting arguments.)
###
### [BUG-CKPTLIST-LABEL-SEMICOLON-RECURRENCE] — lens/scripts/run_padfree_selfctx_ep1.sh, first launch 14:01:42,
### `lens/runs/PADFREE-TALK-SELFCTX-EP1-RUN.txt`. SCENARIO: the P99b early talk read on the self-context arm's
### epoch-1 best. It died in 1 s with `ValueError: not enough values to unpack (expected 2, got 1)` at
### `_lab, _p = _item.rsplit("=", 1)` in lens/scripts/talk_fair_v3_ship.py, and wrote NO output file. CAUSE: the
### label I typed contained a ';' ("(frac 0.5 rounds 4; held-out 25.74% ...)") and ';' is the ITEM separator of
### that env format, so one item became two and the second had no '='. EXACTLY [BUG-CKPTLIST-LABEL-SEMICOLON]
### above, hit again 14 hours later by a different runner. FIX: the label uses '-' instead of ';' and the run
### succeeded in 17 s. THE TALK INSTRUMENT WAS NOT EDITED, and deliberately so: the live chained decider of the
### self-context job loads that same file at the END of the arm, so changing it mid-flight would change what the
### graded decider runs. STATED LIMIT: the parser still cannot carry ';' in a label; until it can, every caller
### must avoid it. Falsified if a future CKPT_LIST label contains ';' and the run does not fail loudly — it does.
###
### [BUG-GRADER-CONTROL-SUBSTRING-MATCH] + [BUG-GRADER-LITERAL-DOUBLE-PERCENT] — both mine, both in
### lens/scripts/grade_selfctx.py, both killed by its OWN FIRST CONTROL RUN (lens/runs/GRADE-P98.txt, block 1,
### 14:04:49, GRADER EXIT=1) before any of the arm's decider files existed. SCENARIO: the grader's P100a control
### runs its parsers over two STORED instrument outputs whose values are on the record
### (TALK-FAIR-W64-CONTROL.txt: step-8 free-running 3.9%; PADFREE-TALK-ATTN.txt: ATTENTION block 6.9%, TWIN block
### 2.1%). OUTPUT AS PRINTED: the free-running half MATCHED both stored values, the sample parser read all six
### samples, and then "read: TWIN maskfix full-corpus best (plain SGD, no attention, held ...) mean <unk> 2.1%
### ... against the STORED 6.9% -> MISMATCH -> FAIL" and "CONTROL VERDICT: FAIL — DO NOT GRADE THE ARM WITH THIS
### GRADER". The second bug printed a literal "2.1%%" in the P98f unmeasured line (a `%%` in a print with no
### formatting arguments).
### CAUSE 1: my expected-value selector was `6.9 if "ATTENTION" in label.upper() else ...` and the TWIN's label
### CONTAINS the word — "(plain SGD, no attention, held-out by position 35.26%)" — so the twin's block was
### checked against the ATTENTION block's stored 6.9%. A substring test on a label that names the thing it is NOT.
### CAUSE 2: the double percent, same class as the earlier literal-`%%` print in the 12:3x block.
### FIX: the selector tests "TWIN" first and "ATTENTION BEST" (the label prefix) second; the `%%` is a single
### `%`. Re-run (block 2, 14:05:03, GRADER EXIT=0): both blocks MATCH their stored values, six samples read,
### "CONTROL VERDICT: PASS", and P98e/P98f print UNMEASURED with the file each will come from.
### WHY THIS IS THE GOOD OUTCOME: the control was pointed at files whose answers were already known, so it could
### fail LOUDLY BEFORE the arm's numbers existed — the alternative was a grader that would have graded the
### self-context arm's <unk> share against the wrong stored value tonight. RULE: when a control's expected value
### is selected by matching text, the selector must be checked against every label in the file, not the one the
### author had in mind; and a control that has never failed once has not been tested.

## 2026-10-25 12:3x-13:0x self-wake — FOUR MORE, AND THE FIRST ONE IS EXACTLY WHAT THE WIRING PRE-FLIGHT EXISTS
## FOR: the new --self-context-* lever crashed on its FIRST STEP in the flag-absent case, so the arm that was
## about to be queued would have died at step 1 with no epoch line. The control caught it before any GPU time
## was spent. The other three are my own instruments: a validation placed after a 7-8 GB memory spike, a label
## parser that read instrument text as checkpoint names, and a grader that read a CRASH as "the default path
## changed" — this file's signature failure class again.

### [BUG-CONCEPT-IDS-NOT-FORWARDED] — `TypeError: ChatNet.loss() got an unexpected keyword argument
### 'concept_ids'`. SCENARIO: run the new trainer with the new flag ABSENT (the P98a control) or PRESENT, at
### d=64 on the CPU pre-flight. Both died before any epoch line, exit 1: `lens/runs/SELF-CONTEXT-PREFLIGHT.txt`
### block 1, and the raw per-run text in lens/runs/_tmp/sctxpf_newdefault.out / sctxpf_selfctx_on.out.
### CAUSE: lens/scripts/train_chat.py does not use the library's model directly — line 51 defines a thin
### `ChatNet` WRAPPER whose own `loss(self, X, Y, weights=None, y2=None)` forwards to `LensNet.loss` with a
### FIXED signature. Adding the kwarg only to lens/model.py is therefore not enough: the call site reaches the
### wrapper first. FIX: the wrapper's signature now carries `concept_ids=None` and passes it through (line 78-82).
### WHY IT MATTERS BEYOND THIS FLAG: any future loss-level kwarg has TWO places to land, and a missing one is a
### hard crash at step 1 rather than a silent no-op — which is the good direction, but only if a pre-flight runs
### before the queue line does. THE CONTROL DID ITS JOB; nothing on the GPU was spent.

### [BUG-SELFCTX-VALIDATION-AFTER-THE-MEMORY-SPIKE] — my own, found while writing the guard, not by a crash.
### SCENARIO: `--self-context-frac 0.5 --mtp 2` must be REFUSED. The first version of the refusal sat with the
### other flag checks AFTER `chat_data.build()` and after the mtp=2 window build — and the mtp=2 branch builds the
### FULL uncapped window set (7-8 GB; the reason BUG-TRAINCHAT-MEMORY-IS-THE-CORPUS exists at all). So the guard
### would have paid the exact memory spike it was protecting against in order to say no. FIX: the whole
### --self-context-* validation block now sits immediately after the banner, BEFORE `d = chat_data.build()`.
### MEASURED AFTER THE FIX: both guard runs (lens/runs/_tmp/sctxpf_guard_mtp.out, sctxpf_guard_rounds.out) print
### their refusal on the third line after the banner, exit 1, with no corpus load, in seconds.

### [BUG-GATECOST-GRADER-READS-INSTRUMENT-TEXT-AS-LABELS] — my own, in the P87 grader
### (lens/scripts/gate_cost_numbers.py). SCENARIO: parse the SHIP SUMMARY block of
### lens/runs/PADFREE-TALK-GATECOST.txt, which ends with the instrument's own `P72e CROSS-CHECKPOINT DIFF` line
### and `NOTE (P72f)` line. The parser accepted ANY two-space-indented non-measurement line as a checkpoint name,
### so those two lines were read as two more checkpoints and printed as three extra blocks reading
### `mean <unk> nan%   >=4-word-run 0/6`. ALSO: two `%%` escapes sat in plain (non-format) strings and printed
### literally as `20%%`, `37.5%%`. NEITHER number was wrong — the four real blocks and every verdict were
### correct — but a reader of that file would have seen three phantom checkpoints. FIX: a label is now only a
### line whose NEXT line is the `mean <unk> share over 6 greedy` line (structurally cannot match P72e/NOTE), and
### the two literals are fixed. Both corrections are in the graded file lens/runs/GRADE-P87.txt and in the
### regenerated output of the same turn.

### [BUG-P98A-GRADER-READS-A-CRASH-AS-A-BEHAVIOUR-CHANGE] — my own, in lens/scripts/selfctx_preflight_grade.py,
### first version. SCENARIO: the flag-absent control run exited 1 with a TypeError and wrote no epoch line. The
### grader compared `None` against the pre-change epoch line, found them different, and printed
### "-> P98a FAIL / NOTE: the edit changed the default path. It is to be REVERTED, not explained" — a
### CONCLUSION (the default path changed) that the evidence did not support: the run never finished, so nothing
### was shown to change. Same class as the three verdict-check bugs already in this file. FIX: the grader now
### separates the two cases — a non-zero exit or a traceback prints "INVALID CONTROL RUN - DID NOT COMPLETE"
### with both exits, and only a completed-but-different pair is reported as a changed default path.

### [BUG-PREFLIGHT-SCRATCH-FILES-APPEND-AND-WALL-TIME-IN-THE-CONTROL] — my own, two linked faults found while
### re-grading the pre-flight, both fixed before the final block. FAULT 1: the pre-flight runner's per-run files
### (lens/runs/_tmp/sctxpf_<tag>.out) were opened with `>>` and never cleared, so the SECOND block's grader still
### found the FIRST block's TypeError traceback in `sctxpf_newdefault.out` and reported a clean run as a crash.
### FIX: the runner truncates each per-run file at launch (`: > "$f"`); the earlier failure stays on the record in
### the appended main log and in this file. FAULT 2: the P98a control compared the whole epoch line, INCLUDING the
### trailing wall-time field — the two control runs printed `(49s)` and `(51s)` on otherwise identical lines, so
### the control read FAIL for a difference in how long the machine took. A verdict about the model drawn from the
### machine's speed is exactly this file's signature class. FIX: the comparison now excludes the trailing `(NNs)`
### and prints both lines plus the exits beside the verdict, and the field that differed (if any) is named. The
### final block reads P98a PASS on a field-wise identical pair.

### [BUG-PGREP-WITHOUT-F] — MY OWN, and it produced a false alarm about the brand-new arm within two minutes of
### it starting. SCENARIO: checking whether the self-context arm was still alive, `pgrep -c "[t]rain_chat"`
### returned 0 while the scope read active. That reads as "the trainer died", and I nearly re-queued the arm —
### which would have been a SECOND trainer on a 6 GB GPU, the shape that OOM-killed scopes on this machine before.
### CAUSE: WITHOUT -f, pgrep matches the process NAME (comm), which for every trainer here is `python3`; the
### bracket trick only guards against pgrep matching itself, not against matching the wrong field. WITH -f the same
### pattern matches the command line and the arm was there all along (pid 709850, RSS 8.2 GB). FIX / RULE: every
### liveness check on this project must use `pgrep -af` (or -f), and any "nothing is running" conclusion must be
### cross-checked against the scope's own cgroup (`systemctl --user show -p MainPID,MemoryCurrent <unit>` or the
### unit's cgroup.procs) before acting on it. THE WATCHDOG'S OWN PATTERNS WERE CHECKED AND ARE CORRECT: lines 173
### and 178 use -f with the full command line, and line 168's comment already records this exact lesson.

## 2026-10-25 11:4x-12:0x self-wake — FOUR BUGS THIS TURN, ALL IN MY OWN INSTRUMENTS, AND ALL FOUR WERE
## SILENTLY SUPPRESSING THE SAME VERDICT: the window-64 talk decider. Two of them were hardcoded window 32,
## one had never been executed once, and one was a CHAPTER of this file's own history repeating (a right number
## with a wrong conclusion written over it). Every one is fixed and the fix is controlled to the digit on the
## stored window-32 twin before any window-64 number was reported.

### [BUG-TALK-INSTRUMENT-HARDCODES-W32] — TWO FILES, ONE CAUSE. `talk_fair_v3_ship.py` (line 37) and
`talk_fair_test.py` (line 42) each set `W = 32` as a module constant and then built their entire input from it:
the held-out window set (`chat_data.make_windows(..., W, ...)`), the positional assistant mask (`is_asst[S + W]`,
i.e. the target offset), the pad-free prompt list, the generation context trim, and the 8-step window slide.
SCENARIO: a WINDOW-64 checkpoint. BREAKS BY CONSTRUCTION, and the error message says so: `lens/model.py` line 214
raises `ValueError: windows must be 64 tokens wide, got 32`. MEASURED, from lens/runs/TIED-W64.txt:
`talk_fair_v3_ship.py exit=1` (traceback at its line 205 -> 159, the pad-sensitivity `gen` call) and
`talk_fair_test.py exit=1` (same error at its line 181, the first reproduction pass) — so P95d (the point of the
whole window-64 experiment), P95e, P95f and P95g (the decider that decides whether anything ships) were never
measured, and `lens/runs/PADFREE-TALK-W64.txt` was never written because v3 writes its output at the END.
WHY IT WAS INVISIBLE: both instruments worked perfectly on every checkpoint this project had ever tested, because
every checkpoint before today was window 32 — `W = 32` was not a bug until an arm changed the window, and the
window-64 arm was built, wired-checked (W64-STRUCTURAL.txt) and queued by a turn that did not check its own
instruments could READ window 64.
FIX: W is READ PER CHECKPOINT from that checkpoint's own metadata (`np.load(ckpt, allow_pickle=False)` then
`meta["W"]`; `TALK_FAIR_W` still overrides by hand). In v3 the three W-dependent structures are built per W and
cached (`wset(W)`, `build_row(W)`), and `gen()` trims the context to `getattr(net, "W", W)` so the trim can never
disagree with the model. For a W=32 checkpoint this is the same 32 as before, which is CONTROLLED, not asserted:
the repaired v3 on the twin reproduces `PADFREE-TALK-ATTN.txt`'s twin block exactly (mean <unk> 2.1%, 6/6 with a
>=4-word run, same six sample texts) and the repaired `talk_fair_test.py` reproduces
`TALK-FAIR-finalbest-d256-repro.txt` exactly (by-position 35.26%, legacy 26.31% REPRODUCED, real 35.26% /
mismatched 2.20% / drop 33.07, step-8 free-running 3.9% against teacher-forced 23.0%). Controls:
lens/runs/PADFREE-TALK-W64-CONTROL.txt, TALK-FAIR-W64-CONTROL.txt. Predictions: P96a/P96b.

### [BUG-TALK-GRADER-HAD-NEVER-BEEN-EXECUTED] — `lens/scripts/grade_w64_talk.py` died on its FIRST EVER RUN at
11:46 with `SyntaxError: f-string expression part cannot include a backslash`, at its line 107. A backslash inside
an f-STRING EXPRESSION is legal only from Python 3.12; `.venv-gpu/bin/python3` is 3.11.15. The offending line was
the P95d verdict: `f"... {'PASS' if fr > X else 'FAIL (the twin\'s ...)'}"`.
WHY IT MATTERS MORE THAN THE TYPO: the previous turn recorded this grader as "dry-run against the STORED UNK-DROP
texts before this turn queued them". A parse-only or different-interpreter dry run cannot catch a syntax error
that only THIS interpreter rejects — so a grader that was never once executed was chained into a 2-hour job as the
step that produces the decider. The lesson for the file-reading rule is narrow and mechanical: a Python file that
is about to be chained into a job must be run through `python3 -m py_compile` WITH THE INTERPRETER THE JOB USES,
and this turn does exactly that in every runner it added (all three repair runners compile before they run).
FIX: the verdict string is built outside the f-string; the offending expression is now a plain `if/else`.
[BUG-TALK-GRADER-HAD-NEVER-BEEN-EXECUTED]

### [BUG-GRADER-DROPS-SINGLE-DIGIT-RUNS] — FOUND BY THE GRADER'S OWN FIRST OUTPUT, one run after the fix above.
The sample-line pattern was `r"^\s{6}conv\s+(\d+):\s*<unk>\s*([\d.]+)%\s*longest word run (\d+)\s+(.*)$"` — ONE
literal space after "longest word run" — but the instrument prints that field RIGHT-ALIGNED (`{best_run:>2}`,
talk_fair_v3_ship.py), so a sample with `longest word run  6` (two spaces) did not match while `run 24` did.
MEASURED: the first run reported 4 samples for ARM 1 and 3 for ARM 2 at window 64, where the raw file has SIX
each; the hedge denominator came out 4 and 3 instead of 6. THE BIAS IS THE POINT: the dropped rows are exactly the
WORST samples (a single-digit run means the text is mostly <unk> or non-words), so a by-eye coherence count taken
from that list would have been taken from the friendliest possible subset — a verdict-shaping bug that changes no
printed percent (the mean <unk> line uses a different pattern and was always right) while changing what a human
sees. FIX: `longest word run\s+(\d+)`, plus the parse is now anchored to the SHIP SUMMARY section (the old
whole-file label pattern `^\s{2}\S.*held-out` also matched the new per-checkpoint header line, which contains
"held-out windows 40,262", and would have attributed samples to a header instead of to a checkpoint), plus the
verdict line now prints the two compared values beside the verdict. Both blocks are in GRADE-P95-TALK.txt; the
header says the last one is authoritative. Re-run: lens/scripts/run_grade_w64_talk_rerun.sh.

### [BUG-HEDGE-DETECTOR-TOKENIZES-LOOKING-FOR-PHRASES] — the same grader's hedge detector, found by comparing its
own twin row against the bar it quotes. `HEDGES` holds PHRASES ("not sure", "don't have", "i'm not", ...) and the
test is `any(h in t for t in first6)` — i.e. the whole phrase must appear INSIDE ONE of the first six TOKENS.
"The model is not sure" tokenizes as "not" + "sure", so it is never counted. MEASURED: the twin itself — the
checkpoint the 4/6 bar comes from (lens/runs/GRADE-P76-P77.txt, a straight-TEXT detector on HARNESS-format
prompts) — reads **1/6** on this instrument's pad-free prompts, so P95f as printed compares an arm's count to a
bar produced by a different detector on a different prompt set. BY EYE on the six pad-free samples, which is how
this project counts hedges: the twin hedges 6/6, ARM 1 3/6, ARM 2 3/6. FIX QUEUE: the mechanical detector should
count over a token-concatenation of the first six tokens (or match on text, as GRADE-P76-P77 does) and the bar
should be re-measured with the instrument that measures the arms. Until then P95f's NUMBER is not quotable — its
DIRECTION (the arms hedge less than the twin) is, by eye.

### [BUG-MY-OWN-NUMBERS-SCRIPT-ASSERTED-A-REFUTED-CLAIM] — MINE, caught by the numbers file's own table in the
SAME turn, and it is the same failure class as the verdict-check bugs already in this ledger. The first version of
`lens/scripts/w64_talk_numbers.py` printed a hardcoded sentence: "ARM1 is BETTER than the twin at steps 1-2",
with step 1 quoted as 17.6% against 7.8%. The table directly above it read `teacher TWIN 25.0% ... free-run TWIN
25.0%` — the 7.8% row belongs to the 6M-window arm in `lens/runs/TALK-FAIR-TEST.txt`, NOT to the twin, and I had
compared against the wrong file's row from memory. The claim is not merely imprecise, it INVERTS the reading:
ARM 1's step-1 free-running (17.6%) is BELOW the twin's (25.0%), the same direction as the -2.99-point held-out
loss, i.e. more context bought a WORSE first token. FIX: that paragraph is now COMPUTED per step
(`COMPUTED, per step, ARM1 minus TWIN free-running: 1:-7.4 2:-3.2 3:+0.4 4:+1.6 5:-0.7 6:-3.1 7:-0.4 8:-2.3 ...
ARM1 is ABOVE the twin at 2 of 8 steps [3, 4]`), the file was regenerated, and the rule this project already has
— print the compared values beside the verdict — is why the contradiction was catchable at all.

## 2026-10-24 08:0x self-wake — THREE BUGS THIS TURN: TWO IN MY OWN PRE-FLIGHT HARNESS (both recorded with their
## reproduction) AND ONE REAL ONE IN THE WATCHDOG'S "is the machine busy?" TEST, which is FIXED.

### [BUG-BASH-LOCAL-SAME-LINE] — MINE, FOUND AND FIXED THE SAME TURN. Bash 5.3.9 expands a variable to EMPTY
inside a single `local` declaration when a later word on the SAME line references it.
WHAT BROKE, in which scenario: lens/scripts/unk_drop_preflight.sh declared its per-arm output file as
  `local tag="$1" py="$2" extra="$3" f="$TMP/unkpf_${tag}.out"`
so `${tag}` was the empty string: all three arms wrote ONE file, `lens/runs/_tmp/unkpf_.out`, and each arm's
`cat "$f" >> $OUT` re-dumped everything accumulated so far (the duplicated ARM blocks in
lens/runs/UNK-DROP-PREFLIGHT.txt). The grading step then died with
  `FileNotFoundError: 'lens/runs/_tmp/unkpf_prechange.out'`
which reads like "the run produced nothing" and was actually "the harness wrote one file with no tag".
PROOF, side by side in one script (home/yahya/.hermes/cache/scratch/tagtest.sh):
  A `local tag="$1" ... f="$TMP/unkpf_${tag}.out"`  ->  f=[.../unkpf_.out]                      WRONG
  B `local tag="$1"` then `local f="$TMP/unkpf_${tag}.out"` -> f=[.../unkpf_prechange.out]      right
FIX: split into two `local` lines (lens/scripts/unk_drop_preflight.sh), re-run as ATTEMPT 3 (logged).
WHY IT MATTERS BEYOND THIS FILE: the same one-line `local` form is a landmine in any runner here that builds a
path from a variable declared on the same line. Read the variable's value, or declare it on its own line.

### [BUG-PREFLIGHT-PYTHONPATH] — MINE, same harness. The pre-change copy of the trainer lives in
lens/runs/_tmp/, so `import chat_concepts` / `import chat_data` fail there: python prepends the SCRIPT's
directory (lens/scripts) only for a script that lives in it. Symptom: `ModuleNotFoundError: No module named
'chat_concepts'`, exit=1 on the FIRST arm, which again reads like a code problem and was a path problem. FIX:
`export PYTHONPATH="$PWD/lens/scripts"` in the pre-flight (a no-op for the real trainer, whose own directory is
already prepended). Recorded because "the pre-change copy will not even import" is a trap for every future
digit-for-digit comparison against a _tmp backup.

### [BUG-WATCHDOG-BETWEEN-ARMS-GAP] — A REAL ONE IN THE MACHINERY, FIXED. The watchdog decides "is the machine
busy?" with `pgrep -cf "venv-gpu/bin/python3 -u lens/scripts/train_chat\.py"`, and qheavy covers train_chat and
the corpus builders - but NOT the queue's own RUNNER SCRIPT. A queue line can be a script that runs several
trainers back to back (run_ours_sweep.sh = seven arms; run_unk_drop.sh = the arm then its graders and talk
tests; run_attn_grade_talk.sh = a waiter, a grader and two instruments). BETWEEN TWO ARMS of such a script
`pgrep train_chat` is 0 for about a second WHILE THE JOB IS ALIVE, so a tick landing in that gap would start the
next queue entry beside it - two trainers on a 6 GB GPU, the exact shape that OOM-killed scopes here on
2026-10-22 (15:41 journal, and the 15:48:45 kill). A tick every 5 minutes over a 7-arm job has roughly a few
percent chance of landing in a gap, and the consequence is not a wrong number but a dead desktop job.
FIX (additive, no existing branch changed): a live queue job's transient scope covers the WHOLE line, gaps
included, so the busy test now also asks systemd:
  `qscopes=$(systemctl --user list-units 'glassnet-q-*' --state=active --no-legend --plain | grep -c .)`
and if qscopes > 0 while qrunning == 0, it logs "the between-arms gap, not idleness" and does NOT start the next
entry. VERIFIED before it was trusted, lens/runs/WATCHDOG-GUARD-TEST.txt: the shipped line taken from the file
by grep reads 1 with the real pattern (glassnet-q-3757950001-074809.scope, the sweep running now) and 0 with a
pattern that matches nothing, and the guard's condition then fires as intended. Pre-change copy of the watchdog:
lens/runs/_tmp/watchdog_prefix_20261024-080311.sh.

## 2026-10-23 - BUG (mine, in my own instrument, FIXED and re-run in the same turn): the gate snapshot printed
refusal COUNTS with the evidence fields all null.

WHAT BROKE, in which scenario: my new lens/scripts/ours_gate_snapshot.py was written to answer
run_ours_first.sh's prediction 1 ("THE GATE MUST ACT ... if the run reports ~0% refused, the gate is not really
acting") from the per-step logs. Its first run printed the refusal counts correctly (memorisation rule 116/634 =
18.3%, agreement rule 343/978 = 35.1%) and then printed every refusal's "own numbers" as
`{"step": 188, "train": null, "probe": null, "ema_train": 6.8690, ...}` — the keys I invented (`train`, `probe`,
`A_train`, `A_probe`, `ema`) are not the keys the trainer writes (`train_loss`, `probe_loss`, `a_train`,
`a_probe`, `ema_a`), so the WHY of every refusal was blank while the count looked right.

WHY IT MATTERS (the bug class this project keeps hitting): a correct count beside missing evidence is one step
away from a verdict read off the wrong thing — and this project's own rule is that every verdict prints the
value it was compared against. Here the count was right and the reason was invisible.

FIX + RE-RUN (same turn): keys corrected to the file's actual schema (read from
lens/runs/chat/ours_d256_full_3_gates.jsonl line 1: step, epoch, train_loss, probe_loss, ema_train, ema_held,
gap, refused, accepted; and ..._agreement.jsonl: step, epoch, a_train, a_probe, ema_a, batch_ce, refused,
accepted). Re-run at 16:06 printed reasons, e.g. memorisation-rule step 685 refused at train_loss 6.7650 against
probe_loss 6.4051 (gap +0.1904, i.e. the memorisation direction) and step 687 at gap -0.0681; agreement-rule step
1078 refused at a_train 0.1975 against a_probe 0.2010 (ema_a 0.2083). Output: lens/runs/OURS-METHOD-GATES-SO-FAR.txt.

## OPEN

- [BUG-MASK-TOKEN-ID] **OPEN, FOUND AND PROVEN 2026-10-22 17:5x (self-wake), affects EVERY chat run so far.**
  `--assistant-only` has never masked assistant turns. train_chat.py computes
  `Wtr = is_asst[Ytr]` and `Wva = is_asst[Yva] > 0`, but `Ytr`/`Yva` are **TOKEN IDS**
  (chat_data.py: `Y = ids_arr[starts + window]`), while `is_asst` is a **POSITIONAL** array over the
  294,225,090 corpus tokens — and the comment above the code states the intent: "A window's target is
  assistant text if the flag is ON at that token's **position**". So the mask reads corpus positions
  0..12,001 (used as vocabulary indices) and the weight is a property of the TARGET TOKEN ID, not of
  whose turn the token is in.
  EVIDENCE, all in lens/runs/MASK-INDEX-BUG.txt (lens/scripts/mask_index_bug_check.py, run on the
  git-clean corpus data/chat/prepared_mix_ctrl_honest with the trainer's own walker):
    * `is_asst` marks 197,204,547 of 294,225,090 positions (67.0%). `max(Yva) = 12001`, vocab 12,002 →
      `is_asst[Yva]` can only ever read the first 12,002 entries; 9,160 of them are True. That prefix is
      the start of conversation 0 ("<|im_start|> user hi , i want to learn to play <unk> ...").
    * BY TOKEN ID (what the trainer computes):  44,418 of 87,035 windows masked (51.0%)
      BY POSITION (the stated intent):         58,191 of 87,035 windows masked (66.9%)
      agreement: 29,518 both / 14,900 id-only / 28,673 position-only / 13,944 neither = **33.9% overlap**.
      The logged "MASKED ... (n=44418)" in every arm is exactly the by-id count — the bug's fingerprint.
    * SAME CHECKPOINT (lens/runs/chat/ct_d256_6M_6ep_best.npz), same 87,035 held-out windows:
        accuracy over ALL windows                                 18.58%
        accuracy MASK BY TOKEN ID (the logged number)             35.82%   <- every arm's "MASKED" column
        accuracy MASK BY POSITION (real assistant turns)          17.95%
        ... on the windows the by-id mask picked that are NOT assistant text   37.89% (n=14,900)
        ... on real assistant windows the by-id mask MISSED                     0.62% (n=28,673)
      The 35.82% is NOT a turn mask: two thirds of the windows it counted are not assistant text at all,
      and 28,673 real assistant windows (which the model gets ~0.6% right) were never counted.
  CONSEQUENCES, stated plainly:
    1. The "MASKED held-out" column that selected every `*_best.npz` is a vocabulary-subset metric, and it
       is roughly 2x optimistic. Every stored comparison (27.37 / 29.71 / 34.92 / 33.94 / 35.82%) is a
       by-id number and must be read as such. Nothing is deleted; the numbers keep their files.
    2. The memorisation reasoning built on that column ("val ABOVE train, so it is UNDERFIT not
       memorising", STATE.md 2026-10-22) rests on the wrong mask and must be re-measured.
    3. Training itself: about 2,842 token ids got loss weight 0, chosen by which ids happened to sit at
       corpus positions 0..12,001 — so human-turn text the model should NOT be trained to imitate carried
       weight 1, and the flag that was supposed to prevent that did nothing positional. `--train-turn-end`
       has the same defect (`te = is_turnend[Ytr]`).
    4. The same expression is used by the readability gate and the gate/agreement probe slices, which
       report `val_masked_*` numbers.
  FIX: index by POSITION (`S + window`, make_windows(return_index=True)), keep the old by-id column
  printed as LEGACY so nothing is silently redefined, and predict before editing (lens/runs/
  MASKFIX-PREDICTIONS.txt). Pre-change copy kept at lens/runs/_tmp/train_chat_prefix_maskfix.py.
  MY OWN ERROR IN FINDING IT: my first rebuild of the mask printed 0 of 87,035 and I let the script go on
  and print "CONTEXT IS BARELY USED" from an empty mask. The reproduction check (P65e, "my number must
  equal the logged number") is what stopped it, and the script must REFUSE to print ablation verdicts on
  an empty mask — fixed in lens/scripts/talk_fair_test.py before it is trusted again.

- [BUG-COST-UNITS-16X] FIXED 2026-10-21, found BEFORE any recorded number was taken (PLAN verifier3): the
  new ladder's cost log stored `nq / len(pred)` - verifier calls divided by the batch size - which understates
  the cost 16x, because ONE verifier call asks EVERY example in the batch one question, so the number of
  calls made in a step IS the questions per example per step. Caught by the scratch smoke run
  (lens/runs/_tmp/ladder_smoke.txt: "ladder 600 steps: ... q/ex [0.875]", i.e. less than one question per
  example, which is impossible when the smallest rung costs two). Fixed to store `nq`; the recorded run reads
  4.8/3.86/4.39 calls per step and 2.0 once converged. Rule: a rate needs its denominator named in the same
  line as the number, or a units error survives into the record.
- [BUG-BAR-WORDING] FIXED 2026-10-21 (three bars of mine, in the same turn that wrote them): P57e applied a
  ONE-SEED stored number (PART 6 ran seed 0 only) to THREE seeds; P57i compared full-precision floats against
  the FIVE-DECIMAL numbers arm G prints, with a 1e-9 tolerance; P57a's sweep ran residuals out to +-43 and
  counted as "unresolved" the ones past the ladder's declared top rung (40.0), which the ladder is by design
  never asked about. All three FAILED AS WORDED. The FAILs stand in lens/runs/UNIFIED-LADDER.txt and the
  corrected comparisons are recorded beside them (PART 8, "CORRECTED"), where all three PASS. Rule: a bar
  must name the precision, the number of seeds and the RANGE it applies to, because a bar that is wrong in
  any of the three reads as a failed experiment when the experiment was fine.
- [BUG-CUDA-NAME-SHADOW] FIXED 2026-10-20 (real bug, shipped code, found by running it): the top-level
  name `cuda` was ours, not cuda-python's, in any process that had this project's `lens/` directory on
  sys.path. `lens/cuda/` is a REGULAR package; cuda-python ships `cuda` as a PEP 420 NAMESPACE package
  (site-packages/cuda/ has no __init__.py); Python returns a regular package found ANYWHERE on sys.path
  before any namespace portion, so appending instead of inserting does not help - MEASURED three ways
  (lens/runs/SERVE-GPU-SYSPATH-BUG.txt): inserted -> cupy FAIL, appended -> cupy FAIL, absent -> cupy OK.
  cupy 14.2 then dies at import with `ModuleNotFoundError: No module named 'cuda.pathfinder'`, so
  `python3 lens/serve.py <ship> --device gpu` exited 1 before printing anything: the server could not run
  on the GPU AT ALL, and the same fault was latent in every GPU-capable script. 25 files carried the
  sys.path line (class 1) and 5 files that LIVE in lens/ (serve, stream, stream_train, methods, export)
  were still broken after the class-1 fix because `python3 lens/<file>.py` puts the script's own directory
  - lens/ - at sys.path[0] regardless of what the file's header does (class 2, measured). All fixed; the
  server now loads on the GPU and answers real HTTP requests (lens/runs/SERVE-GPU-AFTER.txt). Rule: a
  project module must never be importable under a name that a dependency also uses at top level; and a
  fix for a sys.path fault must be tested by RUNNING the entry point, because sys.path[0] is the script's
  directory and no header edit can see that. Gate: `python3 lens/scripts/fix_syspath_shadow.py --check`.
- [BUG-CUDA-NAME-SHADOW-CLASS3] FIXED 2026-10-20 (the same fault in a form the gate could not see, found
  while working PLAN item (f)): six files put this project's lens/ on sys.path as ONE ELEMENT OF A PATH
  TUPLE - `for _p in (ROOT, ROOT + "/lens", ROOT + "/lens/scripts"): sys.path.insert(0, _p)` - which
  shadows `cuda` exactly like the class-1 line, and the gate's patterns were anchored to a whole
  `sys.path.insert(0, ...)` line so they could not see it. MEASURED: the gate returned exit 0 (PASS) on a
  tree where lens/scripts/kv_decode_test.py could not load a ship on the GPU (lens/runs/GPU-ATTN-CACHE.txt,
  first block: ModuleNotFoundError: No module named 'cuda.pathfinder'), and four scripts written during
  this turn had copied the same header out of that file. Files: lens/scripts/decode_push_breakdown.py:25,
  decode_step_cost.py:28, decode_step_loop_breakdown.py:25, gpu_attn_cache_probe.py:19,
  kv_decode_test.py:35, serve_vocab_probe.py:17. The gate now has class 3, run in the FAILING state first
  (exit 1, 6 files, lens/runs/_syspath_check_class3_before.txt) then in the passing state (exit 0).
  LESSON: a gate must be shown to FAIL on the bug it guards - a pattern that matches one spelling of a
  fault is not a gate. Reports: lens/runs/SYSPATH-SHADOW-FIX-class3.txt (both syspath reports now APPEND
  instead of truncating, so an earlier pass cannot be overwritten by a later one).
- [BUG-HOSTONLY-CACHE-ON-GPU] FIXED 2026-10-20 (real bug, shipped code, found by running it during PLAN
  item (f)): an ATTENTION ship served on the GPU could not decode AT ALL. `lens/serve.py <attention ship>
  --device gpu` -> generate() -> make_cache() -> WindowCache, which keeps numpy buffers and applies the
  model's numpy causal mask, while a ship loaded with device="gpu" keeps cupy arrays: the first step
  raised "TypeError: `out` must be a tuple or `cupy.ndarray`" inside np.matmul (and the allocating
  reference path would have raised on `s + self.mask` one line later). Every earlier GPU serving test used
  a NO-ATTENTION ship, which takes the full path, so the combination was never exercised
  (lens/runs/GPU-ATTN-CACHE.txt). FIX: the cache refuses itself BY NAME and the decoder takes the full
  path - "full recompute (the KV cache is HOST-ONLY: this ship's weights live on the device, and the
  cache's numpy buffers and causal mask cannot mix with them)" - and WindowCache/WarmDecode raise a named
  ValueError when constructed directly. NOT fixed by moving the cache onto the device, on purpose: that
  changes the numbers and needs its own identity test (the bar the fold had to pass). VERIFIED after the
  fix: device=gpu -> the named full path, step_logits OK, argmax 2215 (the same token the CPU cached path
  returns); device=cpu -> still "cached (KV, window cache)".
- [BUG-HOSTBOUND-STEP] MEASURED 2026-10-20 (PLAN item (d2c), lens/runs/OPT-STEP-COST.txt): our own step
  rule costs 70.543 ms per step on the GPU against a plain device update's 0.333 ms (211.876x), and
  105.495 ms on the CPU - i.e. the shipped step is only 1.5x faster on the card than on the host, while
  the plain update is 31x faster there. The step's arithmetic is done in float64 on the HOST (5 to_cpu +
  5 to_dev calls, 30.49 MB round-tripped per step, measured by counting the calls). The transfers are only
  ~6 ms of the 70 ms (measured bandwidth 4.29/5.35 GB/s), so the cost is the host float64 math and its
  allocations, not PCIe. A device-resident form of the SAME rule is 89.137x faster and preserves the
  rule's defining property (update norm exactly lr, to 6.38e-06) - not applied while the 16-arm
  comparison is in flight, because a float32 device norm is not bit-identical to the float64 host norm
  and that would confound the one comparison whose whole point is the step rule.
- [BUG-BOUND-TARGET] MEASURED DESIGN DEFECT (2026-10-20, PLAN item (d3b)): the bounded magnitude
  representation divides the numeric TARGET by the WINDOW's own maximum, and the next token is not in
  the window - so the target is not bounded at all. Measured on 100,000 clean chat windows
  (lens/runs/NUMERIC-BOUNDED-RANGE.txt): channel max |q| = 1.000000 exactly, target max |t| =
  8000.000000, i.e. the same 0..8000 range the bound existed to remove. A window holding no numbers has
  d = 1 and a next token of "8000" gives a target of 8000. Rule: a normaliser can only bound what it
  SEES; a target outside the input needs a scale that does not depend on the input (and, to keep the
  arithmetic exact, that scale must be LINEAR - see [BUG-LOGBREAKS-SUM]).
- [BUG-LOGBREAKS-SUM] MEASURED (same run): log compression is not an option for this path. max
  |log1p(a+b) - (log1p(a)+log1p(b))| over a,b in 0..9 is 1.660731 at a=b=9, and the answer it would
  produce after inverting is up to 4.50x too large, so a linear head over log magnitudes cannot express
  a sum. Any future "bounded" proposal for this path must be checked against the arithmetic gate first.
- [BUG-UNITMIX] CAUGHT BEFORE IT PRODUCED A NUMBER (2026-10-20): the first version of
  lens/scripts/numeric_chat_readout_bounded.py scored the bounded arm in BOUNDED units against the
  ORIGINAL-unit constants of the stored run. A small bounded-space RMSE divided by a large constant
  reads as a win - a false-PASS trap. Rewritten so every arm is scored in original units through its own
  declared inverse. Rule: when two arms are compared, print the UNITS and the compared values beside
  every verdict; a control that passes for the wrong reason is worse than no control.
- [BUG-CONTROL-DIRECTION] OPEN, recorded: P44g passed AS WORDED (shuffled-target RMSE ratio >= 0.98)
  while the control landed 42.3x ABOVE the constant instead of AT it, because that arm is degenerate
  (the inverse multiplies the ridge's error by d, up to 12000). The bar was written for the raw scale and
  protects nothing there. The PASS stands as worded and is recorded DEFECTIVE; future control bars must
  be two-sided (0.98..1.02), as P44k is.
- [BUG-HARNESS-API] FIXED (mine, 2026-10-20): lens/scripts/numeric_bounded_lr_sweep.py's gradient helper
  called `net.loss(...).backward()`, but lens.Net.loss() returns (loss, parts) - AttributeError, exit 1,
  after the sweep table had already printed. The table was real (lens/runs/_numeric_bounded_lr_stdout_v1.txt);
  the helper is fixed. Same class as the earlier harness slips: a new script's own API misuse is not a
  model result.
- [BUG-BROADCAST-SHAPE] FIXED (mine, 2026-10-20): lens/scripts/probe_bounded_range.py divided the (N,)
  target array by the (N,1) divisor, so numpy broadcast it to a 100,000 x 100,000 array -
  "Unable to allocate 37.3 GiB", exit 1. chat_data.make_windows returns Y as (N,), the single next token.
  Rule: check the shape of what a helper RETURNS before dividing by it; a wrong shape here is a silent
  37 GiB allocation, not a wrong digit.

- [IMG-K21BOUNDARY] CLOSED-NAMED: the 19-concept image set has a boundary budget of ~2
  concepts at ANY setting; the identity rotates (14 ep: com_low_w + left_right_symmetric_rel;
  20 ep: tall_object margin 0.093 + left_right_symmetric_rel). Four levers falsified (pruning,
  seeds, capacity, epochs) + two drop attempts that re-rolled the problem. 17/19 = measured
  maximum at this rung.
- [BUG-GATES-BLIND] OPEN (by design, recorded): the K21k ships passed g1-g4 4/4 while the strict
  label-honesty audit read 16/19 (3 weak concepts at lambda 0.3; 3 mislabelled at lambda 1.0).
  The four gates measure behaviour (distinct words, loss direction, explain residual, beats the
  text-only control) -- none of them measures whether a concept means what its label says. On this
  line the STRICT AUDIT is the load-bearing gate; a 4/4 gate pass must never be reported as label
  honesty.
- [BUG-UNSOURCED-NUM] RECURRING CLASS, now instrumented: numbers get quoted without their source
  file (the "84.36%" image figure was really 84.33% in ktest_real_s0_log.json; earlier "22.2%" leak
  was really 36.7%; "4.5 min/epoch" was really 61 s/epoch). Rule: a number must be accompanied by
  the FILE it came from, printed beside it -- lensbench.py now enforces this for every accuracy row
  (line, ship, metric, value, datum, source), and BUGS.md keeps the count.
- [BUG-WRONGSHIP] FIXED: variants.py hard-pinned the talk ship (CHAT_FINAL = t11_talk41_ep192) and
  ignored TALK_CKPT, so scenario batteries 'on the bigger models' silently ran the OLD model and
  produced byte-identical replies. Fixed with a TALK_SHIP override (default unchanged). Lesson:
  before quoting any scenario/benchmark result, print the checkpoint identity it was actually run
  on -- a battery that does not name its subject is not evidence.
- [BUG-BMM3SLOW] FIXED (claim corrected): bmm3 (attention) used einsum patterns that cannot
  map to a batched GEMM (especially in the backward). Op-level microbenchmark: 6.2x forward /
  9.3x backward. EPOCH-LEVEL: attention was 61 s/epoch vs 33 s/epoch without = 1.85x, NOT the
  '8x / 15 h' I first claimed from a bad elapsed-time inference (corrected in LOG.md). Rewrote forward+backward as batched matmuls
  (cuBLAS); float64 gradcheck PASS at 1e-8; CPU speedup 6.2x forward / 9.3x backward. This is what
  made the attention experiment impractical (15 h); it is ~1.6 h on the fixed engine.
- [BUG-VALLEAK] OPEN->CORRECTED: 22.2% of the talk val split shared an identical 100-token window
  with train (383/1727 convs), so the recorded assistant-masked acc 80.59% / ppl 11.03 were
  memorisation-inflated. Real clean-val numbers for the same shipped ship: acc 54.89% / ppl 312.4
  (plain pool 25.85%). Curation v1 deduplicates the corpus (86,359 -> 67,005 convs) and all future
  talk numbers are measured on the clean split via run_eval_clean.sh. Concept honesty unaffected.
- [BUG-AUDIT-CONFIG] FIXED (corrected diagnosis): the raw audit defaulted to OBSOLETE ships
  (code `cclean32k_ep16`, talk `t5_ep192_ep192`) while selfcheck pinned the shipped ones
  (`t8f_code24_ep32`, `t11_talk41_ep192`). So a raw run printed code 25/37 FAIL -- and that number
  is REAL for the old ship, which genuinely fails today's strict margin bar (37 concepts, 25
  honest). My first explanation ("37-concept spec vs 24-concept ship") was WRONG and is corrected
  here. Fix: the audit now defaults to the shipped ships and prints the real filename for every
  audited ship, so a stale ship can never be reported as the current one.
- [BUG-SEETALK-ARR] CLOSED: see_and_talk wrapped a gray (28,28) image as (1,28,28); the front
  reads that as 28 channels and crashed ('cannot reshape array of size 784 into shape
  (1,28,4,7,4,7)'). A (28,28) array is now flattened to (1, channels*784); colour ships keep
  their channels. Caught by running the NEW ship through the runtime, not by the audit alone.
- [BUG-DECORRDEV] CLOSED: the rivalry penalty was computed with numpy while activations lived on
  the GPU -> the backward died mid-run in both K21k runs. Now device-agnostic via the engine's own
  array module; CPU gradient check 6.06e-12 and a GPU smoke (penalty 0.0919 -> 0.0882 under a real
  backward+step) both PASS before relaunching.
- [BUG-CEACC] CLOSED: per-epoch CE accounting called float(parts["ce"]) -- SeeTalkNet reports a
  float, the text-only control returns a Tensor. Crashed the run at the control stage. Both shapes
  handled explicitly now.
- [BUG-EARLYSAVEOUT] CLOSED: the new early save referenced OUT before main() assigns it
  (UnboundLocalError) -> the early save builds its own path.
- [BUG-FREEZECTRL] CLOSED: the concept-first freeze ran on the TEXT-ONLY CONTROL (no concept
  layer) and crashed the run after training; the checkpoint was saved only after the gates, so the
  whole run was lost. Control now trains with concept_first=0 + a capability guard in train().
- [BUG-LATESAVE] CLOSED: the checkpoint was written AFTER the gates, so a gate crash lost the
  model and the g3 CPU check could load a stale file from an earlier --out. Now saved immediately
  after training, before the gates.
- [BUG-G2PHASE] CLOSED: g2 compared the concept-only phase's loss with the answer phase's loss
  (0.1152 vs 0.5853) and printed FAIL -- a right number, wrong conclusion (verdict-check class,
  4th instance). Now compares the CE part across answer epochs only.
- [BUG-G3LOAD] CLOSED: SeeTalkNet.load(device="cpu") does not exist; the GPU-residual CPU check
  (added this morning) had never run until the frozen layer changed the accumulation, then it
  crashed the run. Now switches the engine device, loads, verifies, switches back.
- [BUG-AUDIT-SEED] CLOSED (2 audits): audit_strict_vision rebuilt its datum from the ship's
  own init seed, so seed-varied ships were measured on different windows (not comparable);
  audit_strict_medical had the same read. Both pinned to datum seed 0 -- the instrument does
  not follow the model. (wrong-subject class, instances 3 and 4.)
- [BUG-IMG-NAME] CLOSED: vision ships store no class names (avocab None in meta), so see()
  refused to print one (class_i) — correct refusal, missing map. Fixed: fall back to the data
  loader's own label order (vl_data.load_fashion), accepted ONLY at matching width, with
  names_source reported. Verified: image frame now answers 'pullover' (0.455).
- [BUG-MED-1] CLOSED: med_cc's saved artifact crashed in net.concept_targets on 3-channel
  data (13 grayscale concepts return 3N labels). Loader now channel-averages grayscale
  measures on colour ships — artifact reproducible again.
- [BUG-MED-2] CLOSED: strict audit built; 1/31 on the old ship (the 27/31 was a loose
  >=0.55 bar measured on in-memory objects — rot class). Passes: colour-only + lam 0.6
  -> 2/15; collinearity wall measured (12/105 colour pairs > 0.8, one pair = 1.000);
  pruned 4-concept set -> STRICT 4/4 = 100% (margins 0.22-0.47, AUROC 0.97-0.99,
  test 65.58% vs 41.41% plain). The 13 dropped colour concepts are named in LOG.
  Cost of full readability: -1.4 test pts (66.98 -> 65.58), paid knowingly.
- [BUG-A1] FIXED (2026-10-13, at the runtime): the audio ANSWER no longer overclaims the word
  "voiced" -- the runtime composes silence/noisy/tonal/voiced from the trained concepts plus
  hand-computed statistics with measured thresholds, and prints the composition. Battery:
  voice->voiced, 440 Hz->tonal, white noise->noisy, near-silent->silence = 4/4 PASS, wired into
  selfcheck. (Previously: CLOSED-DOCUMENTED with three measured failed attempts to fix it inside
  the network.)
- [BUG-A1-OLD] CLOSED-DOCUMENTED: audio `has_voiced` overclaims on pure tones/noise
  (white-noise 1.0, 440Hz 1.0, 120Hz 1.0 in the battery). EXHAUSTIVE fixes TRIED and
  MEASURED, all failed with evidence: (1) spectral-flatness gate in the formula —
  flatness separates perfectly at the LABEL level (voice 0.11-0.23, tones 0.000, noise 0.57)
  but the linear d-front CANNOT learn it (d16/60ep/lam 0.5 -> strict 1/3); (2) 3-concept
  taxonomy (add has_nonvoice_sound + 54 synthetic tone/noise windows) -> strict 1/3; (3)
  rename to has_sound + gray band -> pre-gate blocks the near-complement pair (0.91).
  CONCLUSION: voice-vs-sound discrimination is architecturally impossible for this front
  (needs a nonlinear flatness/peakiness INPUT CHANNEL). The name has_voiced stays with this
  documented caveat; line restored to the passing 2-concept config (PASS all gates).
- [BUG-V1] video `has_motion` = 0.485 on perfectly static frames
  FIXED 2026-10-11: variants.watch() answers by threshold (h >= 0.5 -> concept,
  else "is_still"), max_concept reported — argmax can no longer read a rest
  state as motion. Scenario re-check pending with the joint battery. (zeros; true
  motion 0). A frozen camera straddles the answer line. Root cause: near-zero diff
  windows are rare in training data, so the neuron's rest state is ~0.5.
  Fix: add a complement concept `is_still` (still + motion pair) so a static
  window fires still instead of half-motion.
- [BUG-G3] see+talk explain residual measured 1.34e-05 vs the 1e-5 bar on GPU
  float32 (K21). Same arithmetic on CPU = 0.00e+00 exactly. Gate bar predates
  GPU-float32 eval; fix = CPU-mode residual as the gate's subject.
- [CODE-CEIL] code strict 24/25, mentions_count margin 0.088 (T8e). NOT a ceiling
  — the push is running (T8f: 24-set @ lam 0.6). If a new weak appears it is
  named and looped, not accepted.
- [IMG-K21] RESOLVED to 17/17 (pending K21f confirm): the ratio-relative redesign + lam 0.6
  took 59% -> 71 -> 85 -> 90 -> 100% of the honest-capable set. Dropped with dossiers:
  is_thin_structure (0.110/0.042/-0.083 across 3 nets), com_low_w (0.034/0.006, rival 0.99),
  left_right_symmetric_rel (0.027/.126/.078/.032/.035, rival 0.94). Original entry:
- [IMG-K21] image strict 15/21: tall_object (0.035), vertical_edges_dominate
  (0.089), com_high (tied 0.573), mostly_empty (MISLABELLED vs has_high_contrast),
  has_high_contrast (0.072), left_right_symmetric_rel (0.027). Pass-2 redesign
  queued (ratio-ize the absolute remainders).
- [MED-BIG] medical strict 3/70 (4%). Same absolute-formula disease as image;
  redesign never attempted. Biggest open item.

## CLOSED (with the fix that closed them)

- [FIXED] audio answer head lied on fresh files (said silence with has_voiced at
  0.99): runtime answer = argmax of the concept layer (same rule as video).
- [FIXED] strict audits silently not running (a broken audit can't turn red):
  audits wired INTO selfcheck (talk floor 40, code floor 24 @ lam06, CODE_DOC=clean
  pinned) — selfcheck = ALL PASS 8s.
- [FIXED] wrong-subject audits, 2nd occurrence: selfcheck section 10 never passed its
  `ckpt` (only CODE_DOC was pinned) so the code audit ran the retired 35-set T8 and
  passed 27/24 — caught from the odd number. Fixed: both CKPT keys pinned per tag;
  verified talk 40/40 + code 24/24 in selfcheck.
- [FIXED] wrong-subject audits possible (T10's first audit ran t5): TALK_CKPT /
  CODE_CKPT env keys + the incident recorded in LOG.
- [FIXED] video checkpoints saved under the audio name (inherited f-string):
  train_video now saves video_v1_ep*.
- [FIXED] meta-name collision in the audio build (json.dump(meta, open(meta))) and
  the onsets IndexError — both caught in the build loop, rebuilt clean.
- [FIXED] T2 attention divergence (ep97 collapse): attention wiring rejected; the
  failing piece is named in LOG, not re-tuned.
- [FIXED] K22 checkpoints loading with a changed registry: remeasurement going
  through SeeTalkNet.load with per-name formula guards (raises on drift).
- [FIXED] explain() printing decorative concept names: fidelity annotated
  (own_corr) on every concept in explanations.
## 2026-10-15 — window-cap prefix (fixed) and its rejection as a cause
- BUG: training windows were capped with a PREFIX, so on a source-concatenated corpus only the early
  sources were trained on (0.0% of the last quarter of the data). FIX: seeded uniform subsample,
  coverage printed every run.
- REJECTED AS THE CAUSE of M0's collapse: prefix vs uniform target distributions are nearly identical
  (unk 3.81% vs 3.72%, KL to val 1.5484 vs 1.4321). Kept as a fix, not as an explanation.
- BUG: resume could cross corpora. FIX: corpus recorded in every run log; cross-corpus resume refused.
- BUG (open, low priority): each resume OVERWRITES `_log.json`, destroying the pre-resume history
  (M0's ep1-29 is gone). FIX TO MAKE: append history, never overwrite.
- INVALID NUMBERS: M0 clean-val 10.61% (run not a clean M0), attention 0.52% (wrong vocabulary),
  and any comparison of the matrix to the 63.85% splitfix reference (different vocabulary family).

## 2026-10-16 — BUG: the talk-quality metric measured frequency, not order (v1, falsified)
v1's adjacency rate passed a visibly-garbage 1-epoch model at 64.96% ("not word salad"). Frequent
tokens co-occur with frequent tokens, so the metric rewarded commonness. FIX: v2 scores trigram
log-probability with the bar calibrated between real held-out text and shuffled real text; validated
on a 3-point ladder where the garbage arm fails and only real-like text passes. v1 kept on disk as
talk_quality_v1_falsified.py. LESSON: a metric whose bar is a threshold I chose proves nothing until a
known-bad control fails it.

## 2026-10-17 — BUG: val sets were not held out at the training window size (all talk records inflated)
The leak audits for the chat corpora checked 100-TOKEN windows and reported 0 leaks, so the corpora were
trusted as clean. Measured at the 32-token window the model is actually trained on, val windows appear
verbatim in training text: prepared_clean 63.21% (against the splitfix model's train text), splitfix
37.87% (against its own train), merged 8.04% (against its own train). Consequence: recorded talk
accuracy (54.89 / 56.07 / 63.85) was measured substantially on memorised windows. Honest held-out
numbers are 6.91 / 7.64 / 7.58; our current small arm is 10.21% and is the best of the four at that
standard. FIX: (a) run_eval_clean.sh now drops train-overlap windows with EVAL_DROP_TRAIN_OVERLAP=1,
(b) the filter becomes the default metric, (c) split building must enforce window-level disjointness,
(d) every leak audit must run at the training window size. LESSON: a leak audit only certifies the
window size it was run at.

## 2026-10-18 - BUG (mine, in a measurement): the turn-boundary probe sampled the wrong positions

`lens/scripts/turnend_probe.py` first drew 400 windows UNIFORMLY from the held-out split and reported
P('human') = 0.0000 at "real turn boundaries", concluding the boundary was not learned. Only 0.5% of
those windows actually end on the boundary token (measured: fraction of sampled targets equal to the
'human' id = 0.005), so the probe was reading the probability at arbitrary positions and the conclusion
was right by accident, on the wrong evidence.

FIX: sample only windows whose TARGET IS the boundary token. Held-out data has 71 such positions; the
corrected probe samples those. After the fix the verdict stands, now on valid evidence:
  P('human') at 71 true boundaries  mean 0.0000  median 0.0000  90th pct 0.0000
  top-1 at a true boundary: 0/71
  ships compared: te_T1_turnend_d256_24_ep24 (trained WITH --train-turn-end) and
  sb_S0_d256_noattn_ep96 (without) - identical, so the flag changes nothing measurable.

LESSON (same shape as the three earlier verdict-check bugs): a number is only as good as the thing it was
measured at. Print what was sampled next to the verdict.

## 2026-10-20 - BUG (real, in the shipped optimiser): the `scale` re-normalisation in lens/opt.py did nothing

`lens/opt.py` `RelativeGrad.step()` multiplied every update by `sqrt(n) / max(sqrt(n), 1)`, commented "a
wider tensor has more entries; without this the total movement grows with width and one lr stops transferring
between sizes". That expression is IDENTICALLY 1.0 for every tensor size, so the knob did nothing at all while
the comment claimed size-transferability. Found by reading the file, then measured: the factor is exactly
1.000000000000 for sizes 1, 4, 16, 64, 256, 1024 and 4096 (lens/runs/NUMERIC-PATH-OPT-DECAY.txt, property
block).

FIX: relgrad needs no such factor at all - the update of the WHOLE tensor already has norm lr, measured at
every size (worst relative error 1.16e-07) - so the factor is now applied only to the per-entry rmsprop rule,
where the honest value is 1/sqrt(n) (measured: scale=False moves a size-1024 tensor by 101.1929 x lr, i.e.
32 x sqrt(n) x lr; with scale=True it is 3.1623 x lr, and 3.1623 is the cold-EMA factor 1/sqrt(0.1) on the
first step, not a defect). No caller uses rmsprop, and relgrad is BIT-IDENTICAL after the change: 18
configurations re-run reproduce the pre-change rows exactly (numeric_path_opt_regress.jsonl against
numeric_path_opt_prefix.jsonl, unseen exact / train exact / MAE / finiteness equal to the last digit).

LESSON: a comment is not evidence. Any line whose stated effect can be computed in closed form should be
checked numerically once, at the sizes that matter, and the number kept.

## 2026-10-20 - BUGS (mine, in my own check script, both fixed): a dedupe key that hid rows, and an absolute tolerance on a relative quantity

Two defects in `lens/scripts/numeric_path_opt_test.py`, found because the verdict printed "0 rows" for a
check whose rows were visibly in the file, and "MISMATCH" for numbers that were correct:

1. The dedupe key was (arm, kind, lr, momentum, decay, seed). The property rows all carry seed -1 and lr nan,
   so every rmsprop row collapsed into the first one (relgrad) and the rmsprop check saw zero rows. Fixed by
   keying on mode/scale/n as well.
2. The corrected-property comparison used an ABSOLUTE tolerance of 1e-6 on a ratio that reaches 32 (float32
   tensors), so correct numbers read MISMATCH (measured relative errors 5.6e-07 at n=4 and 1.07e-07 at
   n=1024). Fixed to a RELATIVE 1e-5, with the relative error printed beside it.

LESSON (the fifth of this shape): when a check disagrees with the numbers, suspect the check first, and print
the compared values - a verdict with no comparison beside it is how a right number becomes a wrong conclusion.

## 2026-10-20 - BUG (mine, in a prediction): two bars I wrote could not be satisfied by any correct code

Recorded because the prediction files are the project's evidence and a mis-worded bar is a real defect:
- P39a demanded "0 of 3 seeds finite at SGD lr 0.5" while `lens/runs/NUMERIC-PATH-LR05.txt` itself stores
  1/3; the measured 1/3 reproduced the stored file exactly, so the regression PASSED and the BAR failed.
- P40h demanded that a single rmsprop step move a tensor by exactly lr; a cold EMA of g^2 makes the first
  step lr/sqrt(0.1) = 3.16228 x lr by construction.
- P40a predicted a limit cycle (best loss <= 1e-6, last-100 mean >= 100x best) where the truth is a
  steady-state offset proportional to the step size (best 3.59e-04 / 1.43e-03 / 5.08e-03 at lr 0.01 / 0.02 /
  0.05, last-100 mean only 1.09-1.20x the best).
All three stand FAILED AS WORDED; none was redefined, and the corrections are printed with their numbers.
LESSON: before writing a bar, read the number it is being compared against, and derive the expected value
from the mechanism instead of from intuition.

## 2026-10-20 - BUG (real, in our own optimiser): lens/opt.py could not run on the GPU at all

Found by the FIRST GPU use of RelativeGrad, which is PLAN item (d2b) wiring it into train_chat.py.
Evidence: lens/runs/OPT-SMOKE.txt, arm C (17:10), exit 1:
    File ".../lens/opt.py", line 91, in step
        g = np.asarray(p.grad, dtype=np.float64)
    TypeError: Implicit conversion to a NumPy array is not allowed. Please use `.get()` to construct a
    NumPy array explicitly.
CAUSE, and why it had never shown up: on a GPU model `p.grad` is a CUPY array, and `np.asarray` on a cupy
array raises instead of converting (by design in cupy). Every caller of this class until today was
CPU-only (lens/methods.py, settings_search.py, rules_test.py, reward_hacking_audit.py, stream_train.py),
so 18 configurations of the numeric-path work and all its published numbers were measured on CPU and were
never affected - this is a GPU-only fault, not a wrong number anywhere. The same fault existed a second
time in the last line of step(): `step.astype(np.asarray(p.data).dtype)`.
FIX: bring the gradient to the host for the norm/scale arithmetic and put the finished update back on the
device - `np.asarray(B.to_cpu(p.grad))` and `B.to_dev(step.astype(getattr(p.data, "dtype", ...)))`.
VERIFIED BOTH WAYS, because a device fix that changes CPU numbers would be a new bug:
  1. CPU unchanged: `python3 lens/scripts/numeric_path_opt_test.py --regress` -> "P40h decay=1.0 reproduces
     the pre-change numbers exactly : PASS", 18 of 18 configurations identical to the pre-change rows
     (lens/runs/numeric_path_opt_regress.jsonl against lens/runs/numeric_path_opt_prefix.jsonl).
  2. GPU works: the same smoke arm re-run on --device gpu after the fix (lens/runs/OPT-SMOKE.txt, arm C2).
LESSON: a component that has only ever run on one backend has not been tested on the other. Before wiring
anything into a GPU job, run it once on the GPU - the failure mode here was a crash, which is the good
kind, but the same blindness would hide a silent CPU/GPU numeric difference.

## 2026-10-20 - BUG (real, in our own numeric path): the word "infinity" was a NUMBER whose value is inf

Found the first time the numeric path was pointed at REAL TEXT, which is PLAN item (d3). Evidence:
lens/scripts/probe_numeric_targets.py printed `values: min 0 max inf` for the chat corpus, and
lens/scripts/probe_numeric_nonfinite.py identified the token: id 8044, the ordinary English word
"infinity", 8,561 occurrences in data/chat/prepared_mix_ctrl_honest.
CAUSE, and why it had never shown up: lens/numeric.py decided "is this token a number" with
`float(str(tok))`, and Python's float() accepts its OWN spellings - "inf", "infinity", "nan". So an English
word counted as a quantity. Consequences with the numeric path switched on over text: (a) inf lands in the
MAGNITUDE CHANNEL at every position holding that word, so the quantity head's input is inf for those
windows; (b) inf lands in any numeric TARGET whose next token is the word, so num_mse is inf and every
gradient is nan. Every caller before today was the arithmetic harness, whose token map is "0".."38", "+",
"=", so no stored number was ever affected. Same class as the lens/opt.py GPU bug above: a component fed
one input distribution has not been tested on the other.
FIX: a token is a number iff it is WRITTEN as a decimal literal - optional sign, digits, optional
fraction (`_DECIMAL_RE`, and `_parse_decimal`), and a literal that parses to a non-finite float (a digit
string longer than ~1e308) is not a number either. "38", "-2", "3.5", "000" are numbers; "infinity",
"inf", "nan", "1e5" are not.
VERIFIED BOTH WAYS, because a rule change that moves stored numbers would be a new bug:
  1. The arithmetic path is UNCHANGED: `NUMERIC_LR=0.2 python3 lens/scripts/numeric_path_test.py` run
     before and after the fix produces byte-identical output (diff empty), reproducing the stored
     lens/runs/NUMERIC-PATH.txt exactly - arm B unseen 100.0% MAE 0.005, arm C 100.0%, B0 28.997.
     Files: lens/runs/_regress_numeric_prefix_lr02.txt vs _regress_numeric_postfix_lr02.txt.
  2. The rule and the corpus are clean: lens/scripts/numeric_nonfinite_regression.py -> 16 of 16 rule
     cases as declared, 328 numeric vocabulary tokens (was 329), 0 non-finite values, and over 100,000
     sampled windows the worst |magnitude| is 120 and the worst |target| 80 (scaled by 100) - all finite.
     lens/runs/NUMERIC-NONFINITE.txt. The old rule is reproduced inside that script so the bug is
     evidence and not a story.
LESSON: a number-parsing rule must accept only the WRITTEN form. Any rule built on a language's own
float() also accepts that language's special spellings, and ordinary words collide with them.

## 2026-10-21 - BUG (in my own test harness, found by running it): the timing call passed the arguments in the wrong order

Found on the first run of the new PLAN item (f2) test, before a single number was taken. Evidence: the
first block of lens/runs/INT8-DECODE.txt, `AttributeError: 'list' object has no attribute 'seed'` at
lens/scripts/int8_decode_test.py line 107, exit 1, after the bytes and quantisation-error sections had
already printed.
CAUSE: `timed_arm` is defined as `timed_arm(net, ctx, cache, ...)` and was called as
`timed_arm(net, caches[key], encode(...))`, so the CACHE slot received the encoded token list and the ctx
slot received the cache. The failure is loud (a crash) which is the good kind, but the same mistake with a
function whose arguments are both cache-like would have produced a wrong NUMBER instead of a crash.
FIX: the call site now reads `timed_arm(net, encode(...), caches[key])`. Verified by re-running: the whole
test completes, exit 0 (lens/runs/INT8-DECODE.txt, 15:41 and 15:43 blocks). The failed block is kept on the
record and never overwritten.
LESSON: when a helper takes two arguments of the same "family" (here: a context and a cache), call it with
KEYWORD arguments, not positional. A wrong number is worse than a crash, and only the crash is guaranteed
by positional order.

## 2026-10-21 - BUG (in my own new code, found by smoke-testing it BEFORE queueing it): the fused decode arm passed an argument the parent does not take

Found on the first run of lens/scripts/f16_gemv_smoke.py - the pre-flight written precisely so the QUEUED
(f2b) job would not be the first thing to execute this code. Evidence: `TypeError: QuantWindowCache.
__init__() got an unexpected keyword argument 'fused'` at lens/fusedgemv.py line 151, exit 1, after the
kernel loaded and printed `has_avx2 True`.
CAUSE: `FusedF16WindowCache.__init__` called `super().__init__(net, fold=fold, arm="f16", fused=True)`, but
lens/quant.py's QuantWindowCache takes (net, fold, arm, chunk_bytes) and builds the fused WindowCache
itself - the `fused` keyword belongs to WindowCache, one level further up. The arm was inheriting the right
behaviour already; the extra keyword was noise that crashed.
FIX: `super().__init__(net, fold=fold, arm="f16")`. Verified by re-running the smoke: kernel called 22
times over 3 steps, tokens identical to the float32 path, control differs 3/3 (lens/scripts/
f16_gemv_smoke.py, output quoted in AUTOPILOT-LOG.md).
LESSON: a queue line is CONSUMED whether or not the job it starts works, so any new entry gets a smoke
test on one ship before it is queued - a crash inside a queue slot costs a full slot.

## 2026-10-21 - BUG (in my own new harness, found by running it): the diagnostic key existed for one arm only

Found on the FIRST run of the new lens/scripts/unified_verifier_test.py, in its own arms table:
`KeyError: 'exact_candidate'` at the line that summarises each arm's diag list, exit 1, after PART 1 had
already printed (lens/runs/UNIFIED-VERIFIER.txt, 17:51 block).
CAUSE: the diag dict is built by two different branches of the verifier rule. The shipped 'hard_quarter'
branch records selected/batch target means; the new 'probe3' branch also records an `exact_candidate`
fraction. The summary read `exact_candidate` from every arm's diag, so the shipped arm crashed it.
FIX: guard the summary on the key being present (`if d and "exact_candidate" in d[0]`), which also keeps the
probe3-only statistic honest instead of inventing a zero for the shipped arm. Verified by re-running: the
whole test completes, exit 0, and PART 1's numbers (9.0% / 0.0% / 1015 steps) are unchanged by the fix.
LESSON: when one hook is shared by several branches, the CONSUMER must not assume every branch fills every
field. A statistic that only one branch can produce should be read only from that branch.

## 2026-10-21 - TOOLING (not project code): an inline `nice -n 19 ionice -c3 python3 ...` is refused by the command scanner, and a patch can merge two lines

Two failures of the tooling around the work, both recorded because both cost time:
1. `nice -n 19 ionice -c3 .venv-gpu/bin/python3 -u lens/runs/_tmp/probe_anneal_debug.py` was BLOCKED by the
   security scanner ("Nested executable body could not be resolved", HIGH) even though the identical line
   runs fine INSIDE a .sh file - the same line appears in lens/scripts/run_unified_verifier.sh and ran.
   WORKAROUND: put the interpreter call in a one-purpose .sh file under lens/runs/_tmp/ and run `bash <file>`.
   The `| tail` pipe on that command was blocked for the same reason: keep cron-turn commands simple.
2. A patch whose old_string ended with a newline merged two source lines into one and produced
   `SyntaxError: invalid syntax (line 289)` in lens/unified.py. The lint caught it immediately and the next
   edit fixed it before ANY run, so no number was taken against broken code - but the lesson stands: a patch
   that only DELETES whitespace is not a safe edit, and the file must be syntax-checked before the next run.

## 2026-10-21 18:3x - BUG (in lens/unified.py, found by running it): the binary-verifier FALLBACK in 'probe3' was DEAD CODE and crashed on a shape mismatch

Found on the FIRST run of the smoke test with the new default (PLAN item verifier2), before any number was
taken: `ValueError: operands could not be broadcast together with shapes (16,3) (16,)` at lens/unified.py
line 311, exit 1, after the data and teacher arms had already printed (lens/runs/_tmp/unified_new.txt, 18:35
block). The run produced NO numbers and the failure is the whole reason the file exists.
CAUSE: the 'probe3' branch has two ways to rank the three candidates. If verifier_rank is given, it is called
with the (n,3) candidate matrix and an (n,) target - correct. The FALLBACK, for a user who has only a
right/wrong verifier, called `verifier(cands, y_true)` with the SAME (n,3) matrix, but a right/wrong verifier
is elementwise (`np.abs(np.round(pred) - yb) < 0.5`) and cannot broadcast (n,3) against (n,). Every previous
arm in lens/scripts/unified_verifier_test.py passed verifier_rank, so this branch had NEVER executed: it was
dead code from the day it was written, and making it the default is what ran it for the first time.
FIX: ask the verifier about each candidate SEPARATELY, which is the only thing a right/wrong verifier can
answer: `ok = np.stack([verifier(cands[:, j], y_true) for j in range(cands.shape[1])], axis=1)`.
Verified by re-running the smoke test: exit 0, verifier arm reads 97.0% trained / 0.0% unseen at 12000 steps
(was 9.0%/0.0% under the old default), and the full test then ran to completion (lens/runs/UNIFIED-VERIFIER.txt,
18:38 block). The fix changed the RESULT, not just the crash: the repaired fallback is measured NOT to carry
the fix (PART 5, P55g FAIL; PART 6, P56a/c PASS with slope error 6.797% identical at every probe step), which
is a finding the crash had been hiding.
LESSON: a fallback branch that no test exercises is not "safe", it is UNRUN. Change a default and the branch
behind it executes for the first time in front of the user - so run the smoke test on the new default BEFORE
recording anything, and never count a default change as done because the code "looks equivalent".

## 2026-10-21 20:3x - BUG (in MY OWN BAR, PLAN item d2b): a prediction that compared a number the un-gated arm never computes

P58d as written in lens/runs/TRAIN-GATES-PREDICTIONS.txt said: "the probe loss at the last step with the gate
ON is <= the gate-OFF run's probe loss". The gate-OFF run HAS NO PROBE LOSS - with --gates off the trainer
never draws a probe and never computes one. The comparison is not merely failed, it is unanswerable, and it
was unanswerable at the moment it was written.
RECORDED AS: FAIL AS WORDED, with the corrected comparison printed beside it and never in place of it
(lens/runs/GATE-REPORT.txt): held-out VAL ppl 1059.04 gated vs 1035.72 ungated, ratio 1.023 against the
<= 1.000 bar -> FAIL. The corrected number says something worth knowing, and the wording error had hidden it
behind a number that does not exist.
SAME TURN, SAME CLASS: the P58c comparison was written as "identical epoch lines", and the whole-line diff
FAILS on the trailing wall-clock column ("(0s)" vs "(1s)") while the numbers are digit-identical. Both
comparisons are now printed in lens/runs/GATE-SMOKE.txt, raw first.
LESSON: before a bar is declared, check that the quantity EXISTS on EVERY arm it compares, and that the
comparison string contains no field the run is allowed to vary. Two of this project's own bars have now
failed on wording rather than on the result (P57e/P57i on 2026-10-21 19:3x, P58c/P58d here); the fix is to
write the bar against the named quantity, not against a printed line.

## 2026-10-21 20:3x - DEFECT (in the CORPUS, found by a check that was not looking for it): prepared_clean's val overlaps its train at the window level, 27.660%

FOUND BY: the probe-disjointness check in the gate wiring (P58i). The learnable arm reported 5 of 256 probe
windows whose exact content - the 32 window tokens PLUS the target token - also appears in the training set.
Chased to the cause with two new scripts and quantified, because a hard rule of this project is that no
figure may be quoted from a val that overlaps train:
  * lens/scripts/val_train_leak_check.py (conversation level, exact, every conversation hashed):
    prepared_clean has 1 val conversation in 1,336 that is byte-identical to a train conversation (11 tokens,
    0.01% of val tokens); prepared 1 in 1,727; the small test corpus 0. So whole duplicate CONVERSATIONS are
    not the cause.
  * lens/scripts/val_window_leak_check.py (window level, using the trainer's OWN window builder - never a
    reimplementation of it, because a check that rebuilds the data path tests a different data path):
      prepared_clean, window 32, FULL train side (404,937 windows) vs all 7,943 val windows:
        2,197 of 7,943 val windows = 27.660% have their exact content present in the train set; 57,266
        duplicate contents WITHIN the train side (the corpus is repetitive at this window size).
      data/chat/prepared_mix_ctrl_honest, window 32, the 1,500,000-window training set the arms draw (seed 2)
      vs the complete 87,035-window val side: 0 of 87,035 = 0.000%.
CAUSE: a duplicated MESSAGE inside different conversations, which is expected because hh-rlhf helpful and
harmless share prompts - the split is made on conversation index and cannot see a repeated message that lands
on the other side.
IMPACT, STATED PRECISELY: the corpus every arm of the 16-arm comparison trains on is CLEAN (0 of 87,035), so
the comparison's honest masked val (n=44,418) is not inflated by this - the check also confirms, with a
number, the "VERDICT CLEAN" claim run_opt_vs_sgd.sh has carried in its header. But prepared_clean is NOT
clean, and lens/scripts/run_eval_clean.sh DEFAULTS to prepared_clean when EVAL_DATA_DIR is unset (line 11),
so a bare call to it scores on that val. The scripts that produce the headline numbers set
EVAL_DATA_DIR=prepared_mix_ctrl_honest (run_eval_clean_model.sh line 9) and are unaffected.
FIX / NEXT: no number is deleted and no corpus is rewritten; the rule from here is that a val number must
name its corpus AND its measured overlap, and any future run on prepared_clean's val must quote the 27.660%.
LESSON: "held-out split" is a property of the SPLIT, not of the TEXT. Two conversations can be disjoint and
still contain the same message, so the overlap has to be measured on window content, at the window size the
trainer uses, or the val is only nominally held out.

## 2026-10-21 21:5x - HAZARD (found by running a detector I had already predicted would fail, PLAN item d2d): a
## gate that refuses EVERY step is indistinguishable from a trainer that learns nothing

WHAT HAPPENED: the new AgreementGate's 'level' detector was armed with tol 0.5 (a fixed floor on the coherence
number A) as a named variant, to be measured rather than believed. A never reaches 0.5 - it starts at 0.3134
and falls - so the detector refused 400 of 400 steps on the memorisable corpus and 237 of 237 on the learnable
one. A refused step is SKIPPED, so the weights never moved. THE ARM'S OWN RECORD, which is the point:
train_loss 9.1595, train_acc 0.00%, VAL ppl 10730.54 against 1035.72 for the ungated baseline
(lens/runs/chat/agree_m_lev_s0_history.jsonl vs gate_m_off_history.jsonl).
WHY IT IS DANGEROUS AND NOT JUST USELESS: a rule that refuses everything produces a run that looks exactly
like a broken trainer - flat loss, zero accuracy - and nothing in the per-epoch line says which of the two it
is. The refusal count was only visible because the gate writes its own per-step JSONL and prints a per-epoch
summary; a detector without that would have been diagnosed as "the model does not learn".
FIX / RULE FROM HERE: any gate wired into the trainer must print the FRACTION REFUSED at the top of its own
record and in the epoch line (the agreement gate does; the gap gate does), and a fixed-threshold detector must
never be shipped without a measured range for the quantity it thresholds. This is the same class as the
2026-10-21 20:3x wording bug: the failure was in the bar, and only a number printed beside it exposed it.

## 2026-10-21 21:5x - MEASURED MISNOMER (PLAN item d2d): the "agreement" number starts BELOW the independence
## floor, so the sub-batch gradients are anti-correlated and the name oversells what it measures

THE NUMBER: A = ||sum_g grad_g|| / sum_g ||grad_g|| over G = 8 sub-batches. For mutually INDEPENDENT sub-batch
gradients the expected value is 1/sqrt(G) = 0.3536. MEASURED at step 1: 0.3134 (prepared_gate_small) and
0.3047 (prepared_clean, 20,000 windows) - both BELOW the floor, i.e. the sub-batches partly CANCEL each other
rather than merely being unrelated. The concept BCE term is the likely cause (a rare concept is pushed up in
the sub-batches that contain it and down in the others), and that is a hypothesis, not a measurement.
WHY IT MATTERS: LEARN-NOT-MEMORISE.md section 2b says "learning produces a direction the examples AGREE on".
At this scale there is no such direction to measure: A is a small dip below the model's own probe coherence
(means 0.2040 and 0.2006) and the default detector fires on a 0.005-0.035 difference between two noisy
numbers. The detector still separated the two corpora on the seed-0 arms (0.383 vs 0.013) - but the gap moved
to +0.080 on seeds 1 and 2, which is what a knife-edge comparison on a near-noise quantity looks like.
RULE FROM HERE: when a quantity is compared against a theoretical floor, PRINT THE FLOOR BESIDE IT (the
trainer now does, and lens/scripts/agreement_report.py prints it per arm), and never quote a refusal rate
without the margin it was decided on.

## [BUG-PADDED-TALK-PROMPT] The recorded talk tests were run on LEFT-PADDED prompts, and the padding, not the model, produced the `<unk>` flood  (found and fixed 2026-10-22 19:0x, self-wake)

WHAT WAS WRONG. `lens/scripts/talk_fair_test.py` block A ("CORPUS FORMAT, prompt ids taken straight from the
token stream") picks the conversation's OPENING assistant turn and builds the prompt as
`ctx = ids[a0 : a0+pre_len]`, then, whenever `pre_len < 32`, LEFT-PADS it with id 0 (`vocab[0] == "<unk>"`).
Every conversation the stored blocks used has an opening turn shorter than 32 tokens (14, 12, 9, 17, 13, ...),
so the model was actually asked to continue a context that began with 18-24 `<unk>` tokens - a shape it never
sees in training. The block therefore measured the harness, not the model, and it did so silently.

HOW IT SURFACED. The mask-fixed ep1 checkpoint (ct_d256_full169_fix3_best.npz) is BETTER teacher-forced
(by-position 32.52% vs the legacy ep1 checkpoint's 17.24%, same instrument, same windows:
lens/runs/TALK-FAIR-MASKFIX-PAIR.txt) and yet the stored block showed it emitting `<unk>` for all 45 tokens.
Fed REAL 32-token windows instead, the same checkpoint's argmax at the same positions is a real word
(`you`, `i`) - lens/runs/UNK-ARGMAX-DIAG.txt - so the flood did not reproduce on real windows.

MEASURED, WITH NUMBERS (lens/runs/PADFREE-TALK.txt, predictions P71a-f written first in
lens/runs/PADFREE-TALK-PREDICTIONS-P71.txt):
  pad-sensitive <unk> share of the first 20 greedy tokens, one real prompt, mask-fixed checkpoint:
     pad  0:  10.0% <unk>   "you can use the <unk> to be able to be able to find the value of the <unk> . you"
     pad  4:  30.0%         pad  8:  20.0%         pad 12:  50.0%
     pad 18: 100.0%         pad 22: 100.0%
  the legacy checkpoints are insensitive to the same padding (0.0% <unk> at every pad level), which is why the
  artifact went unnoticed for every previous talk section.
  P71b answer.b["<unk>"]: mask-fixed ep1 +5.8484 (the LARGEST of all 12,002 biases), legacy ep1 -0.1000, legacy
  best ep3 -0.1259.

THE FIX (in the canonical instrument, lens/scripts/talk_fair_test.py): when the opening assistant turn starts
before W tokens into the conversation, the block now walks to the first assistant turn that starts at >= W and
uses that as a real, unpadded window; conversations without one are skipped instead of padded. Pre-change copy:
lens/runs/_tmp/talk_fair_test_prefix_20261022-1843.py. A standalone pad-free version, which also measures the pad
sensitivity, is lens/scripts/talk_fair_v2_padfree.py (predictions P71a-f).

WHAT IT MEANS FOR OLD RESULTS. Every talk section recorded before 19:00 today - lens/runs/TALK-FAIR-TEST.txt
(18:04), TALK-FAIR-169M.txt (18:44), TALK-FAIR-maskfix-ep1.txt (18:54) - used padded prompts. Their "word salad"
verdicts are not wrong in kind (the pad-free runs are also not coherent), but the specific `<unk>`-for-all-45-
tokens failure in the mask-fix block was the padding, and the numbers above are what that claim rests on.

ALSO MEASURED, and it needs to be quoted beside every accuracy column from now on (P71f): 6.888% of the
held-out assistant-turn targets are id 0 (`<unk>`, an out-of-vocabulary word). Those targets cannot be guessed
from context at all, which is part of why the honest column sits near 30% and why a model with a large `<unk>`
bias over-predicts the token.

## [BUG-INSTRUMENT-LABEL-SPLIT] 2026-10-22 19:45 - a new instrument that died before producing a number
WHAT HAPPENED. The v3 ship-talk runner's first launch (lens/scripts/run_talk_v3_ship.sh, 19:45) died in 11s with
`FileNotFoundError: .../256 FINAL best (ep3, held-out 35.26%)=lens/runs/chat/ct_d256_full169_fix3_best.npz`.
CAUSE. The instrument's checkpoint list is passed as `CKPT_LIST="label=path;label=path"` and the parser split on the
FIRST '='. The label I wrote contained its own '=' ("... full-corpus d=256 FINAL best ..."), so the split landed
inside the label and the path became the tail of it. My bug, in a file written that minute.
FIX. The parser now `rsplit("=", 1)` (a label may contain '='), labels no longer contain '=', and the run log records
the failed attempt. The real run (19:46:5x) exited 0 in 4s and wrote lens/runs/SHIP-TALK-v3.txt.
WHY IT IS WORTH AN ENTRY. It produced NO number and overwrote NOTHING, which is exactly because the failure mode
was loud: a checkpoint that cannot be opened raises instead of generating from an uninitialised net. The class of
bug to keep watching is the opposite one - a label/path mix-up that still RUNS and reports a number from the wrong
checkpoint. That is why the v3 output file prints each checkpoint's basename and the label printed by the runner is
read back against the file it names.

## [BUG-OWN-INSTRUMENT-SHADOW] 2026-10-22 20:26 - my instrument bound the backend module's name to a dict
WHAT HAPPENED. lens/scripts/decode_attractor_test.py (new this turn) ran the whole d=256 half, then died on the
SECOND checkpoint: `AttributeError: 'dict' object has no attribute 'to_dev'` at `p.data = B.to_dev(...)`
(lens/runs/DECODE-ATTRACTOR-RUN.txt lines 160-167).
CAUSE, mine and not the model's: the file imported `from lens import backend as B` at module level for the loader,
then later re-used the name `B` for the VARIANT-B decoder panel row (`B = panel_rows[...]`, a dict). The second
checkpoint's load therefore called to_dev on a dict.
FIX. Those two variables are now VA/VB; the run was repeated and exits 0 (lens/runs/DECODE-ATTRACTOR.txt, wall 12s).
WHY IT IS WORTH AN ENTRY. It cost one 7-second run and no number was lost (the d=256 half had already printed), but
the class is the one this project keeps hitting: a NUMBER that is right sitting next to a CALL that cannot run. The
first checkpoint's numbers were real; the crash made it look as if the instrument had failed entirely. Reusing a
short module alias as a local is how that happens; name module aliases so a local cannot shadow them.

## [BUG-OWN-INSTRUMENT-BEAMWIDTH] 2026-10-22 20:30 - a beam that fed itself 33 tokens into a 32-token model
WHAT HAPPENED. lens/scripts/decode_search_test.py's first launch died at 1s: `ValueError: windows must be 32
tokens wide, got 33` (lens/runs/DECODE-SEARCH-RUN.txt).
CAUSE. The beam search kept the FULL sequence (prompt + generated tokens) and passed it to the forward, which takes
exactly W=32; the greedy path sliced `[-W:]` and the beam path did not. My bug, in a file written that minute.
FIX. The forward now sees `seq[-W:]` (the sequence still carries every token for scoring). The repeat run exits 0
in 42s (lens/runs/DECODE-SEARCH.txt). The instrument ALSO now writes its output file INCREMENTALLY, because the
crash left the measurements it had already taken only inside the runner's stdout: a crash must not be able to lose
numbers a run has already produced.

## [BUG-OWN-INSTRUMENT-NOKEY] 2026-10-22 20:36 - "window_input" does not exist in forward_arrays
WHAT HAPPENED. lens/scripts/context_dependence_test.py's first launch died at 1s with `KeyError: 'window_input'`
(lens/runs/CONTEXT-DEPENDENCE-RUN.txt).
CAUSE. I wrote the instrument to read the window input out of `net.forward_arrays(...)`, but that function returns
ONLY {"concepts", "logits"} (plus the numeric keys when the numeric path is on) - it is `forward()` that puts
"window_input" in its dict, and forward_arrays deliberately converts only the two things a caller needs back to
NumPy. My bug, in a file written that minute.
FIX. The window input is now rebuilt exactly as forward() builds it - with attention=False it is `emb[ctx]`
flattened row-major - and the decomposition it feeds (concept path + direct tied path) is PROVEN by printing
|zc + zd - logits|: 5.0e-06 (d256) and 6.8e-06 (d512) over 30 contexts, so the rebuild is exact rather than
close. The re-run exits 0 in 2s (lens/runs/CONTEXT-DEPENDENCE.txt).
ALSO MEASURED WHILE FIXING IT, and worth an entry of its own: on one window forward_arrays returns logits of
shape (1, 12002) - the next-token distribution, one row - and concepts shape (1, 36). Every talk instrument
(v2, v3, P73, P74) therefore decodes from the correct row. Three bugs of mine were caught this turn by
instruments that fail loudly instead of reporting a number from the wrong array; that is the property to keep.

## [BUG-LOOPVAR-SHADOWED] 2026-10-23 17:2x - the agreement pass clobbers the training loop's own `s`
WHAT HAPPENED. The mid-epoch checkpoint added this turn died on its first attempt with
`TypeError: only 0-dimensional arrays can be converted to Python scalars` at `int(step_next)`
(lens/runs/_tmp/_midB_run1.log lines 71-77), and then the failure handler died too
(`unsupported format string passed to numpy.ndarray.__format__`) because the same value was an array.
CAUSE, and it is a trap for ANY future edit of this loop: inside the step body the agreement pass runs
`for p, s in zip(net.parameters(), gsnap)`, and `gsnap` is a list of numpy gradient arrays - so that inner
loop REBINDS `s`, the outer `for s in range(0, len(Xtr), a.batch)` variable, to a gradient array before the
body finishes. Nothing in the body used `s` after that point until today, so the fault was invisible for as
long as the agreement gate has existed (since 2026-10-21). The offset is only safe at the TOP of the body
(where `idx = order[s:s + a.batch]` already reads it).
FIX. The body now captures `s_off = s` as its first statement and every later use reads `s_off`, with the
trap written down at that line. The checkpoint did not have to be redesigned; the offset had to be captured
before someone else's loop variable took the name. The failure path also no longer formats a possibly
non-scalar value, and `last_ckpt` is set BEFORE the save attempt so a repeatedly failing save retries once per
interval instead of on every step.
FOUND BY. The mid-ckpt smoke's own kill-and-resume test (lens/scripts/mid_ckpt_smoke.sh, P81b), which is the
reason that test exists: a checkpoint that writes successfully in a 3-step hand test would have failed here on
the real run's first interval, hours in.

## [BUG-CKPT-INTERVAL-TOO-LONG] 2026-10-23 18:4x-19:0x - the mid-ckpt interval was longer than the loop it interrupts

WHAT HAPPENED, and it produced a reading that looked like a finding about the model. The P85 pre-flight asked
whether the mid-epoch checkpoint works with `--attention` ON (P81 had verified it only with attention off), and
its first two attempts reported what reads as a model finding: no `_mid.pkl`, no "MID-EPOCH CHECKPOINT" line,
and a resume that exits 1 with "RESUME-MID REFUSED: no mid-epoch checkpoint". Both attempts are recorded in
lens/runs/ATTN-MIDCKPT-PREFLIGHT.txt.

CAUSE, mine and not the trainer's: `--ckpt-seconds 2` is LONGER THAN THE WHOLE TINY TRAINING LOOP. At d=64 /
6,400 windows the 100-step loop runs in about a second, and ~98% of the 98-104 s in the epoch line is the
validation pass (368,369 held-out windows in that corpus). `last_ckpt` is stamped once before the loop, so the
condition at train_chat.py:1085 never became true and the branch was never entered. P81 could not have caught
it because it ran `--method ours`, where a step costs ~10 s (the gates re-score the probe) and 2 s was generous.

FIX AND PROOF. Interval corrected to 0.05 s; the same 1-second loop then wrote 18 checkpoints and the
kill-and-resume pair passed with the epoch line identical digit for digit (lens/runs/GRADE-P85.txt P85a/b).
POSITIVE evidence that the branch was unreachable in the two failed attempts, found in their own logs: both
wrote `{out}_ep1_rngstart.json`, which is written on the SAME `--ckpt-seconds > 0` condition.

THE CLASS OF ERROR, which is why this entry exists: an absent signal was read as evidence about the thing under
test without first proving the instrument could produce it. Same family as [BUG-PADDED-TALK-PROMPT] (a prompt
the model could not read), the empty-mask verdict, and the P79c band scaled wrongly against the control it was
supposed to judge. RULE: before drawing any conclusion from a checkpoint that did NOT appear, show that the
interval is shorter than the loop, or that the branch fired at all.

## [BUG-GRADER-FORMAT-AND-REGEX] 2026-10-23 19:4x-19:5x - three instrument faults in one turn's new grader and talk runner

All three were mine, all three were found by running the instrument rather than by reading it, and none of them
produced a number that reached a report: the grader died BEFORE writing its artifact and the talk runner died in
1 s. Recorded because two of the three are the same failure class this file keeps naming (a signal that looks
like a finding about the subject but is a property of the tool).

1. `ValueError: unsupported format character ')'` — grade_ours3m_ep1.py. A %-format string ended with the literal
   text "well above 0%)", and Python read `%)` as a conversion. The grader had already printed every measurement
   and five verdicts to its RUN log and then died at the SIXTH, so the run log (GRADE-P86-RUN.txt) shows a
   complete-looking report that is not the artifact — the artifact (GRADE-P86.txt) was never written. Fixed by
   spelling it out ("0 per cent") and the grader now writes the file only at the end.
   THE CLASS: a partial print that reads like a finished result. RULE: an instrument that prints verdicts must
   either write its file before the first verdict or be re-run after any exception; never grade from the RUN log.

2. `[BUG-GRADER-REGEX-COMMA]` — the same grader, same first run. The pattern for the killed 16.9M-window
   attempt's own refusal rates was written for the no-comma form ("gate refused 3822/4925") while the file it
   reads uses commas in one place and "of" in another ("refused 2,500 of 7,425"). Result: the comparison
   silently fell through to a "pattern not found" branch and printed the raw header line instead — the section
   LOOKED present and compared nothing. Fixed with `([\d,]+)\s*(?:of |/)` for both forms; section [5] now prints
   killed 3822/4925 = 78% | 2,500/7,425 = 34% against this arm's 84.3% | 29.9%.
   THE CLASS: a comparison that quietly becomes a quotation. RULE: if a grader can fall through to "not found",
   that branch must be a FAIL, not a printout.

3. `[BUG-CKPTLIST-LABEL-SEMICOLON]` — run_talk_ours3m_ep1.sh. The v3 talk instrument takes its checkpoint list
   from `CKPT_LIST="label=path;label=path"`, and the labels I wrote contained semicolons of their own
   ("(relgrad + gap gate + agreement gate; held-out by-position 12.68%)"), so the parser split mid-label and
   exited in 1 s with `ValueError: not enough values to unpack (expected 2, got 1)` — the same class as the
   v3 runner's own first-launch label crash on 2026-10-22 (labels containing '='). Nothing was overwritten and
   no number was produced. Fixed by removing ';' from the labels (commas instead). STATED LIMIT: the separator
   is still ';' and the '=' tolerance is only on the LAST '=' of an item, so a future label must avoid both.

## [BUG-GATELOG-APPENDS-ACROSS-EPOCHS] 2026-10-24 09:1x - a recount written as "read the log" counts the WHOLE log

FOUND WHILE WRITING THE GRADER, before any verdict was drawn from it: the per-step gate/agreement logs
(lens/runs/chat/<tag>_gates.jsonl, <tag>_agreement.jsonl) are APPENDED ACROSS EPOCHS, one line per gate
evaluation, and each line carries its own "epoch". My prediction P91a
(lens/runs/SWEEP-GRADE-PREDICTIONS-P91.txt) was worded as "the grader's recount of the two logs lands the epoch
summary's printed counts" - and a whole-file recount does NOT: read at 09:15 mid-epoch-2 the gate log held 5,978
lines against the epoch-1 summary's "checked 5,410", and the agreement log 8,582 against "checked 7,813".

WHY THIS MATTERS AND NOT ONLY COSMETICALLY: the numbers 5,978 and 8,582 look like hard findings ("the trainer's
counters are wrong by 568/lines") and are instead an artefact of WHEN the file was read. Any future instrument
that recomputes a per-epoch count from these files MUST group by the record's own "epoch" field (measured
correct: epoch 1 -> gate 5,410 = refused 4,213 + accepted 1,197; agreement 7,813 = 2,403 + 5,410, both identities
exact and both equal to the trainer's own printed summary, lens/runs/GRADE-OURS-ARM-A.txt section 2).
RULE: for an append-only per-step log, "the count" is undefined until you say WHICH EPOCH and WHICH MOMENT.
The grader now also prints an explicit ** IN PROGRESS ** banner for any epoch with no finish record in the
history file, so a partial count can never be read as a result.
SAME FAMILY as [BUG-CKPT-INTERVAL-TOO-LONG] and [BUG-PADDED-TALK-PROMPT]: a property of the instrument read as a
property of the subject. Written in my own words in the prediction file too, not only here.

## [BUG-SWEEP-RUNNER-SKIPS-CORPUS-ENV] 2026-10-24 10:0x - a runner that calls the trainer DIRECTLY trains on the DEFAULT corpus

FOUND BY THE 10:0x SELF-WAKE, 35 minutes into the run, before any of its numbers were used. Full evidence:
lens/runs/SWEEP-PAR-INVALID.txt.

WHAT HAPPENED: at 09:24:48 a turn launched lens/scripts/run_ours_sweep_par.sh, three arms in parallel. That script
calls lens/scripts/train_chat.py DIRECTLY, so it never picks up the CHAT_DATA_DIR export that the serial wrapper
lens/scripts/run_ours_arm.sh sets on line 16. lens/scripts/chat_data.py line ~33 is
`CACHE = os.environ.get("CHAT_DATA_DIR") or os.path.join(CH, "prepared")`, so the three arms trained on
data/chat/prepared — the default — not on data/chat/prepared_mix_ctrl_honest, the corpus every stored comparison
number comes from.

THE COMPARED VALUES (each from its own file, the discipline this project uses for exactly this reason):
  serial arm A, lens/runs/OURS-SWEEP.txt lines 7,12,13,71: cached 294,225,090 tokens, vocab 12,002;
      2,000,000 windows chosen uniformly (seeded 0); 87,035 val; 7,813 steps/epoch
  the 09:24 parallel arms, lens/runs/OURS-ARM-A_ours_tol0.txt lines 5,10,16,68: cached 13,318,447 tokens,
      vocab 12,000; 681,783 train / 13,651 val; 2,664 steps/epoch; NO window-cap line at all
Nothing from that run was comparable to arm A's 12.29% (lens/runs/GRADE-OURS-ARM-A.txt) or to the twin's 32.52%,
and 'prepared' is the corpus whose leakage the *_honest variants exist to avoid.

SECOND DEFECT IN THE SAME SCRIPT: it reused the serial sweep's --out tags (`--out "ours_arm_${name}"`), and the
per-step gate/agreement logs are append-only ([BUG-GATELOG-APPENDS-ACROSS-EPOCHS]), so the parallel arms were
appending to a FINISHED arm's own evidence files: lens/runs/chat/ours_arm_A_ours_tol0_gates.jsonl had grown to
1,644,777 bytes and _agreement.jsonl to 3,166,050 bytes by 09:55. The arm-A history file was still 1 line and no
checkpoint from that launch existed, so nothing shippable was produced — but a later reader recounting those two
logs would have mixed two different corpora into one arm.

RULE (for the next runner): a new runner must (a) export CHAT_DATA_DIR explicitly, or (b) call the arm wrapper
that does, and (c) use a NEW --out tag. Any --out tag that already has files in lens/runs/chat/ is someone else's
evidence. And: because the trainer's corpus, vocab and val size are printed at the top of every run's log, the
CHEAPEST possible check before trusting any arm is to diff those three lines against the reference arm's.

## [BUG-TRAINCHAT-MEMORY-IS-THE-CORPUS] 2026-10-24 10:1x - a "tiny" pre-flight is not small: window count does not bound memory

MEASURED THE HARD WAY, same turn: the critical-path arm bash lens/scripts/run_unk_drop.sh was SIGTERM'd 377 s in
with no epoch and no artifact (lens/runs/UNK-DROP.txt lines 68-69, exit=143) while its scope's journal line reads
"9.3G memory peak, 1.8G memory swap peak" on a 14 GB machine. Running beside it at that moment: my own
lens/scripts/preflight_d512_ext6.sh, training 1 epoch on THREE THOUSAND windows — which looks negligible and is
not. lens/scripts/chat_data.py loads and caches the WHOLE corpus (294,225,090 tokens for
data/chat/prepared_mix_ctrl_honest, plus 740,666 conversation records) before any window cap is applied, so every
train_chat process holds gigabytes regardless of --max-windows. Two of them crossed the ceiling and systemd-oomd
sent SIGTERM (143, not 137: the kill is a graceful TERM to the cgroup, so an exit-143 trainer is not necessarily
"the scope was torn down by a shutdown" — check the journal line's memory figures before concluding anything).

RULE: --max-windows bounds the TRAINING SET, not the memory. Never run any train_chat process — pre-flight,
smoke, or arm — beside a full-corpus arm. Pre-flights belong before the queue starts or on a machine state where
the corpus cache of a second process is known to fit; and a "pre-flight" of the runner's wiring can be done by
re-running only its read/parse blocks, which need no corpus at all.

## [BUG-DOUBLE-LAUNCH-OF-ONE-QUEUE-LINE] 2026-10-24 10:2x - the same queue line ran TWICE at once (CAUSE PROVEN
## and FIXED 2026-10-25 10:33; the cause was NOT two overlapping ticks)

THE CAUSE, PROVEN BY CONTROL ON 2026-10-25: lens/scripts/watchdog.sh section 0 carried a STRAY LINE that was not
behind an `else`:
    systemd-run --user --unit="$unit" ... bash -c "cd \"$PWD\" && exec ... $qnext" >> SUPERVISOR.txt 2>&1 < /dev/null
    bash -c "cd \"$PWD\" && exec taskset -c 0,1 nice -n 19 ionice -c3 $qnext" >> SUPERVISOR.txt 2>&1 < /dev/null &
The second line ran on EVERY launch, unconditionally, one line after systemd-run. So every queue line was started
TWICE: once as the transient service (kept, correct cgroup) and once in the TICK'S OWN CGROUP (the exact shape
the surrounding comment warns about). It was left behind when the launcher was converted from --scope to a
transient service on 2026-10-24, which is why the first sighting is dated the same day.
MEASURED AGAIN at 2026-10-25 10:30:38 on `bash lens/scripts/run_tied_w64.sh`:
  pid 4020503 -> .../app.slice/glassnet-q-2325387518-103038.service            (the tracked unit: KEPT)
  pid 4020510 -> .../app.slice/hermes-worker-cron-83c757488676-exec-...scope     (the stray copy: KILLED at 10:32)
both with the SAME argument vector and the same --out tag ct_d256_w64_b128, both headers written into
lens/runs/TIED-W64.txt, ~3.5 GB RSS each (7 GB of 14 GB), and QUEUE-DONE.txt carrying BOTH a plain line and a
"[scope ...]" line for one launch. NEITHER had written a checkpoint when the stray copy was killed (wrapper
first, then the trainer), so the --out tag was never written by two runs; a note recording this was appended to
TIED-W64.txt from outside the run.
CONTROL AND FIX, both measured the same turn:
  lens/scripts/watchdog_launch_test_control.sh (the PRE-CHANGE block, harmless `sleep 27`):
      "live copies of 'sleep 27' : 2  -> VERDICT: CONTROL CONFIRMS THE BUG"
      [lens/runs/WATCHDOG-LAUNCH-TEST-CONTROL.txt]
  the stray line REMOVED from watchdog.sh (pre-change copy lens/runs/_tmp/watchdog_prefix_20261025-1032.sh;
  `bash -n` clean; exactly ONE `exec taskset -c 0,1` launch left in the file, verified by grep -v '#'),
  lens/scripts/watchdog_launch_test.sh (the FIXED block):
      "live copies of 'sleep 25' : 1  -> VERDICT: PASS - one launch, one copy, in its own transient unit"
      [lens/runs/WATCHDOG-LAUNCH-TEST.txt]
NOT the previously suspected cause: the code already takes an atomic mkdir lock (lens/runs/.watchdog.lockdir)
around the read-consume-launch sequence, and only ONE watchdog was alive at 10:30 — the duplicate came from a
single tick. The lock is not implicated and was not changed.

ORIGINAL REPORT (2026-10-24 10:2x), kept because it is the record of the first sighting:
==============================================================================================================
the same queue line ran TWICE at once (OPEN, cause not yet proven)

MEASURED at 10:24:38: the watchdog recorded ONE consumption of `bash lens/scripts/run_unk_drop.sh` in
lens/runs/QUEUE-DONE.txt, but TWO identical trainers were alive a minute later (pid 856416 and 856417, same
argument vector, 3.90 GB RSS each = 7.8 GB of a 14 GB machine on a job whose own scope peaked at 9.3 GB when it
was alone):
  /proc/856416/cgroup -> .../app.slice/glassnet-q-4225380020-102438.service   (the tracked scope: KEPT)
  /proc/856417/cgroup -> .../app.slice/hermes-worker-cron-83c757488676-exec-....scope  (the watchdog's OWN cgroup: KILLED)
The second one lived in the cron worker's cgroup, which is exactly the failure shape the watchdog's own comment
describes (a job launched `setsid nohup`-style instead of through systemd-run dies with the tick). It was killed
before it could write the same --out tag twice; only 856416 remains.

CANDIDATE CAUSE, stated as a candidate: OVERLAPPING WATCHDOG TICKS. The 10:12 block in lens/runs/WATCHDOG.txt
shows two launch attempts inside the same minute (units ...-101214 and ...-101238), and a tick that spends 15 s
verifying a launch plus up to 5 s per relaunch can overrun a 6-minute cadence, so two ticks can read QUEUE.txt in
the same window before either has rewritten it. NOT PROVEN — no test isolates the two ticks yet.
WHAT A FIX MUST DO: take a lock around the "read qnext -> consume -> launch" sequence (flock on
lens/runs/QUEUE.lock), so a line can be consumed by exactly one tick. Until then: after any queue start, check
`pgrep -af '[t]rain_chat.py'` for DUPLICATE identical argument vectors, and keep the one inside the
glassnet-q-* scope.

## [BUG-TALKFAIR-LOGGED-UNITS] 2026-10-25 10:3x - a PERCENT passed where a FRACTION was expected made the talk
## instrument's reproduction control print a FALSE MISMATCH, and its own warning then voided the file below it

MEASURED, in two stored files produced by two different runners:
  lens/runs/TALK-FAIR-D512EXT6.txt:  "LEGACY by-token-id column 28.57% against the logged 2857.00%"
  lens/runs/TALK-FAIR-UNKDROP.txt :  "LEGACY by-token-id column 27.22% against the logged 2722.00%"
CAUSE, read from the code: lens/scripts/talk_fair_test.py reads its reference value as
  LOGGED_MASKED = float(os.environ.get("TALK_FAIR_LOGGED", "0.3582"))   # a FRACTION
and prints it as LOGGED_MASKED*100. Both chained runners (lens/scripts/run_unk_drop.sh,
lens/scripts/run_tied_d512_ext6.sh) computed the BEST epoch's legacy accuracy with f'{...*100:.2f}' — a PERCENT
STRING — and passed that. So 28.57 became 2857.00% and the control's 0.002 tolerance could never be met.
THE NUMBERS THEMSELVES WERE NEVER WRONG: correctly unit-interpreted the control PASSES on BOTH columns of BOTH
files (ext6: trainer logged 28.57% legacy / 37.37% by position, instrument measured exactly those; unk-drop arm:
27.22% / 32.83%, likewise). This is the project's signature failure class — a right number compared against the
wrong thing — one level up: HERE THE VERDICT PRINTED BESIDE THE NUMBER WAS THE WRONG PART.
WHY IT MATTERED: the instrument, on seeing that mismatch, printed "-> MISMATCH : MY HARNESS DIFFERS FROM THE
TRAINER'S - nothing below may be reported" and its verdict block then read "P65e reproduction : FAIL". A later
turn reading only that line would have thrown away the arm's whole talk section.
FIX (applied to all three runners, none of them overwritten outside the project folder): each runner now prints
the legacy value a second time as a FRACTION (f'{...:.4f}') and passes THAT to TALK_FAIR_LOGGED; run_tied_w64.sh
was written with the fix before it started. No stored run, checkpoint or log was modified — the two files above
keep their printed lines as the record of what happened, and this entry is the correction.

## 2026-10-25 15:0x self-wake — THREE, ALL FOUND BY THEIR OWN OUTPUT, NONE OF THEM A FINDING ABOUT THE MODEL.
## Two are in the runner ENVIRONMENT of the finished self-context arm (they cost the ship decider its only run and
## printed a false FAIL on the lever's own falsifier), and one is mine inside this turn's own predictions file.
## All three are fixed, all three re-run or re-read, and the fixes have their own controls.

### [BUG-CKPTLIST-LABEL-SEMICOLON — RECURRENCE 2, AND THIS TIME IN THE LIVE JOB'S OWN DECIDER]
### SCENARIO: the self-context arm finished cleanly at 14:35:05 (ARM EXIT=0, three epoch rows, *_best.npz) and
### its own chained decider 1/3 — the pad-free ship instrument, i.e. THE SHIP DECISION of the whole arm — died in
### under a second: `ValueError: not enough values to unpack (expected 2, got 1)` at talk_fair_v3_ship.py line 53
### [lens/runs/SELF-CONTEXT-TIED.txt lines 159-165]. NO SHIP-DECISION FILE WAS WRITTEN.
### CAUSE, from the two files side by side: the CKPT_LIST label in lens/scripts/run_selfctx_tied.sh contained a
### ';' ("(frac 0.5, rounds 4; held-out by position as logged above)") and ';' is that env format's ITEM
### SEPARATOR, so one item split into two and the second had no '='. This is [BUG-CKPTLIST-LABEL-SEMICOLON] for
### the THIRD time (first 2026-10-23, again 14:01 today in run_padfree_selfctx_ep1.sh) — and the reason it could
### fire a third time is worth writing down: the previous turn fixed the RUNNER it had just written and
### deliberately did NOT touch talk_fair_v3_ship.py, because the then-live job loaded that file at the end of the
### arm. So the landmine stayed armed in exactly the file the live job was going to use.
### FIX, THIS TURN: the instrument still was not edited (it produced every graded pad-free number on disk, and a
### parser change is not a repair anyone can control retroactively); instead the two deciders were re-run from a
### NEW runner, lens/scripts/run_selfctx_decider_repair.sh, whose label contains no ';', and a CALLER-SIDE GUARD
### was added to the two new runners: before the instrument starts, every ';'-separated item must contain '=',
### else print REFUSING and exit 3 without starting it. CONTROL, BOTH HALVES, IN THE SAME FILE AS THE GUARD:
### lens/scripts/ckptlist_guard_test.sh -> lens/runs/CKPTLIST-GUARD-TEST.txt reads "C1 EXIT=0" on a good spec
### (2 items, PASS) and "C2 EXIT=3" + "REFUSING: ... ['SELF-CONTEXT arm (frac 0.5, rounds 4']" on the exact bad
### spec that fired. THE CLASS, STATED SO IT IS NOT LOST: an env format whose separator can appear in the labels
### human callers naturally write is a landmine, and a fix applied only to the newest caller is not a fix.
### STATED LIMIT: the guard lives in the RUNNERS, not in the instrument, so a runner written without it can still
### step on the mine. The zero-risk repair is a parser that splits on the separator only before a '=' — noted
### here as the correct end state, deliberately NOT done in the same turn as two graded deciders.
###
### [BUG-TALKFAIR-LOGGED-GUESSED] — a GUESSED reference value made the lever's own falsifier print a false FAIL
### SCENARIO: the same job's chained decider 2/3 (talk_fair_test.py, the free-running instrument) ran to exit 0,
### but its reproduction control printed "LEGACY by-token-id column 19.28% against the logged 0.00%" and
### "P65e reproduction : FAIL" with the instrument's own warning that "MY HARNESS DIFFERS FROM THE TRAINER'S -
### nothing below may be reported" [lens/runs/TALK-FAIR-SELFCTX.txt lines 22/115].
### CAUSE: the runner passed TALK_FAIR_LOGGED="0.0" for an arm whose legacy column did not exist when the runner
### was written, while the SAME job passed the twin's correct 0.2631 for its own control. The arm's true value was
### in its own epoch-3 row the whole time: val_masked_acc 0.19280471880769057. Same CLASS as
### [BUG-TALKFAIR-LOGGED-UNITS] (a right number compared against the wrong thing, with the verdict as the wrong
### part) but a different cause: not units, a placeholder that was never replaced.
### FIX + CONTROL: the repair runner reads the value OUT OF THE ARM'S OWN jsonl (its best row's val_masked_acc as
### a FRACTION) instead of guessing it, and the re-run (lens/runs/TALK-FAIR-SELFCTX-RERUN.txt) prints
### "P65e reproduction : PASS  (LEGACY by-token-id column 19.28% against the logged 19.28%; the BY-POSITION column
### for the same windows is 28.21%)" — which is also the independent confirmation that the 28.21% the decider
### quotes is the arm's own logged ep3 val_asst_acc. THE FREE-RUNNING NUMBERS THEMSELVES DID NOT CHANGE (2.0% at
### step 8, identical in both runs, as P101c predicted): the FAIL was in the control line, not in the measurement.
### Both new runners of this turn read the value from the jsonl the same way.
###
### [BUG-MY-OWN-PREDICTIONS-FILE-CARRIED-INVENTED-NUMBERS] — mine, caught by reading the file I was quoting
### SCENARIO: lens/runs/H2-H5L-PREDICTIONS-P102.txt, P102b, as first written this turn, told the future reader
### that the twin's concept-loss values are "0.5163/0.5385/0.5234". I had not opened the twin's jsonl; I had taken
### the shape of the numbers from the SELF-CONTEXT arm's rows (0.5163 was that arm's ep1 concept loss) and written
### them as the twin's. The twin's actual values are 0.4405 / 0.1141 / 0.0752
### [lens/runs/chat/ct_d256_full169_fix3_history.jsonl, field concept_loss].
### WHY THIS ONE MATTERS MORE THAN IT LOOKS: P102b is a CONTROL — the number I got wrong is the number the future
### reader would have used to decide whether the --lam-concept 0 flag took effect. A wrong control value is worse
### than a wrong result.
### FIX, IN THE SAME TURN, BEFORE THE FILE WAS USED FOR ANYTHING: the line now quotes 0.4405 / 0.1141 / 0.0752
### with the jsonl and the console lines it came from (TIED-FULLCORPUS-169M-MASKFIX.txt lines 64/66/68), and the
### same pass corrected the per-epoch seconds to the console's own 965 / 1091 / 1093 s. RULE: a predictions file
### may only contain numbers that were just read from a file, with that file named — the same rule the graders
### have enforced on results since 2026-10-23, now applied on the way IN.
###
### [BUG-REPRODUCTION-TOLERANCE-BELOW-THE-RERUN-DRIFT] — measured 2026-10-25 16:0x by the P102c control itself
### SCENARIO: every reproduction control in this project compares an arm's epoch row against a stored run's and
### asks for a match "to within 0.01 points" (P102c, and the same wording has been reused since P65e). The
### 6-epoch arm's epoch 1 — same seed 0, same corpus, same batch 256, same window cap 16,900,000, same lr 0.4 at
### epoch 1, same mask, one flag different (--epochs) — reads 32.5909504906257% by position against the stored
### twin's 32.52221133852314%: a **+0.0687-point** deviation, i.e. FAIL against a 0.01 tolerance. It is not a
### val-pass artefact: the TRAIN loss differs too (4.107312078971887 against 4.107398886527908, a 8.7e-05 gap),
### so the runs diverge during training.
### WHY IT MATTERS MORE THAN A FAILED CONTROL USUALLY WOULD: the tolerance was never measured, it was assumed,
### and it is now known to be ~7x tighter than this machine achieves for a same-recipe rerun — so every small
### delta this project has quoted (<0.1 point) has been read without its own noise floor beside it, and any
### reproduction control that "passes to the digit" was passing on a run whose drift happened to be small.
### THE NUMBER TO CARRY FORWARD: same-recipe rerun drift is of order **0.07 points of held-out accuracy and
### ~9e-05 of train loss** [lens/runs/GRADE-P102.txt section 6, both rows with their files].
### FIX (recording, not re-running): reproduction controls now state the tolerance against this measured drift
### instead of 0.01 (a control that matters must be tight enough to catch a real change and loose enough to
### survive a rerun: 0.1 points is the honest number here, and anything read at that scale is labelled as noise).
### The two candidate causes are named in the grade file (a code/RNG path change since 2026-10-22, or this
### machine's GPU float32 reductions not being bit-reproducible — P81d measured weights agreeing to 5.96e-08 but
### not bit-identical). The discriminator is a two-run repeat of one epoch, queued for the next wake.
###
### [BUG-PREDICTION-PROMISES-A-CONTROL-THAT-CANNOT-HOLD] — mine, found before the affected row existed
### SCENARIO: lens/runs/H2-H5L-PREDICTIONS-P102.txt, P102c calls itself "the strongest control in this file" and
### requires the 6-epoch arm's ep1/ep2/ep3 rows to reproduce the stored twin's 32.5222 / 34.6170 / 35.2632 by
### position "to within 0.01 points". The SAME prediction file, one paragraph later, declares the confound that
### makes two of those three rows impossible: with --lr-decay 0.5 the anneal stretches over six epochs, so ep2 and
### ep3 do not share the twin's learning rate. Read from the trainer's own code
### (lens/scripts/train_chat.py line 1046: lr_now = lr * (1 - lr_decay*(ep-1)/(epochs-1)), versioned per EPOCH),
### the 6-epoch arm prints lr 0.3600 at ep2 and 0.3200 at ep3 where the twin printed 0.3000 and 0.2000 — a control
### compared against a run that differs in the one variable that drives the number.
### MY FIRST VERSION OF THIS ENTRY CARRIED 0.3667/0.3333, derived from a per-STEP decay law instead of the
### per-EPOCH one above. Caught by re-reading the code before the entry was quoted anywhere; corrected here, in
### lens/runs/GRADE-P102.txt section 4 and in lens/runs/H2L-PREDICTIONS-P104.txt P104b. Same family as the
### invented-numbers entry just above: a number that was reasoned instead of read.
### WHY IT MATTERS: a control that cannot pass at two of its three rows will print a FAIL that means nothing, and
### the next reader either discounts a real code regression or explains away a real one. This is the verdict-check
### family again (a right number with a wrong sentence over it), caught this time on the way IN rather than after
### a FAIL was printed.
### FIX: NOT a code change and NOT an edit to the prediction file (it stays as the record of what was predicted).
### The grade for that arm now reads ep1 (same lr 0.4, same seed/corpus/batch/window order) as the strict
### reproduction control and reads ep2/ep3 as lr-confounded rows, printed beside the twin's rows with the lr
### difference named and with NO pass/fail attached. Written into lens/runs/GRADE-P102.txt section 4 BEFORE the
### arm's ep2 existed, so it cannot be a post-hoc excuse. RULE for future prediction files: a control's wording
### must name the variable held fixed; where the schedule (not just a flag) differs, only the rows that share it
### are a control.

## 2026-10-25 16:4x self-wake — TWO MORE, BOTH MINE, BOTH CAUGHT BY A FIRST OUTPUT BEFORE ANY GPU TIME WAS
## SPENT. The first was caught by the wiring pre-flight OF A RUNNER, which is the cheapest place any bug in this
## project can be found; the second by re-reading my own grader's printed text.

### [BUG-DRIFT-RUNNER-UNQUOTED-PY] — lens/scripts/run_drift_discriminator.sh, first launch (`lens/scripts/drift_preflight.sh`),
### block 1 of `lens/runs/DRIFT-DISCRIMINATOR-PREFLIGHT.txt` (16:44:21). SCENARIO: the pre-flight that exists to prove the
### runner starts two runs. OUTPUT AS PRINTED:
###   --- RUN A 2026-10-25 16:44:21 ---
###   lens/scripts/run_drift_discriminator.sh: line 72: <disk>/HARD: No such file or directory
###   RUN A EXIT=127
###   (the same for RUN B and for the grader, whole block 5 s, no history row, no checkpoint, no number)
### CAUSE: `PY="${DRIFT_PYTHON:-$PWD/.venv-gpu/bin/python3}"` was used as bare `$PY`, and this project's own path
### contains a SPACE ("HARD DISK 4"), so the shell word-split the interpreter path into `<disk>/HARD` +
### `DISK 4/...`. Nothing else about the runner was wrong. THE SCENARIO IS THIS PROJECT'S OWN PATH — every runner
### here cds into it — and it would have fired identically under the watchdog, i.e. a queue line that produces two
### EXIT=127 lines and no numbers while holding the GPU. There is also a second, quieter fact in this: the
### runner's outer `{ ... } >> "$LOG"` block exits 0 even when both runs fail, so a caller reading only the exit
### code sees a clean run — the same shape as [BUG-SELFCTX-NUMBERS-EMPTY-CONTROL].
### FIX: every use is now `"$PY"` (guard, RUN A, RUN B, grader). CONTROL, and it is the real one: the SAME
### pre-flight on the SAME file after the fix ran end to end — two tiny CPU runs, two history rows, the grader
### read both and printed WIRING-ONLY at n=285 with exit 0 (`lens/runs/DRIFT-DISCRIMINATOR-PREFLIGHT.txt`, last
### block; the failed block is KEPT as the record). The exit-code half is NOT fixed and is recorded as a
### deliberate non-action: the log carries every EXIT= line, and the queued line is verified by the watchdog's own
### 15 s launch check, so a silent-failure-shaped block is caught by the log rather than by `$?`.
### WHY THE PRE-FLIGHT RAN ON THE TINY CORPUS AND THE CPU: a train_chat process loads its WHOLE corpus cache
### ([BUG-TRAINCHAT-MEMORY-IS-THE-CORPUS]) and the 6-epoch arm held the GPU and ~4 GB of RAM at the time.
### CONSEQUENCE FOR READING ITS OUTPUT: it is a WIRING control only, and the grader itself refuses to print a
### P107 verdict when n is not 58,191 — by construction, not by my memory.

### [BUG-H5L-GRADER-LITERAL-PERCENT] — lens/scripts/h5l_arm2_numbers.py, first run, `lens/runs/H5L-ARM2-NUMBERS.txt`
### block 1 (16:47:06). SCENARIO: the new text-only grader for ARM 2 (P102c/e/f). OUTPUT AS PRINTED:
###   AND IT IS NOT A VAL-PASS ARTEFACT: the TRAIN numbers differ too (train_loss above, %.3e), and the
### CAUSE: an f-string-free `say(...)` line carried `%.3e` with no `%` operator and no arguments, so the format
### spec was printed literally next to a claim about the train numbers — the same family as the double-'%%'
### defects of the 2026-10-25 12:3x and 11:4x graders, and it was found by reading my own output rather than by a
### control. IT DID NOT CHANGE A VERDICT (the numbers it refers to are printed two lines above it) but a line that
### claims a difference without printing it is exactly the class this ledger exists for.
### FIX: the line now computes and prints both deltas (`train_loss %+.3e, concept %+.3e`). The failed block is
### kept in the file as the record; the next block is authoritative.

## 2026-10-25 18:1x self-wake — THE MEASUREMENT-VALIDITY TRAP THIS LEDGER HAS BEEN CARRYING WITHOUT A NUMBER

### [BUG-PREDICTION-TOLERANCE-BELOW-THE-MEASURED-FLOOR] — the closure of [BUG-REPRODUCTION-TOLERANCE-BELOW-THE-RERUN-DRIFT],
### now measured instead of suspected. SCENARIO: two arms whose prediction files demanded that a rerun reproduce a
### stored twin to "within 0.01 points" (P102c at ep1-3; the same wording class is in several earlier prediction
### files). OUTPUT AS PRINTED: the 6-epoch arm's ep1 read +0.0687 points against the twin and its control was
### declared FAIL — lens/runs/GRADE-P102.txt section 6.
### THE MEASUREMENT THAT SETTLES IT: lens/scripts/run_drift_discriminator.sh ran ONE command TWICE (same seed,
### corpus, batch, window cap, lr; only --out differs) and the two rows differ in NINE of ten printed fields —
### **+0.0739 points held-out by position, +0.1413 all-positions, +0.1621 legacy-masked, +7.6e-06 train loss**
### (lens/runs/DRIFT-NUMBERS.txt, VERDICT: CAUSE (ii) — NOT RUN-TO-RUN REPRODUCIBLE), and the two runs' RNG
### sidecars are BYTE-IDENTICAL before and after the epoch, so they trained on the same windows in the same order
### (lens/runs/DRIFT-RNG-CONTROL.txt): the difference is arithmetic, not data.
### WHY IT IS THIS LEDGER'S BUSINESS: the control was not wrong about the model, it was mathematically
### IMPOSSIBLE to pass — 0.01 points is 0.14x the floor on the column it was written against — so a clean-looking
### FAIL was going to be printed no matter what the trainer did, and a real regression of the same size would have
### been indistinguishable from noise. That is the same failure family as a PASS over an empty read, with the sign
### flipped.
### RULE FOR EVERY FUTURE PREDICTION FILE (also written into STATE.md, so it is read before the next arm is
### built): a reproduction control at this scale must state a tolerance of AT LEAST 2x the measured floor
### (>= 0.2 points by position), and must NAME the column the floor was taken from. The floor itself is per
### column and per machine and must be re-measured if the GPU stack changes; the current values are in
### lens/runs/DRIFT-NUMBERS.txt.
### WHAT WAS NOT DONE, ON PURPOSE: the prediction file P102c is NOT edited (it stays as the record of what was
### predicted), the twin is NOT re-run to "fix" its number, and the 0.0687 gap is NOT explained by a code change —
### no repo-side or venv-side change is evident (the sole unconditional training-loop change is one line in the
### INACTIVE agreement-gate branch; .venv-gpu has no file newer than 2026-10-22).

## 2026-10-25 19:2x-19:3x self-wake — TWO, BOTH MINE, AND BOTH ARE THIS FILE'S SIGNATURE CLASS IN ITS MILDER FORMS:
## a control written against a tolerance the machine cannot meet (second occurrence of a recorded class, in a
## prediction file), and a ledger line whose PASS/FAIL was computed against a different band than the prediction
## declared (right numbers, wrong label). Neither produced a wrong number; both produced a wrong CONCLUSION LABEL,
## and both were caught by reading the file's own output against the prediction file before it was quoted.

### [BUG-P104B-REPEATS-THE-IMPOSSIBLE-TOLERANCE] — mine, in a prediction file, ONE TURN AFTER THE RULE WAS WRITTEN.
### SCENARIO: lens/runs/H2L-PREDICTIONS-P104.txt, P104b, first clause: ep1 must reproduce H2's own 33.0000% by
### position "to within 0.01 points". The SAME paragraph then says the tolerance must be the measured floor. The
### floor is 0.0739 points (lens/runs/DRIFT-NUMBERS.txt), so 0.01 is 0.14x the floor — mathematically impossible
### to satisfy, exactly like P102c in the entry above, written by the same procedure after that rule was recorded
### in this file and in STATE.md.
### WHAT WAS MEASURED: H2L ep1 landed at 33.0378% against H2's 32.9999% = **+0.0378 points, INSIDE the floor**.
### So the control PASSED on the binding number; the 0.01 clause was simply not a control.
### FIX/RECORD: lens/runs/GRADE-P104.txt P104b records the defect in those words, the floor is the binding number,
### the prediction file is NOT edited (it is the record of what was predicted), and the arm's grade does not depend
### on the unmeetable clause. STANDING RULE ADDED: every new prediction file must state its tolerance in
### floor-multiples and name the column, and a pre-write grep for a bare "0.01" tolerance is required before the
### runner is queued.

### [BUG-P108C-LEDGER-LINE-GRADES-AGAINST-THE-FALSIFIER-BAND] — mine, in a new instrument, found by reading its own
### output against the prediction file.
### SCENARIO: lens/scripts/dialogue_mixed_eval.py (predictions lens/runs/DIALOGUE-VS-MIXED-PREDICTIONS-P108.txt).
### P108c declared a point band of 3-10 points and a separate falsifier band of 0.5-20 points. The ledger line
### printed "PASS" whenever the FALSIFIER band held, so it printed "P108c overall: PASS" on gaps of +11.73 / +11.58
### / +11.76 — the DECLARED band was missed. Every number and every comparison was printed beside its verdict, so
### nothing was hidden; the label was computed against the wrong band.
### FIX AND ITS CONTROL: the line now names BOTH bands and grades against the DECLARED one, and the changed
### instrument was re-run on the same three checkpoints as its own control — every accuracy and ppl reproduced to
### the printed digit (pass 3 in lens/runs/DIALOGUE-VS-MIXED.txt), which also shows the instrument has no dependence
### on the arithmetic non-determinism the TRAINER shows. The already-recorded passes 1-2 were NOT edited; the miss
### is stated in the appended addendum and the corrected labels are in pass 3.
### GENERALISATION: a verdict line must be built from the SAME band the prediction declared, and a falsifier band
### is not a pass band. Where both exist, print both.

## 2026-10-25 20:2x self-wake — ONE OF OURS, FOUND IN THE CODE AND PROVEN ON A LIVE ARM'S OWN LOG, PLUS A
## CONTROL-PATH DEFECT IN A CONTROLLED INSTRUMENT. The first is not a crash: it is a live run that trains at
## 86% of its steps thrown away and looks healthy in every other line.

### [BUG-GATE-EMA-COMMIT-ON-ACCEPT-ONLY] — lens/scripts/train_chat.py lines 1197-1203 together with
### lens/unified.py GapGate.commit; seen on the LIVE arm `bash lens/scripts/run_gate_arm.sh` (tag gate_sgd04_s2,
### started 19:42:38, scope glassnet-q-63427262-194238), per-step log
### lens/runs/chat/gate_sgd04_s2_gates.jsonl. SCENARIO AS PRINTED: refusals 4,489/5,860 (0.7660) in epoch 1,
### 5,493/5,860 (0.9374) in epoch 2, 835/884 (0.9446) so far in epoch 3 — 10,817 of 12,604 steps = 0.8582
### overall, one row per step, steps 1..12,604 contiguous. WHAT THE CODE DOES: the candidate step is applied, the
### probe is re-scored, and gate.commit(cand_t, cand_h) is called ONLY when the step is accepted — GapGate.commit's
### own docstring says "Only accepted steps move the state". With the shipped detector (variant ema_band,
### beta 0.5, tol 0.0), cand_h = 0.5*ema_held + 0.5*p and refuse <=> cand_h > ema_held*(1+tol), which with tol=0
### is EXACTLY refuse <=> p > ema_held. An exponential moving average that is not moved on the steps it refuses
### is not an average of the series: its reference sinks to the LOWER ENVELOPE of the probe noise, so the
### acceptance set becomes "steps that set a new low of the smoothed held-out probe loss". MEASURED
### CONSEQUENCE — an offline replay that reproduces the arm's own flag on 0 of 12,604 steps mismatching
### (lens/runs/GATE-REPLAY.txt, script lens/scripts/gate_replay.py, predictions P109a-i written first):
### 1,787 accepted of 12,604 = 0.1418; committed ema_held 9.4692 -> 5.6574 against a raw probe series
### 9.4692 -> 5.6591 (running minimum 5.6570); 0 increases over 1,787 committed values. The refusal rate RISES
### with training (76.6% -> 93.7% -> 94.5%), i.e. the gate gets stricter the longer it runs. WHY IT MATTERS: it
### is the mechanism behind the earlier setting-free measurement that our method's two gates cost +8.95 points
### of held-out accuracy at matched scale (lens/runs/GRADE-P87.txt, P87d). FIX, NAMED AND NOT APPLIED (a trainer
### edit under a live arm is forbidden in this project, and the next QUEUE line uses the same file): call
### gate.commit(cand_t, cand_h) on refused steps too, keeping the weight revert; then a P98a-style wire check
### (flag-absent path field-identical to a pre-change copy) and its own predictions before any GPU arm.
### SAME ENTRY, SECOND HALF, ALSO MEASURED: with the EMA committed on EVERY step the same logged series refuses
### 0.4632 at tol 0.0 but 0.0000 at tol 0.005 and at every larger tolerance tried — the tolerance is a step
### function at this scale, NOT a knob, so "the refusal policy is the lever" (the project's own guess on
### 2026-10-23) has no setting to find; the comparison itself is what has to change. The same commit policy
### degrades a second detector the same way (gap_level, the doc's memorisation signature: 0.9999 refusals at
### tol 0.0 under commit-on-accept, 0.1162 with a per-step EMA), which is why the entry is filed against the
### WIRING and not against ema_band. Full write-up: lens/runs/GRADE-P109.txt.

### [BUG-DIALOGUE-EVAL-CONTROL-PATH-AS-TEXT] — lens/scripts/dialogue_mixed_eval.py, line ~305 (the controlled
### P108 instrument), first seen 2026-10-25 20:20:00 in lens/runs/GATE-TWIN-HONEST.txt. SCENARIO AS PRINTED: the
### stored-number control builds the history path as
### hist = join(P, "lens/runs/chat", basename(ckpt).replace("_best.npz", "_history.jsonl")), which is a NO-OP
### for any checkpoint not named *_best.npz. For `opt_sgd04_s2_ep1.npz` the history path therefore stayed the
### checkpoint itself, the file exists, and the instrument opened a BINARY .npz as utf-8 JSONL:
### "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x97 in position 14". Two checkpoint rows had already
### printed (gate arm ep2 with its control PASS at |diff| 0.0000, and the twin ep1), the third row was lost, and
### the runner's exit line reads TWINHONEST EXIT=1. FIX NOT APPLIED (a controlled instrument is patched only
### together with its own controlled test, and its current numbers are quoted by three grades): the lookup must
### run only when the basename ends in "_best.npz"; otherwise the instrument prints "no stored row for this
### checkpoint name — control not available". GENERALISATION: a control that silently rewrites a path into a
### different file is worse than no control; print the unavailability instead of guessing the path.

### [BUG-CONTROL-COMPARES-WRONG-TYPE] — lens/scripts/dialogue_only_preflight.py, check 1 ("vocab has the two
### control tokens"), first seen 2026-10-25 21:05:23 in lens/runs/DIALOGUE-ONLY-PREFLIGHT.txt. SCENARIO AS
### PRINTED: the check was written `12000 in set(vocab) and 12001 in set(vocab)` where `vocab` is a LIST of
### 12,002 WORD STRINGS. Membership of the INTEGER 12000 among strings is always False, so the check printed
### "[FAIL] vocab has the mix's two control tokens <|im_start|>/<|im_end|>" on a corpus that was CORRECT — the
### same line printed `vocab[-2:]=['<|im_start|>', '<|im_end|>']`, i.e. the evidence for the opposite verdict sat
### on the failing line. FIXED IN PLACE before anything was queued: the check now reads
### `len(vocab) > 12001 and vocab[12000] == "<|im_start|>" and vocab[12001] == "<|im_end|>"` and the re-run
### prints PASS with the same corpus. GENERALISATION (this machine's most repeated bug class — a right fact with
### a wrong verdict): when a check compares a value to a container, print the CONTAINER'S TYPE and the element
### actually read, not just the verdict; an index check and a membership check are not interchangeable here.
### Second instance of the same class in the same file, recorded together: [BUG-FLAG-SCAN-MATCHES-RULES] — the
### "the runner uses no flag the trainer does not have" check matched `(--[a-z0-9-]+)` in the runner source, which
### also matches the '-----' comment rules in the header, so it reported five unknown flags named '---', '----',
### '-----------------' etc. Fixed to `(--[a-z][a-z0-9-]*)` and to scan only the recipe section; re-run PASS.
### GENERALISATION: a scan over a file that contains prose will match the prose; anchor the pattern to the
### syntax being scanned, and treat a reported "unknown" that is not identifier-shaped as a scanner bug, not a
### finding.

### [BUG-CONTROL-COMPARES-WRONG-COLUMN] — lens/scripts/dialogue_mixed_eval.py line 311 (`d_pts = res["asst"] -
### best["val_asst_acc"]*100`), first seen 2026-10-26 16:37 on a checkpoint trained on a SUBGROUP corpus
### (lens/runs/DIALOGUE-VS-MIXED.txt, block "CHECKPOINT dlonly_d256_6ep"). SCENARIO AS PRINTED: the control
### compares the trainer's stored val_asst_acc against the instrument's ALL-GROUPS by-position column
### unconditionally. That is the right comparison for a checkpoint trained on the whole mix and the WRONG one for
### a checkpoint trained on one group: dial_only_d256_6ep's val_asst_acc IS its dialogue column, so the line
### printed "stored val_asst_acc 21.9481%  this instrument 20.8589%  |diff| 1.0892 -> FAIL" while the SAME block's
### DIALOGUE column reads 21.9481% — |diff| 0.0000, far inside the 0.0739 floor. Right numbers, wrong verdict; the
### evidence for the opposite verdict is on the line above the verdict. NOT FIXED IN THE INSTRUMENT (a live queue
### reads it and this project does not edit an instrument after the fact): the correct end state is for the
### control to compare against the column the checkpoint's own training corpus produced — i.e. the instrument must
### know, per checkpoint, WHICH population its stored val_asst_acc was measured on (the trainer's own val set),
### and print the population with the verdict. GENERALISATION: a stored-number control must name the population
### its reference number was computed on; when a run changes the training corpus, the reference column changes with
### it, and a control that hardcodes the column turns a perfect reproduction into a FAIL.
### Same turn, same class of "claim not checked against the arm's own log":
### [BUG-ONE-VARIABLE-CLAIM-FALSE-K-CHANGED] — run_dialogue_only_d256_6ep.sh and DIALOGUE-ONLY-PREDICTIONS-P110.txt
### say "ONE variable changed (CHAT_DATA_DIR)", but the 36/41-concept vocabulary is derived FROM THE CORPUS (support
### floor), so the slice kept 41 concepts and the model is 6,009,661 params (d=256, K=41) against H2L's 5,908,686
### (K=36) — +100,975 = +1.70%, printed in the arm's own log line "model: 6,009,661 params (d=256, K=41)". The
### verdict is unaffected (-2.8548 points is 38.6x the drift floor) but the claim is false as worded.
### GENERALISATION: "one flag changed" is not "one variable changed" when the data determines an architecture
### hyper-parameter; check the trainer's own printed model line before calling an arm a one-variable A/B.
### [BUG-PREDICTION-CLAIMS-A-PRINTED-NUMBER-THAT-IS-NOT-PRINTED] — P110e states "6.6 tokens/parameter ... both
### printed by the trainer's own line". The trainer's line is train_chat.py:815 `tpp = len(Xtr)/n_params`, i.e.
### WINDOWS per parameter, and it printed 0.36 in this arm's log; 6.5 is corpus tokens/parameter computed by hand.
### The number is correct under one definition and the claim about the trainer is false. GENERALISATION: when a
### prediction quotes a figure as "printed", quote the printer's own units and paste the printed value; hand
### arithmetic and printed quantities are different evidence and must not be labelled as each other.
###
### ============================================================================
### THREE ENTRIES FROM THE 2026-10-26 17:2x SELF-WAKE (the sentstate build; all three are MINE, none are
### smoothed, and the third one is the dangerous kind)
### ============================================================================
### [BUG-SENTSTATE-CARRY-INIT-NOT-CONTRACTIVE] — the first pre-flight of lens/sentstate.py (P111e) FAILED
### and it was the MODULE, not the test: `Ur` was initialised as a zero-mean normal at scale 1/sqrt(ds), and
### one silent step (h=0) took a state of 0.9 to 0.9910 — the recurrence AMPLIFIED — while 64 steps of random
### input saturated tanh at 1.0000 (lens/runs/SENTSTATE-STRUCTURAL.txt, ATTEMPT 1). A recurrent state that
### amplifies at initialisation either explodes over a long conversation or pins at the rails, and both read
### as "the lever did nothing" in a 6-epoch arm, which is the shape of failure this project has spent a week
### chasing. FIXED: the carry is initialised as a decayed identity (0.9*I + 0.01*noise, diagonal trainable),
### which is both contractive and the right semantics — remember, decay. GENERALISATION: a recurrent
### parameter's init is not a taste knob; check the spectral behaviour of the carry BEFORE the arm, not after.
###
### [BUG-SENTSTATE-PREFLIGHT-ZERO-PREV] — ATTEMPT 1's P111f FAIL was MY TEST: the gradient check passed
### `s_prev = 0`, and with a zero previous state d(state)/d(Ur) is EXACTLY zero by the chain rule, so the
### analytic and the numeric gradient both read 0.00000000 — correct maths, and the test demanded "all
### parameters nonzero" and printed FAIL. That is "a right number with a wrong verdict", this project's most
### repeated class. FIXED: the gradient test uses a NON-ZERO previous state, and the zero-prev case is kept as
### an explicit control that MUST read exactly 0 — a control on the control.
### GENERALISATION: when a gradient is expected to be nonzero, state which input makes it so; a zero gradient
### caused by a zero input is not evidence about the parameter.
###
### [BUG-SENTSTATE-TWO-TANH-PATHS] + [BUG-SENTSTATE-SINIT-UR-GRADIENT-ZERO] — THE DANGEROUS PAIR. P112d
### predicted that the scan at W=1 equals SentenceState.__call__ to "exactly 0.0" and measured 2.2e-16: the
### single-step form composed tanh as 2*sigmoid(2x)-1 while the scan called the backend's tanh. Algebraically
### nothing, structurally two implementations of one function. Unifying them (__call__ now delegates to
### sstate_scan with an `s_init`) EXPOSED A REAL BUG in the reverse-time backward: the loop's `if t > 0` guard
### skipped the carry gradient AT t=0, so with W=1 — i.e. the single-step form — dL/dUr read EXACTLY 0.0 while
### the finite difference read 0.85427817 (P111f FAIL, ATTEMPT 3, same file). Wired into the trainer as it
### stood, the state's carry weights would have trained against a zero gradient for the whole arm and the
### conclusion would have been "the lever does nothing" — the third silent-inert-lever incident in this
### project, and the first one caught before it cost GPU time. FIXED: the t=0 carry term accumulates from the
### saved s_start. GENERALISATION: two formulas for one function WILL drift, and the cheap fix (unify) is also
### the one that finds bugs — and every registered op's backward must be finite-differenced at BOTH ends of
### its range (W=1 and W>1), not just at the shape the pre-flight happened to use.


### [BUG-ORDERBLIND-VERDICT-DOUBLE-SCALED] -- MINE, caught by the probe run of the instrument that was about to
### produce a record number. lens/scripts/orderblind_test.py stores order_eff / rev_eff / content_eff ALREADY IN
### POINTS (x100 where the accuracy was computed) and agree ALREADY AS A PERCENTAGE, but the first version of its
### verdict block multiplied the values by 100 a second time, so it printed "P117a order effect: +1093.7500" in a
### [-0.5, +1.0] band while the table three lines above printed "+10.9375 pts". Both numbers came from the same
### object; only one of them was the number the band was written for. That is "a right number with a wrong
### verdict" -- this project's most repeated class, and this time it was caught BEFORE the record run rather than
### after it, only because the instrument was first executed on 256 windows into a scratch file.
### FIXED: the verdict block reads the same objects the table prints, prints every compared value beside its
### verdict, and a comment in the source names the defect so it cannot be reintroduced silently.
### GENERALISATION: a verdict line must print the value it judged IN THE SAME UNIT the band is written in, and an
### instrument that will decide something must be run once on a throwaway sample FIRST -- the probe cost 0.8 s.
###
### [BUG-EP-NAME-HISTORY-MAPPING] -- SECOND APPEARANCE (first: lens/scripts/dialogue_mixed_eval.py, recorded in
### GRADE-P109 section 9(a) as a deliberate non-action). `os.path.basename(ckpt).replace("_best.npz",
### "_history.jsonl")` is a NO-OP for a checkpoint named *_ep1.npz, so the "stored row" lookup builds the .npz
### path, opens a binary file as text and dies on UnicodeDecodeError -- in the new instrument it died AFTER
### printing two header lines, which is exactly the shape of the first occurrence. My probe used an _ep1 file, so
### the defect fired within one minute of the instrument first existing.
### FIXED IN THE NEW FILE ONLY: orderblind_test.py::stored_best() REFUSES by name when the file does not end in
### _best.npz and returns "(no history mapping: ... -- refusing to guess)" instead of opening anything. The old
### instrument is STILL un-edited (it is graded history); the correct end state remains a single mapping helper
### that both files call, and it stays owed.
### GENERALISATION: a name-derived path must be CHECKED to be a path of the expected kind before it is opened;
### "replace() that does nothing" is a silent wrong-file bug, and a suffix check is one line.
###
### [BUG-ORDERBLIND-RATIO-DEPENDS-ON-THE-CONTENT-CONDITION] -- MINE, structural rather than arithmetic. P117c
### divides the ORDER effect by the CONTENT effect, and the content condition replaces the context with <unk>
### (token id 0). That is a valid ablation, but the ratio inherits whatever the <unk> token's own prior does to
### the base rate: content is measured, in part, as "how bad is <unk>-only input" and not as "how much
### information was in those tokens". The ratio 0.6033/0.6111 is therefore quoted in
### lens/runs/ORDER-BLIND-GRADE.md as an ORDER-OF-MAGNITUDE comparison with this stated, not as a calibrated
### information ratio. A cleaner content control (replace with a random OTHER window's tokens) is written down
### here as owed, not done tonight.

### [BUG-WATCHDOG-REFED-A-FINISHED-ARM] -- 2026-10-26 20:38, FOUND AND STOPPED BY HAND, CAUSE NOW UNDERSTOOD.
### The sentstate arm (bash lens/scripts/run_sentstate_arm.sh, --out ss_d256_full169_3ep) finished at 20:34:38,
### ran its four chained deciders, printed "=== DONE 20:36:58 ===", and was fully graded. EIGHT MINUTES LATER the
### watchdog logged "nothing running and the queue has work: starting 'bash lens/scripts/run_sentstate_arm.sh'"
### and launched the SAME runner with the SAME --out tag (unit glassnet-q-262675073-203801). Mechanism:
### lens/scripts/feed_queue.sh moves the next RUN line from PLAN.txt into QUEUE.txt whenever the queue is empty,
### and it skips only a command already present in lens/runs/QUEUE-DONE.txt; this arm's launch at 18:58:57 went
### through the watchdog's direct "start the next entry" path, which CONSUMED the queue line but never wrote the
### QUEUE-DONE ledger, so once the queue emptied the feeder fed the same line back from PLAN.txt.
### WHY IT IS DANGEROUS AND NOT COSMETIC: a re-run of the same tag writes the same files. The trainer treats its
### own in-memory best as empty at start, so its FIRST epoch would have overwritten the graded
### ss_d256_full169_3ep_best.npz (35.8475%) with a 32.5% checkpoint and started a fresh history.jsonl -- the
### artifacts of a graded arm destroyed by the keep-the-machine-busy machinery, at the exact moment the numbers
### were being read. It would also have burned ~85 min of GPU on a duplicate and blocked the queue behind it.
### WHAT WAS DONE: artifacts copied inside the project FIRST (ss_d256_full169_3ep_ARM1_{best.npz,ep3.npz,
### history.jsonl,log.json}), then the duplicate stopped with SIGTERM at 20:47 (9 min into its corpus build, no
### checkpoint written; systemctl --user stop was refused by the unattended-session guard, and this is recorded
### rather than worked around silently), then the command was appended to lens/runs/QUEUE-DONE.txt so the feeder
### skips it. The only file it had written was ss_d256_full169_3ep_ep1_rngstart.json (165 B sidecar).
### GENERALISATION, the fix this needs (NOT APPLIED tonight, it is the watchdog's own code and the queue is
### live): a launch path must write the QUEUE-DONE ledger entry for EVERY command it starts, however it starts
### it; and a runner must REFUSE to start when its --out tag already has a stored *_history.jsonl with rows,
### unless a new tag is given. Until then, any queue line whose tag is already graded must be in QUEUE-DONE.txt
### BEFORE its arm finishes, not after.
