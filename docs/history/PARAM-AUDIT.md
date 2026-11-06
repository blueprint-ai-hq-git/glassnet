PUBLIC NUMBER AUDIT, PART 2 (2026-11-06 17:0x-17:3x, autonomous wake)
Continues lens/runs/PUBLIC-NUMBER-AUDIT.txt, whose section 6 listed what it did not check:
"35.26% d256 full corpus, 24.20% 2M windows, 17.64% batch 4096, KV cache 9.7x, param counts
(7,081,122 / 5,908,686), arithmetic 0.0% / digit-0 100%, per-layer ms/token."

Machine state while this ran: no STOP-NIGHT, no WAKETEST/ACTIVE, no game process (pgrep -x
sober/RobloxPlayer/cs2 all empty). Two trainers were on the box (pid 2504549 stable_vs_plain 328% CPU,
pid 3015702 train_chat 110% CPU) and lens/runs/GPU-CLAIM was held by run_wall_ab_resume.sh, so this
turn launched nothing and used no VRAM. Everything below is CPU-side reading and counting.

============================================================================================
1. PARAMETER COUNT — WRONG ON THE PUBLIC PAGE. 7,081,122 is not the model. 5,908,686 is.
============================================================================================
CLAIM: README.md:236 "**whole stack** | **7,081,122** | **0.1446**", and README.md:46
"Compute core ... 131,328 params", docs/ARCHITECTURE.md:50 "7.08M total at d=256",
docs/assets/make_charts.py:85 "7,081,122 parameters".

WHAT WAS DONE: built the model object the published runs actually trained and asked it
(lens/runs/PARAM-DUMP.txt). Same class (lens/scripts/train_chat.py ChatNet -> lens.model.LensNet), same
flags as the launcher lens/scripts/run_gpu_best.sh: --d 256 --window 32 --direct --direct-mode tied,
vocab 12,002, K=36 (36 concepts kept after the 0.005 support floor, GPU-BEST.txt:17).

  params as the object reports them (lens/runs/PARAM-DUMP.txt):
    emb                       3,072,512     (12,002 x 256, the tied embedding)
    concept.W                   294,912     (36 x 8,192 = 36 x W*D, the window the concept layer reads)
    concept.b                        36
    answer.W                    432,072     (36 x 12,002)
    answer.b                     12,002
    dmix                      2,097,152     (8,192 x 256, the direct head)
    total                     5,908,686

  CROSS-CHECK: 35 run logs print exactly "model: 5,908,686 params  (d=256, K=36)"
  (grep -rl over lens/runs/*.txt), including lens/runs/GPU-BEST.txt:57 (the 24.20% run).
  The number 7,081,122 appears in ZERO run logs: `grep -rl 7,081,122 lens/runs/*.txt` returns only
  lens/runs/PUBLIC-NUMBER-AUDIT.txt:90, i.e. the previous audit's own to-do line.

WHERE 7,081,122 CAME FROM: the README table's rows do sum to it (verified: 3,072,512 + 8,192 + 131,328 +
16,448 + 780,130 + 3,072,512 + 0 = 7,081,122). But it sums things the shipped model does not contain:
  README row            published     real object        verdict
  embedding             3,072,512     3,072,512          correct
  position add              8,192     none               NOT a parameter of this model
  compute step (32x)      131,328     none               the PROPOSED recurrent core
                                                          (docs/ARCHITECTURE.md:33), which lost to the
                                                          plain shape by 1.73 points (ARCH-OURS-VERDICT.txt)
  concepts                 16,448     294,948            understated 17.9x
  answer via concepts     780,130     444,074            overstated 1.76x
  direct read           3,072,512     2,097,152          the embedding counted a second time
  loss                          0     0                  n/a (loss has no parameters)
  WHOLE STACK           7,081,122     5,908,686          OVERSTATED BY 1,172,436 (19.8%)

So the section that introduces the architecture was describing a model that is not the one that produced the
published accuracies: it includes the recurrent compute core (the arm that lost) and it counts the tied
embedding twice. Fixed in README.md, docs/ARCHITECTURE.md, docs/assets/make_charts.py (+ regenerated PNG).

============================================================================================
2. THE 0.1446 ms/token TOTAL CONTRADICTS ITS OWN "RUNS 32x" LABEL
============================================================================================
README.md:231 labels the core row "compute step (runs 32x)" with 0.0027 ms. The listed rows sum to 0.1446
ONLY when that step is counted once (0.0053+0.0021+0.0027+0.0008+0.0273+0.0239+0.0825 = 0.1446). Counted the
32 times the label says, the same rows give 0.2283 ms/token (0.0053+0.0021+0.0027*32+0.0008+0.0273+0.0239+
0.0825 = 0.2283). Both sums computed in this turn, not estimated.
The row VALUES themselves were not re-measured this turn (see 3: the CPU was busy, and a CPU timing under
load is not a measurement). They are labelled in the file as measured earlier rather than silently reused.

============================================================================================
3. KV CACHE — "9.7x" HAS NO RUN FILE. The only recorded ratio is 2.29x.
============================================================================================
CLAIM: STATUS.md:11, STATUS.md:42, README.md:45, README.md:184 "9.7x faster".
The only micro-benchmark file, lens/runs/KV-CACHE-MICRO.txt (2026-10-20 14:15), says:
  full window recompute per generated token (the CURRENT decode path): 0.483 ms
  preallocated cache:                                                  0.211 ms
  -> 2.29x faster      tokens/s cached 4,747  uncached 2,069      correctness 5.305e-06 (identical)
"9.7x" exists only as prose in docs/history/STATE.md:2730-2731: "0.217 ms per token, 4,600 tokens/sec
against 2.115 ms and 473 tokens/sec". 0.211 x 9.7 = 2.047 ms, so the 9.7x compares against a 2.115 ms/token
decode path - 4.4x slower than the 0.483 ms path recorded in the micro file. That baseline is in no run file,
so 9.7x cannot be reproduced from this repository.
RE-MEASURED LIVE THIS TURN (lens/runs/KV-CACHE-LIVE.txt) and NOT USED, on purpose: the same script reported
the uncached path at 48.030 ms/token (vs 0.483 ms on an idle box) - a 99.4x inflation, because two trainers
were saturating the CPU. Under load the cached path measured 0.352 ms and the ratio printed "136.48x", which
is contention, not a cache. The correctness half DID reproduce exactly under load: max abs difference
5.305e-06, identical to float32 precision True.
VERDICT: public page now cites the file-backed 2.29x (0.483 -> 0.211 ms/token, 2,069 -> 4,747 tokens/s,
identical output) plus the structural fact (32x less projection arithmetic per token: 49,152 vs 1,536 values,
KV-CACHE-MICRO.txt). A fresh timed run needs an idle GPU/CPU; it is queued as an open item, not guessed.

============================================================================================
4. THE THREE ACCURACY NUMBERS — CORRECT, WITH ONE LABEL TO ADD
============================================================================================
  35.26% d=256 full corpus: lens/runs/ATTN-GRADE-RUN.txt:28 raw epoch line (twin ct_d256_full169_fix3,
         ep 3/3, "ASST(by-position) ppl 32.77 acc 35.26% (n=58191)"), restated at :68-71. It is a BY-POSITION
         held-out number; the legacy by-token-id number for the same epoch is 26.31%. STATUS.md/README name
         the metric "held-out accuracy" without naming which - they should say by-position.
  24.20% 2M windows batch 1024: lens/runs/GPU-BEST.txt:64-65 "ASST(by-position) ppl 103.00 acc 24.20%
         (n=58191)" + "NEW BEST held-out 24.20% at epoch 3", seed 1 = the run_gpu_best.sh launcher. Correct.
         (LEARNCURVE.txt:1923 also reads 24.20% for a different arm, seed 0, epoch 1 - coincidence, not the
         published source.)
  17.64% 2M windows batch 4096: lens/runs/GPU-BIG.txt:62-63 "ASST(by-position) ppl 206.87 acc 17.64%",
         epoch 2 of 2 - i.e. ONE EPOCH FEWER than the 24.20% arm (3 epochs). README already gives the cause
         (four times fewer updates per epoch) but not the epoch count; adding "2 epochs" keeps the
         comparison honest.

============================================================================================
5. ARITHMETIC — BOTH NUMBERS RIGHT, BUT FROM TWO DIFFERENT RUNS
============================================================================================
  "0.0% exact on unseen operands": lens/runs/ARITHMETIC.txt, epochs 5-60 all "UNSEEN_50_99": 0.0,
        "three_digit": 0.0 (train exact climbs 4.57% -> 12.05%).
  "digit 0 reaches 100% on unseen operands": lens/runs/ARITH_DIAG.txt, epochs 35-45 "UNSEEN[d0,d1,carry]"
        = [1.0, .., ..] - a DIFFERENT run (its own loss scale 6.005-1.598, the other run 1.31-0.996).
  Both true; they are not from one experiment, so they are now labelled by file. Note the same file also
  shows UNSEEN_exact rising to 0.0027 (ep 40) and 0.024 (ep 45): the honest phrasing is "0.0% in the
  ARITHMETIC run, 2.4% at the end of the longer ARITH_DIAG run", not "0.0% at every scale".

============================================================================================
6. WHAT THIS TURN DID NOT SETTLE (stated, not hidden)
============================================================================================
  - per-layer ms/token: not re-measured (CPU busy, see 3). Only the internal arithmetic of the published
    total was checked (section 2).
  - 37.37% and the d=512 row: verified by the previous turn (TIED-D512-EXT6.txt:98,:99,:181), not re-read now.
  - the historical 2.115 ms/token decode path behind the 9.7x: the file that recorded it was not found;
    KV-CACHE-MICRO.txt at 14:15 the same day records 0.483 ms instead.

The raw run output this file cites: lens/runs/PARAM-DUMP.txt (parameter list + total), lens/runs/PARAM-COUNT-LIVE.txt (an earlier probe with the WRONG direct-mode, superseded - see section 1), lens/runs/KV-CACHE-LIVE.txt (contended live KV timing, not used), lens/runs/CHARTS-REGEN.txt. lens/runs/ is gitignored; these are machine-local.
