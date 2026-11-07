# WHY IT IS WORD SALAD, AND THE FIX (2026-10-20)

## THE MEASURED CAUSE (from our own code, not from theory)
    self.concept = Dense(D * W, K)     # 8,192 numbers (32 positions x 256)  ->  41 concept values
    self.answer  = Dense(K, V)         # 41 concept values  ->  12,002 vocabulary logits
    model.py line 5, our own docstring: "The answer layer sees ONLY the concepts"
A standard model's output layer reads the full hidden state (256-4096 numbers of TOKEN-LEVEL detail). Ours
reads 41 numbers of SEMANTIC LABELS. The 41 named concepts ("is_a_question", "mentions_weather") throw away
token-level identity ON PURPOSE, and writing the next word requires exactly that identity: which word, which
inflection, which article. So the word salad is structural: the information needed was deleted before the
answer layer ever saw it. This is also why all 450 settings failed - they were tuning a pipe whose water had
already been thrown away. No learning rate fixes a bottleneck that removed the information.

## CORRECTION TO MY OWN EARLIER EXPLANATION (the maintainer caught this)
I said "give the answer layer a path from the hidden state, like other models". the maintainer: "our model does not have
a hidden state, so what will it read". Correct. What our model HAS in every configuration, attention or not, is
the TOKEN-LEVEL WINDOW REPRESENTATION: W x D = 32 x 256 = 8,192 numbers, which the concept layer already reads
before compressing it. So the fix needs no hidden state to exist: the answer layer must read those 8,192 numbers
IN ADDITION to the 41 concepts. Readability is preserved (the concepts still explain), and fluency becomes
possible (the token-level detail is back).

## AND WE DO HAVE A PER-TOKEN STATE NOW - IT IS THE CACHE
The hot tier keeps, per token: key, value, and the finished contextual output O (D numbers). That IS a per-token
hidden state. It can be carried across turns instead of recomputed. So the layers are:
    per-token state      -> the cache (BUILT: correct at 5.3e-06, 9.7x faster, 4,600 tokens/sec)
    what it means        -> the named concepts (readable)
    what it said         -> the warm concept memory (~1.3 bytes/token: 1 GB ~ 835M tokens, no fixed cap)
    exact wording        -> the cold token store (2 bytes/token: 1 GB ~ 536M tokens)
    consistency          -> it remembers its OWN output through those tiers, so it cannot drift from code
                            into "a house in a story" - the drift IS the missing memory

## THE FIXES the maintainer ASKED FOR, IN HIS WORDS
1. "carries everything ... make a token we can decode easily instead of words, because numbers can be stored
   easily": represent content as NUMBERS (quantities) rather than word tokens, so it is compact and decodes
   exactly. This is the numeric path applied to representation, not only to arithmetic. Stored as int8/int16,
   which is what makes unlimited memory cheap.
2. "make some way so it carries on what it is, so it can make sentences proper and understandable for itself":
   a running sentence state - WHAT I AM SAYING - carried across positions, made of named concepts so it is
   readable, and fed to the answer layer beside the token-level representation.
3. "make a connection for each word, so it knows what it is trying to say, that is why it is answering ...
   each neuron that has learned stored something and it should have its label": every unit carries a LABEL and
   a ROLE. Extend the concept labelling from whole-window detectors to per-token roles (subject/verb/object/
   number/name), each audited against its own evidence by the existing false-fire gate. A unit that cannot
   justify its label is DROPPED, never shipped with a disclaimer.
4. "remember what it output so he can be consistent": memory of its own output through the hot/warm/cold tiers,
   which is also what makes unlimited context useful rather than just large.

## ORDER OF WORK (the maintainer's instruction: our method first, settings second)
Because settings found for standard training will not transfer to our own method, the order is:
  A. Put OUR training method into the real trainer (lens/unified.py into lens/scripts/train_chat.py), with OUR
     step rule (lens/opt.py) wired in - the trainer still does plain SGD by hand today.
  B. Fix the bottleneck: answer layer reads the 8,192-number token-level representation beside the 41 concepts.
  C. The sentence state ("what I am saying") + per-token role labels, both readable and auditable.
  D. The memory tiers with NO FIXED CAP: sized from what the machine has and grown on demand.
  E. ONLY THEN the settings sweep, run ON OUR METHOD, across the 1M to 10B ladder.
  F. The thinking block (standard tags, filled with named concepts) goes in BEFORE the next long run.

## WHAT I WILL NOT DO
No more smoke-test-level work presented as progress. No accuracy claim without a measured line naming its file
and its validation split. No gate called PASS unless it is PASS, and no gate trusted until it has been shown to
FIRE on a corpus that can actually be memorised.

## THE BOTTLENECK EXPERIMENT - FIRST RUN WAS INCONCLUSIVE, MY ERROR (2026-10-20)

experiments/bottleneck_proof.py, task = predict a token by IDENTITY (chance 2.0%):
    A concepts only (our model today)        4.6%
    B concepts + direct tied path (the fix)  4.7%
The fix did NOT help, and the reason is a mistake in the experiment, not a rescue for the old design: I gave the
direct path the LAST token's embedding, while the answer is the token in the MIDDLE of the window. So the direct
path had no access to the answer either, and B could not win. The experiment as written proves nothing.

CORRECTED DESIGN TO TEST (next): the direct path must be POSITION-AWARE, not last-position-only. Each position
gets its own low-rank projection of its own embedding into the vocabulary:
    logits = answer(concepts) + sum_i ( E[x_i] @ Wo_i )
Cost at window 8, d=16, vocab 50 is 6,400 parameters; at our sizes (window 32, d=256, vocab 12,002) a low-rank
form is required to stay inside the 50M cap. Prediction to write BEFORE the corrected run: A stays at chance,
B learns the identity task. If B does not, the diagnosis is wrong and that result stands as the evidence.

## THE CORRECTED EXPERIMENT - DIAGNOSIS CONFIRMED, FIX PROVEN (2026-10-20)

experiments/bottleneck_proof2.py, task = predict the MIDDLE token BY IDENTITY, chance = 2.0%:
    A concepts only (our model today)                  held-out   5.9%
    B concepts + position-aware direct path (the fix)  held-out  77.6%
Same data, same seed, same step count. 13x improvement on a task where semantics are useless by construction.

VERDICT: the diagnosis holds. Our answer layer cannot see WHICH token, only what KIND of token it was, and
that is why the output is word salad. Giving every position its own path into the vocabulary fixes it. This is
the first hard evidence for the change, on a controlled task - NOT a chat claim.

WHAT THIS MEANS FOR THE REAL MODEL: the answer layer must read the token-level window representation
(W x D = 32 x 256 = 8,192 numbers) beside the 41 concepts, with a PER-POSITION path. At our real sizes the
naive form is too big (8,192 x 12,002 = 98M parameters, over the 50M cap), so it must be low-rank: per position,
project the embedding to a small rank r, then to the vocabulary. Readability is preserved because the concepts
stay and explain() still reads them.

## WHAT THE GGUF FILE ACTUALLY CONTAINS (verified with a correct parser, 2026-10-20)
GGUF v3, 15.65 MB, 5 tensors, 10 metadata entries. The metadata is honest and self-describing:
    general.architecture = lensnet        lensnet.vocab_size = 12000   lensnet.d = 256
    lensnet.window = 32                  lensnet.n_concepts = 41      lensnet.attention = False
    lensnet.concept_names = array[41] of strings (the readable names)
    lensnet.loader_note = "requires an implementation of the lensnet architecture; not runnable in..."
tensors:
    emb        (12000, 256)     concept.W  (8192, 41)     concept.b (41,)
    answer.W   (41, 12000)      answer.b   (12000,)
THE BOTTLENECK IS VISIBLE IN THE FILE ITSELF: concept.W squeezes 8,192 numbers into 41, and answer.W reads
ONLY those 41 to produce 12,002 vocabulary logits. That is the word salad, in the tensor shapes.

NOTE ON MY OWN ERROR: my first GGUF parser read 8 bytes for a 4-byte uint32 field and printed a garbage vocab
size of 38,654,717,664. The FILE was fine; the PARSER was wrong. Fixed and re-verified before making any claim
about the file.

## THE FIX IMPLEMENTED AND VERIFIED IN THE REAL MODEL (2026-10-21)

lens/model.py now has an optional POSITION-AWARE DIRECT PATH: the answer layer reads the token-level window
representation beside the concepts. `logits = answer(concepts) + relu(x @ dproj) @ dout`, where x is the
flattened window (W*D). Multiplying the whole flattened window by a (W*D, rank) matrix gives EVERY position its
own projection, summed, so the readout is position-aware without a per-position vocabulary matrix.
COST: W*D*rank + rank*V = 32*256*16 + 16*12,002 = 323,104 parameters at our sizes, inside the 50M cap. The
naive form (W*D -> V) would be 8,192*12,002 = 98M and is over budget, which is why it is low-rank.
Off by default: every existing checkpoint still loads byte-identical. Wired through api.Net/save/load so the
flag and its rank travel with the checkpoint.

MEASURED IN THE REAL MODEL, REAL ENGINE (experiments/direct_path_real_model.py, identity task, chance 2.0%):
    direct=False (our model today)   held-out  63.1%   params 2,282   final loss 2.396 (stuck)
    direct=True  (position-aware)    held-out 100.0%   params 3,706   final loss 0.177
MY PREDICTION WAS WRONG IN ITS NUMBER: I predicted direct=False would sit "near chance" and it reached 63.1%,
because in this small setup the category counts leak some identity. The comparison held; the specific number
did not. Recorded, not hidden.

NEXT TEST THAT MATTERS: the honest chat question. A real training run WITH the direct path, then the same
honest val measurement and the same talk test, compared against 28.30% (no-attention, no-direct, d=256/12ep).
The prediction is written before the run in the runner script.

## THE LOW-RANK DIRECT PATH: MEASURED, AND IT DID NOT WORK (2026-10-21)

Full 12-epoch run on the clean corpus, d=256, same setting as the baseline (lens/runs/DIRECT-BEST.txt):
                        final masked val   peak            final train loss
  baseline (no direct)      28.30%        28.52% (ep9)        3.48
  direct, rank 16           28.06%        28.56% (ep8)        3.58
Predictions written before the run: "val holds >=28%" HELD (barely); "train loss goes below 3.48" FAILED - it
came out WORSE at 3.58. The talk test (the stated decider) is still pure word salad:
  "hello, how are you?"    -> "* 5 * / 2 / 4 * * * sample input / * * * * 5 = = 5 ! * * 5 * * * 5 *"
  "what is your name?"     -> "` ` ` your task from it takes each ` b ` as it follows from each element `"
  "i am feeling sad today" -> "m = \" imagine her left with her school - while trying them with right right"
VERDICT: the rank-16 direct path alone does not fix the word salad. Recorded as a negative result, not softened.

WHY, FROM ARITHMETIC: a rank-16 bottleneck cannot say WHICH of 12,002 tokens it was. Sixteen numbers are far
too few. In the toy test rank 8 was enough because the vocabulary was 50 tokens, and that is exactly why the
toy result did not transfer. The fix was right in principle and under-parameterised in practice.

## THE TIED, FULL-RANK VERSION - BUILT AND CHECKED (2026-10-21)
lens/model.py, direct_mode="tied": every position mixes its own D numbers into D numbers (full rank, D=256),
then the readout is the model's OWN embedding matrix transposed - TIED, so identity flows by construction and
the readout costs ZERO new parameters. Cost: W*D*D = 2,097,152 parameters (~5.9M total, inside the 50M cap).
A new op was needed for this: lens/tensor.py now has `transpose` with its own backward, so the gradient can
reach the embedding matrix. Without it the tied path would be a dead end that trains nothing.
VERIFIED (experiments/tied_path_check.py, gradients checked in FLOAT64 as standard practice):
  dmix                numeric -0.004351  analytic -0.004351  rel diff 4.14e-09
  emb (tied readout)  numeric +0.013375  analytic +0.013375  rel diff 2.15e-09   -> PASS
And on the identity task in the real model (experiments/tied_identity.py): tied 100.0% at 4,330 parameters
against lowrank 100.0% at 5,130 - the tied version does it with FEWER parameters because the readout is shared.

## ERRORS IN MY OWN WORK THIS TURN (all found, all fixed, none hidden)
1. My first tied-path test wrapped a raw array in a fresh Tensor, detaching it from the graph, so every analytic
   gradient came back nan. The MODEL was fine; the TEST was wrong.
2. The same test printed PASS on nan, because max(0.0, nan) returns 0.0 in Python. A verdict must never read
   PASS on a non-finite number; it now fails explicitly.
3. I reported "transpose added" after an edit that silently did NOT land - I had matched the wrong function
   signature and my check only tested syntax, not that the code was present. The op was then written in the
   engine's real house style and verified by importing and calling it.
4. The gradient check first failed at 6.9e-03 in float32. That was the finite difference being noisy, not a bad
   gradient: float32 resolution near a loss of 4.3 is about 4e-7. Re-checked in float64: 4.1e-09.

## THE TIED FULL-RANK RUN FINISHED - 2026-10-21 19:54 (lens/runs/DIRECT-TIED.txt, exit 0)

d=256, window 32, 12 epochs, clean corpus, same setting as both earlier runs.
                     final masked val   peak                 final train loss   final train acc
  baseline no-direct     28.30%         28.52% (ep9)             3.48               ~24.8%
  direct rank-16         28.06%         28.56% (ep8)             3.58               ~24.8%
  TIED full-rank         27.37%         29.71% (ep4)             1.69               44.14%
PREDICTIONS, written before the run:
  1. "train loss beats 3.58, closer to or below 3.48" -> HELD, and by a lot: 1.69. The model can finally FIT
     token-level detail: train accuracy jumped from ~24.8% to 44.14%.
  2. "val holds at or above 28.30%" -> FAILED at the end (27.37%), though its PEAK 29.71% at epoch 4 beats the
     baseline peak (28.52%) by 1.19 points. The model overfits after epoch 4: train 44.14% against masked val
     27.37%, and val perplexity climbs from 86.7 at ep3 to 300.8 at ep12.
  3. "the talk test decides" -> STILL WORD SALAD, but the failure mode CHANGED: it now emits real English words
     and code-shaped text instead of the old "* 5 * / 2 / 4 * * * sample input" fragments, e.g.
     "can you help me learn to cook?" -> "try my idea here are playing with your niche ! yes welcome ! what are
     your knowledge ! here are playing today !" - real words, no sense yet.
VERDICT: the answer-layer bottleneck was REAL and is now measurably relieved (the model can express token-level
detail; it could not before). It is NOT the only problem. The new binding constraint is visible in the numbers:
the new capacity went into MEMORISING the training text rather than generalising - train accuracy 44.14% against
masked val 27.37%, peak at epoch 4, perplexity tripling afterwards. That is exactly the memorisation problem
the maintainer keeps naming, now measured with a number attached, and it is the next thing to attack (early stop at the
peak plus the learn-not-memorise work), together with the model being small for the data (0.36 tokens/parameter).
