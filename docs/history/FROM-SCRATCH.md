# Everything from scratch - inventory and the two gaps (2026-10-20)

The rule: we own a network we fully understand, so every part around it must be ours too and must be
compatible with it. This file is the honest inventory, with the two real gaps at the bottom.

## OURS ALREADY (written by hand in this project, no library)
- tensors + backward pass (lens/tensor.py, lens/backend.py) - gradient-checked, worst abs error ~1.3e-09
- layers: Dense, ReLU, sigmoid (lens/model.py)
- ATTENTION: CausalSelfAttention with our own Wq/Wk/Wv/Wo, our own causal mask, our own matmuls
- concept layer (named units) and answer layer
- losses: next-token cross-entropy, concept loss, sparsity penalty
- optimiser: lens/opt.py RelativeGrad (scale-invariant, one setting that does not depend on model size)
- MemorisationGuard (the safety net that finds memorisation by itself)
- tokenizer (word level, built from our own corpus in lens/scripts/chat_data.py)
- trainer (lens/scripts/train_chat.py) and autopilot (lens/autopilot.py)
- data judge (lens/scripts/data_judge.py)
- streaming encoder (lens/stream.py) - memory flat 75.6 MB from 1,000 to 1,000,000 tokens
- export (.npz/.safetensors/.gguf) and the local server (lens/serve.py, 127.0.0.1:8199)

## GAP 1 - NO KV CACHE (asked for 2026-10-20)
Current state: lens/decode.py calls net.forward_arrays([ctx]) with the FULL window for every generated token,
so token N re-computes all N-1 tokens again. Nothing is remembered.

What to build: our own incremental cache.
  - keep K and V for every token already seen; for a new token compute only its own q,k,v
  - attend the new query against the cached keys/values (causal by construction: the cache only holds the past)
  - the concept layer and answer layer still read the same contextualised tokens, so explain() is unchanged

Three numbers this must produce, and it does not count as done without them:
  1. bytes kept per token in the cache (and total cache size for a 512-token context)
  2. seconds per generated token, WITH and WITHOUT the cache, same prompt and seed
  3. a CORRECTNESS test: the cached output must be IDENTICAL to the old full-recompute output, token for
     token, on the same seed. "Close enough" is a fail. A cache that changes the answer is a bug.

## GAP 2 - OUR OPTIMISER IS BUILT BUT NOT WIRED IN
Current state: lens/opt.py has RelativeGrad, but lens/scripts/train_chat.py line ~396 still does
`p.data = p.data - lr_now * p.grad` by hand. So the model trains with plain SGD and our own optimiser sits unused.

STATUS 2026-10-20 16:2x - WIRED INTO THE NUMERIC PATH, MEASURED, AND IT NEEDS A DECAY SCHEDULE
  PLAN item (d2) is done: our step rule is now used and measured on the numeric path, against plain SGD on the
  same harness, same seeds. Numbers: lens/runs/NUMERIC-PATH-OPT.txt and lens/runs/NUMERIC-PATH-OPT-DECAY.txt
  (predictions written first: lens/runs/NUMERIC-PATH-OPT-{PREDICTIONS,DECAY-PREDICTIONS,EXT-PREDICTIONS}.txt).
  WHAT IT DOES WELL: it CANNOT overflow. 12 of 12 configurations finite at every step size tried, against
  plain SGD which is all-finite at only 2 of its 4 step sizes (lr 0.5 gave 2 of 3 seeds non-finite). Its
  admissible window is 10x (0.005..0.05) against SGD's 2x (0.1..0.2), and the per-tensor update norm is
  exactly lr at every size (measured 1..4096, worst relative error 1.16e-07) - the size-independence the
  docstring claims is real.
  WHAT IT DID NOT DO: with a CONSTANT step it is non-monotonic in the step size (unseen exact
  100.0 / 54.7 / 66.7 / 100.0% at lr 0.005 / 0.01 / 0.02 / 0.05) and it did not even fit a 5-parameter linear
  regression (19-28% of its own training set) that SGD fits to 100%. Cause, measured: a constant normalised
  step settles at a steady-state OFFSET from the solution proportional to the step size.
  FIX, MEASURED: a `decay` knob (lr * decay**step, default 1.0 = constant, bit-identical to before). With
  decay 0.999: unseen exact 100.0% with 3/3 seeds finite at EVERY step size from 0.005 to 0.5 (a 100x range),
  the spread across lr 0.01/0.02/0.05 collapses from 45.3 points to 0.0, and the 5-parameter regression goes
  to 100%. So "one setting works at any step size" is true for our rule WITH a decay schedule, and false for
  a constant step.
  STILL OPEN: train_chat.py does not use it yet (PLAN item (d2b)). The comparison ours-vs-SGD on the honest
  val, same seed, is the next thing to measure, and it must be run with the decay on.

What to build: our own adaptive step, no borrowed Adam.
  - normalised direction (already in RelativeGrad) + momentum + per-parameter scaling
  - the learning rate must not need re-tuning when the model size changes: that is the whole point, because
    a hand-set learning rate per size cannot travel up to 10B parameters
  - tested honestly: same seed, same data, same epochs, ours vs plain SGD, decided by the HONEST masked val
    accuracy on the disjoint val (never the train accuracy)

## ORDER
1. KV cache (with its three numbers)
2. own optimiser wired in and tested
3. then back to the numeric path, which is still the change that fixes rule-learning


## GAP 1 STATUS - KV CACHE BUILT AND CORRECT, SPEED NOT YET (2026-10-20)

lens/kvcache.py exists and was tested against the model's OWN attention block on a real attention ship
(sb_B1_d512_attn_ep96, d=512, window=32):
  CORRECTNESS: PASS. max absolute difference 5.305e-06 (float32 rounding) between the cached step-by-step
    path and the model's full recompute. Identical at float32 precision, not merely close.
  MEMORY: 6,144 bytes per token (K, V and the finished contextual output O), 3.0 MB for a 512-token context.
    Per generated token the old path touches W*D*3 = 49,152 values; the cached path touches D*3 = 1,536
    (32x less work per token).
  SPEED: FAIL. Full window recompute 0.51 ms; cached one-token-at-a-time 7.66 ms = 14x SLOWER.
    Cause is the implementation, not the idea: one Python call per token, a growing np.concatenate each step,
    and a fresh softmax over the whole cache every step, versus one batched matmul in the model path.
    FIX: preallocate the cache arrays and batch the step (no concatenate), then re-measure. The cache does not
    count as done until it is both correct AND faster.

NOTE FOR ANYONE TESTING THIS: the recent clean models (cl_*) are NO-ATTENTION, so they have no attention block
and the cache has nothing to hook into. 591 attention checkpoints exist (sb_B1_d512_attn_*, t9_attn, t12_attn)
and one of those must be used to test the cache.

## GAP 3 - THE THINKING BLOCK (asked for 2026-10-20)

the maintainer: "we will need to use tags so he can [do] thinking and we can hide the thinking or make a separate block
for thinking and the output ... everybody does it from start, that was the whole problem of this".

Every other model puts a private reasoning block before the answer (<|think|> ... <|/think|>) and designs it in
from the beginning, because a model trained without one cannot learn to use it well afterwards. We do not have
one at all, and it must be added BEFORE the next long training run, not after.

OUR DESIGN (ours, not a copy): the thinking block contains the NAMED CONCEPTS. The concept layer already
produces a readable trace, so the private thinking is a list of human-readable units rather than an opaque blob:

    <|think|> mentions_weather, is_a_question <|/think|> it might rain later, take a jacket

Three properties nobody else has: the thinking is readable by construction; it is CHECKABLE, because the
existing false-fire gate can test whether the named concepts really fired on that window; and it can be hidden
from the user by a flag while staying auditable by us.

WHAT IT NEEDS:
  1. two control tokens added to the vocabulary beside <|im_start|> (12000) and <|im_end|> (12001)
  2. a data pass that writes the concept trace into the think block for each assistant turn. The model's own
     concept layer generates the trace, so the model is taught by its own readable layer, not by hand-written
     text from us. That also means the trace cannot lie about what the model actually computed.
  3. a retrain with the block present from the first step, and a gate: on random held-out windows, do the
     concepts named in the think block actually fire? If they do not, the think block is decoration and the
     gate must FAIL it.


## GAP 4 - UNLIMITED CONTEXT: REMEMBER CONCEPTS, NOT TOKENS (asked for 2026-10-20)

the maintainer: "the KV cache and the context window, can we get unlimited context windows ... that would be a crazy
feature ... everything so small in size, it fits in 7 MB ... even 1 gigabyte would be around a billion tokens".

THE ARITHMETIC THAT DECIDES IT (1 GB of memory) - CORRECTED 2026-10-20 to the MEASURED figures for the
current ship (cl_d256_clean_24_ep24: 36 named concepts, window 32), file lens/runs/WARM-MEMORY.txt:
  raw Keys+Values+outputs at d=512: 6,144 bytes/token              -> ~175,000 tokens      (everyone else's wall)
  our concepts, float32: 36 values per 32-token window = 4.5 B/token -> ~222,000,000 tokens
  our concepts, 1 byte per concept: 36 B/window = 1.125 B/token     -> 888,888,888 tokens  ("about a billion")
  our concepts, +1 byte for the exact top-1 name: 37 B/window = 1.156 B/token -> 864,864,864 tokens  <- SHIPPED
  raw tokens losslessly, vocab 12,002 fits in int16 = 2 B/token     -> ~536,000,000 tokens (exact wording kept)
  The earlier "41 values per window / 1.28 B/token / 835M" was arithmetic for a 41-concept window, not a
  measurement; the numbers above are measured and the file is named. At the stream encoder's OVERLAPPING
  stride (16 instead of 32) every figure doubles: 2.312 B/token shipped, 432,432,432 tokens per GB.

WHY THIS IS POSSIBLE FOR US AND NOT FOR THEM: they must keep per-token Keys and Values. We already have a
readable summary layer (36 named concepts per window in the current ship), so we can store MEANINGS instead
of words.

THREE TIERS, ALL OURS:
  1. HOT: exact K/V/O for the last window (32 tokens) - small, fast, exact. BUILT (lens/kvcache.py, verified
     correct at 5.3e-06; speed rewrite still pending).
  2. WARM: concept memory for the whole conversation, ~1.2 bytes/token, effectively unlimited, and READABLE:
     we can print what it remembers because it is named concepts, not an opaque vector. BUILT 2026-10-20
     (lens/warmmemory.py). MEASURED on the held-out split (lens/runs/WARM-MEMORY.txt): 1.156 bytes/token at
     stride 32 with the exact top-1 name kept (1.125 without it), 1 GB = 864,864,864 tokens, top-1 token
     identical to the float32 path on 99.25% of 2,000 held-out windows, recall of the first 1,000 windows
     from stored bytes alone 99.60%, and a 1M-token store that is exactly 1,152,000 bytes. The first version
     read the wrong concept NAME on 6.35% of windows and FAILED its own bar; 100% of those flips were float32
     near-ties within 1/255, and keeping the argmax index in one extra byte fixed it to 100.00% exactly.
     NOT YET USED inside a generation loop: wiring it into lens/decode.py is still open, and nothing claims it.
  3. COLD: lossless tokens on disk (int16), any exact passage re-readable. TO BUILD.

HONEST LIMITATION: the warm tier remembers meanings, not exact wording, so it cannot quote a sentence back
verbatim. That is precisely why tier 3 exists. Nothing in any tier is hidden: every stored value is either a
named concept or the original token.

## GAP 5 - THE SETTINGS REGISTRY, OURS, NOT ONE LEARNING RATE (asked for 2026-10-20)

the maintainer: "why are you just thinking about learning rate, there I like 50 billion different settings we have to go
through and make our own so it only learns and still does not hide anything, what about those, even the
gradient".

WHAT THIS MEANS: not one knob. A registry of knobs we define ourselves, each implemented by us and inspectable,
then swept systematically to find the best per size (the standing "one best setting per size, easy mode uses it,
hard mode exposes every line" requirement).

THE KNOBS WE OWN (each gets its own implementation and its own measurement):
  - the step rule: plain, normalised, agreement-gated, momentum, per-parameter scaling
  - THE GRADIENT ITSELF: we wrote the backward pass, so any gradient can be printed per unit and per concept,
    and it is numerically checked (current worst absolute error ~1.3e-09). It is not a black box.
  - learning-rate schedule: warmup, decay shape, constant
  - concept loss weight, sparsity penalty, activation penalty
  - attention on/off, window size, concept unit count, width, depth
  - thinking-block weight, turn-end weighting, multi-token prediction
  - memory tiers: what stays hot/warm/cold and at what precision

RULE FOR EVERY SETTING (standing): the prediction is written BEFORE the run, the result is recorded with the
file it came from, and the gate is binary. "Passed in spirit" is banned. Three seeds before anything is called
a win.
