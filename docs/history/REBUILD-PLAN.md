# REBUILD — EVERYTHING FROM SCRATCH

**Started 2026-10-06 by owner instruction: "start over and make everything from scratch, libraries,
every single thing, even the GPU compatibility, and fully readable — it takes more time but we get
more control, and each thing is tested fully."**

**Rule carried over: DELETE NOTHING.** The rebuild lives in `glasscore/` alongside the existing
work. Every old run, test and result stays as reference. Nothing outside GlassNet is touched.

---

## WHAT "FROM SCRATCH" MEANS HERE

| layer | we write | we do NOT use |
|---|---|---|
| 0 array | our own ndarray wrapper: shape, stride-free broadcast, dtype rules | numpy for compute (numpy only for loading data and checking our answers) |
| 1 autograd | our own graph, our own backward, **no reference cycles** | torch, autograd, jax |
| 2 GPU | our own CUDA: `ctypes` + driver API + NVRTC-compiled kernels | cupy, torch, numba |
| 3 readable | our own concept layer and exact `explain()` | anything pretrained |
| 4 library | our own train/explain/save/load | sklearn, keras |

**The one allowance:** NumPy is used **only** as a reference oracle to check our own layer-0
results, and to read `.npz` files. If our array layer disagrees with numpy, that is a bug in our
layer and the test says so.

## WHY OUR OWN GPU LAYER (the honest cost)

The current code uses **CuPy** — a raw GPU array backend, which the original brief allowed. Writing
our own is a real step up in control and a real step up in cost.

* **Cost:** this is the biggest single piece. Rough estimate **1–2 weeks** of sessions, not hours.
* **Risk:** CUDA driver details, memory management, and kernel correctness are where this can burn
  time. It is the one layer I expect to be hard.
* **Mitigation:** our GPU layer is checked **against CuPy** (same op, same numbers, tolerance
  stated) and **against numpy** on small cases. If our layer is wrong, a known-good implementation
  catches it. That is how "tested fully" is possible here rather than hoped for.
* **If it stalls:** it stalls visibly, with a measurement, and I propose alternatives rather than
  quietly falling back.

## THE GATES (each layer has one, none is skipped)

* **L0 gate** — broadcasting, reductions and dtype rules match numpy on a fixed random battery;
  every mismatch printed, zero mismatches required.
* **L1 gate** — every op's gradient matches central differences (rel tol 1e-5), and a 2-layer net's
  full gradient check passes.
* **L2 gate** — every GPU op matches the CPU result elementwise within a stated tolerance, and
  matches CuPy where available; a small net trains to the same loss curve on both backends.
* **L3 gate** — the standing meter (AUROC ≥ 0.55, contribution ≥ 0.25, firing floor 2%, ≥ 50% of
  concepts, lie probe caught) plus coverage **100.0000%** measured, plus `explain()` residual
  printed.
* **L4 gate** — a scenario suite: train, save, load, explain, fail-loudly-on-misuse. Every scenario
  prints PASS/FAIL with its numbers.

**Every gate runs before the next layer starts. No layer is "done in spirit."**

## "FULLY READABLE" AS A DESIGN REQUIREMENT

Readable means the **whole system**, not just the concepts:

* every module has a plain-English docstring saying what it is for and what it does not do
* every non-obvious line says **why**, not what
* no hidden state, no magic constants without a comment naming their source
* `explain()` is exact and asserts it — contributions must sum to the model's own score
* any layer can be run and inspected on its own

## ORDER OF WORK

1. **L0 array core** + gate
2. **L1 autograd** (cycles fixed — the known latent issue in the old engine) + gate
3. **L2 GPU layer** (the big one) + gate
4. **L3 readable architecture** + gate
5. **L4 library** + gate

## WHAT WE KEEP FROM THE OLD WORK

Not the code — the **findings**, so the rebuild does not relearn them:

* all 9 concept targets are **linearly readable at AUROC 1.0000**; targets are sound
* `is_a_dialogue_line` fails at **every** size; `pronoun_subject` fails at D≥64
* accuracy is **degenerate** on this corpus (`<unk>` = 29.64% of targets) — perplexity is the metric
* the readable model costs a **stable ~12–14 ppl**; that is the known blocker
* a best-LM checkpoint can be an untrained model — **state which epoch any audit measured**
* **every wrong number so far was caught by a control, a contradiction, or a self-check**

Those are the facts the rebuild inherits. The code starts clean.
