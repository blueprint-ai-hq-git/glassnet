# Rebuilding the copy path structurally does NOT make it immune - refuted (2026-11-13)

Freezing the copy path on language batches kept held-out copy at 77.3 +/- 1.4 (`docs/COPY-KEEP-LEARNING.md`) but costs 0.85
language points, because language is not allowed to touch those tensors. The obvious successor was to stop trading: build the copy
path so language cannot erode it - an **identity readout** (nothing learned to overwrite) plus the copy path's **own embedding
table**. `lens/model.py` records that the identity readout alone lifted held-out copying from 71.5% to 85.3%. This tests whether
that construction survives language training with NOTHING frozen.

It does not. The hypothesis is refuted, and the first version of the experiment was void.

## First, the void arm (kept, not deleted, because the mistake is easy to repeat)
The first script enabled `--sep-emb` and `--identity-pointer` on an already-trained checkpoint. `--sep-emb` creates a **freshly
random** (12002, 512) `pemb` table and the identity replaces the trained `ptrW`, so the copy path had nothing trained to read from
BEFORE any language batch. Its own log says so on line 10 of `logs/struct/struct_id.log`:

```
before training: taught 18.0% | held-out 0.0%
[step 500..4000] copy accuracy: TAUGHT 0.0%  HELD-OUT 0.2%
```

Measuring a 4000-step "decay" of a path that was already at 0.0% before training began is meaningless. The error was changing a
trained model's configuration at load time and expecting the ability to survive untrained tensors. The arms were stopped and
archived, not re-run unchanged.

## The corrected design
Give the new construction its own training first, exactly as the memorisation-split experiment did with its warm-up
(`docs/MEM-SPLIT.md`):

- **Phase A (warm-up)**: copy batches only, 3000 steps, so `pemb` and the identity readout learn to copy.
- **Phase B (test)**: language-only, 4000 steps, **nothing frozen**.

Predictions were written in the script header before the run:
`P1` warm-up recovers held-out copy to >= 60%; `P2` Phase B costs held-out copy < 7 points against the -14.2-point reference decay
(71.0 -> 56.8, `logs/keep/keep_none.log`); `P3` language after Phase B >= 16%.

## Result
```
PHASE A warm-up (copy-only, identity readout + own embedding, 3000 steps)
before training:      taught 18.0%   held-out 0.0%
step  500:            taught 78.3%   held-out 0.0%
step 1000:            taught 67.7%   held-out 0.0%
step 1500:            taught 60.3%   held-out 0.0%
step 2000:            taught 56.5%   held-out 0.0%
step 2500:            taught 52.3%   held-out 0.0%
step 3000:            taught 50.5%   held-out 0.0%

PHASE B language-only from the warmed copy path, nothing frozen, 4000 steps
step 3000:            taught  0.7%   held-out 0.2%   language 25.20%
step 3500:            taught  0.8%   held-out 0.2%   language 25.52%
step 4000:            taught  0.8%   held-out 0.2%   language 26.15%
```

**P1 FALSIFIED - and not marginally.** Held-out copy is **0.0% for the entire 3000-step warm-up**. The identity readout plus its
own embedding cannot copy a word it has never been trained on, at any point, even when copy batches are the only thing it sees.
P2 became untestable: there was no held-out ability left to decay. P3 passed strongly (language **26.15%**, the highest number
recorded for any arm in this line).

Two further facts from the run:
- **Taught-word copy peaks at 78.3% and then erodes to 50.5% during the warm-up itself**, while held-out stays at zero. The
  construction memorises the trained words and loses even that - it is a lookup, not a copying mechanism.
- **Phase B then takes taught copy from 50.5% to 0.8%** in 4000 language-only steps. A copy path built this way is not immune to
  language; it is more fragile than the shared one it replaced.

## What this changes
- The `lens/model.py` note that the identity readout lifts held-out copy from 71.5% to 85.3% is **not a drop-in recipe**. Under the
  conditions tested here (config swapped onto a trained checkpoint, then warmed up on copy batches) it produces zero held-out
  copy. Whatever that measurement was, it was not this change applied this way, and the note should not be read as an available win
  until the difference is identified.
- **The best copy protection measured so far remains the per-batch freeze**: 77.3 +/- 1.4 held-out at a cost of 0.85 language
  points (`docs/COPY-KEEP-LEARNING.md`). Gradient discipline works; this particular structural rebuild does not.
- Language reaching 26.15% here - against 16.2-17.7% in every arm that kept a working copy path - is consistent with the project's
  standing observation that copy and language compete for the same parameters. It is not a win: that 26.15% came with a dead copy
  path.

## Reproduction
`lens/scripts/run_struct2_arms.sh` (predictions in the header), logs in `logs/struct2/`, checkpoints `warm_id_*` and
`id_after_lang_*` in `lens/runs/chat/`. The void first attempt is `lens/scripts/run_struct_arms.sh` with logs in `logs/struct/`.
