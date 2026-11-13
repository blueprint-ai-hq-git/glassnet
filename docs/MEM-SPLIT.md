# The separated memorisation store works: 96.1 points of forgetting -> 0.0 (2026-11-13)

The problem in the project's own record: copy ability peaked at **70.2%** held-out at step 1000 and decayed to **46.5%** by step
4000 while language kept improving, and no checkpoint had both. Yahya's proposal was to keep memorised facts in their own part of
the model so that training the language part cannot overwrite them. This is the smallest experiment that can settle it.

## Setup
- **Memory task:** 128 fixed key -> value pairs. Trained to 100% first (3000 steps).
- **Language task:** a deterministic 2-gram chain over the same vocabulary. Plenty of signal, learnable.
- **Phase 2 is language ONLY** - no rehearsal of the facts. That is the situation that produced the decay.
- **Arm `joint`** (what the project does today): the key is pushed through the same trunk and the same output head as the
  language task, so language gradients rewrite the facts.
- **Arm `split`:** the facts live in their own store - their **own key embedding** plus their **own read-out**. Nothing is shared
  with the language network, so no language gradient can reach them.

**Prediction, written before the run:** the joint arm's recall falls by at least 20 points while language improves; the split arm
holds recall at 97% or better at the same language accuracy.

## Result, three seeds each
```
arm      recall after memory warm-up   recall after 1500 language-only steps   language
joint            100.0%                        3.9%  (drop 96.1 +/- 0.0)          100.0%
split            100.0%                      100.0%  (drop  0.0 +/- 0.0)          100.0%
```
Per-seed joint recall: 3.9% / 3.9% / 3.9%. Per-seed split recall: 100% / 100% / 100%. **Prediction passed on both halves**, and
language accuracy is identical between arms, so the split is not buying memory with language.

The experiment reproduces the project's decay in shape and makes it far more extreme (96 points rather than 24), which is what a
128-fact demand against a 24-wide hidden layer should do - the joint model has no room for both.

## Why the split holds, stated as a mechanism and not as luck
The separated store reads the key through **its own embedding** and writes through **its own read-out**. The language network has
no parameter that appears in the memory path, so the memory gradient and the language gradient are updates to disjoint parameter
sets. Forgetting by interference is therefore impossible by construction, not merely unlikely.

## Two design errors of mine that had to be fixed before the test meant anything
1. **Rehearsal every step.** The first version trained the facts again on every language step, in both arms. That prevents
   forgetting by construction, and it measured exactly nothing: both arms reached ~100% recall and 100% language. A test that
   cannot fail is not a test. Phase 2 is now language-only.
2. **A shared key embedding.** The first split arm shared the key embedding with the language network, which is a half-measure:
   language training then rewrote the representation the facts were read from, and recall froze at 17.6% instead of holding. The
   store now owns its embedding.

Both are recorded because they are the same class of mistake the project keeps catching: a comparison that differs in two places,
or a setup where the outcome is guaranteed.

## What this does and does not prove
- **Proves:** for facts that must be recalled exactly, an isolated store beats one shared with a learning model by 96.1 points of
  retention at equal task performance, on three seeds, in a controlled setting.
- **Does not prove:** that this scales to the real corpus copy task (70.2% -> 46.5%), or that an isolated store helps *language*
  in any way - language is identical between arms here. Porting this to the real copy task, with its measured 70.2% peak, is the
  next experiment.
- The store also costs parameters that do nothing for language, which is exactly the trade Yahya described: a part of the model
  that exists to remember, and a part that exists to learn.

## Reproduction
`.venv-gpu/bin/python3 tools/mem_split.py --arm {joint,split} --keys 128 --hidden 24 --lr 0.12 --warm 3000 --steps 1500 --seed
{0,1,2}`; results in `runs/mem/mem_<arm>_s<seed>.json`, logs in `logs/mem/`. Seeds differ (checked: step-300 joint language is
87.8 / 85.5 / 82.2%, so the runs are genuinely different); the identical final numbers are the converged floor of a collapse and a
structurally immune store, not a dead patch.
