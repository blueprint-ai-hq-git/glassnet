# The cheapest freeze: concept.W alone is nearly free, the pointer alone is the best copy (2026-11-13)

Freezing the copy path on language batches protects held-out copy by about 2 points (+1.98 over five seeds; +2.00 replicated from a
second starting checkpoint) but costs **-0.73 points of language**. The mean-ablation on the copy checkpoint said which tensor should
carry that protection: removing `concept.W` cost **15.0 points of held-out copy** while changing taught words by +1.0, whereas
removing the pointer matrices destroyed copying entirely. So this tests each half on its own.

## Arms (3 seeds each, identical to the working arms: mixed 50/50 stream, 4000 steps, from `early_s0_ep1000`, one nice-19 core)
```
arm                frozen                      held-out copy           language
none               -                           74.02                   16.30
concept.W only     concept.W                    75.30 +/- 1.01          16.46 +/- 0.03
pointer only       ptrW, Wq, Wk                 77.20 +/- 1.40          15.70 +/- 0.15
full freeze        ptrW, Wq, Wk, concept.W      75.97 (5 seeds)         15.56 (5 seeds)
```

## Paired against the same seeds of the unfrozen arm
```
arm              copy gain   copy t   seeds up   language change   language t   seeds up
concept.W only    +1.28      2.5        3 of 3       +0.16           0.9           (no cost)
pointer only      +2.07      5.2        3 of 3       -0.79           5.3           -
full freeze       +1.95      6.2        5 of 5       -0.74            -            -
```

**Both predictions passed, and the split is cleaner than expected:**
- **`concept.W` alone buys +1.28 points of copy and costs nothing** - language is 16.46 against 16.30 unfrozen, i.e. no measurable
  price. Measured on three seeds with a language spread of 0.03, so the absence of a cost is not noise.
- **The pointer alone buys +2.07 points, the largest copy gain measured in this line, for -0.79 language** - slightly more copy and
  slightly more cost than the four-tensor freeze.
- **Freezing `concept.W` on top of the pointer makes copy WORSE** (75.97 against 77.20) while costing the same language. The extra
  tensor is not free protection, it is a small loss.

The mechanism reading is consistent with `docs/COPY-REAUDIT.md` and `docs/COPY-WEIGHTS.md`: `concept.W` is the generalising part of
the copy path (worth 15 points of held-out copy, worth nothing to memorised answers) and the pointer matrices are what language
training overwrites most (drift 0.57 against 0.27 for `concept.W`). Protecting the generalising component alone is enough to hold
the ability.

## What the ship configuration is now
```
goal                                    configuration            copy      language
maximum held-out copy                   freeze ptrW, Wq, Wk      77.20     15.70
free copy protection (no language cost) freeze concept.W          75.30     16.46
```
Both beat doing nothing on copy; the choice is now a real engineering trade with measured numbers on each side rather than a single
"readable costs accuracy" story.

## Caveats
- Three seeds per subset arm. The copy gaps are 2.5-5.2 standard errors, but the *language* difference between the pointer arm and the
  concept arm (-0.79 against +0.16) is the number to trust least here: it comes from separate arms and a single evaluation per run.
- Late held-out copy is the mean of the last three evaluations, as in every other arm in this line; within-run spread reaches
  69.2-76.8 on single runs, which is why no arm is judged on one checkpoint.

## Reproduction
`lens/scripts/run_freeze_subset.sh` (prediction and falsifier in the header), logs `logs/sub/{concept,ptr}_s{0,1,2}.log`,
checkpoints `sub_concept_s*` / `sub_ptr_s*` in `lens/runs/chat/`.
