# HIDDEN_DEFAULTS.md — every number living inside the auto-pilot (Phase 3)

Per the maintainer's rule: *"some numbers still exist inside the defaults. List them.
Nothing secret."* This file lists EVERY numeric default the auto-pilot uses,
what it does, and why that number. The code's single source of truth is
`AUTO_PILOT_DEFAULTS` in `glassnet/autopilot.py`; `phase3_autopilot.py`
verifies this file covers every key and refuses to run if any is missing.

## Architecture (inherited unchanged from Phase 2 manual tuning)
| default | value | meaning | why that number |
|---|---|---|---|
| num_concepts | 14 | name-tagged concept neurons | Phase 2 sweep: 14 documented pixel concepts |
| num_rules | 30 | rules in the sparse rule layer | Phase 2 sweep: best accuracy at R=30 |
| concept_loss_alpha | 1.0 | weight of concept-name supervision | Phase 2 sweep: fidelity 0.81 at same accuracy |
| l1_lambda | 0.005 | rule-connection sparsity strength (sum-\|w\|) | Phase 2 sweep: real sparsity without collapsing |
| prune_keep_connections | 3 | connections kept per rule after pruning | the brief: 2–3 concepts per rule |

## Data handling
| default | value | meaning | why that number |
|---|---|---|---|
| batch_size | 128 | images per training step | same as Phase 1 baseline (apples-to-apples) |
| eval_train_sample | 5000 | images used for epoch train-accuracy | full 60k is wasteful; 5k is stable |

## Work budget
| default | value | meaning | why that number |
|---|---|---|---|
| max_epochs | 12 | hard cap on training distance | manual run used 10; +2 epochs of runway for lr cuts |
| min_learning_rate | 0.0001 | below this the pilot stops | smaller steps move nothing (float32 floor) |
| max_restarts | 3 | explosion-restart budget | sanity cap; >3 explosions = broken setup |

## The watcher's eyes and hands
| default | value | meaning | why that number |
|---|---|---|---|
| start_learning_rate | 0.1 | initial step size | same as the manual tuning's winner |
| check_every_batches | 100 | how often the pilot looks at the loss | ~1/5 of an epoch: responsive but not jittery |
| eval_every_batches | 469 | full eval ~once per epoch | 60000 / 128 = 469 batches per epoch |
| loss_ema_decay | 0.98 | smooth average for judging spikes | 50-batch half-life smooths mini-batch noise |
| flat_improvement_min | 0.005 | relative drop needed to count as improving | smaller would cut lr on noise, larger would wait forever |
| flat_patience_windows | 5 | flat windows before lowering step size | ~1 epoch of flatness = a real plateau |
| lr_cut_factor_flat | 0.5 | step-size multiplier when flat | classic halving |
| spike_factor | 3.0 | loss above 3× smooth = spike | mini-batch CE rarely exceeds this without cause |
| lr_cut_factor_spike | 0.7 | step-size multiplier after a spike | mild correction (not a restart) |
| explode_restart_lr_factor | 0.5 | step-size multiplier for a full restart | classic halving |
| explode_jump_factor | 20.0 | window loss > 20× the previous window = explosion | order-of-magnitude jump is never healthy; below that it's a spike |
| test_gap_threshold | 0.10 | train−test gap beyond this = overfit | manual run's gap was <0.01; 0.10 is a clear alarm |
| early_stop_no_improve_evals | 2 | consecutive evals with an overfit gap (train−test > 0.10) → stop early | 1 eval would stop on noise; 2 is decisive |
| silent_activation_min | 0.03 | mean firing below this = "never fires" | sigmoid rules that average below 3% fire on ~nothing |
| concept_rescue_fidelity_floor | 0.55 | name fires on its label below this AUROC → name-rescue boosts it | matches the audit's broken-name threshold (fidelity < 0.55) |
| concept_rescue_boost | 1.0 (OFF) | BCE multiplier for rescued concept names | R2-013 measured the reflex FAILING: x5 boost cost −7 pts accuracy and did NOT save the drifted names → switched OFF by default; code + controls remain (documented experiment, can be re-enabled) |

## Reproducibility (also defaults, also listed)
| default | value | meaning |
|---|---|---|
| input_dim | 784 | input width: 784 pixels (Fashion-MNIST, PathMNIST) or 2048 bag-of-words (Banking77 text) |
| seed | 0 | model-init seed (runner uses seeds 0, 1, 2 for the mean±std gate) |
| data stream seed | seed*777+1 | training shuffle |
| restart seeds | seed*1000+1..3 | per-restart re-init (deterministic restarts) |
| eval sample seed | step+100000 | deterministic per-eval sampling |

## Change history
- 2026-10-01: initial set. All values chosen from the Phase 2 logs/sweeps;
  every change to these numbers goes in this table with a new row, never silently.

## Phase 4 — data judge defaults (also listed, nothing secret)
| default | value | meaning |
|---|---|---|
| good_count | 1200 | good rows in the hand-labeled pile (120 per class) |
| bad_per_type | 150 | bad rows per corruption type |
| corruption_count | 8 | corruption types: noise_full, noise_partial, scrambled, rotated90, upside_down, half_bled, box_overlay, salt_pepper |
| injected_dupes / _empty / _gray / _nan | 40 / 20 / 20 / 20 | junk injected ONLY to prove the cleaner works |
| empty_mean_max | 0.01 | mean pixel below this = blank/empty |
| flat_std_max | 0.005 | pixel std below this = constant/gray junk |
| dup_l1_max | 0.02 | mean abs diff below this = near-duplicate |
| dup_bucket_size / dup_levels | 14 / 4 | near-dup fingerprint: 14×14 downsample, 4 quantization levels |
| hidden | 32 | judge hidden units (784→32→1) |
| epochs | 10 | judge training epochs (was 6 in the failed v1 gate: 8 epochs = 72% unseen-type vs 54.7% at 6, measured in experiments/judge_features.py) |
| batch_size | 64 | judge batch size |
| learning_rate | 0.1 | judge step size |
| train_fraction | 0.6 | 60% of the cleaned pile trains the judge, 40% held out |
| ensemble_seeds | [0,1,2] | gate runs 3 seeds; the gate verdict uses their MEAN probability |
| pile seed / split seed | 42 / 7 | reproducibility of the pile and the split |

## Change history (Phase 4)
- 2026-10-01: epochs 6 → 10 after the v1 gate FAILED the unseen-type test
  (54.67% ± 4.45%). The feature experiment showed epoch count, not feature
  space, was the main lever (72.2% pixels unseen-type at 8 epochs); the
  concept-based judge generalized better on unseen types (75.2%) but worse
  on held-out rows (71.6% vs 79.8%) — pixels kept, epochs raised, and an
  explicit 3-seed mean-probability ensemble is the official gate verdict.

## Phase 6 — evolution-strategy trainer defaults (listed, nothing secret)
| default | value | meaning |
|---|---|---|
| batch_size | 128 | images per evaluation batch |
| budget_batches | 5628 | = 12 epochs × 469 batches — same teaching budget as SGD |
| sigma_start | 0.05 | initial mutation noise (per weight) |
| sigma_min | 0.0001 | floor on mutation noise |
| sigma_grow / sigma_shrink | 1.15 / 0.85 | 1/5-rule multipliers |
| onefifth_window | 50 | evaluations per success-rate measurement |
| onefifth_high / onefifth_low | 0.22 / 0.18 | success rate above/below which sigma grows/shrinks |
| concept_loss_alpha / l1_lambda | 1.0 / 0.005 | the SAME objective the SGD trainer uses |
| eval_every_batches | 469 | test-accuracy cadence (same as SGD) |
| prune_keep_connections | 3 | same post-training pruning as SGD |