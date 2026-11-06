# CONCEPT TARGETS (supervised concept loss) — PRE-REGISTERED

**Frozen before any run of the auxiliary concept loss.** Derived from **TRAIN
text only**; val and test text were never read by the derivation script.

## Method (fixed in advance)

- Source: TRAIN split windows only (`d["Xtr"]`, 182377 windows, vocab 1501).
- `train_pos_frac(w)` = fraction of TRAIN windows containing `w` at least once.
- A word is a candidate if it is in the vocabulary, **not** in the STOP list,
  and has length >= 2. **No frequency floor** — this is the owner's "top
  frequent words" rule, applied literally.
- Rank candidates by train_pos_frac descending; take the top 16.
- Target word i is assigned to concept column i (0..15).
- **Columns 16..K-1 are NOT supervised** and are the circularity control.

## The targets

| concept column | target word | train_pos_frac | train count | pos_weight |
|---|---|---|---|---|
| 0 | `And` | 0.0705 | 12856 | 13.2 |
| 1 | `he` | 0.0464 | 8456 | 20.6 |
| 2 | `The` | 0.0337 | 6144 | 28.7 |
| 3 | `To` | 0.0319 | 5824 | 30.3 |
| 4 | `him` | 0.0292 | 5320 | 33.3 |
| 5 | `That` | 0.0246 | 4488 | 39.6 |
| 6 | `But` | 0.0210 | 3824 | 46.7 |
| 7 | `KING` | 0.0204 | 3720 | 48.0 |
| 8 | `good` | 0.0198 | 3616 | 49.4 |
| 9 | `For` | 0.0183 | 3336 | 53.7 |
| 10 | `My` | 0.0176 | 3208 | 55.9 |
| 11 | `they` | 0.0174 | 3176 | 56.4 |
| 12 | `I'll` | 0.0166 | 3024 | 59.3 |
| 13 | `more` | 0.0155 | 2832 | 63.4 |
| 14 | `As` | 0.0151 | 2760 | 65.1 |
| 15 | `What` | 0.0149 | 2720 | 66.1 |

## Why the control columns exist

Supervising a concept on a word makes the audit's AUROC clause easier for that
column — the neuron is being taught to detect the word the audit then asks about.
That is circular. The unsupervised columns in the SAME model are the honest
check: if their pass fraction does not move, any gain is supervision, not
readability. Both are reported separately.

## The auxiliary loss

`loss = task_loss + lam_concept * mean_over_supervised_columns BCE(z_k, y_k)`

where `z_k` is concept k's pre-activation and `y_k` is the window's binary
contains-target for that column. `lam_concept = 0` (default) leaves the old
model bit-for-bit intact.

**Balanced BCE (fixed in advance).** The targets are imbalanced (base rates
0.170 down to 0.015), and a plain BCE would be minimised by leaving rare
concepts permanently off — which would make them unnameable and fail the audit
for a reason that has nothing to do with readability. So the positive terms are
weighted by `pos_weight = (1 - p) / p`, where `p` is that column's TRAIN base
rate. This is the standard balanced-BCE weighting, computed on TRAIN only.

Artifact: `runs/t3/concept_targets.json`
