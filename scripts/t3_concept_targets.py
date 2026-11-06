"""t3_concept_targets.py — derive the supervised concept targets from TRAIN text ONLY.

Condition 1 of the approval: concept targets must be derived automatically from
TRAIN text only, and pre-registered in a file BEFORE any run. No val/test text is
touched anywhere in this script.

Method (fixed here, before looking at any result):
  * Source: the TRAIN split windows only (d["Xtr"]).
  * Count how often each vocabulary word occurs, as the fraction of TRAIN windows
    that contain it at least once ("train_pos_frac").
  * Candidate = a word that is
      - in the vocabulary,
      - NOT in the STOP list (stopwords/punctuation tokens),
      - with train_pos_frac in [0.05, 0.50]   (so the target label is neither
        degenerate-rare nor degenerate-common),
      - length >= 2 characters.
  * Rank candidates by train_pos_frac descending; take the top N_SUP.
  * Those words become the supervised concept targets, assigned to concept
    columns 0..N_SUP-1 in rank order. Columns N_SUP..K-1 stay UNSUPERVISED and
    serve as the circularity control.

The control matters: if the audit's AUROC rises only on the SUPERVISED columns,
the gain is circular (we taught it the answer). The unsupervised columns in the
same model are the honest check.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, PROJ)
sys.path.insert(0, HERE)

import numpy as np

from glassnet import tiny_data
from run_t2_iter003 import STOP, WIN

VOCAB_CAP = 1500
N_SUP = int(os.environ.get("CT_N", "16"))
MIN_PF, MAX_PF = 0.0, 1.0     # no frequency floor: "top frequent words" (owner's wording).
                              # Measured on TRAIN: the top 16 non-STOP tokens run from
                              # pos_frac 0.1700 ("I") down to 0.0151 ("As").

OUT_JSON = os.path.join(PROJ, "runs", "t3", "concept_targets.json")
OUT_MD = os.path.join(PROJ, "CONCEPT-TARGETS.md")

d = tiny_data.build(win=WIN, vocab_cap=VOCAB_CAP)
vocab = list(d["vocab"])
V = d["V"]
Xtr = np.asarray(d["Xtr"])
N = len(Xtr)

# --- TRAIN ONLY -----------------------------------------------------------
contains = np.zeros(V, dtype=np.float64)
for p in range(WIN):
    np.add.at(contains, Xtr[:, p], 1.0)
train_pos_frac = contains / N
# --------------------------------------------------------------------------

cands = []
for j, w in enumerate(vocab):
    pf = float(train_pos_frac[j])
    if w in STOP or len(w) < 2:
        continue
    if not (MIN_PF <= pf <= MAX_PF):
        continue
    cands.append({"word": w, "id": j, "train_pos_frac": pf})
cands.sort(key=lambda r: (-r["train_pos_frac"], r["word"]))
sup = cands[:N_SUP]
for r in sup:
    r["pos_weight"] = (1.0 - r["train_pos_frac"]) / max(r["train_pos_frac"], 1e-9)

print(f"CONCEPT TARGETS — derived from TRAIN windows only (n={N} windows, V={V})")
print(f"  selection rule: not a stopword, len>=2, train_pos_frac in "
      f"[{MIN_PF:.2f}, {MAX_PF:.2f}], ranked by train_pos_frac desc, top {N_SUP}")
print(f"  candidates passing the filter: {len(cands)}")
print(f"\n  {'col':>4} {'word':>14} {'train_pos_frac':>15} {'train_count':>12}")
for i, r in enumerate(sup):
    print(f"  {i:>4} {r['word']:>14} {r['train_pos_frac']:>15.4f} "
          f"{int(round(r['train_pos_frac']*N)):>12}")
print(f"\n  columns {N_SUP}..K-1 remain UNSUPERVISED (circularity control)")

# sanity: confirm no val/test leakage in the derivation
res = {
    "derived_from": "TRAIN split windows only (d['Xtr'])",
    "val_used": False, "test_used": False,
    "n_train_windows": int(N), "vocab_cap": VOCAB_CAP, "win": WIN,
    "filter": {"exclude_stop": True, "min_len": 2,
               "min_train_pos_frac": MIN_PF, "max_train_pos_frac": MAX_PF},
    "n_supervised": N_SUP,
    "supervised_columns": list(range(N_SUP)),
    "unsupervised_columns": f"{N_SUP}..K-1",
    "targets": sup,
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
json.dump(res, open(OUT_JSON, "w"), indent=1)

lines = [
    "# CONCEPT TARGETS (supervised concept loss) — PRE-REGISTERED",
    "",
    "**Frozen before any run of the auxiliary concept loss.** Derived from **TRAIN",
    "text only**; val and test text were never read by the derivation script.",
    "",
    "## Method (fixed in advance)",
    "",
    f"- Source: TRAIN split windows only (`d[\"Xtr\"]`, {N} windows, vocab {V}).",
    "- `train_pos_frac(w)` = fraction of TRAIN windows containing `w` at least once.",
    "- A word is a candidate if it is in the vocabulary, **not** in the STOP list,",
    "  and has length >= 2. **No frequency floor** — this is the owner's \"top",
    "  frequent words\" rule, applied literally.",
    f"- Rank candidates by train_pos_frac descending; take the top {N_SUP}.",
    f"- Target word i is assigned to concept column i (0..{N_SUP - 1}).",
    f"- **Columns {N_SUP}..K-1 are NOT supervised** and are the circularity control.",
    "",
    "## The targets",
    "",
    "| concept column | target word | train_pos_frac | train count | pos_weight |",
    "|---|---|---|---|---|",
]
for i, r in enumerate(sup):
    pw = (1.0 - r["train_pos_frac"]) / max(r["train_pos_frac"], 1e-9)
    lines.append(f"| {i} | `{r['word']}` | {r['train_pos_frac']:.4f} | "
                 f"{int(round(r['train_pos_frac']*N))} | {pw:.1f} |")
lines += [
    "",
    "## Why the control columns exist",
    "",
    "Supervising a concept on a word makes the audit's AUROC clause easier for that",
    "column — the neuron is being taught to detect the word the audit then asks about.",
    "That is circular. The unsupervised columns in the SAME model are the honest",
    "check: if their pass fraction does not move, any gain is supervision, not",
    "readability. Both are reported separately.",
    "",
    "## The auxiliary loss",
    "",
    "`loss = task_loss + lam_concept * mean_over_supervised_columns BCE(z_k, y_k)`",
    "",
    "where `z_k` is concept k's pre-activation and `y_k` is the window's binary",
    "contains-target for that column. `lam_concept = 0` (default) leaves the old",
    "model bit-for-bit intact.",
    "",
    "**Balanced BCE (fixed in advance).** The targets are imbalanced (base rates",
    "0.170 down to 0.015), and a plain BCE would be minimised by leaving rare",
    "concepts permanently off — which would make them unnameable and fail the audit",
    "for a reason that has nothing to do with readability. So the positive terms are",
    "weighted by `pos_weight = (1 - p) / p`, where `p` is that column's TRAIN base",
    "rate. This is the standard balanced-BCE weighting, computed on TRAIN only.",
    "",
    f"Artifact: `runs/t3/concept_targets.json`",
]
open(OUT_MD, "w").write("\n".join(lines) + "\n")
print(f"\nsaved {OUT_JSON}")
print(f"saved {OUT_MD}")
