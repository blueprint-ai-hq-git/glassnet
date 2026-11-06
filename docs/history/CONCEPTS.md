# CONCEPTS.md — the named neurons of the Concept layer (Phase 2)

Every concept is computed straight from the pixels by the formula below.
Nothing is learned about the concept definitions — learning only maps pixels
to concept VALUES. All formulas live in `glassnet/concepts.py` (the code is
the source of truth; this file is the human-readable copy).

> **Rung 3 (text, Banking77):** `glassnet/text_concepts.py` defines the
> WORD concepts below — fixed lexicons/formulas, tokenised with
> `[a-z']+` on lowercased text, min-max normalised on TRAIN only.

## The 14 concepts

X is a 28×28 image in [0,1] (0 = white background, 1 = black ink). mean(A)
= average of the pixel values in region A.

| name | formula (exact) | plain meaning |
|------|-----------------|---------------|
| ink_amount | mean(X) | how much dark ink |
| symmetry | 1 − 2·mean(\|X − X_fliplr\|) | mirror-symmetric left/right |
| top_heavy | (mean(top 14 rows) − mean(bottom 14 rows) + 1)/2 | more ink up top |
| horizontal_edges | mean(\|X[:,1:] − X[:,:-1]\|) | vertical stripes / edges |
| vertical_edges | mean(\|X[1:,:] − X[:-1,:]\|) | horizontal stripes / edges |
| quad_UL_ink | mean(X[0:14, 0:14]) | ink in upper-left quarter |
| quad_UR_ink | mean(X[0:14, 14:28]) | ink in upper-right quarter |
| quad_LL_ink | mean(X[14:28, 0:14]) | ink in lower-left quarter |
| quad_LR_ink | mean(X[14:28, 14:28]) | ink in lower-right quarter |
| center_ink | mean(X[7:21, 7:21]) | ink in the middle 14×14 |
| left_heavy | (mean(left 14 cols) − mean(right 14 cols) + 1)/2 | more ink on the left |
| border_ink | mean of the 4 border rows/cols | ink touching the frame |
| tall_shape | (fraction of 28 ROWS containing any pixel > 0.25) | silhouette height |
| wide_shape | (fraction of 28 COLUMNS containing any pixel > 0.25) | silhouette width |

Normalisation: each raw column is min–max scaled to [0,1] using TRAIN-set
min/max only (test uses the same scale). The exact lo/hi values used are
stored in `data/concepts.npz` and printed in `docs/concept-class-stats.md`.

Per-class means of every concept (shows which concepts separate which
clothing classes) → `docs/concept-class-stats.md`.

## How fidelity is audited (the "does it mean what its label says?" check)

For concept neuron k on the held-out test split:
- **fidelity** = AUROC of (neuron k's activation) vs (true concept k target
  binarised at >0.5). Written by hand in glassnet/eval.py — no sklearn.
  1.0 = perfect match with its name, 0.5 = random, <0.5 = it detects the
  ABSENCE of its name.
- **cross-leak**: AUROC of neuron k vs EVERY concept target; if another
  concept beats the namesake, the neuron secretly leans elsewhere.
- **lie-neuron control**: a neuron is deliberately trained with its target
  replaced by a DIFFERENT concept, and the audit must flag it. This is the
  sanity test that catches "a neuron secretly detecting something else".

## Honest caveats
- Fidelity measures whether a neuron tracks its NAME — not whether the
  name is useful for the task. A faithful-but-unused concept gets near-zero
  rule weights; that is reported separately.
- Concepts are engineered, so near-perfect fidelity is the designed result;
  the interesting failures show up in the ablations (no concept supervision)
  and in the lie-neuron control.
## The 12 text concepts (Banking77, rung 3) — exact lexicons

Counts are over the tokenised utterance; normalised [0,1] on TRAIN only.

| name | exact definition | plain meaning |
|------|------------------|---------------|
| word_count | number of tokens | how long the utterance is |
| avg_word_len | mean token length (chars) | long/formal words vs short slang |
| question_start | 1 if first token ∈ {when,what,where,why,how,which,who,do,does,did,can,could,would,should,is,are,am,will} | the user starts with a question word |
| polite | count of {please, kindly, thanks, thank} | politeness |
| card_family | count of {card, cards, debit, credit, atm, visa, mastercard} | mentions of cards |
| fee_family | count of {fee, fees, charge, charges, charged, overcharged, chargeback} | mentions of fees |
| money_family | count of {money, payment, payments, paid, pay, refund, cash, deposit, withdrawals, withdrawal, withdraw, transfer, transfers, transferred, sent, send, topup, top} | money/transfer talk |
| account_family | count of {account, accounts, balance, bank, banking, statement, statements, card, cards, pin} | account talk |
| problem_family | count of {problem, issue, wrong, broken, error, failed, failing, work, works, working, usable, unusable} | complaints/problems |
| time_family | count of {when, how long, days, day, week, weeks, hours, hour, schedule, appointment, opening, open} | timing/availability |
| punctuation_frac | non-alphanumeric chars / utterance length | em-dashes, ? ! pressure |
| uppercase_frac | uppercase letters / alphabetic letters | shouting/emphasis |

**Measured but excluded:** `digit_frac` (tokens containing a digit) — the
entire Banking77 corpus contains ZERO digits (verified: 0/9,038 train
utterances), so the column is constant and the name can never fire. Kept in
the registry, excluded from every loop set, documented as dead-on-this-task.
