# The "probe disjointness SKIPPED" line above 200k windows is not a hole: the split makes it structural

2026-11-07 14:50 IST. Verified from the module the trainer imports (`lens/scripts/chat_data.py`, `import chat_data`
at `train_chat.py:50`), not from a temp copy.

## The line that looks alarming

`train_chat.py:783-787`:

    # The probe must be disjoint from the training windows. Checked exactly on window CONTENT (the
    # window's tokens plus its target token), because the drawn indices alone do not prove it. Above
    # 200,000 train windows the walk is skipped and the skip is PRINTED with the count - a stated limit,
    # never a silent pass.
    if len(Xtr) <= 200000:
        ...

Every data-scale arm this project has run - 500k, 2M, and the 16.9M-window headline - exceeded 200,000 train
windows, so **none of them printed a disjointness walk**. Someone reading those logs sees a stated limit and has
to decide whether the numbers above it are trustworthy. This note answers that, and the answer is yes, for a
structural reason rather than a checked one.

## Why the sets cannot overlap

1. **The split is assigned per conversation**, one flag per conversation:
   `split.append(1 if i in val_idx else 0)` (`chat_data.py:133`).
2. **Windows are selected per conversation by that flag**: `idx = [j for j in range(len(bounds) - 1) if
   split[j] == which]` (`chat_data.py:165`).
3. **Windows never cross a conversation boundary** - that is `make_windows`' own contract, stated in its
   docstring ("Windows that do NOT cross a conversation boundary -- otherwise the model learns to continue
   someone else's sentence, which is a silent data-leak-style defect"), and enforced by building each
   conversation's windows from its own `bounds[j] .. bounds[j+1]` slice with `stride = window // 2`
   (`chat_data.py:166`).

So a train window and a val window cannot share tokens: they come from different conversations. The content walk
above 200k windows is a **redundant second check**, and its absence costs nothing at this scale.

One related property, also structural: the vocabulary is built from **train text only** (`chat_data.py:119`,
"vocab built from TRAIN text only"), so the val set cannot influence tokenisation either.

## What this does not prove

* It does not check **corpus-level duplication** - a val conversation could be a near-copy of a train
  conversation (the same dialogue scraped twice). That is a different, harder check, and it has not been run at
  any scale. If it were a problem it would inflate every held-out number in this project, including 37.37%.
* The 500k/2M/16.9M arms also cap windows **per conversation** proportionally, so the val conversations are not
  a random sample of the corpus in any special way - but that is a sampling question, not a leakage one.

## Why this is being written down now

The d=512 cost probe (`lens/runs/PROBE-D512.txt`) prints `windows 16,900,000 train / 87,035 val` and, like every
arm above 200k, no disjointness walk. Rather than leave that gap to be worried about later, it is now resolved
from the code: **disjoint by construction, not skipped by accident.**
