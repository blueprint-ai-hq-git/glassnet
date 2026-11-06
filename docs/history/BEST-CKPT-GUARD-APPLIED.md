# Best-checkpoint clobber guard: applied to the trainer, and the report line that lied about it

2026-11-06 17:57 IST. Queue item `bash lens/scripts/run_apply_best_ckpt_guard.sh`, run through the chain
(label `apply_ckpt_guard`) with no `train_chat.py` alive. Log: `lens/runs/BEST-CKPT-GUARD-APPLY.txt`.

## What changed, with hashes

    lens/scripts/train_chat.py  f10dc89ee4f3377e98f1f7c17d86b5876d3637c8b7b352c910eda36c016b3a0b
                             -> e5acd0aafc8d17c361f2964b002ac808358150b13a7c0323be8f88349a47fa1d

Backup of the pre-patch file, hash-compared before the write:
`lens/runs/archive/train_chat_prepatch_20261106_175744/train_chat.py` (sha256 f10dc89e..., identical to
the pre-patch file). Nothing was deleted.

`diff -u` against that backup is exactly two hunks: one import at line 49, and the guarded call before the
epoch loop at line 1016 (`if mid_state is None: ckpt_guard.guard_existing_best(OUT, a.out, log=log)`). A
resumed run is deliberately untouched - it restores its best memory from the mid-state.

    .venv-gpu/bin/python3 -m py_compile lens/scripts/train_chat.py   -> OK
    .venv-gpu/bin/python3 tests/test_ckpt_guard.py                   -> 4/4 PASS
    .venv-gpu/bin/python3 lens/scripts/apply_best_ckpt_guard.py --check -> patched: import=True call=True

## The prediction, and the part of it that did not hold

P1 said the apply "reports `patched: import=True call=True`". It printed:

    patched: import=False call=False  post-write check=True

Both numbers were real - `import=False call=False` were the reads taken BEFORE the patch - but the sentence
stapled to them said the patch had failed while the same line said it had worked. Third instance of the
same class this evening (after the `wall_ab_grade.py` P2 sign bug): right numbers, wrong sentence. So P1
is recorded as **not reproduced as written**; the state it described is true, and it was proven by the
`--check` line above, not by the apply line.

Fixed in `lens/scripts/apply_best_ckpt_guard.py`: the "patched:" line now prints the post-write counts and
the pre-patch reads are on their own "before patch:" line. Verified on a COPY of the pre-patch backup
rather than on the live trainer: `before patch: import=False call=False`, then
`patched: import=True call=True  post-write check=True`, then a second run `already patched - nothing to
do (no file written)`, and the copy's final sha256 is e5acd0aa... - byte-identical to the real patched
file. Nothing was patched twice to prove it.

P2 (tests stay 4/4) held. P3 (second apply is a no-op) held.

## Added because tonight produced the problem three times

`run_apply_best_ckpt_guard.sh` and `run_guard_e2e.sh` now take a `flock -n` on
`lens/runs/.lock_<script-name>` before doing anything, and exit 0 with a REFUSED line if another copy
holds it. Reason, measured: when a short chain item ends, the chain wakes an agent immediately, and two
woken agents read the same queue line. Three duplicate-agent collisions happened today; one swept an
in-flight edit into an unrelated commit (68557e5) and an earlier one clobbered a checkpoint. The lock is
cheap and it is the only thing in the loop that makes a duplicate harmless instead of destructive.

## Still open

`run_guard_e2e.sh` is the next queue item: two 1-epoch CPU runs on 1,500 windows under tag `guard_e2e`,
proving the guard fires at the CALL SITE (the unit test only proves the function). Its predictions P1-P4
were written in `lens/runs/QUEUE.txt` before either run.
