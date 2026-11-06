#!/usr/bin/env python3
"""Apply the best-checkpoint clobber guard to lens/scripts/train_chat.py. Idempotent, refuses on drift.

The guard code itself lives in lens/ckpt_guard.py and is tested by tests/test_ckpt_guard.py, so this
script only has to make two exact edits:

  1. import it (next to the other `import lens...` lines),
  2. call it once, before the epoch loop, when the run is FRESH (no --resume-mid state).

Why it must run when no arm is training: docs/history/WALL-AB-ARMS.md says editing train_chat.py
between arms makes the arms non-identical, and the wall A/B is the one experiment that needs them
identical. The wrapper script checks for a live train_chat.py process and exits 3 without touching
anything if one is running.

Usage: .venv-gpu/bin/python3 lens/scripts/apply_best_ckpt_guard.py [--check]
       --check only reports whether the patch is present; nothing is written.
"""
import argparse
import hashlib
import os
import py_compile
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TARGET = os.path.join(ROOT, "lens", "scripts", "train_chat.py")
BACKUP_DIR = os.path.join(ROOT, "lens", "runs", "archive", "train_chat_prepatch_" +
                          time.strftime("%Y%m%d_%H%M%S"))

IMPORT_ANCHOR_LINE = "import lens                                                # noqa: E402"
IMPORT_ADDED_LINE = ("import lens.ckpt_guard as ckpt_guard                    # noqa: E402  (clobber guard)")

CALL_ANCHOR = "    hist = []\n    gstep = 0\n"
CALL_ADDED = """    # ---- BEST-CHECKPOINT CLOBBER GUARD (2026-11-06, docs/history/WALL-AB-ARMS.md "Provenance defect") ----
    # Best tracking starts empty on every fresh run, so a restarted run rewrites <out>_best.npz from its
    # own epoch 1 and can silently replace an earlier run's better checkpoint (measured on wall_off_s0:
    # 13.00% epoch 3 replaced by an aborted second attempt's epoch 1). A resumed run restores its best
    # memory from the mid-state and is left alone. Nothing is deleted: the old file is moved into
    # archive/best_clobber_guard_<stamp>/ with a .sha256 receipt and its hash is printed here.
    if mid_state is None:
        ckpt_guard.guard_existing_best(OUT, a.out, log=log)
    hist = []
    gstep = 0
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    src = open(TARGET).read()
    has_import = "import lens.ckpt_guard as ckpt_guard" in src
    has_call = "ckpt_guard.guard_existing_best(OUT, a.out, log=log)" in src

    if args.check:
        print("patched: import=%s call=%s" % (has_import, has_call))
        return 0 if (has_import and has_call) else 1

    if has_import and has_call:
        print("already patched - nothing to do (no file written)")
        return 0
    if has_import != has_call:
        print("REFUSING: half-patched file (import=%s call=%s) - inspect by hand" % (has_import, has_call))
        return 4
    if src.count(IMPORT_ANCHOR_LINE) != 1:
        print("REFUSING: import anchor appears %d times (need exactly 1) - train_chat.py has drifted, "
              "patch by hand" % src.count(IMPORT_ANCHOR_LINE))
        return 5
    if CALL_ANCHOR not in src or src.count(CALL_ANCHOR) != 1:
        print("REFUSING: call anchor not unique (count=%d) - train_chat.py has drifted, patch by hand"
              % src.count(CALL_ANCHOR))
        return 6

    before = sha256(TARGET)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup = os.path.join(BACKUP_DIR, "train_chat.py")
    shutil.copy2(TARGET, backup)
    if sha256(backup) != before:
        print("REFUSING: backup hash mismatch, nothing changed")
        return 7

    new = src.replace(IMPORT_ANCHOR_LINE, IMPORT_ANCHOR_LINE + "\n" + IMPORT_ADDED_LINE, 1) \
             .replace(CALL_ANCHOR, CALL_ADDED, 1)
    tmp = TARGET + ".tmp_patch"
    with open(tmp, "w") as f:
        f.write(new)
    try:
        py_compile.compile(tmp, doraise=True)
    except py_compile.PyCompileError as e:
        os.unlink(tmp)
        print("REFUSING: patched file does not compile: %s" % e)
        return 8
    os.replace(tmp, TARGET)
    after = sha256(TARGET)

    # the patched file must still be a superset of the original: both anchors' neighbours present
    check = open(TARGET).read()
    post_import = check.count("import lens.ckpt_guard as ckpt_guard")
    post_call = check.count("ckpt_guard.guard_existing_best(OUT, a.out, log=log)")
    ok = (post_import == 1 and post_call == 1)
    print("train_chat.py sha256 %s -> %s" % (before, after))
    print("backup kept at %s (same sha256 as the pre-patch file)" % backup)
    # has_import/has_call are the PRE-patch reads. Printing them on the "patched:" line made a correct
    # patch read as a failure ("patched: import=False call=False  post-write check=True" on 2026-11-06
    # 17:57, which contradicted the prediction P1 word for word). Report the post-write counts instead.
    print("before patch: import=%s call=%s" % (has_import, has_call))
    print("patched: import=%s call=%s  post-write check=%s"
          % (post_import == 1, post_call == 1, ok))
    return 0 if ok else 9


if __name__ == "__main__":
    sys.exit(main())
