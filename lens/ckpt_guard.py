"""Best-checkpoint clobber guard: one small function with a test behind it.

WHY THIS EXISTS (measured, not theorised - docs/history/WALL-AB-ARMS.md, "Provenance defect"):

    train_chat.py starts its best-epoch tracking empty (best_val = None) on every fresh run, so a
    restarted run writes <out>_best.npz again from its OWN epoch 1 and silently replaces an earlier
    run's better checkpoint with a worse one. That happened to wall_off_s0 on 2026-11-06 17:11: its
    13.00% epoch-3 file was replaced by an aborted second attempt's epoch-1 file (sha256
    714d15c6...), and the epoch-3 file only survived because a separate ep3.npz happened to exist.

WHAT IT DOES: before a FRESH run (no --resume-mid state) writes any best checkpoint, if
<out>_best.npz already exists it is MOVED into

    <out>/archive/best_clobber_guard_<stamp>/<out>_best.npz.<sha8>

with a .sha256 receipt written beside it, and the hash printed in the log. Nothing is deleted.
The hash is re-checked after the move, so a receipt can never describe bytes that are not there.

WHAT IT DOES NOT DO: it does not decide which checkpoint is better, does not compute any number, and
does not touch the weights. A resumed run (mid-state present) is left alone on purpose: its best
memory is restored from the mid-state, so it cannot regress an earlier best by accident.
"""
import hashlib
import os
import shutil
import time


def sha256_file(path):
    """sha256 of a file, in 1 MiB chunks, returned as hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def guard_existing_best(out_dir, tag, log=print, stamp=None, dry_run=False):
    """Move an existing <tag>_best.npz aside so a fresh run cannot overwrite it.

    Returns None when there was nothing to guard (the common case: a first run of that tag).
    Returns a dict when a file was moved:
        {"from": <old path>, "to": <archived path>, "sha256": <hash before the move>,
         "sha256_after": <hash re-measured at the destination>, "bytes": <size>}
    Raises RuntimeError if the hash does not survive the move - a lost receipt is worse than none.
    """
    path = os.path.join(out_dir, tag + "_best.npz")
    if not os.path.exists(path):
        return None

    before = sha256_file(path)
    size = os.path.getsize(path)
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(out_dir, "archive", "best_clobber_guard_" + stamp)
    dest = os.path.join(dest_dir, "%s_best.npz.%s" % (tag, before[:8]))

    if dry_run:
        log("  CLOBBER GUARD (dry run): %s exists (sha256 %s, %d bytes) - would move to %s"
            % (os.path.basename(path), before, size, dest))
        return {"from": path, "to": dest, "sha256": before, "sha256_after": None, "bytes": size}

    os.makedirs(dest_dir, exist_ok=True)
    shutil.move(path, dest)
    after = sha256_file(dest)
    with open(dest + ".sha256", "w") as f:
        f.write("%s  %s\n" % (after, os.path.basename(dest)))
    if after != before:
        raise RuntimeError("clobber guard lost bytes moving %s -> %s (%s != %s)"
                           % (path, dest, before, after))
    log("  CLOBBER GUARD: %s_best.npz already existed (sha256 %s, %d bytes) - moved to %s so this "
        "fresh run cannot replace another run's checkpoint. Nothing deleted; receipt written."
        % (tag, before, size, dest))
    return {"from": path, "to": dest, "sha256": before, "sha256_after": after, "bytes": size}
