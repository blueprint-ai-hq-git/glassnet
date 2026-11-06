"""Tests for lens/ckpt_guard.py - the best-checkpoint clobber guard.

The guard exists because a restarted run replaced wall_off_s0's 13.00% epoch-3 best.npz with an
aborted second attempt's epoch-1 file (docs/history/WALL-AB-ARMS.md, "Provenance defect"). These
tests use a throwaway directory and fake bytes; no model, no GPU, no training.
"""
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lens.ckpt_guard import guard_existing_best, sha256_file   # noqa: E402


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(payload)


def test_nothing_to_guard_is_a_no_op():
    with tempfile.TemporaryDirectory() as d:
        assert guard_existing_best(d, "arm_x", log=lambda *_: None) is None
        assert os.listdir(d) == []


def test_existing_best_is_moved_not_deleted_and_bytes_survive():
    with tempfile.TemporaryDirectory() as d:
        payload = b"epoch-3 weights, pretend these are 23 MB" * 1000
        _write(os.path.join(d, "arm_s0_best.npz"), payload)
        want = hashlib.sha256(payload).hexdigest()

        rec = guard_existing_best(d, "arm_s0", log=lambda *_: None, stamp="TESTSTAMP")

        # the live path is free again, so the fresh run's first epoch can write there
        assert not os.path.exists(os.path.join(d, "arm_s0_best.npz"))
        # the old checkpoint still exists, at the archive path, byte-identical
        assert rec["to"] == os.path.join(d, "archive", "best_clobber_guard_TESTSTAMP",
                                        "arm_s0_best.npz." + want[:8])
        assert os.path.exists(rec["to"])
        assert sha256_file(rec["to"]) == want == rec["sha256"] == rec["sha256_after"]
        assert rec["bytes"] == len(payload)
        # and a receipt sits beside it naming the same hash
        with open(rec["to"] + ".sha256") as f:
            assert f.read().strip().endswith(os.path.basename(rec["to"]))
            f.seek(0)
            assert f.read().startswith(want)


def test_second_run_guards_the_first_runs_leftover():
    """The exact wall_off_s0 shape: run A leaves a best, run B starts fresh and must not overwrite it."""
    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "a_best.npz"), b"run A, epoch 3, 13.00%")
        first = guard_existing_best(d, "a", log=lambda *_: None, stamp="T1")
        _write(os.path.join(d, "a_best.npz"), b"run B, epoch 1, 7.54%")
        second = guard_existing_best(d, "a", log=lambda *_: None, stamp="T2")

        assert first["to"] != second["to"]
        assert first["sha256"] != second["sha256"]
        assert open(first["to"], "rb").read() == b"run A, epoch 3, 13.00%"
        assert open(second["to"], "rb").read() == b"run B, epoch 1, 7.54%"


def test_dry_run_leaves_everything_in_place():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a_best.npz")
        _write(p, b"weights")
        rec = guard_existing_best(d, "a", log=lambda *_: None, dry_run=True)
        assert os.path.exists(p)
        assert rec["sha256_after"] is None
        assert not os.path.exists(os.path.join(d, "archive"))


if __name__ == "__main__":
    test_nothing_to_guard_is_a_no_op()
    test_existing_best_is_moved_not_deleted_and_bytes_survive()
    test_second_run_guards_the_first_runs_leftover()
    test_dry_run_leaves_everything_in_place()
    print("test_ckpt_guard: 4/4 PASS")
