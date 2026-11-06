"""TEST GATE 1 — every hand-written gradient vs finite differences.

Run:  python3 tests/test_gradcheck.py
Exit code 0 = gate PASS, 1 = gate FAIL.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from glassnet.gates import run_gate1  # noqa: E402

ok = run_gate1(verbose=True)
sys.exit(0 if ok else 1)