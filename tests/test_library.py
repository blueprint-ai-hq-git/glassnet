"""Basic library tests (Phase 5). Plain asserts, no frameworks.

Run:  python3 tests/test_library.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from glassnet.gates import run_gate1  # noqa: E402
from glassnet.api import GlassNet  # noqa: E402
from glassnet.data import load as load_pixels  # noqa: E402

DATA_DIR = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "runs", "tests")
os.makedirs(OUT, exist_ok=True)

_tiny = None


def tiny_data():
    global _tiny
    if _tiny is None:
        Xtr, ytr, _, _ = load_pixels(DATA_DIR)
        idx = np.concatenate([np.where(ytr == c)[0][:20] for c in range(10)])
        _tiny = (Xtr[idx], ytr[idx])
    return _tiny


def test_gate1_still_passes():
    assert run_gate1(verbose=False) is True


def test_construct():
    m = GlassNet("fashion-mnist", out_dir=OUT)
    assert m.model is None
    assert m.class_names[1] == "trouser"
    try:
        GlassNet("unknown-task")
        assert False, "should have raised"
    except ValueError:
        pass


def test_train_mini_explain_and_report():
    X, y = tiny_data()
    m = GlassNet("fashion-mnist", out_dir=OUT)
    info = m.train((X, y), epochs=1)
    assert "actions" in info and "post_prune_acc" in info
    acc = m.accuracy()
    assert 0.0 <= acc <= 1.0
    # explain by array
    text = m.explain(X[0])
    assert "I said" in text and "rule #" in text
    # explain with a saved picture
    png = os.path.join(OUT, "test_report.png")
    m.explain(X[0], save=png)
    assert os.path.exists(png) and os.path.getsize(png) > 0
    # explain by index needs the built-in test set -> use custom split store
    assert m.X_test is not None and len(m.X_test) > 0


def test_judge_integration():
    X, y = tiny_data()
    m = GlassNet("fashion-mnist", out_dir=OUT)
    rep = m.train_judge(epochs=1)
    assert "total_removed" in rep and rep["total_removed"] == 100
    scores = m.judge(np.zeros((6, 784), np.float32))
    assert scores.shape == (6,)
    assert float(scores.min()) >= 0.0 and float(scores.max()) <= 1.0


def test_save_load():
    X, y = tiny_data()
    m = GlassNet("fashion-mnist", out_dir=OUT)
    m.train((X, y), epochs=1)
    path = m.save()
    m2 = GlassNet.load(path)
    assert m2.model is not None
    t = m2.explain(m2.X_test[0])
    assert "I said" in t


def test_20_line_gate_script():
    src = open(os.path.join(ROOT, "examples", "example_20_lines.py")).read()
    lines = [l for l in src.splitlines()
             if l.strip() and not l.strip().startswith("#")
             and not l.strip().startswith('"""')]
    lines = [l for l in lines if l.strip() != '"""']
    assert len(lines) <= 20, f"gate script is {len(lines)} lines, must be <= 20"


def test_evo_switch():
    """Phase 6: the experimental optimizer behind the switch."""
    X, y = tiny_data()
    m = GlassNet("fashion-mnist", out_dir=OUT)
    info = m.train((X, y), epochs=1, optimizer="evo")
    assert "post_prune_acc" in info and "final_sigma" in info
    assert 0.0 <= m.accuracy() <= 1.0
    try:
        m.train((X, y), epochs=1, optimizer="not-a-thing")
        assert False, "should have raised"
    except ValueError:
        pass


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\nLIBRARY TESTS: {len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()