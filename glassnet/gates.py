"""glassnet.gates — GATE 1: every hand-written gradient vs finite differences.

Shared by the test script AND by every training runner: no run may train
unless this gate passes ("numbers produced under a failed check are void").
"""
import sys

import numpy as np

from .tensor import set_default_dtype, Tensor, add, mul, matmul, relu, sigmoid
from .layers import Dense, Sequential
from .losses import mse, cross_entropy, bce_logits, l1_sum
from .gradcheck import check


def _rng(i):
    return np.random.default_rng(i * 101 + 13)


def _gather_tests():
    tests = []

    def add_test(name, build_loss, params):
        tests.append(check(build_loss, params, name=name))

    # --- single ops, float64 so the comparison is exact -------------------
    def t_add():
        a = Tensor(_rng(0).standard_normal((2, 3)), requires_grad=True, name="a")
        b = Tensor(_rng(1).standard_normal((3,)), requires_grad=True, name="b")  # broadcast
        return lambda: mse(add(a, b), np.zeros((2, 3))), [a, b]
    name, params = "add (with broadcast)", None
    bl, ps = t_add(); add_test(name, bl, ps)

    def t_mul():
        a = Tensor(_rng(2).standard_normal((2, 3)), requires_grad=True, name="a")
        b = Tensor(_rng(3).standard_normal((2, 3)), requires_grad=True, name="b")
        return lambda: mse(mul(a, b), np.zeros((2, 3))), [a, b]
    bl, ps = t_mul(); add_test("mul", bl, ps)

    def t_matmul():
        a = Tensor(_rng(4).standard_normal((3, 4)), requires_grad=True, name="a")
        b = Tensor(_rng(5).standard_normal((4, 2)), requires_grad=True, name="b")
        return lambda: mse(matmul(a, b), np.zeros((3, 2))), [a, b]
    bl, ps = t_matmul(); add_test("matmul", bl, ps)

    def t_relu():
        x = Tensor(_rng(6).standard_normal((4, 3)), requires_grad=True, name="x")
        return lambda: mse(relu(x), np.zeros((4, 3))), [x]
    bl, ps = t_relu(); add_test("relu", bl, ps)

    def t_sigmoid():
        x = Tensor(_rng(7).standard_normal((4, 3)), requires_grad=True, name="x")
        return lambda: mse(sigmoid(x), np.zeros((4, 3))), [x]
    bl, ps = t_sigmoid(); add_test("sigmoid", bl, ps)

    # --- whole nets, the form every trainer actually uses ------------------
    def t_net_ce():
        rng = _rng(8)
        x = Tensor(rng.standard_normal((5, 3)), name="x")
        y = np.array([0, 1, 2, 1, 0])
        net = Sequential(Dense(3, 4, rng=rng, name="l1", dtype=np.float64), relu,
                         Dense(4, 3, rng=rng, name="l2", dtype=np.float64))
        return lambda: cross_entropy(net(x), y), net.parameters()
    bl, ps = t_net_ce(); add_test("Dense->relu->Dense->cross_entropy", bl, ps)

    def t_xor_mse():
        rng = _rng(9)
        x = Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
                   name="xor_x")
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)
        net = Sequential(Dense(2, 4, rng=rng, name="h", dtype=np.float64), sigmoid,
                         Dense(4, 1, rng=rng, name="o", dtype=np.float64), sigmoid)
        return lambda: mse(net(x), y), net.parameters()
    bl, ps = t_xor_mse(); add_test("sigmoid-net on XOR (mse)", bl, ps)

    def t_bce():
        rng = _rng(10)
        x = Tensor(rng.standard_normal((6, 4)), name="x")
        t = rng.random((6, 3))
        net = Dense(4, 3, rng=rng, name="l", dtype=np.float64)
        return lambda: bce_logits(net(x), t), net.parameters()
    bl, ps = t_bce(); add_test("Dense->bce_logits (concept supervision)", bl, ps)

    def t_l1():
        w = Tensor(_rng(11).standard_normal((3, 5)), requires_grad=True, name="w")
        return lambda: l1_sum(w), [w]
    bl, ps = t_l1(); add_test("l1_sum (sparsity penalty)", bl, ps)

    return tests


def run_gate1(verbose=True):
    """Runs all gradient checks in float64, restores float32, returns bool."""
    set_default_dtype(np.float64)
    try:
        tests = _gather_tests()
    finally:
        set_default_dtype(np.float32)
    ok = all(t["ok"] for t in tests)
    if verbose:
        print(f"{'gradient check':44s} {'max|abs|err|':>12s} {'max rel err':>12s}  verdict")
        for t in tests:
            print(f"{t['name']:44s} {t['worst_abs']:12.3e} {t['worst_rel']:12.3e}  "
                  f"{'PASS' if t['ok'] else 'FAIL'}")
        print("GATE 1:", "PASS ✔" if ok else "FAIL ✘")
    return ok


def gate1():
    """Call at the top of every training runner. Exits the process on failure."""
    if not run_gate1():
        print("GATE 1 FAILED — refusing to train with broken gradients.")
        sys.exit(1)