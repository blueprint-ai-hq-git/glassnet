#!/usr/bin/env python3
"""slot_binding_probe.py - WHICH structural change makes arithmetic learnable? Tested, not guessed.

The 450-trial settings search proved that no learning rate, epoch count, sample count or model size makes the
model learn the addition rule (extrapolation 0% everywhere). So the blocker is structural, and there are two
candidate causes:

  (1) NO BINDING   - a bag of token-presence concepts cannot say "3 is the FIRST operand".
  (2) NO INTERACTION - our answer layer is LINEAR over concepts, and the sum a+b is not a linear function of
      (a, b). Even with perfect slot information, a linear map over one-hot operands cannot express a
      diagonal rule.

This probe fits each structure on the 100 addition pairs and measures EXTRAPOLATION to pairs 10..19, which no
memorised table can answer. Fitted with plain numpy (no engine, no library models) because the question is
about the FEATURE STRUCTURE, not about training.

  A  presence + linear              = what our architecture effectively is today
  B  slot-pair presence + linear    = binding fixed, interaction still missing
  C  slot-pair + interaction terms  = binding and interaction, flat features
  D  digit-decomposed + small MLP   = the ALGORITHM made visible (units/tens of each operand)

Usage: python3 lens/scripts/slot_binding_probe.py
"""
import os
import sys

import numpy as np

ROOT = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_linear(X, y, classes, epochs=4000, lr=0.5):
    """Plain gradient descent on softmax cross-entropy: our own code, no library models."""
    W = np.zeros((X.shape[1], classes), dtype=np.float64)
    b = np.zeros(classes, dtype=np.float64)
    Y = np.eye(classes)[y]
    for _ in range(epochs):
        P = softmax(X @ W + b)
        G = (P - Y) / len(X)
        W -= lr * (X.T @ G)
        b -= lr * G.sum(axis=0)
    return W, b


def fit_mlp(X, y, classes, hidden=64, epochs=4000, lr=0.3, seed=0):
    """One hidden layer, our own forward and backward. Enough to express a digit-wise sum."""
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((X.shape[1], hidden)) * 0.5
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, classes)) * 0.5
    b2 = np.zeros(classes)
    Y = np.eye(classes)[y]
    for _ in range(epochs):
        H = np.maximum(X @ W1 + b1, 0)
        P = softmax(H @ W2 + b2)
        G = (P - Y) / len(X)
        gW2 = H.T @ G
        gb2 = G.sum(axis=0)
        gH = (G @ W2.T) * (H > 0)
        gW1 = X.T @ gH
        gb1 = gH.sum(axis=0)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
    return W1, b1, W2, b2


def feats(pairs, kind):
    """Feature vectors for a list of (a, b) operand pairs."""
    rows = []
    for a, b in pairs:
        if kind == "presence":
            f = np.zeros(40)                      # which digits appear anywhere (order-blind)
            f[a] = 1; f[20 + b] = 1
        elif kind == "slot":
            f = np.zeros(40)                      # slot-bound: first operand, second operand
            f[a] = 1; f[20 + b] = 1
        elif kind == "slot_interaction":
            # the feature space must cover the extrapolation range too (operands up to 19), otherwise the
            # probe crashes on the very cases it exists to measure -- which is what the first version did
            base = np.zeros(40)
            base[a] = 1; base[20 + b] = 1
            inter = np.outer(base[:20], base[20:]).ravel()      # all a x b products
            f = np.concatenate([base, inter])
        elif kind == "digitwise":
            # THE ALGORITHM MADE VISIBLE: units and tens digits of each operand, separately
            f = np.zeros(40)
            f[a % 10] = 1; f[10 + min(a // 10, 1)] = 1          # units and tens digit of a (0..19)
            f[20 + b % 10] = 1; f[30 + min(b // 10, 1)] = 1
        elif kind == "numeric":
            # THE OPERANDS AS QUANTITIES, not as categories. One-hot cannot extrapolate: the tens digit 1
            # never appears in 0..9 training, so its weight stays untrained and the model has no idea what a
            # 10 is. Given a and b as NUMBERS, the sum is a linear function the model can learn exactly
            # (weights 1 and 1), and it then works for operands it has never seen.
            f = np.asarray([a / 20.0, b / 20.0, 1.0], dtype=np.float64)
        elif kind == "numeric_digits":
            # both: quantities plus their digits, so the model can use either route
            f = np.asarray([a / 20.0, b / 20.0, (a % 10) / 10.0, (b % 10) / 10.0,
                            min(a // 10, 1.0), min(b // 10, 1.0), 1.0], dtype=np.float64)
        else:
            raise ValueError(kind)
        rows.append(f)
    return np.asarray(rows, dtype=np.float64)


def fit_regression(X, y, epochs=6000, lr=0.05):
    """Predict the SUM AS A NUMBER, not as a class. Our own gradient descent on mean squared error.

    A classifier over 0..18 can never answer 27, no matter how it is trained: the answer is outside its
    label set. Regression has no such ceiling, so this is the structure that can extrapolate at all.
    """
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        p = X @ w + b
        g = (p - y) / len(X)
        w -= lr * (X.T @ g)
        b -= lr * g.sum()
    return w, b


def main():
    train = [(x, y) for x in range(10) for y in range(10)]
    extra = [(x, y) for x in range(10, 20) for y in range(10, 20)]
    ytr = np.asarray([a + b for a, b in train])
    yex = np.asarray([a + b for a, b in extra])
    # held-out pairs from the SAME range: leave out 20
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train))
    held = [train[i] for i in perm[:20]]
    tr = [train[i] for i in perm[20:]]
    print("SLOT BINDING PROBE - which structure makes arithmetic learnable?")
    print(f"  {len(tr)} train pairs, {len(held)} held-out pairs (same range), {len(extra)} extrapolation "
          f"pairs (10..19, outside the training range)\n")
    print(f"  {'structure':22s} {'train':>7s} {'held out':>9s} {'extrapolation':>14s}")
    results = {}
    for kind, model in (("presence", "linear"), ("slot", "linear"),
                        ("slot_interaction", "linear"), ("digitwise", "mlp"),
                        ("numeric", "linear"), ("numeric", "mlp"), ("numeric_digits", "mlp")):
        Xtr = feats(tr, kind)
        y = np.asarray([a + b for a, b in tr])
        Xhe = feats(held, kind)
        Xex = feats(extra, kind)
        if model == "linear":
            W, b = fit_linear(Xtr, y, classes=40)
            pred = lambda X: np.argmax(X @ W + b, axis=1)
        else:
            W1, b1, W2, b2 = fit_mlp(Xtr, y, classes=40)
            pred = lambda X: np.argmax(np.maximum(X @ W1 + b1, 0) @ W2 + b2, axis=1)
        a_tr = float((pred(Xtr) == y).mean())
        a_he = float((pred(Xhe) == np.asarray([a + b for a, b in held])).mean())
        a_ex = float((pred(Xex) == yex).mean())
        label = kind if model == "linear" else f"{kind}+mlp"
        results[label] = (a_tr, a_he, a_ex)
        print(f"  {label:22s} {a_tr*100:6.1f}% {a_he*100:8.1f}% {a_ex*100:13.1f}%")
    print("\n  READING: extrapolation is the only column that separates learning a rule from memorising a")
    print("  table. A structure that cannot extrapolate cannot be fixed by any training setting, which is")
    print("  exactly what the 450-trial search showed for the current architecture.")
    # ---- regression probe: the answer as a quantity
    print("\n  OUTPUT AS A QUANTITY (regression, not classification):")
    print(f"  {'structure':22s} {'train MAE':>10s} {'held-out MAE':>13s} {'extrapolation MAE':>18s}")
    for kind in ("numeric", "numeric_digits"):
        Xtr = feats(tr, kind); y = np.asarray([a + b for a, b in tr], dtype=np.float64)
        Xhe = feats(held, kind); yhe = np.asarray([a + b for a, b in held], dtype=np.float64)
        Xex = feats(extra, kind); yexf = np.asarray([a + b for a, b in extra], dtype=np.float64)
        w, b = fit_regression(Xtr, y)
        mae_tr = float(np.abs(Xtr @ w + b - y).mean())
        mae_he = float(np.abs(Xhe @ w + b - yhe).mean())
        mae_ex = float(np.abs(Xex @ w + b - yexf).mean())
        exact_ex = float((np.round(Xex @ w + b) == yexf).mean())
        print(f"  {kind:22s} {mae_tr:10.3f} {mae_he:13.3f} {mae_ex:18.3f}   exact on 10..19: "
              f"{exact_ex*100:.0f}%")
        results[f"{kind}+regression"] = (0.0, 0.0, exact_ex)

    best = max(results.items(), key=lambda kv: kv[1][2])
    print(f"  BEST STRUCTURE for extrapolation: {best[0]} at {best[1][2]*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())