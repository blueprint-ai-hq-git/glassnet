"""glassnet/api.py — Phase 5: the library front door.

    model = GlassNet("fashion-mnist")   # built-in task, or your own
    model.train()                       # auto-pilot, zero settings
    print(model.explain(5))             # why did it say that? (text)
    model.explain(5, save="why.png")    # ... and a picture
    model.accuracy()                    # held-out accuracy
    model.train_judge(); model.judge(X) # the data judge (Phase 4)
"""
import json
import os

import numpy as np

from .tensor import sigmoid  # noqa: F401  (re-exported for power users)
from .data import load as load_fmnist, LABELS
from .concepts import load as load_concepts, compute as compute_concepts
from .autopilot import train_autopilot, AUTO_PILOT_DEFAULTS
from .phase2_model import GlassNetPhase2
from .eval import accuracy as acc_fn
from .report import receipt, draw_explain_figure
from .judge import build_pile, Cleaner, JudgeModel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(HERE, "data")
DEFAULT_OUT = os.path.join(HERE, "runs", "library")


class GlassNet:
    """The whole system behind three words: GlassNet / train / explain."""

    def __init__(self, task="fashion-mnist", seed=0, out_dir=None):
        if task != "fashion-mnist":
            raise ValueError(
                f"task {task!r} unknown. Built-in tasks: 'fashion-mnist'. "
                f"Your own data works too: model.train((X, y)).")
        self.task = task
        self.seed = int(seed)
        self.out_dir = out_dir or DEFAULT_OUT
        os.makedirs(self.out_dir, exist_ok=True)
        self.model = None
        self.info = None
        self.X_test = None
        self.y_test = None
        self.judge_model = None
        self._concept_lo = None
        self._concept_hi = None
        self.class_names = LABELS

    # ------------------------------------------------------------- training
    def train(self, data=None, epochs=None, optimizer="sgd", silent=False):
        """Train the readable architecture. Zero required settings.

        data = None        -> built-in Fashion-MNIST (downloads once, cached).
        data = (X, y)      -> your own images: X {(N,28,28) or (N,784)}
                              floats in [0,1], y = integer class labels.
        epochs = int       -> optional work-budget override (tests/tight runs).
        optimizer = "sgd"  -> the normal auto-pilot trainer (default; Phase 3).
        optimizer = "evo"  -> Phase 6 experimental (1+1)-Evolution Strategy,
                              NO gradients; the normal one is never deleted.
        Returns the trainer's info dict (accuracies + every action taken).
        """
        if data is None:
            Xtr, ytr, Xte, yte = load_fmnist(DATA_DIR)
            Ctr, Cte, lo, hi = load_concepts(DATA_DIR)
            self._concept_lo, self._concept_hi = lo, hi
            self.X_test, self.y_test = Xte, yte
        else:
            X, y = data
            X = np.asarray(X, np.float32)
            if X.ndim == 3:
                X = X.reshape(len(X), -1)
            if X.ndim != 2 or X.shape[1] != 784:
                raise ValueError("images must be 28x28 greyscale (784 floats)")
            y = np.asarray(y)
            if len(y) != len(X):
                raise ValueError("X and y must have the same length")
            n = len(X)
            order = np.random.default_rng(42 + self.seed).permutation(n)
            tr, te = order[: int(0.8 * n)], order[int(0.8 * n):]
            Xtr, ytr = X[tr], y[tr]
            Xte, yte = X[te], y[te]
            Ctr, lo, hi = compute_concepts(Xtr)
            Cte, _, _ = compute_concepts(Xte, lo, hi)
            self._concept_lo, self._concept_hi = lo, hi
            self.X_test, self.y_test = Xte, yte

        defaults = dict(AUTO_PILOT_DEFAULTS)
        if epochs is not None:
            defaults["max_epochs"] = int(epochs)

        if optimizer == "evo":
            from .evo import train_evo, EVO_DEFAULTS
            ed = dict(EVO_DEFAULTS)
            if epochs is not None:
                ed["budget_batches"] = int(epochs) * (len(Xtr) // ed["batch_size"])
            net, info = train_evo(Xtr, ytr, Ctr, Xte, yte, Cte,
                                  seed=self.seed,
                                  log_actions=not silent, defaults=ed)
        elif optimizer == "sgd":
            net, info = train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte,
                                        seed=self.seed,
                                        log_actions=not silent,
                                        defaults=defaults)
        else:
            raise ValueError(f"unknown optimizer {optimizer!r} — "
                             f"use 'sgd' (default) or 'evo' (experimental)")
        self.model = net
        self.info = info
        self.save()
        return info

    # ------------------------------------------------------------- explain
    def explain(self, example, save=None):
        """Why did the model answer? Returns the plain-text receipt.
        example = int index (into the held-out test set) or an image array.
        save = 'path.png' also writes the visual report (concepts, rules,
        contributions, class probabilities)."""
        if self.model is None:
            raise RuntimeError("train() first — nothing to explain yet")
        if isinstance(example, (int, np.integer)):
            i = int(example)
            if self.X_test is None or not (0 <= i < len(self.X_test)):
                raise ValueError(f"example index {i} out of range")
            x = self.X_test[i]
            head = (f"test example #{i} "
                    f"(true label {LABELS[int(self.y_test[i])]})")
        else:
            x = np.asarray(example, np.float32)
            if x.ndim == 3:
                x = x.reshape(-1)
            if x.size != 784:
                raise ValueError("image must be 28x28 greyscale (784 floats)")
            head = "provided image"
        text = receipt(self.model, x, head=head)
        if save:
            draw_explain_figure(self.model, x, head, str(save))
        return text

    def accuracy(self):
        """Held-out test accuracy of the trained model."""
        if self.model is None:
            raise RuntimeError("train() first")
        return acc_fn(self.model, self.X_test, self.y_test)

    # -------------------------------------------------------------- judge
    def train_judge(self, epochs=None, silent=False):
        """Train the Phase 4 data judge on the hand-labeled good/bad pile.
        Returns the cleaner report (what junk was removed)."""
        Xtr, ytr, _, _ = load_fmnist(DATA_DIR)
        rng = np.random.default_rng(42)
        pile = build_pile(Xtr, ytr, rng)
        _, Xc, yc, _, rep = Cleaner().clean(pile["X"], pile["y"], pile["types"])
        good_i = np.where(yc == 0)[0]
        bad_i = np.where(yc == 1)[0]
        r2 = np.random.default_rng(7)
        gtr = r2.choice(good_i, int(0.6 * len(good_i)), replace=False)
        btr = r2.choice(bad_i, int(0.6 * len(bad_i)), replace=False)
        tri = np.concatenate([gtr, btr])
        self.judge_model = JudgeModel(np.random.default_rng(self.seed))
        self.judge_model.train(Xc[tri], yc[tri], epochs=epochs)
        return rep

    def judge(self, X):
        """Score rows with the data judge: >= 0.5 suspicious, < 0.5 fine.
        Returns one float per row."""
        if self.judge_model is None:
            raise RuntimeError("train_judge() first")
        X = np.asarray(X, np.float32)
        if X.ndim == 3:
            X = X.reshape(len(X), -1)
        return self.judge_model.prob_bad(X)

    # ---------------------------------------------------------- save/load
    def save(self, path=None):
        if self.model is None:
            raise RuntimeError("train() first")
        path = path or os.path.join(self.out_dir, "model.npz")
        self.model.save(path)
        json.dump({"task": self.task, "seed": self.seed,
                   "concept_lo": [float(v) for v in self._concept_lo],
                   "concept_hi": [float(v) for v in self._concept_hi]},
                  open(os.path.join(self.out_dir, "model-meta.json"), "w"),
                  indent=1)
        return path

    @classmethod
    def load(cls, path, out_dir=None):
        meta_path = os.path.join(os.path.dirname(path), "model-meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        m = cls(meta["task"], seed=meta["seed"], out_dir=out_dir)
        m.model = GlassNetPhase2.load(path)
        m._concept_lo = np.asarray(meta["concept_lo"], np.float32)
        m._concept_hi = np.asarray(meta["concept_hi"], np.float32)
        if meta["task"] == "fashion-mnist":
            _, _, Xte, yte = load_fmnist(DATA_DIR)
            m.X_test, m.y_test = Xte, yte
        return m