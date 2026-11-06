"""Phase 2 architecture: Concept layer (named neurons) -> Sparse rule layer
-> Answer layer.

loss = class cross-entropy + alpha * concept BCE + lam * L1 on the
concept->rule connections. After training, weak rule connections are pruned
to the top-`keep` per rule.

Phase 3's auto-pilot trainer will drive this same model (see phase2_train).
"""
import numpy as np

from .tensor import Tensor, sigmoid, relu
from .layers import Dense
from .losses import cross_entropy, bce_logits, l1_sum
from .concepts import NAMES as DEFAULT_CONCEPT_NAMES, K as DEFAULT_K


class GlassNetPhase2:
    """concept (784->K, sigmoid) -> rules (K->R[, sigmoid]) -> answer (R->10).
    Flags: concept_hidden>0 = a learned ReLU hidden layer feeds the named
    concept readouts (learned-but-audited; names stay on the outputs);
    rule_linear=True = rules are plain weighted sums (no sigmoid saturation)
    while keeping L1 + top-3 pruning for readability."""

    def __init__(self, K=None, R=20, rng=None, concept_names=None,
                 concept_hidden=0, rule_linear=False, no_rules=False,
                 classes=10, input_dim=784):
        if K is None:
            K = DEFAULT_K
        rng = rng if rng is not None else np.random.default_rng()
        self.K, self.R = int(K), int(R)
        self.classes = int(classes)
        self.input_dim = int(input_dim)
        self.concept_hidden = int(concept_hidden or 0)
        self.rule_linear = bool(rule_linear)
        self.no_rules = bool(no_rules)
        self.concept_names = list((concept_names or DEFAULT_CONCEPT_NAMES)[:K])
        if self.concept_hidden:
            self.concept_gate = Dense(input_dim, self.concept_hidden, rng=rng,
                                      name="concept_gate")
            self.concept = Dense(self.concept_hidden, K, rng=rng, name="concept")
        else:
            self.concept = Dense(input_dim, K, rng=rng, name="concept")
        if self.no_rules:
            self.rules = None
            self.answer = Dense(K, classes, rng=rng, name="answer")
        else:
            self.rules = Dense(K, R, rng=rng, name="rules")
            self.answer = Dense(R, classes, rng=rng, name="answer")

    def forward(self, x):
        if self.concept_hidden:
            z_c = self.concept(relu(self.concept_gate(x)))
        else:
            z_c = self.concept(x)        # pre-sigmoid (bce_logits wants this)
        h = sigmoid(z_c)             # concept activations — the named meanings
        if self.no_rules:
            r = h                    # classes read the named concepts directly
            logits = self.answer(h)
        else:
            if self.rule_linear:
                r = self.rules(h)    # linear rules (still pruned to 2-3)
            else:
                r = sigmoid(self.rules(h))
            logits = self.answer(r)
        return {"logits": logits, "concepts": h, "concept_pre": z_c, "rules": r}

    def loss(self, x, y, c, alpha=0.5, lam=0.02, return_parts=False, f=None,
             include_ce=True):
        f = f or self.forward(x)
        ce = cross_entropy(f["logits"], y) if include_ce else None
        cb = bce_logits(f["concept_pre"], c,
                        w=getattr(self, "_concept_w", None))
        if self.no_rules:
            l1 = lam * 0.0
        else:
            l1 = l1_sum(self.rules.W)
        total = (cb * 0.0 if not include_ce else ce) + alpha * cb + lam * l1
        return (total, ce, cb, l1) if return_parts else total

    def parameters(self):
        if self.concept_hidden:
            heads = (self.concept_gate.parameters() + self.concept.parameters())
        else:
            heads = self.concept.parameters()
        if self.no_rules:
            return heads + self.answer.parameters()
        return heads + self.rules.parameters() + self.answer.parameters()

    def predict(self, x):
        return self.forward(Tensor(x))

    def prune_rule_connections(self, keep=3):
        """Keep only the top-`keep` absolute connections per rule — no-op
        when the rule layer is absent (no_rules variant)."""
        if self.rules is None:
            return
        W = self.rules.W.data
        for r in range(W.shape[1]):
            idx = np.argsort(np.abs(W[:, r]))[::-1][:keep]
            vals = W[idx, r].copy()
            W[:, r] = 0.0
            W[idx, r] = vals

    def connection_stats(self, threshold=1e-4):
        if self.rules is None:
            return {"avg": 0.0, "min": 0, "max": 0,
                    "dead_rules": 0, "counts": []}
        W = self.rules.W.data
        counts = (np.abs(W) > threshold).sum(axis=0)
        return {"avg": float(counts.mean()), "min": int(counts.min()),
                "max": int(counts.max()),
                "dead_rules": int((counts == 0).sum()),
                "counts": [int(c) for c in counts]}

    def count_parameters(self):
        return sum(p.data.size for p in self.parameters())

    def save(self, path, **extra):
        layers = [self.concept]
        if self.concept_hidden:
            layers.insert(0, self.concept_gate)
        if not self.no_rules:
            layers.append(self.rules)
        layers.append(self.answer)
        out = {p.name: p.data for layer in layers
               for p in layer.parameters()}
        out["concept_hidden"] = np.int32(self.concept_hidden)
        out["rule_linear"] = np.int32(1 if self.rule_linear else 0)
        out["no_rules"] = np.int32(1 if self.no_rules else 0)
        out["input_dim"] = np.int32(self.input_dim)
        out.update(extra)
        np.savez(path, **out)

    @staticmethod
    def load(path, K=None, R=None):
        z = np.load(path)
        if K is None:
            K = z["concept.W"].shape[1]
        if R is None:
            R = (z["rules.W"].shape[1]
                 if "rules.W" in z.files else 0)
        ch = int(z["concept_hidden"]) if "concept_hidden" in z.files else 0
        rl = bool(int(z["rule_linear"])) if "rule_linear" in z.files else False
        nr = bool(int(z["no_rules"])) if "no_rules" in z.files else False
        n_c = int(z["answer.W"].shape[1])
        din = int(z["input_dim"]) if "input_dim" in z.files else 784
        net = GlassNetPhase2(K=K, R=R, concept_hidden=ch, rule_linear=rl,
                             no_rules=nr, classes=n_c, input_dim=din)
        layers = [net.concept]
        if ch:
            layers.insert(0, net.concept_gate)
        if net.rules is not None:
            layers.append(net.rules)
        layers.append(net.answer)
        for layer in layers:
            for p in layer.parameters():
                p.data[:] = z[p.name].astype(np.float32)
        return net


class NoRulesModel:
    """Ablation A3: concepts -> classes DIRECTLY (the rule layer removed)."""

    def __init__(self, K=None, rng=None):
        if K is None:
            K = DEFAULT_K
        rng = rng if rng is not None else np.random.default_rng()
        self.concept = Dense(784, K, rng=rng, name="concept")
        self.direct = Dense(K, 10, rng=rng, name="direct")

    def forward(self, x):
        z_c = self.concept(x)
        h = sigmoid(z_c)
        return {"logits": self.direct(h), "concepts": h, "concept_pre": z_c}

    def loss(self, x, y, c, alpha=0.5, return_parts=False, f=None):
        f = f or self.forward(x)
        ce = cross_entropy(f["logits"], y)
        cb = bce_logits(f["concept_pre"], c,
                        w=getattr(self, "_concept_w", None))
        total = ce + alpha * cb
        return (total, ce, cb) if return_parts else total

    def parameters(self):
        return self.concept.parameters() + self.direct.parameters()

    def predict(self, x):
        return self.forward(Tensor(x))

    def count_parameters(self):
        return sum(p.data.size for p in self.parameters())