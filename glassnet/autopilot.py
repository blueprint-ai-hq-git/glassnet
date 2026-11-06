"""glassnet/autopilot.py — Phase 3: the auto-pilot trainer.

The user types NOTHING. A watcher checks every few batches and acts on the
four signs from the brief:
  - loss stops falling  -> lower step size
  - loss explodes       -> restart smaller
  - train high / test low -> stop early (keep the best-test weights)
  - neurons that never fire -> flag during the run, prune silent rules at end

Every action the auto-pilot takes is logged in plain English. Every number
living inside the defaults is listed in HIDDEN_DEFAULTS.md (nothing secret;
phase3_autopilot verifies the file covers every key).
"""
import numpy as np

from .tensor import Tensor
from .optim import SGD
from .eval import accuracy
from .phase2_model import GlassNetPhase2

AUTO_PILOT_DEFAULTS = {
    # ---- architecture (inherited from Phase 2 manual tuning, unchanged) ----
    "num_concepts": 14,
    "num_rules": 30,
    "concept_loss_alpha": 1.0,      # weight of the concept-name supervision
    "l1_lambda": 0.005,             # strength of the rule-sparsity penalty
    "prune_keep_connections": 3,    # connections kept per rule after pruning

    # ---- data handling ----
    "batch_size": 128,
    "eval_train_sample": 5000,      # train-accuracy sample per epoch eval

    # ---- work budget ----
    "max_epochs": 12,               # hard cap on training distance
    "min_learning_rate": 1e-4,      # below this the pilot stops (too small to matter)
    "max_restarts": 3,              # explosion-restart budget

    # ---- the watcher's eyes and hands ----
    "input_dim": 784,               # input width (pixels 784 / text BOW 2048)
    "start_learning_rate": 0.1,     # same start as the manual run
    "concept_rescue_fidelity_floor": 0.55,  # name fires on its label below this -> boost
    "concept_rescue_boost": 1.0,    # OFF by default: R2-013 showed boosting drifted
                                    # names costs class accuracy without saving the
                                    # names (-7 pts). Switchable for future work.
    "check_every_batches": 100,     # how often the pilot looks at the loss
    "eval_every_batches": 469,      # ~once per epoch (60000 / 128)
    "loss_ema_decay": 0.98,         # smooth average used to judge spikes
    "flat_improvement_min": 0.005,  # relative loss drop needed to count as "improving"
    "flat_patience_windows": 5,     # consecutive flat windows -> lower step size
    "lr_cut_factor_flat": 0.5,      # step-size multiplier when loss is flat
    "spike_factor": 3.0,            # loss above 3x its smooth average -> spike
    "lr_cut_factor_spike": 0.7,     # step-size multiplier after a spike
    "explode_restart_lr_factor": 0.5,  # step-size multiplier for a full restart
    "explode_jump_factor": 20.0,    # window loss > 20x the previous window = explosion
    "test_gap_threshold": 0.10,     # train_acc - test_acc beyond this = overfit
    "early_stop_no_improve_evals": 2,  # evals without a new best -> stop early
    "silent_activation_min": 0.03,  # mean firing below this = "never fires"
}


def _snapshot(net):
    layers = [net.concept]
    if net.concept_hidden:
        layers.insert(0, net.concept_gate)
    if net.rules is not None:
        layers.append(net.rules)
    layers.append(net.answer)
    return {p.name: p.data.copy()
            for layer in layers
            for p in layer.parameters()}


def _restore(net, snap):
    layers = [net.concept]
    if net.concept_hidden:
        layers.insert(0, net.concept_gate)
    if net.rules is not None:
        layers.append(net.rules)
    layers.append(net.answer)
    for layer in layers:
        for p in layer.parameters():
            p.data[:] = snap[p.name]


def _act(actions, step, msg):
    line = f"[auto-pilot @step {step:6d}] {msg}"
    actions.append((int(step), msg))
    print(line, flush=True)


def _auroc(scores, pos):
    """Area under the ROC curve (Mann-Whitney, no tie correction)."""
    s = np.asarray(scores, dtype=np.float64)
    pos = np.asarray(pos, dtype=bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(len(s))
    return (ranks[pos].mean() - (n_pos - 1) / 2.0) / n_neg


def train_autopilot(Xtr, ytr, Ctr, Xte, yte, Cte, seed=0,
                    log_actions=True, defaults=None, concept_warn=True):
    """Train the Phase 2 architecture with zero user settings.
    `defaults` (optional, for controls/tests) OVERRIDES individual keys of
    AUTO_PILOT_DEFAULTS — the rest stay."""
    d = dict(AUTO_PILOT_DEFAULTS)
    if defaults:
        d.update(defaults)
    d["seed"] = int(seed)
    n = Xtr.shape[0]
    bpe = n // d["batch_size"]
    total_steps = d["max_epochs"] * bpe
    actions = []

    rng_model = np.random.default_rng(int(seed))
    rng_data = np.random.default_rng(int(seed) * 777 + 1)
    net = GlassNetPhase2(K=d["num_concepts"], R=d["num_rules"], rng=rng_model,
                             concept_hidden=d.get("concept_hidden", 0),
                             rule_linear=d.get("rule_linear", False),
                             no_rules=d.get("no_rules", False),
                             classes=d.get("num_classes", 10),
                             input_dim=d.get("input_dim", 784))
    lr = float(d["start_learning_rate"])
    opt = SGD(net.parameters(), lr=lr)
    net._concept_w = None           # uniform per-concept BCE weights at start

    _act(actions, 0, f"MISSION START — seed {seed}, step size {lr}, "
                     f"budget {d['max_epochs']} epochs (max {total_steps} batches), "
                     f"architecture untouched from Phase 2")

    order = rng_data.permutation(n)
    ema, win_acc = None, []
    best_win, flat_streak = float("inf"), 0
    restarts = 0
    best_test, best_snap = 0.0, None
    gap_streak = 0
    prev_epoch = -1
    stopped = False
    eval_log = []
    quiet_flags = 0
    silent_pruned = []
    max_steps_needed = total_steps

    for step in range(total_steps):
        if stopped:
            break
        epoch = step // bpe
        if epoch != prev_epoch:
            prev_epoch = epoch
            order = rng_data.permutation(n)
            _act(actions, step, f"epoch {epoch + 1}/{d['max_epochs']} begins "
                                f"(data reshuffled)")
        i = (step % bpe) * d["batch_size"]
        idx = order[i:i + d["batch_size"]]
        xb_t = Tensor(Xtr[idx], name="batch")
        yb, cb_t = ytr[idx], Ctr[idx]

        opt.zero_grad()
        f = net.forward(xb_t)
        total = net.loss(xb_t, yb, cb_t, alpha=d["concept_loss_alpha"],
                         lam=d["l1_lambda"], f=f)
        total.backward()
        opt.step()
        bl = float(total.data)
        ema = bl if ema is None else d["loss_ema_decay"] * ema + (1 - d["loss_ema_decay"]) * bl
        win_acc.append(bl)

        # ---------------- watcher: loss checks ----------------
        if (step + 1) % d["check_every_batches"] == 0:
            win = float(np.mean(win_acc))
            win_acc = []

            # EXPLOSION (non-finite OR order-of-magnitude jump) -> restart smaller
            if not np.isfinite(win) or (best_win < float("inf") and
                                         win > best_win * d["explode_jump_factor"]):
                if restarts < d["max_restarts"]:
                    restarts += 1
                    lr = lr * d["explode_restart_lr_factor"]
                    _act(actions, step, f"LOSS EXPLODED (non-finite) — restarting "
                                         f"from scratch with a smaller step size "
                                         f"{lr:.4f} (restart {restarts}/{d['max_restarts']})")
                    rng_model = np.random.default_rng(int(seed) * 1000 + restarts)
                    net = GlassNetPhase2(K=d["num_concepts"], R=d["num_rules"],
                                                             rng=rng_model,
                                                             concept_hidden=d.get("concept_hidden", 0),
                                                             rule_linear=d.get("rule_linear", False),
                                                             no_rules=d.get("no_rules", False),
                                                             classes=d.get("num_classes", 10),
                                                             input_dim=d.get("input_dim", 784))
                    net._concept_w = None
                    opt = SGD(net.parameters(), lr=lr)
                    rng_data = np.random.default_rng(int(seed) * 777 + 1 + restarts)
                    order = rng_data.permutation(n)
                    ema, best_win, flat_streak, win_acc = None, float("inf"), 0, []
                    prev_epoch = -1
                else:
                    _act(actions, step, "LOSS EXPLODED but restart budget spent — "
                                        "giving up")
                    stopped = True
                continue

            # FLAT -> lower step size
            if win < best_win * (1.0 - d["flat_improvement_min"]):
                best_win = win
                flat_streak = 0
            else:
                flat_streak += 1
                if flat_streak >= d["flat_patience_windows"] and lr > d["min_learning_rate"]:
                    lr = lr * d["lr_cut_factor_flat"]
                    opt = SGD(net.parameters(), lr=lr)
                    _act(actions, step, f"loss stopped falling ({flat_streak} "
                                         f"windows flat, best window loss "
                                         f"{best_win:.4f}) — lowered step size to {lr:.4f}")
                    flat_streak = 0

            # SPIKE -> lower step size
            if ema and win > ema * d["spike_factor"] and lr > d["min_learning_rate"]:
                lr = lr * d["lr_cut_factor_spike"]
                opt = SGD(net.parameters(), lr=lr)
                _act(actions, step, f"loss spiked ({win:.4f} vs smooth {ema:.4f}) "
                                     f"— lowered step size to {lr:.4f}")

        # ---------------- watcher: once-per-epoch full eval ----------------
        if (step + 1) % d["eval_every_batches"] == 0:
            rng_eval = np.random.default_rng(step + 100_000)
            pick = rng_eval.choice(n, min(d["eval_train_sample"], n), replace=False)
            train_acc = accuracy(net, Xtr[pick], ytr[pick])
            test_acc = accuracy(net, Xte, yte)

            acts_c = net.predict(Xte)["concepts"].data.mean(axis=0)
            sr = []
            if not getattr(net, "no_rules", False):
                acts_r = net.predict(Xte)["rules"].data.mean(axis=0)
                sr = [int(r) for r in range(len(acts_r))
                      if acts_r[r] < d["silent_activation_min"]]
            sc = [net.concept_names[k] for k in range(len(acts_c))
                  if acts_c[k] < d["silent_activation_min"]]
            if sr:
                quiet_flags += 1
                _act(actions, step, f"silent rule neurons (mean firing "
                                     f"< {d['silent_activation_min']}): {sr}")
            if sc:
                quiet_flags += 1
                names_list = net.concept_names
                _act(actions, step, "silent CONCEPT neurons: " +
                     str([(names_list[k] if k < len(names_list)
                           else f"concept{k}") for k in sc]))

            # NAME-RESCUE: a concept neuron not firing on its own label gets
            # its supervision boosted (per-concept BCE weight) — the remedy
            # for run-to-run name drift seen in R2-000/R2-012.
            if Cte is not None and d.get("concept_rescue_boost", 1.0) > 1.0:
                z_c = net.predict(Xte)["concept_pre"].data
                K = z_c.shape[1]
                w = np.ones(K, dtype=np.float32)
                for k in range(K):
                    thr = float(np.median(Cte[:, k]))   # balanced split
                    if _auroc(z_c[:, k], Cte[:, k] > thr) < d[
                            "concept_rescue_fidelity_floor"]:
                        w[k] = d["concept_rescue_boost"]
                cur = getattr(net, "_concept_w", None)
                if cur is None or not np.array_equal(cur, w):
                    if np.any(w > 1.0):
                        rescued = [net.concept_names[k] for k in range(K)
                                   if w[k] > 1.0]
                        _act(actions, step, "concept(s) NOT firing on their "
                             "own label: " + str(rescued) + " — supervision "
                             f"boosted x{d['concept_rescue_boost']} "
                             "(name-rescue reflex)")
                    net._concept_w = w

            eval_log.append({"step": step, "epoch": epoch + 1, "lr": lr,
                             "train_acc": train_acc, "test_acc": test_acc,
                             "silent_rules": sr})
            if test_acc > best_test:
                best_test = test_acc
                best_snap = _snapshot(net)
                _act(actions, step, f"new best test accuracy {test_acc*100:.2f}% "
                                     f"— snapshot saved")

            # TRAIN >> TEST -> stop early (the spec's overfit sign)
            gap = train_acc - test_acc
            if gap > d["test_gap_threshold"]:
                gap_streak += 1
            else:
                gap_streak = 0
            if gap_streak >= d["early_stop_no_improve_evals"]:
                _act(actions, step, f"train {train_acc*100:.1f}% vs test "
                                     f"{test_acc*100:.1f}% ({gap*100:.1f} pts "
                                     f"overfit gap) for {gap_streak} consecutive "
                                     f"evals — STOPPING EARLY, keeping best-test weights")
                stopped = True

    # ---------------- finish ----------------
    if best_snap is not None:
        _restore(net, best_snap)
        _act(actions, max_steps_needed, f"finished — restored best-test snapshot "
                                        f"({best_test*100:.2f}%)")

    pre_prune_acc = accuracy(net, Xte, yte)
    if not getattr(net, "no_rules", False):
        net.prune_rule_connections(keep=d["prune_keep_connections"])
        acts_r = net.predict(Xte)["rules"].data.mean(axis=0)
        silent = np.where(acts_r < d["silent_activation_min"])[0]
        if len(silent):
            net.answer.W.data[silent, :] = 0.0
            silent_pruned = [int(s) for s in silent]
            _act(actions, max_steps_needed, f"pruned {len(silent)} silent rules "
                                            f"(never fire): {silent_pruned}")
    post_prune_acc = accuracy(net, Xte, yte)
    stats = net.connection_stats()
    if not getattr(net, "no_rules", False):
        _act(actions, max_steps_needed, f"pruned rule connections to top-"
            f"{d['prune_keep_connections']} "
            f"(avg {stats['avg']:.2f} conns/rule, "
            f"{stats['dead_rules']} dead rules)")
    else:
        _act(actions, max_steps_needed, "no rule layer (concepts -> classes "
                                        "directly); no pruning step")

    info = {"seed": int(seed), "pre_prune_acc": pre_prune_acc,
            "post_prune_acc": post_prune_acc, "best_test": best_test,
            "restarts": restarts, "quiet_flags": quiet_flags,
            "silent_rules_pruned": silent_pruned,
            "actions": actions, "eval_log": eval_log,
            "final_lr": lr, "conns_avg": stats["avg"],
            "dead_rules": stats["dead_rules"]}
    return net, info