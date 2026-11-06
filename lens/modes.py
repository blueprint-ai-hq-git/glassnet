"""lens/modes.py - EASY MODE and HARD MODE, and the best-setting table behind them.

The user's requirement (2026-10-18): "easy mode ... just few lines of code and the hard mode you have to just
write every single line how you want it, and how many sequential tokens you need in and out in RL and for the
SFT and everything in the hard mode exactly setting up."

EASY MODE - a few lines, every setting taken from the best-known recipe for that size:

    import lens.modes as modes
    net = modes.easy("data/chat/prepared_mix_ctrl_honest", size="auto")   # the whole thing
    print(modes.why_easy())                                              # where those numbers came from

HARD MODE - nothing is chosen for you. Every line is written out, including the sequence lengths for the
input and the output of RL and SFT, which the user asked for by name:

    net = modes.hard(
        data="data/chat/prepared_mix_ctrl_honest",
        method="rl", d=512, window_in=64, window_out=8,
        epochs=40, batch=128, opt="relgrad", lr=0.01,
        lam_concept=0.3, lam_act=0.5, max_gap=0.35,
        rl_seq_in=128, rl_seq_out=16, rl_samples=8, rl_reward="math_exact",
    )

Hard mode REFUSES to run if any required setting is missing. That is deliberate: the whole point of hard mode
is that you wrote it, so a silently-filled default would be a lie about what was run.
"""
import json
import os
import subprocess

ROOT = "/run/media/maintainer/HARD DISK 4/GlassNet"
RECIPES_PATH = os.path.join(ROOT, "lens", "recipes.json")

# every setting hard mode insists on, grouped so the error message can say what is missing
REQUIRED = ("method", "d", "window_in", "window_out", "epochs", "batch", "opt", "lr",
            "lam_concept", "lam_act", "max_gap")
METHOD_SPECIFIC = {"rl": ("rl_seq_in", "rl_seq_out", "rl_samples", "rl_reward"),
                   "sft": ("sft_seq_in", "sft_seq_out"),
                   "distill": ("teacher", "distill_temp")}


def recipes():
    with open(RECIPES_PATH) as f:
        return json.load(f)


def recipe_for(size="auto"):
    """The best-known setting for a size. Never invents one: if a size was not measured, it says so and uses
    the nearest measured rung, so the caller knows the number came from somewhere else."""
    r = recipes()
    table = r["recipes"]
    key = str(size)
    if key in table:
        # a recipe that EXISTS is not the same as a recipe that was MEASURED. Several rungs are carried
        # forward from a smaller size with a predicted scale adjustment, and saying "measured" about those
        # would misreport the source of the numbers, which is this project's most common bug class.
        src = str(table[key].get("measured_from", ""))
        if src.lower().startswith("pending") or "PENDING MEASUREMENT" in str(table[key].get("note", "")):
            return dict(table[key]), (f"d={key} settings are CARRIED from a smaller rung with a predicted "
                                      f"scale adjustment -- NOT yet measured ({src[:70]})")
        return dict(table[key]), f"measured recipe for {key}"
    if key.isdigit():
        want = int(key)
        known = sorted(int(k) for k in table if k.isdigit())
        near = min(known, key=lambda k: abs(k - want))
        return dict(table[str(near)]), (f"NOT MEASURED at d={want}; using the nearest measured rung "
                                        f"d={near} -- the run is labelled with that")
    if key in ("auto", "None"):
        d = r["default"]
        return dict(table[d]), (f"auto -> d={d}, the largest rung whose settings were actually "
                               f"MEASURED (larger rungs are queued, not measured)")
    raise ValueError(f"unknown size {size!r}")


def _print_plan(mode, cfg, why):
    print(f"lens.modes.{mode}: {why}")
    print(f"  method {cfg['method']:<8} d {cfg['d']:<5} window_in {cfg['window_in']:<4} "
          f"window_out {cfg['window_out']:<4} epochs {cfg['epochs']:<4} batch {cfg['batch']:<4} "
          f"opt {cfg['opt']:<8} lr {cfg['lr']}")
    print(f"  lam_concept {cfg['lam_concept']}  lam_act {cfg['lam_act']}  max_gap {cfg['max_gap']} "
          f"(0 disables the memorisation net)")
    for k in METHOD_SPECIFIC.get(cfg["method"], ()):
        print(f"  {k} {cfg.get(k)}")
    if cfg.get("note"):
        print(f"  note: {cfg['note']}")


def _run(cfg, data, out=None):
    """Dispatch to the REAL implementation of the chosen method.

    The first version of this function passed `--method rl --rl-seq-in ...` to train_chat.py. None of those
    flags exist there, so hard mode with method="rl" would have died on unknown arguments. Verified with:
        grep -nE 'add_argument\("--(method|rl-seq' lens/scripts/train_chat.py   ->   nothing
    So each method now calls code that exists, and an unimplemented method raises instead of pretending.
    """
    out = out or cfg.get("out") or f"{cfg['method']}_d{cfg['d']}"
    method = cfg["method"]
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["CHAT_DATA_DIR"] = os.path.abspath(data)

    if method == "sft":
        # REAL: the supervised trainer, the same one behind every chat run in this project.
        cmd = [f"{ROOT}/.venv-gpu/bin/python3", "-u", f"{ROOT}/lens/scripts/train_chat.py",
               "--d", str(cfg["d"]), "--window", str(cfg["window_in"]),
               "--epochs", str(cfg["epochs"]), "--batch", str(cfg["batch"]),
               "--lr", str(cfg["lr"]), "--out", out, "--assistant-only",
               "--lam-concept", str(cfg["lam_concept"]), "--lam-act", str(cfg["lam_act"])]
        if cfg.get("max_gap"):
            cmd += ["--max-gap", str(cfg["max_gap"])]
    elif method == "rl":
        # REAL: lens/methods.py, rejection sampling with an exact verifier. Its own settings are used, and
        # rl_seq_in / rl_seq_out are passed through as the sampling window and how many answer tokens are
        # produced, which is what the user asked to be able to set explicitly.
        cmd = [f"{ROOT}/.venv-gpu/bin/python3", "-u", f"{ROOT}/lens/methods.py",
               "--d", str(cfg["d"]), "--samples", str(cfg["rl_samples"]),
               "--rounds", str(cfg.get("rl_rounds", 3)), "--pretrain", str(cfg.get("rl_pretrain", 300)),
               "--lr", str(cfg["lr"]), "--seq-in", str(cfg["rl_seq_in"]), "--seq-out", str(cfg["rl_seq_out"]),
               "--reward", str(cfg["rl_reward"]), "--seed", str(cfg.get("seed", 0))]
    elif method == "distill":
        # REAL: run_distill.py, with a gradient-checked KL backward and a teacher precondition.
        cmd = [f"{ROOT}/.venv-gpu/bin/python3", "-u", f"{ROOT}/lens/scripts/run_distill.py",
               "--teacher", str(cfg["teacher"]), "--d", str(cfg["d"]), "--temp", str(cfg["distill_temp"])]
    else:
        raise NotImplementedError(
            f"method {method!r} has no implementation in this repo. Implemented: sft (train_chat.py), "
            f"rl (lens/methods.py), distill (lens/scripts/run_distill.py). Refusing to run a plan that "
            f"cannot execute.")

    print(f"  running: {' '.join(cmd[:5])} ...")
    r = subprocess.run(cmd, cwd=ROOT, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"{method} exited {r.returncode}; the real error is in the output above")
    if method != "sft":
        return None          # rl and distill report their own measured numbers; no checkpoint is implied
    runs = os.path.join(ROOT, "lens/runs/chat")
    ships = sorted([f for f in os.listdir(runs) if f.startswith(out)],
                   key=lambda f: os.path.getmtime(os.path.join(runs, f)))
    if not ships:
        raise RuntimeError(f"training exited 0 but wrote no checkpoint named {out}* -- not claiming success")
    return os.path.join(runs, ships[-1])


def easy(data, size="auto", **overrides):
    cfg, why = recipe_for(size)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    _print_plan("easy", cfg, why)
    return _run(cfg, data)


def hard(data, **cfg):
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"hard mode needs every setting written out; missing {missing}. "
                         f"Required: {REQUIRED}")
    for k in METHOD_SPECIFIC.get(cfg["method"], ()):
        if k not in cfg:
            raise ValueError(f"method {cfg['method']!r} also needs {k}; nothing is filled in for you")
    if cfg["method"] not in METHOD_SPECIFIC:
        raise ValueError(f"method must be one of {sorted(METHOD_SPECIFIC)}, got {cfg['method']!r}")
    _print_plan("hard", cfg, "hard mode: you supplied every setting")
    return _run(cfg, data)


def why_easy(size="auto"):
    cfg, why = recipe_for(size)
    return {"why": why, "settings": cfg}