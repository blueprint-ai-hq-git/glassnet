"""make_everything_file.py — bundle the ENTIRE GlassNet project into ONE
markdown file on the Desktop: every line of code, every doc, every report,
every number. (Yahya's order: "everything ... in one file ... on the desktop".)

Run:  python3 make_everything_file.py
"""
import datetime
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DESKTOP = os.path.expanduser("~/Desktop")
OUT = os.path.join(DESKTOP, "GlassNet-EVERYTHING.md")

EXPLICIT = [
    # ---- docs first ----
    "README.md", "PHASES.md", "PROBLEMS.md", "HIDDEN_DEFAULTS.md",
    "CONCEPTS.md", "DATA_LICENSES.md", "LICENSE", "pyproject.toml",
    "docs/BEGINNER.md", "docs/concept-class-stats.md",
    # ---- the engine (every line of glassnet/) ----
    "glassnet/__init__.py", "glassnet/tensor.py", "glassnet/layers.py",
    "glassnet/losses.py", "glassnet/optim.py", "glassnet/gradcheck.py",
    "glassnet/gates.py", "glassnet/eval.py", "glassnet/data.py",
    "glassnet/concepts.py", "glassnet/phase2_model.py",
    "glassnet/autopilot.py", "glassnet/judge.py", "glassnet/evo.py",
    "glassnet/api.py", "glassnet/report.py",
    # ---- phase scripts ----
    "phase1_xor.py", "phase1_fmnist.py", "phase2_train.py",
    "phase2_audit.py", "phase2_explain.py", "phase2_ablations.py",
    "phase3_autopilot.py", "phase3_controls.py", "phase4_judge.py",
    "phase6_comparison.py", "verify_all.py",
    "experiments/judge_features.py",
    # ---- tests + examples ----
    "tests/test_gradcheck.py", "tests/test_library.py",
    "examples/example_20_lines.py", "examples/example_evo_switch.py",
    # ---- result reports (the measured numbers) ----
    "runs/phase1/BASELINE.md", "runs/phase2/PHASE2-BASELINE.md",
    "runs/phase2/PHASE-2-REPORT.md", "runs/phase2/audit.md",
    "runs/phase2/ablations.md", "runs/phase2/explain.md",
    "runs/phase3/PHASE3-REPORT.md", "runs/phase3/CONTROLS.md",
    "runs/phase4/PHASE4-REPORT.md", "runs/phase5/PHASE5-REPORT.md",
    "runs/phase6/PHASE6-REPORT.md", "runs/verify/VERIFY-ALL.md",
    # ---- small data summaries ----
    "runs/phase1/baseline_summary.json", "runs/phase2/config.json",
    "runs/phase4/summary.json", "runs/phase6/summary.json",
]
GLOBS = ["runs/phase1/baseline_seed*.csv",
         "runs/phase2/final_seed*.json",
         "runs/phase3/eval_seed*.json",
         "runs/phase3/actions_seed*.log",
         "runs/phase6/evo_actions_seed*.log"]
SKIP_WALK = {"data", "__pycache__", ".git"}


def collect():
    entries = [os.path.join(HERE, p) for p in EXPLICIT]
    for g in GLOBS:
        entries += sorted(glob.glob(os.path.join(HERE, g)))
    seen, out = set(), []
    for f in entries:
        if f in seen:
            continue
        seen.add(f)
        if os.path.exists(f):
            out.append(f)
    return out


def lang_for(path):
    ext = os.path.splitext(path)[1].lower()
    return {".py": "python", ".md": "markdown", ".json": "json",
            ".csv": "csv", ".log": "text", ".toml": "toml",
            ".txt": "text", ".npz": "text"}.get(ext, "text")


def inventory():
    rows = []
    for root, dirs, names in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in SKIP_WALK]
        base = os.path.relpath(root, HERE)
        for n in sorted(names):
            p = os.path.join(root, n)
            rows.append((os.path.join(base, n) if base != "." else n,
                         os.path.getsize(p)))
    return sorted(rows)


def main():
    files = collect()
    if not os.path.isdir(DESKTOP):
        os.makedirs(DESKTOP, exist_ok=True)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    L = ["# GLASSNET — EVERYTHING (one file)", "",
         f"Bundled {now} · project root: `{HERE}` · owner: Yahya", "",
         "This single file contains every line of code written for the "
         "project, every document (readme, phases, problems board, hidden "
         "defaults, concept formulas, licenses), and every measured result "
         "report with the real numbers. Pictures and checkpoints are binary "
         "and live in the project (`runs/**/`) — they are listed in the "
         "inventory at the bottom. Nothing here is estimated; every number "
         "was produced by a real run.", "",
         "## Headline numbers (fresh verify run, 13/13 gates green)", "```",
         "P1 baseline (opaque net) : 86.69% ± 0.59%",
         "P2 readable arch         : 79.04% ± 1.97%  (3.00 conns/rule, "
         "12/14 faithful neurons, lie-neuron control caught)",
         "P3 auto-pilot (0 settings): 80.16% ± 0.98%  (beats manual on all "
         "3 seeds; 4/4 reflexes proven)",
         "P4 data judge            : held-out 79.69% · unseen-type 71.48% "
         "(chance = 50%) · cleaner removed 100/100 injected junk",
         "P5 library               : 11-line script, 7/7 tests",
         "P6 experiment            : SGD 80.16% ± 0.98% vs (1+1)-ES "
         "34.73% ± 2.04% — fair fight, SGD wins",
         "```", "",
         "## Index (everything inside this file)", ""]
    for i, f in enumerate(files, 1):
        rel = os.path.relpath(f, HERE)
        L.append(f"{i:3d}. `{rel}`")
    L.append("")
    for f in files:
        rel = os.path.relpath(f, HERE)
        L.append(f"---")
        L.append(f"## 📄 `{rel}`")
        L.append("")
        L.append(f"```{lang_for(f)}")
        with open(f, encoding="utf-8", errors="replace") as fh:
            L.append(fh.read().rstrip("\n"))
        L.append("```")
        L.append("")

    L.append("## 📦 Full project inventory (all files, with sizes)")
    L.append("")
    total = 0
    for rel, size in inventory():
        total += size
        L.append(f"- {rel}  ({size:,} bytes)")
    L.append("")
    L.append(f"Total project size: {total:,} bytes "
             f"({total/1048576:.1f} MB, including the binary dataset cache "
             f"in data/). Binary artifacts (checkpoints `.npz`) and `__pycache__` "
             f"are listed but not embedded here — the code that reads them is." )
    L.append("")
    open(OUT, "w", encoding="utf-8").write("\n".join(L))
    print(f"[ok] wrote {OUT}")
    print(f"[ok] {len(files)} files embedded · "
          f"{os.path.getsize(OUT):,} bytes total")


if __name__ == "__main__":
    main()