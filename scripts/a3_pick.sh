#!/bin/bash
# a3_pick.sh — apply the frozen selection rule to the screened+confirmed combos.
# Fixes the bug in a3_tune.sh: the screen rows use keys ppl/acc (built from val_ppl/val),
# while the confirm files use the raw model schema (val_ppl/val). Mixing them raised
# KeyError: 'val_ppl'. This reads each with its own key names.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/Hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY=python3
"$PY" - <<'EOF'
import json, os
scr = json.load(open("runs/t3/a3_screen.json"))
cands = []
for t in scr["top2"]:
    lc, lr = t["lc"], t["lr"]
    # seed 0 from the screen (keys: ppl, acc)
    ppls, accs = [t["ppl"]], [t["acc"]]
    f = f"runs/t3/models_a3_cfm_lc{lc}_lr{lr}.json"
    if os.path.exists(f):
        for x in json.load(open(f)):
            if x["model"] == "readable":            # keys: val_ppl, val
                ppls.append(x["val_ppl"]); accs.append(x["val"])
    cands.append({"lc": lc, "lr": lr, "ppl": sum(ppls) / len(ppls),
                  "acc": sum(accs) / len(accs), "n_seeds": len(ppls)})
# frozen SELECTION-RULE.md: validation perplexity primary, tie-break accuracy within 1%
# relative, then lower lr
cands.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4), float(x["lr"])))
print("  candidates on the 3-seed mean:")
for c in cands:
    print(f"    lc={c['lc']} lr={c['lr']}  {c['n_seeds']} seeds  ppl {c['ppl']:.4f}  "
          f"acc {c['acc']:.2f}%")
w = cands[0]
json.dump({"lr": w["lr"], "lc": w["lc"], "ppl": w["ppl"], "acc": w["acc"],
           "n_seeds": w["n_seeds"], "candidates": cands,
           "rule": "frozen SELECTION-RULE.md: val ppl, tie-break acc within 1% rel, then lower lr",
           "budget_note": "9 combos screened on seed 0; top 2 confirmed on seeds 1,2",
           "screen_all": scr["screened"]},
          open("runs/t3/a3_selection.json", "w"), indent=1)
print(f"  WINNER: lr={w['lr']} lam_concept={w['lc']}  ({w['n_seeds']} seeds, "
      f"ppl {w['ppl']:.4f}, acc {w['acc']:.2f}%)")
EOF