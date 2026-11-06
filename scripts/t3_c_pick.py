"""t3_c_pick.py — apply the frozen SELECTION-RULE.md to Target C's screened combos.

The screen rows use keys ppl/acc; the confirm files use the raw model schema (val_ppl/val).
They are read with their own key names -- mixing them raised KeyError: 'val_ppl' once
already, so the lesson is applied here.
"""
import glob
import json
import os

R = "runs/t3"
rows = []
for f in sorted(glob.glob(os.path.join(R, "models_c_scr_*.json"))):
    tag = os.path.basename(f)[len("models_"):-len(".json")]
    lc = tag.split("_lc")[1].split("_")[0]
    lr = tag.split("_lr")[1]
    m = json.load(open(f))
    rd = [x for x in m if x["model"] == "readable"]
    if not rd:
        continue
    rows.append({"lc": lc, "lr": lr, "ppl": rd[0]["val_ppl"], "acc": rd[0]["val"]})

rows.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4), float(x["lr"])))
print("  screened combos, sorted by the frozen rule:")
for r in rows:
    print(f"    lc={r['lc']} lr={r['lr']}  val ppl {r['ppl']:.4f}  val acc {r['acc']:.2f}%")
if not rows:
    raise SystemExit("no screened combos found")

# confirm the top 2 on seeds 1,2 and choose on the 3-seed mean
cands = []
for t in rows[:2]:
    lc, lr = t["lc"], t["lr"]
    ppls, accs = [t["ppl"]], [t["acc"]]
    f = os.path.join(R, f"models_c_cfm_lc{lc}_lr{lr}.json")
    if os.path.exists(f):
        for x in json.load(open(f)):
            if x["model"] == "readable":
                ppls.append(x["val_ppl"])
                accs.append(x["val"])
    cands.append({"lc": lc, "lr": lr, "ppl": sum(ppls) / len(ppls),
                  "acc": sum(accs) / len(accs), "n_seeds": len(ppls)})
cands.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4), float(x["lr"])))
print("\n  candidates on the mean:")
for c in cands:
    print(f"    lc={c['lc']} lr={c['lr']}  {c['n_seeds']} seeds  ppl {c['ppl']:.4f}  "
          f"acc {c['acc']:.2f}%")
w = cands[0]
json.dump({"lr": w["lr"], "lc": w["lc"], "ppl": w["ppl"], "acc": w["acc"],
           "n_seeds": w["n_seeds"], "candidates": cands, "screen_all": rows,
           "rule": "frozen SELECTION-RULE.md: val ppl, tie-break acc within 1% rel, then lower lr"},
          open(os.path.join(R, "c_selection.json"), "w"), indent=1)
print(f"\n  WINNER: lr={w['lr']} lam_concept={w['lc']}")
