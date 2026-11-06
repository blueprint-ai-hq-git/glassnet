"""t3_c_top2.py — pick the TOP 2 combos from Target C's screen, by the frozen rule.

Run BEFORE the confirm step (the full pick, t3_c_pick.py, needs the confirm files and is run
after). Keeps the pre-registered "screen on seed 0, confirm the top 2 on seeds 1-2" discipline.
"""
import glob
import json
import os

R = "runs/t3"
rows = []
for f in sorted(glob.glob(os.path.join(R, "models_e_scr_*.json"))):
    tag = os.path.basename(f)[len("models_"):-len(".json")]
    lc = tag.split("_lc")[1].split("_")[0]
    lr = tag.split("_lr")[1]
    m = json.load(open(f))
    rd = [x for x in m if x["model"] == "readable"]
    if rd:
        rows.append({"lc": lc, "lr": lr, "ppl": rd[0]["val_ppl"], "acc": rd[0]["val"]})
if not rows:
    raise SystemExit("no screened combos found")
rows.sort(key=lambda x: (round(x["ppl"], 3), -round(x["acc"] / 100.0, 4), float(x["lr"])))
print("  screen, sorted by the frozen rule:")
for r in rows:
    print(f"    lc={r['lc']} lr={r['lr']}  ppl {r['ppl']:.4f}  acc {r['acc']:.2f}%")
print("  TOP2: " + " ".join(f"{t['lc']}:{t['lr']}" for t in rows[:2]))
json.dump({"screened": rows, "top2": rows[:2]},
          open(os.path.join(R, "e_screen.json"), "w"), indent=1)
