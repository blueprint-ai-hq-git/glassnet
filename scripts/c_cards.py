"""c_cards.py — write the Target C blind reader cards (R4) as a fillable file.

Reads the cards already built by t3_c_detail.py (fixed shuffle seed 0, 5 firing + 5
non-firing), so the reader sees exactly what the key says.
"""
import json
import os

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(PROJ, "runs", "t3")
d = json.load(open(os.path.join(R, "name_precision_c.json")))
cards = d["cards"]

L = []
L.append("# READER TEST CARDS — TARGET C (R4)")
L.append("")
L.append("**The one metric in GlassNet judged by a human, not computed.**")
L.append("**Official reader: you, or the external reviewer. NOT the agent** — it wrote the names,")
L.append("so scoring its own test would be circular.")
L.append("")
L.append("## What to do")
L.append("")
L.append("Each card gives a concept's **name** and **10 windows in random order**. Mark **Y** if")
L.append("you think that neuron fires on the window, **N** if not. **Exactly 5 of the 10 fire** —")
L.append("that is the only hint, and it is a fair one.")
L.append("")
L.append("**8 of 10 correct (0.80) or better = the name genuinely tells a reader something.**")
L.append("Chance is 5/10. The answer key is `runs/t3/reader_test_key_c.json` — do not open it")
L.append("until the cards are filled in.")
L.append("")
L.append("---")
L.append("")
for card in cards:
    L.append(f"## Concept {card['col']}: `{card['name']}`")
    L.append("")
    L.append("| # | window | fires? (Y/N) |")
    L.append("|---|---|---|")
    for w in card["windows"]:
        L.append(f"| {w['n']} | `{w['text']}` |  |")
    L.append("")
    L.append("Your score: ___ / 10")
    L.append("")
    L.append("---")
L.append("")
L.append("## Summary")
L.append("")
L.append("| concept | name | your score | ≥8/10? |")
L.append("|---|---|---|---|")
for card in cards:
    L.append(f"| {card['col']} | `{card['name']}` | ___/10 |  |")
L.append("")
L.append("**Readability under R1–R4 = the count of concepts passing all four.** Report it as a")
L.append("count, never a vibe.")

open(os.path.join(PROJ, "READER-TEST-CARDS-C.md"), "w").write("\n".join(L))
print(f"wrote READER-TEST-CARDS-C.md  ({len(cards)} cards x 10 windows)")
print("answer key: runs/t3/reader_test_key_c.json (reader must not open it)")
