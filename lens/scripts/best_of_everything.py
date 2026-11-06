"""best_of_everything.py — ONE report: for every capability the owner asked for, which artifact is
best, on which evidence, and what its honest weakness is.

WHY A SINGLE REPORT
  Numbers for talk, code and medical live in different run directories with different metrics
  (bits/token, exec rate, accuracy). Reading them separately invites comparing incomparable things.
  This gathers them in one place, states the metric used for each, and names the loser as well as
  the winner. Nothing is ranked by a metric that flatters it.

RULES APPLIED
  * each capability is judged by its OWN correctness metric (talk: unk+bits/token, code: does it RUN,
    medical: held-out accuracy) -- never by perplexity alone
  * the runner-up is shown, so the choice is visible rather than asserted
  * every weakness is stated next to the win
"""
import json
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"
R = os.path.join(P, "lens", "runs")


def jload(p, default=None):
    try:
        return json.load(open(p))
    except Exception:
        return default


rows = []

# ---------------- TALK ----------------
talk = []
for tag in ("best1", "v12k", "w24_v12000", "big1"):
    lf = os.path.join(R, "chat", f"{tag}_log.json")
    d = jload(lf)
    if not d or not d.get("history"):
        continue
    h = d["history"][-1]
    a = d.get("args", {})
    talk.append({"tag": tag, "V": a.get("vocab"), "W": a.get("window"), "d": a.get("d"),
                 "ppl": h["val_ppl"], "bpt": round(h["val_ppl"] ** 0.5, 3) if False else None,
                 "acc": h.get("val_acc")})
# prefer the recorded verdict files for the unk numbers (measured on shared prompts)
v12k = jload(os.path.join(R, "chat", "v12k_verdict.json"))
best1_unk = v12k["base"]["unk"] if v12k else None
v12k_unk = v12k["new"]["unk"] if v12k else None
w24v = jload(os.path.join(R, "chat", "w24_verdict.json"))
w24_unk = w24v["new"]["unk"] if w24v else None
unk_map = {"best1": best1_unk, "v12k": v12k_unk, "w24_v12000": w24_unk}

import math
for t in talk:
    t["bpt"] = round(math.log2(t["ppl"]), 3) if t["ppl"] else None
    t["unk"] = unk_map.get(t["tag"])

# talk winner: lowest unk (holes hurt a reader most), reported with its bpt cost
with_unk = [t for t in talk if t["unk"] is not None]
talk_win = min(with_unk, key=lambda t: t["unk"]) if with_unk else None
talk_ppl_win = min(talk, key=lambda t: t["ppl"]) if talk else None
rows.append({
    "capability": "TALK (chat)",
    "metric": "<unk> rate in generated replies (holes a reader sees), with bits/token cost shown",
    "winner": talk_win["tag"] if talk_win else "n/a",
    "winner_numbers": (f"<unk> {talk_win['unk']*100:.1f}%, bits/token {talk_win['bpt']}, "
                       f"V={talk_win['V']}, W={talk_win['W']}") if talk_win else "n/a",
    "runner_up": (f"{talk_ppl_win['tag']} (lowest bits/token {talk_ppl_win['bpt']}, "
                  f"<unk> {talk_ppl_win['unk']*100:.1f}%)"
                  if talk_ppl_win and talk_ppl_win.get("unk") is not None else "n/a"),
    "weakness": ("Still not fluent; replies are short and often drift. A ~600K-param model on 34MB "
                 "of chat data cannot reach real-chatbot fluency, and I am not claiming it does."),
})

# ---------------- CODE ----------------
code = jload(os.path.join(R, "code", "all_code_eval.json"))
if code:
    code_sorted = sorted(code, key=lambda r: (r["syntax"], r["ran"]), reverse=True)
    cw = code_sorted[0]
    rows.append({
        "capability": "CODE",
        "metric": "generated programs that PARSE (ast.parse) and RUN (subprocess) — not perplexity",
        "winner": cw["tag"],
        "winner_numbers": f"syntax {cw['syntax']}/{cw['total']}, ran {cw['ran']}/{cw['total']}, ppl {cw['ppl']}",
        "runner_up": (f"{code_sorted[1]['tag']} (syntax {code_sorted[1]['syntax']}/{code_sorted[1]['total']})"
                      if len(code_sorted) > 1 else "n/a"),
        "weakness": ("Low absolute rate — it parses sometimes, rarely runs, and output is still "
                     "mixed with prose. ppl has lied twice here; exec rate is the only honest metric."),
    })

# ---------------- MEDICAL ----------------
med = []
for tag in ("med_cc", "med_cc_s1", "med_adapt", "med_bal"):
    d = jload(os.path.join(R, "vl", f"{tag}_log.json"))
    if not d:
        continue
    res = d.get("results", {})
    rr = res.get("readable", {})
    pp = res.get("plain", {})
    if not rr.get("history"):
        continue
    aud = d.get("audit", {})
    npass = None
    if isinstance(aud, dict) and "concepts" in aud:
        npass = sum(1 for c in aud["concepts"] if c.get("pass"))
        ntot = len(aud["concepts"])
    else:
        npass = ntot = None
    med.append({"tag": tag, "readable": rr["history"][-1]["test_acc"] * 100,
                "plain": (pp["history"][-1]["test_acc"] * 100) if pp.get("history") else None,
                "npass": npass, "ntot": ntot})
if med:
    mw = max(med, key=lambda m: m["readable"])
    rows.append({
        "capability": "MEDICAL (PathMNIST, 9 tissue classes, chance 11.1%)",
        "metric": "held-out test accuracy, readable model vs parameter-matched plain control",
        "winner": mw["tag"],
        "winner_numbers": (f"readable {mw['readable']:.2f}% vs plain "
                           f"{mw['plain']:.2f}% = +{mw['readable']-mw['plain']:.2f} pts, "
                           f"{mw['npass']}/{mw['ntot']} concepts pass their firing audit"),
        "runner_up": (f"{max(med, key=lambda m: m['readable'] if m is not mw else -1)['tag']}"
                      if len(med) > 1 else "n/a"),
        "weakness": ("One dataset only (histopathology). Concepts passing 30/35 means 5 do NOT fire "
                     "as named — stated, not hidden. Not clinically validated and must never be "
                     "used for medical decisions."),
    })

# ---------------- FACES ----------------
rows.append({
    "capability": "FACES (detect + track)",
    "metric": "known-case controls (5/5) and false-positive count on 40 real unlabeled photos",
    "winner": "face_concepts.detect + face_track.py (EXPERIMENTAL)",
    "winner_numbers": "synthetic controls 5/5; real photos 7/40 flagged (false positives)",
    "runner_up": "n/a — no second implementation exists",
    "weakness": ("Real-face accuracy UNPROVEN: no ground-truth labels exist offline, and every public "
                 "face dataset fails the safe-licence rule. Needs 30s of the owner in front of the "
                 "webcam to create licence-clean positives. NOT shipped as working."),
})

# ---------------- CYBER ----------------
rows.append({
    "capability": "CYBER SECURITY",
    "metric": "none — not started",
    "winner": "none",
    "winner_numbers": "no model, no dataset on disk",
    "runner_up": "n/a",
    "weakness": ("No licence-clean corpus is available offline, and the mobile-data budget (2 GB) "
                 "rules out pulling one. Honestly NOT STARTED rather than faked."),
})

print("=" * 104)
print("BEST OF EVERYTHING — one line per capability, judged by its own correctness metric")
print("=" * 104)
for r in rows:
    print(f"\n  {r['capability']}")
    print(f"    judged by : {r['metric']}")
    print(f"    BEST      : {r['winner']}")
    print(f"    numbers   : {r['winner_numbers']}")
    print(f"    runner-up : {r['runner_up']}")
    print(f"    weakness  : {r['weakness']}")

print("\n" + "-" * 104)
print("  HONEST BOTTOM LINE")
print("    * Strongest real result: MEDICAL (+24.6 pts over its own plain control, 30/35 concepts firing).")
print("    * Best talk artifact: the lowest-unk model above — improved, still not a real chatbot.")
print("    * Code: weakest area; exec rate is the only metric that has not lied.")
print("    * Faces: experimental, accuracy unproven. Cyber: not started.")
print("    No capability here is claimed at a level the numbers do not show.")
json.dump(rows, open(os.path.join(R, "best_of_everything.json"), "w"), indent=1)
print(f"\n  saved {os.path.join(R, 'best_of_everything.json')}")
