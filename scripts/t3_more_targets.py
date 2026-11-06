"""t3_more_targets.py — derive MORE meaningful concept targets from TRAIN only.

Target D showed that six concepts is too few to carry a whole model: two of them collapsed
under the load. Target C showed that unnamed columns do the work when they exist. The untested
middle is MORE named concepts.

This keeps the 2.5% TRAIN-support floor EXACTLY as it is (changing a rule after seeing results
is banned) and simply tests a larger batch of human-meaningful categories against it. Whatever
clears the floor qualifies; whatever does not is reported, not hidden.
"""
import json
import os
import re
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet import tiny_data

R = os.path.join(PROJ, "runs", "t3")
SUPPORT_BAR = 0.025
WIN = 8

DATA = tiny_data.build(win=WIN, vocab_cap=1500)
Xtr = np.asarray(DATA["Xtr"])
VOCAB = DATA["vocab"]
V = DATA["V"]
N = len(Xtr)
VIDX = {w: i for i, w in enumerate(VOCAB)}

print("=" * 100)
print(f"MORE MEANINGFUL TARGETS — TRAIN only ({N} windows), support floor {SUPPORT_BAR:.1%} UNCHANGED")
print("=" * 100)

# A larger batch of human-nameable ideas. Every word is checked against the real vocabulary
# first, so a category is never named after a token the corpus does not contain.
CANDIDATES = {
    # --- the six that already qualified (kept, so the set is a superset) ---
    "pronoun_subject":      ["he", "she", "they", "we", "I"],
    "pronoun_object":       ["him", "her", "them", "us", "me"],
    "is_a_negation":        ["not", "no", "never", "nor", "none", "nothing"],
    "is_archaic_address":   ["thou", "thee", "thy", "thine", "ye"],
    "contains_a_wh_word":   ["what", "why", "how", "when", "where", "who"],
    # --- new candidates ---
    "mentions_kinship":     ["father", "mother", "son", "daughter", "brother", "sister",
                             "wife", "husband", "child", "uncle"],
    "mentions_death":       ["death", "dead", "die", "dies", "kill", "killed", "slain",
                             "grave", "dying", "murder"],
    "mentions_love":        ["love", "loves", "loved", "loving", "dear", "sweet", "kiss",
                             "lover", "beloved"],
    "mentions_time":        ["time", "hour", "day", "night", "morrow", "morning", "evening",
                             "years", "year", "moment"],
    "mentions_nature":      ["sun", "moon", "stars", "sea", "earth", "sky", "wind", "rain",
                             "fire", "water"],
    "mentions_authority":   ["sir", "lord", "master", "madam", "mistress", "sire",
                             "grace", "worship"],
    "is_an_exclamation":    ["O", "Alas", "Fie", "Come", "Well", "Nay", "Ay"],
    "mentions_a_number":    ["two", "three", "four", "five", "ten", "twenty", "thousand",
                             "hundred", "first", "second"],
    "mentions_seeing":      ["see", "sees", "saw", "seen", "look", "looks", "looked",
                             "behold", "eyes"],
    "mentions_speech":      ["say", "says", "said", "speak", "speaks", "spoke", "tell",
                             "told", "talk", "word"],
    "mentions_knowing":     ["know", "knows", "knew", "known", "think", "thinks", "thought",
                             "believe", "doubt", "sure"],
    "mentions_giving":      ["give", "gives", "gave", "given", "take", "takes", "took",
                             "bring", "brought", "send"],
    "mentions_money":       ["gold", "money", "coin", "ducats", "pay", "pounds", "crowns",
                             "wealth"],
    "mentions_royalty":     ["king", "queen", "prince", "princess", "duke", "lord",
                             "lady", "majesty", "crown", "duchess"],
    "mentions_a_body_part": ["hand", "hands", "eye", "eyes", "heart", "head", "face",
                             "tongue"],
    "is_a_greeting":        ["hail", "welcome", "farewell", "adieu", "good", "morrow"],
}

results = []
for name, words in CANDIDATES.items():
    present = [w for w in words if w in VIDX]
    if not present:
        results.append({"name": name, "support": 0.0, "n_present": 0, "words": [],
                        "qualifies": False, "reason": "no tokens in vocab"})
        continue
    ids = [VIDX[w] for w in present]
    m = np.zeros(N, dtype=bool)
    for i in ids:
        m |= (Xtr == i).any(axis=1)
    s = float(m.mean())
    results.append({"name": name, "support": s, "n_present": len(present),
                    "n_words": len(words), "words": present, "ids": ids,
                    "qualifies": bool(s >= SUPPORT_BAR),
                    "reason": "" if s >= SUPPORT_BAR else "below floor"})

# speaker-label union (already known to qualify at 23.19%) -- recompute so the file is complete
sp = [w for w in VOCAB if re.fullmatch(r"[A-Z][A-Za-z' ]*:", w or "")]
sp_ids = [VIDX[w] for w in sp]
m = np.zeros(N, dtype=bool)
for i in sp_ids:
    m |= (Xtr == i).any(axis=1)
dialogue = {"name": "is_a_dialogue_line", "support": float(m.mean()),
            "n_present": len(sp), "n_words": len(sp), "words": ["<any speaker label>"],
            "ids": sp_ids, "qualifies": True, "reason": ""}

qual = [r for r in results if r["qualifies"]]
qual.append(dialogue)
qual.sort(key=lambda r: -r["support"])

print(f"\n{'concept':<24} {'support':>9}  {'tokens':>8}  verdict")
for r in sorted(results, key=lambda z: -z["support"]):
    mark = "QUALIFIES" if r["qualifies"] else r["reason"]
    print(f"{r['name']:<24} {r['support']:>9.4f}  {r['n_present']:>3}/{r['n_words']:<4}  {mark}")
print(f"{dialogue['name']:<24} {dialogue['support']:>9.4f}  {dialogue['n_present']:>3}/{dialogue['n_words']:<4}  QUALIFIES (speaker-label union)")

print("\n" + "=" * 100)
print(f"QUALIFYING: {len(qual)} concepts at the UNCHANGED {SUPPORT_BAR:.1%} floor")
for i, r in enumerate(qual):
    print(f"  {i:>2} {r['name']:<24} support {r['support']:.4f}")
print("=" * 100)

json.dump({"method": "extended meaningful targets, TRAIN only, floor unchanged",
           "support_bar": SUPPORT_BAR, "qualifying": qual, "all": results + [dialogue]},
          open(os.path.join(R, "meaningful_targets_ext.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meaningful_targets_ext.json  ({len(qual)} qualifying)")
