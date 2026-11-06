"""t3_meaningful_targets.py — rung 2: derive MEANINGFUL concept targets from TRAIN only.

Replaces single-word clusters with ideas a human actually recognises:
  * SPEAKER   - who is speaking (the text carries speaker labels: TRANIO:, GRUMIO: ...)
  * PRONOUN   - subject/object pronoun sets
  * WH        - question words (what/why/how/when/where/who)

TRAIN split only. Nothing is downloaded. Support is reported before anything trains, so the
target set can be judged honestly. Target support >= 2.5% is the threshold A3 established.
"""
import json
import os
import re
import sys
from collections import Counter

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np
from glassnet import tiny_data

RUNDIR = os.path.join(PROJ, "runs", "t3")
SUPPORT_BAR = 0.025
WIN = 8

DATA = tiny_data.build(win=WIN, vocab_cap=1500)
Xtr = np.asarray(DATA["Xtr"])
VOCAB = DATA["vocab"]
V = DATA["V"]
N = len(Xtr)
print("=" * 104)
print(f"RUNG 2 — MEANINGFUL TARGETS FROM TRAIN ONLY  ({N} train windows, V={V})")
print("=" * 104)

# does the tokeniser keep '?' at all? (decides whether "question" is an honest name)
has_qmark = any("?" in w for w in VOCAB)
_qname = '"is a question"' if has_qmark else '"contains a wh-word" (HONEST: no ? token)'
print(f"does any vocab token contain '?': {has_qmark}  "
      f"-> a question concept must be named {_qname}")

# ---- speaker labels: uppercase tokens ending in ':' ----
speaker_tokens = [w for w in VOCAB if re.fullmatch(r"[A-Z][A-Za-z' ]*:", w or "")]
print(f"\nspeaker-like tokens found in vocab: {len(speaker_tokens)}")
sp_ids = {w: VOCAB.index(w) for w in speaker_tokens}
present = np.zeros((N, len(speaker_tokens)), dtype=bool)
for j, w in enumerate(speaker_tokens):
    present[:, j] = (Xtr == sp_ids[w]).any(axis=1)
supp = present.mean(axis=0)
order = np.argsort(-supp)
print(f"\n{'speaker':<16} {'TRAIN support':>14}  {'qualifies (>=2.5%)':>19}")
speakers = []
for j in order:
    w = speaker_tokens[j]
    q = bool(supp[j] >= SUPPORT_BAR)
    print(f"{w:<16} {supp[j]:>14.4f}  {('YES' if q else 'no'):>19}")
    if q:
        speakers.append({"name": f"speaker_{w.rstrip(':')}", "words": [w],
                        "ids": [sp_ids[w]], "support": float(supp[j])})

# ---- pronouns ----
PRON = {
    "pronoun_subject": ["he", "she", "they", "we", "I"],
    "pronoun_object": ["him", "her", "them", "us", "me"],
}
pron_targets = []
print("\npronoun sets:")
for name, ws in PRON.items():
    ids = [VOCAB.index(w) for w in ws if w in VOCAB]
    got = [w for w in ws if w in VOCAB]
    if not ids:
        print(f"  {name:<18} NO TOKENS IN VOCAB")
        continue
    m = np.zeros(N, dtype=bool)
    for i in ids:
        m |= (Xtr == i).any(axis=1)
    s = float(m.mean())
    print(f"  {name:<18} {got}  support {s:.4f}  "
          f"{'QUALIFIES' if s >= SUPPORT_BAR else 'below 2.5%'}")
    pron_targets.append({"name": name, "words": got, "ids": ids, "support": s})

# ---- wh / question words ----
WH = ["what", "why", "how", "when", "where", "who"]
ids = [VOCAB.index(w) for w in WH if w in VOCAB]
got = [w for w in WH if w in VOCAB]
m = np.zeros(N, dtype=bool)
for i in ids:
    m |= (Xtr == i).any(axis=1)
wh_support = float(m.mean())
print(f"\nwh-word set: {got}  support {wh_support:.4f}  "
      f"{'QUALIFIES' if wh_support >= SUPPORT_BAR else 'below 2.5%'}")

targets = []
targets += [t for t in pron_targets if t["support"] >= SUPPORT_BAR]
if wh_support >= SUPPORT_BAR:
    targets.append({"name": "contains_a_wh_word",
                    "words": got, "ids": ids, "support": wh_support})

# --- the two BROAD versions of the speaker/question ideas -------------------------------
# Individual speakers are far too rare to clear 2.5% (top speaker is ~0.13%), but the UNION
# of all speaker labels is broad, and "this is a dialogue line" is a genuine, human-nameable
# idea. Same for questions: '?' IS a token, so "is a question" is an honest name.
union_ids = [sp_ids[w] for w in speaker_tokens]
m = np.zeros(N, dtype=bool)
for i in union_ids:
    m |= (Xtr == i).any(axis=1)
dialogue_support = float(m.mean())
print(f"\nUNION of all {len(speaker_tokens)} speaker labels: support {dialogue_support:.4f}  "
      f"{'QUALIFIES' if dialogue_support >= SUPPORT_BAR else 'below 2.5%'}")
if dialogue_support >= SUPPORT_BAR:
    targets.append({"name": "is_a_dialogue_line", "words": ["<any speaker label>"],
                    "ids": union_ids, "support": dialogue_support})

if has_qmark:
    q_tokens = [w for w in VOCAB if "?" in (w or "")]
    q_ids = [VOCAB.index(w) for w in q_tokens]
    qm = np.zeros(N, dtype=bool)
    for i in q_ids:
        qm |= (Xtr == i).any(axis=1)
    q_support = float(qm.mean())
    print(f"'?' is not standalone; {len(q_tokens)} tokens carry it, e.g. {q_tokens[:6]}")
    print(f"question-marked tokens as a set: support {q_support:.4f}  "
          f"{'QUALIFIES' if q_support >= SUPPORT_BAR else 'below 2.5%'}")
    if q_support >= SUPPORT_BAR:
        targets.append({"name": "is_a_question", "words": q_tokens[:8], "ids": q_ids,
                        "support": q_support})

# --- more genuinely meaningful categories (each is an idea a human would name) -----------
SEMANTIC = {
    "is_a_negation":        ["not", "no", "never", "nor", "none", "nothing"],
    "is_archaic_address":   ["thou", "thee", "thy", "thine", "ye"],
    "mentions_royalty":     ["king", "queen", "prince", "princess", "duke", "lord",
                             "lady", "majesty", "crown", "duchess"],
    "mentions_money":       ["gold", "money", "coin", "ducats", "pay", "pounds",
                             "crowns", "wealth"],
    "mentions_a_body_part": ["hand", "hands", "eye", "eyes", "heart", "head", "face",
                             "tongue"],
    "is_a_greeting":        ["hail", "welcome", "farewell", "adieu", "good", "morrow"],
}
print("\nsemantic categories:")
for name, ws in SEMANTIC.items():
    ids_ = [VOCAB.index(w) for w in ws if w in VOCAB]
    got_ = [w for w in ws if w in VOCAB]
    if not ids_:
        print(f"  {name:<24} NO TOKENS IN VOCAB")
        continue
    m_ = np.zeros(N, dtype=bool)
    for i in ids_:
        m_ |= (Xtr == i).any(axis=1)
    s_ = float(m_.mean())
    q_ = s_ >= SUPPORT_BAR
    print(f"  {name:<24} {s_:.4f}  {'QUALIFIES' if q_ else 'below 2.5%'}   "
          f"({len(got_)}/{len(ws)} tokens present)")
    if q_:
        targets.append({"name": name, "words": got_, "ids": ids_, "support": s_})

print("\n" + "=" * 104)
print(f"QUALIFYING MEANINGFUL TARGETS: {len(targets)}")
for t in sorted(targets, key=lambda z: -z["support"]):
    print(f"   {t['name']:<26} support {t['support']:.4f}  words {t['words']}")
print("=" * 104)

out = {"method": "rung 2 meaningful targets, TRAIN only",
       "support_bar": SUPPORT_BAR, "has_qmark": has_qmark,
       "n_speaker_tokens": len(speaker_tokens),
       "targets": targets,
       "all_speaker_supports": {speaker_tokens[j]: float(supp[j]) for j in range(len(speaker_tokens))}}
json.dump(out, open(os.path.join(RUNDIR, "meaningful_targets.json"), "w"), indent=1)
print(f"\nsaved runs/t3/meaningful_targets.json  ({len(targets)} targets)")
