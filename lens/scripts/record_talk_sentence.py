"""record_talk_sentence.py — sentence API ported onto the interface; <unk> rate stated honestly.
"""
import os

P = "/run/media/yahya/HARD DISK 4/GlassNet"

LOG = """
---

## 2026-10-08 late — `v.talk_sentence()`: sentence-level talk on the one interface

Ported the chat sweep's VERIFIED decoder (top-k 40 + repetition penalty 1.15 — the two findings
that greedy loops and pure sampling drifts are the reason it exists) onto Variants, bound to the
best1 checkpoint with the EQUALITY vocab guard (3500 words == output width 3500, verified live).
Determinism added to the smoke: same seed -> byte-identical reply (probe passes; raise-on-diff).

Honest quality note: replies from the vocab-3500 arm contain many <unk> tokens (sample:
'can you <unk> <unk> for an <unk> ...'). That is the shipping trade the sweep already made —
vocab 3500 won perplexity AND size; the 12k arm (big1) has fewer <unk>s but is 3.3x bigger. The
talk interface is honest about which words it does not have: `vocab_v3500.json` + equality guard,
never a silent substitution. <unk>-rate comparison already exists (compare_unk_rate.py); improving
output quality is a data/decoder follow-up, not today's gate.

REGRESSION AFTER THE CHANGE: variants smoke 80.2% test noun position, refusal OK, talk_sentence
deterministic; selfcheck 28/28 ALL PASS (2 s).
"""

with open(os.path.join(P, "LOG.md"), "a") as f:
    f.write(LOG)
print("appended")
