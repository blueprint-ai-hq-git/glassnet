"""g2_check_concept_set.py — WHICH concept set did the G2 sweep actually train on?

CONCEPT_TARGETS is a DICT (7 keys), not a list, so "7 targets" was a misread. The dict carries
n_taught and words. The real question is which set the loader chose.

run_t2_iter003._load_concept_targets defaults to T2_CONCEPT_SET="words" (single-word targets).
Target E set T2_CONCEPT_SET=meaningful_ext explicitly. If the G2 runner did not, the model trained
on a DIFFERENT concept set than the audit compared against -- which produces exactly the near-chance
AUROC that G2 reported.
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import numpy as np  # noqa: E402

for k in ("T2_CONCEPT_SET", "T2_N_TAUGHT"):
    os.environ.pop(k, None)          # clear whatever the shell had
os.environ["T3M_IMPORT_ONLY"] = "1"
os.environ["T2_LAM_CONCEPT"] = "0.1"

from glassnet.tensor import set_backend   # noqa: E402
import run_t2_iter003 as M                # noqa: E402

set_backend("numpy")

print("=" * 92)
print("WHICH CONCEPT SET DOES THE LOADER PICK WHEN T2_CONCEPT_SET IS UNSET? (what G2 did)")
print("=" * 92)
ct = M.CONCEPT_TARGETS
print(f"type            : {type(ct).__name__}   (a DICT -- 'len' gives keys, not targets)")
print(f"keys            : {sorted(ct)}")
print(f"kind chosen     : {ct['kind']!r}")
print(f"n_taught        : {ct['n_taught']}")
print(f"n_targets avail : {ct['n_targets']}")
print(f"words           : {ct['words'][:6]}{' ...' if len(ct['words']) > 6 else ''}")

print()
print("Target E used T2_CONCEPT_SET=meaningful_ext and trained on the 9 idea targets:")
print("  pronoun_subject, is_a_dialogue_line, pronoun_object, is_a_negation,")
print("  is_archaic_address, contains_a_wh_word, mentions_speech, mentions_giving, mentions_knowing")

if ct["kind"] == "words":
    print()
    print("VERDICT: the G2 sweep trained on the SINGLE-WORD set ('words'), NOT the 9 idea targets")
    print("         that its audit compared against. Different sets, so columns do not correspond")
    print("         and every concept scores near chance. THE G2 SWEEP IS VOID.")
else:
    print(f"\nVERDICT: kind is {ct['kind']!r} -- investigate further.")
