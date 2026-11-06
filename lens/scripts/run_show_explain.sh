#!/bin/bash
# run_show_explain.sh — show the improved explain(), including its honesty footer.
#
# G1's remaining gap was that the explanation did not state its own limits: it printed a word list
# without saying how much of the neuron that list covers. The footer added to concepts.py fixes
# that. This prints the result on the winning wide model so it can be judged, not just asserted.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1

nice -n 19 ionice -c3 .venv-gpu/bin/python3 - <<'PY'
import sys, json
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet/scripts")
import numpy as np
import lens
from lens.concepts import concept_word_list
from glassnet import tiny_data

DATA = tiny_data.build(win=8, vocab_cap=1500)
VOCAB, Xva = DATA["vocab"], np.asarray(DATA["Xva"])

for tag, path in [("WIDE 41-concept model (the winner)", "lens/runs/opt1_wide60/readable.npz"),
                  ("9-concept model (the original)", "lens/runs/blocker/readable.npz")]:
    print("=" * 100)
    print(tag)
    print("=" * 100)
    net = lens.load(path, device="cpu")
    print(lens.explain_text(net, Xva[7], vocab=VOCAB, top_k=3))
    print()

    print("  --- the word list, WITH its coverage number ---")
    for k in range(min(3, net.K)):
        terms, frac = concept_word_list(net, k, top=40)
        body = "  ".join(f"{v:+.2f}*{VOCAB[i] if i < len(VOCAB) else 'unk'}" for i, v in terms[:6])
        print(f"  {net.concepts[k].name:<24} {body}")
        print(f"  {'':<24} top-40 covers {frac*100:.1f}% of this neuron; "
              f"{(1-frac)*100:.1f}% is NOT in any word list")
    print()
PY
