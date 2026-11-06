#!/bin/bash
# Runtime proof for the speech line: drive the SAME runtime object that talks/sees/listens and ask
# it to recognize spoken words, with the model's own explanation. Real WAV files, no synthetic input.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import glob
import sys
sys.path.insert(0, "/run/media/yahya/HARD DISK 4/GlassNet")
from lens.scripts.variants import Variants

P = "/run/media/yahya/HARD DISK 4/GlassNet"
rt = Variants(speech=P + "/lens/runs/speech/speech_readable_s0_drop-noisy.npz", device="cpu")
hits = 0
for w in ("yes", "no", "stop", "go", "up", "down", "left", "right"):
    f = sorted(glob.glob(f"{P}/data/external/speech_commands_raw/extracted/{w}/*.wav"))[0]
    r = rt.recognize_word(f, why=False)
    hits += int(r["word"] == w)
    print(f"  clip {w:5} -> predicted '{r['word']:5}' p={r['probability']:.3f}  "
          f"concepts {r['concepts']}")
print(f"  one clip per word, hit {hits}/8 (this is a smoke, not an accuracy measurement -- "
      f"the measured number is 37.43% on 6,243 official-val clips)")
print("  full explanation for one clip:")
print("   ", rt.recognize_word(sorted(glob.glob(P + '/data/external/speech_commands_raw/extracted/go/*.wav'))[1])["why"].replace("\n", "\n    "))
PY