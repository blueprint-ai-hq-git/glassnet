"""combo_matrix.py — the FULL combination test: every subset of the four modalities
(text, image, audio, video) answered through the shipped joint runtime on REAL files.
Reports each combination's answer, the exact concept activations, and any refusal.

The point (owner's standing demand): the joint model works in EVERY scenario combination,
not just each line alone. A failure here is a bug, named.
"""
import json
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants as V          # noqa: E402
import video_data             # noqa: E402
from lens import audio as A   # noqa: E402

FILES_IMG = [f for f in ("/home/yahya/cam_frame13.png",) if os.path.exists(f)]
FILES_AUD = [f for f in ("/home/yahya/amar_slang_p1.ogg", "/home/yahya/status_update.ogg")
             if os.path.exists(f)]
FILES_VID = [f for f in ("/home/yahya/cam_fashion.mp4", "/home/yahya/xcool_edit.mp4")
             if os.path.exists(f)]
TEXT = "Human: hello , how are you ?\n\nAssistant:"


def main():
    os.makedirs(os.path.join(PROJ, "lens", "runs", "combos"), exist_ok=True)
    vr = V.build_default()  # see / see_talk / talk ships
    # NEWEST BY MTIME (the project's documented rule -- a name-sort picked a different
    # experiment's artifact on the first matrix run; wrong-subject loads are the rot class)
    def _newest(d):
        fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
        return max(fs, key=os.path.getmtime) if fs else None

    av = _newest(os.path.join(PROJ, "lens", "runs", "audio"))
    vv = _newest(os.path.join(PROJ, "lens", "runs", "video"))
    if av:
        net_a = A.load(av, device="cpu")
        names_a = [c.name for c in getattr(net_a, "concepts", [])]
        print(f"  audio ship: {os.path.basename(av)}  concepts {names_a}")
        if names_a and names_a != ["is_silence", "has_voiced"]:
            print(f"  WARNING: the newest audio artifact is NOT the shipped 2-concept line "
                  f"({names_a}) -- named, not hidden")
        vr.models["audio"] = (net_a, None)
    if vv:
        net_v = A.load(vv, device="cpu")
        names_v = [c.name for c in getattr(net_v, "concepts", [])]
        print(f"  video ship: {os.path.basename(vv)}  concepts {names_v}")
        vr.models["video"] = (net_v, None)

    have = {"text": True, "image": bool(FILES_IMG), "audio": bool(FILES_AUD),
            "video": bool(FILES_VID)}
    missing = [k for k in have if not have[k]]
    for k in missing:
        print(f"  MISSING real file for {k} -> that column is skipped (named, not hidden)")

    subs = [["text"], ["image"], ["audio"], ["video"]]
    subs += [["text", "image"], ["text", "audio"], ["text", "video"],
             ["image", "audio"], ["image", "video"], ["audio", "video"]]
    subs += [["text", "image", "audio"], ["text", "image", "video"],
             ["text", "audio", "video"], ["image", "audio", "video"]]
    subs += [["text", "image", "audio", "video"]]

    rows = []
    for sub in subs:
        sub = [s for s in sub if have[s]]
        if len(sub) < 1:
            continue
        try:
            r = vr.react(
                text_window=TEXT if "text" in sub else None,
                image=FILES_IMG[0] if "image" in sub else None,
                audio_in=FILES_AUD[0] if "audio" in sub else None,
                video_in=FILES_VID[0] if "video" in sub else None)
            mods = sorted(r.keys()) if isinstance(r, dict) else None
            ok = isinstance(r, dict)
            dtl = {k: (v if isinstance(v, str) else str(v)[:70]) for k, v in
                   (r.items() if isinstance(r, dict) else [])}
        except Exception as e:
            ok = False
            dtl = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        rows.append({"combo": "+".join(sub), "ok": ok, "detail": dtl})
        print(f"\n  COMBO [{'+'.join(sub):<27}] -> {'OK' if ok else 'FAIL'}")
        for k, v in dtl.items():
            print(f"      {k}: {v}")

    n_ok = sum(1 for r in rows if r["ok"])
    out = {"n_combos": len(rows), "n_ok": n_ok, "missing_modalities": missing, "rows": rows}
    json.dump(out, open(os.path.join(PROJ, "lens", "runs", "combos", "combo_matrix.json"),
                        "w"), indent=1)
    print(f"\n  COMBO MATRIX: {n_ok}/{len(rows)} combinations answered.")
    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())