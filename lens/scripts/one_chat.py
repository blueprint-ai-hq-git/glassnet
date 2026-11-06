"""one_chat.py — ONE chat that answers text, image, audio and video in a single session.

This is the owner's ask made usable: type a sentence and it talks back; type a file path and
the right line answers it (by extension). Every answer is the model's OWN arithmetic --
`react()` unions the shipped lines and each reply carries its exact concept activations.
Nothing is fused or hidden.

Usage
    .venv-gpu/bin/python3 -u lens/scripts/one_chat.py            # interactive
    printf 'hello\\n/home/yahya/cam_fashion.mp4\\n' | python one_chat.py   # scripted
"""
import os
import sys

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import variants as V          # noqa: E402
from lens import audio as A   # noqa: E402


def _newest(d):
    fs = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".npz")]
    return max(fs, key=os.path.getmtime) if fs else None


def build():
    vr = V.build_default()
    av = _newest(os.path.join(PROJ, "lens", "runs", "audio"))
    vv = _newest(os.path.join(PROJ, "lens", "runs", "video"))
    if av:
        vr.models["audio"] = (A.load(av, device="cpu"), None)
    if vv:
        vr.models["video"] = (A.load(vv, device="cpu"), None)
    return vr


def route(line):
    """Which modality is this input? Extension decides for files; text otherwise."""
    low = line.strip().lower()
    if low.endswith((".png", ".jpg", ".jpeg", ".bmp")) and os.path.exists(line.strip()):
        return "image"
    if low.endswith((".ogg", ".wav", ".mp3", ".flac", ".m4a")) and os.path.exists(line.strip()):
        return "audio"
    if low.endswith((".mp4", ".mkv", ".webm", ".mov")) and os.path.exists(line.strip()):
        return "video"
    return "text"


def render(kind, r):
    if kind == "text":
        return f"me: {r.get('reply', '')}"
    if kind == "image":
        return (f"i see: {r.get('answer')} (confidence {r.get('confidence')})"
                f"  concepts: {r.get('fired_concepts')}")
    if kind == "audio":
        return (f"i hear: {r.get('answer')}  concepts: {r.get('fired_concepts')}"
                f"  (max {r.get('max_concept')})")
    if kind == "video":
        return (f"video: {r.get('answer')}  concepts: {r.get('fired_concepts')}"
                f"  (max {r.get('max_concept')})")
    return str(r)


def main():
    vr = build()
    print("one chat ready. capabilities:", sorted(vr.models.keys()))
    print("type a sentence, or a path to a .png / .ogg / .mp4. 'exit' to stop.\n")
    while True:
        try:
            line = input("you: ")
        except EOFError:
            break
        if not line.strip():
            continue
        if line.strip().lower() in ("exit", "quit", "q"):
            break
        kind = route(line)
        try:
            if kind == "text":
                r = vr.talk_sentence(f"Human: {line.strip()}\n\nAssistant:", n_new=16)
            elif kind == "image":
                r = vr.react(image=line.strip())["see"]
            elif kind == "audio":
                r = vr.react(audio_in=line.strip())["audio"]
            else:
                r = vr.react(video_in=line.strip())["video"]
            print("  " + render(kind, r))
        except Exception as e:
            print(f"  [{kind}] refused: {type(e).__name__}: {str(e)[:140]}")
    print("bye")


if __name__ == "__main__":
    main()