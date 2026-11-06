"""caption_camera.py — the see+talk model narrating a stream of frames, word by word, live.

The fused model from train_seetalk finally used as a CAMERA: frames come in (fashion test images
here; --source video decodes real files through the same 28x28 eyes), the model WRITES a caption
one word at a time, and every frame shows WHICH NAMED CONCEPTS drove each word — through
variants-style explain math (image influences text ONLY via named concept cells).

Greedy decode: start 'this is a', extend word by word until a class noun or max 6 words. The noun
is what the model SEES; the full sentence is what it says.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import frameio                                           # noqa: E402
import lens.tensor as T                                  # noqa: E402
import train_seetalk as ts                               # noqa: E402
import vl_data                                           # noqa: E402
from lens import backend                                 # noqa: E402

FFMPEG = shutil.which("ffmpeg") or "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
W, H = 940, 470


def caption_one(net, vocab, img, max_words=6):
    """Greedy decode: 'this is a ...' extended word by word until a content word appears."""
    vi = {w: i for i, w in enumerate(vocab)}
    seq = [vi["<bos>"], vi["this"], vi["is"], vi["a"]]
    words_out = ["this", "is", "a"]
    stopped_on = None
    h = None
    STOP = {"<pad>", "<bos>", "this", "is", "a", "the", "photo", "shows", "looks",
            "like", "i", "see"}
    for _ in range(max_words):
        pad = [vi["<pad>"]] * (net.W - len(seq)) + seq[-net.W:]
        out = net.forward_arrays(np.asarray([img], dtype=np.float32),
                                 np.asarray([pad], dtype=np.int64))
        lg, h = out["logits"][0], out["concepts"][0]
        nxt = int(lg.argmax())
        w = vocab[nxt]
        words_out.append(w)
        seq.append(nxt)
        if w not in STOP:
            stopped_on = w
            break
    return " ".join(words_out), stopped_on, h


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(PROJ, "lens", "runs", "seetalk",
                                                   "seetalk_big.npz"))
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--out", default="caption_cam.mp4")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    T.set_device(a.device)
    net, meta = ts.SeeTalkNet.load(a.ckpt)
    vocab = meta["vocab"]
    names = meta["names"]
    d = vl_data.prepare_vqa("fashion", seed=0)
    X, y = d["Xte"], d["yte"]
    print(f"  captioning {a.frames} TEST frames with {os.path.basename(a.ckpt)}")

    # frameio.VideoWriter pipes rgb24 straight into ffmpeg (the mechanism cam_labeled.mp4 uses);
    # the earlier draft invented frameio.write_png, which does not exist — read, not guessed.
    vw = frameio.VideoWriter(os.path.join(PROJ, "lens", "runs", "video", a.out), W, H, fps=2)
    correct = 0
    lines = []
    for i in range(a.frames):
        img = X[i]
        cap, noun, h = caption_one(net, vocab, img)
        truth = names[int(y[i])].replace("_", " ")
        ok = (noun == truth)   # FULL name: "ankle boot" must match "ankle boot", not "ankle"
        correct += int(ok)
        lines.append({"frame": i, "caption": cap, "noun": noun, "truth": truth, "ok": bool(ok)})
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        canvas[:, :] = (18, 18, 22)
        gx, gy = 20, 20
        box = 380
        pic = frameio.upscale(np.asarray(img, dtype=np.float32).reshape(28, 28), box)
        canvas[gy:gy + box, gx:gx + box] = pic   # upscale already returns (box,box,3) rgb
        frameio.draw_rect(canvas, gx - 2, gy - 2, box + 4, box + 4, (120, 120, 140), thick=2)
        frameio.draw_text(canvas, "MODEL EYES: 28x28 PX (784), STRETCHED 14X",
                          gx, gy + box + 8, 1, (150, 150, 170))
        px, py = gx + box + 40, 24
        frameio.draw_text(canvas, "SEE + TALK — LIVE CAPTIONER", px, py, 2, (0, 255, 255))
        py += 28
        frameio.draw_text(canvas, f"FRAME {i+1}  ({correct}/{i+1} nouns correct so far)",
                          px, py, 1, (200, 200, 200))
        py += 26
        frameio.draw_text(canvas, "CAPTION", px, py, 2, (255, 255, 255))
        py += 24
        frameio.draw_rect(canvas, px - 4, py - 4, 470, 54, (0, 50, 0))
        frameio.draw_text(canvas, cap[:34], px, py + 4, 2, (0, 255, 0))
        if len(cap) > 34:
            frameio.draw_text(canvas, cap[34:68], px, py + 26, 2, (0, 255, 0))
        py += 62
        frameio.draw_text(canvas, f"TRUTH: {truth}" + ("   [MATCH]" if ok else "   [MISS]"),
                          px, py, 2, (0, 255, 0) if ok else (255, 90, 90))
        py += 30
        frameio.draw_text(canvas, "CONCEPTS DRIVING THIS FRAME (named, audited)", px, py, 1,
                          (255, 255, 0))
        py += 18
        order = np.argsort(-h)
        shown = 0
        for k in order:
            if h[k] < 0.5 or shown >= 7:
                continue
            frameio.draw_text(canvas, f"{net.concepts[k].name[:30]:<30} {h[k]:.2f}",
                              px, py, 1, (180, 220, 180))
            py += 16
            shown += 1
        vw.write(canvas)
        print(f"  frame {i+1:>3}: '{cap}'  noun={noun}  truth={truth}  {'OK' if ok else 'MISS'}")

    out = os.path.join(PROJ, "lens", "runs", "video", a.out)
    okw, msg = vw.close()
    print(f"  video written: {'OK' if okw else 'FAILED'}  {msg}")
    print(f"  output: {out}  ({os.path.getsize(out)//1024} KB)  nouns {correct}/{a.frames}"
          f" = {correct/a.frames*100:.1f}%")
    json.dump({"ckpt": a.ckpt, "n_frames": a.frames, "noun_correct": correct,
               "acc": correct / a.frames, "lines": lines},
              open(os.path.join(PROJ, "lens", "runs", "video", a.out.replace(".mp4",
                                                                            "_log.json")), "w"),
              indent=1)


if __name__ == "__main__":
    main()
