"""virtual_camera.py — the LIVE VIDEO pipeline. Frames in, the model's own answer drawn on them.

WHAT THIS IS
------------
A virtual camera: a stream of frames is fed through the trained vision model one at a time, and the
model's output is drawn back onto each frame -- its answer, its confidence, and which NAMED concepts
fired. The result is written as a real .mp4.

NO PHYSICAL CAMERA IS TOUCHED. The frames come from a source the script chooses:
    --source synthetic   moving shapes generated in code (no data needed)
    --source fashion     Fashion-MNIST test images scrolling past (MIT)
    --source pathmnist   PathMNIST medical images (CC BY 4.0)
    --source video PATH  an existing video file, decoded with ffmpeg

This is the "live video / virtual camera" half of the brief, and it is the honest version of it: the
model sees frames it has never seen, in order, and every label is its own.

WHY THE OVERLAY IS READABLE
---------------------------
It draws only what the model actually produced: the answer word, its probability, and the top
firing concepts WITH their formulas available in the log. Nothing is invented for display.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "scripts"))

import frameio                                            # noqa: E402
import lens.tensor as T                                   # noqa: E402
import vl_data                                            # noqa: E402
from lens import backend as B                             # noqa: E402
from lens.vision import load_vision, VLNet, check_vocab_match   # noqa: E402

OUT = os.path.join(PROJ, "lens", "runs", "video")
os.makedirs(OUT, exist_ok=True)

# LAYOUT — the first version put a 560x560 image inside a 480-tall canvas and raised
# "could not broadcast input array from shape (560,560,3) into shape (460,560,3)". The canvas is
# now sized from its parts so the two cannot disagree: image box + panel + margins.
IMG_BOX = 420                     # the picture is drawn at IMG_BOX x IMG_BOX
PAD = 480                         # right-hand panel with the labels
M = 20                            # margin
FY = IMG_BOX + 2 * M              # canvas height
FX = IMG_BOX + PAD + 2 * M        # canvas width


def synthetic_frames(n, seed=0):
    """Moving shapes: no dataset needed, so the pipeline can be tested before any data exists.

    Each frame has a thin bright bar that moves; the model's job is to say what it sees. This is
    deliberately NOT from the training distribution -- it shows what the model does on input it has
    never seen, which is the honest thing to show for a "live camera".
    """
    rng = np.random.default_rng(seed)
    for i in range(n):
        img = np.zeros((28, 28), dtype=np.float32)
        x = int(3 + (i % 22))
        y = int(3 + (i * 3) % 22)
        img[y:y + 2, 3:25] = 0.9                      # horizontal bar
        if i % 2:
            img[3:25, x:x + 2] = 0.7                  # vertical bar
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        yield np.clip(img, 0, 1).reshape(-1), None


def dataset_frames(which, n, seed=0, task="vqa1", color=False):
    """Cycle through the QUESTION TEMPLATES, one per frame.

    The first version fed `Qtr[0]` on every frame -- the same "what is in the photo" question for the
    whole clip. That is not a fair demonstration of a VL model, and worse, it HID a defect: the
    frame showed `asks_name` firing at 0.89 on a "what" question, and with only one question in the
    clip there was no way to see whether that concept tracks its name at all. Cycling the questions
    makes both the demo and the defect visible.

    `task` matters: each VL task has its own question set AND its own answer vocabulary (vqa2 adds
    `yes`/`no`). Feeding a vqa2 model vqa1 data put the model's 14 answer classes against a 12-word
    vocabulary -- the mismatch that produced `IndexError`, and that would have printed the wrong
    class name silently if the sizes had happened to agree.

    Yields (image, (true_answer_id, question_text, question_ids)). The true answer is the model's own
    answer token, so the comparison is always against the same vocabulary the model was trained on.
    """
    if task == "vqa2":
        import vl_data2
        base = vl_data.prepare_vqa(which, max_train=None, seed=seed, color=color)
        if base is None:
            raise SystemExit(f"data '{which}' unavailable")
        d = vl_data2.prepare_vqa2(base, max_train=None, seed=seed)
    elif task == "vqa3":
        import vl_data3
        base = vl_data.prepare_vqa(which, max_train=None, seed=seed, color=color)
        if base is None:
            raise SystemExit(f"data '{which}' unavailable")
        d = vl_data3.prepare(base, "medical" if which != "fashion" else "fashion", seed=seed)
    else:
        d = vl_data.prepare_vqa(which, max_train=None, seed=seed, color=color)
        if d is None:
            raise SystemExit(f"data '{which}' unavailable")
    X, Q, A = d["Xte"], d["Qte"], d["Ate"]
    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    for i in idx:
        qtext = " ".join(d["qvocab"][t] for t in Q[i] if t != 0)
        yield X[i], (int(np.asarray(A[i]).reshape(-1)[0]), qtext, Q[i])
    return d


def video_frames(path, n, size=28):
    """Decode a real video with ffmpeg into greyscale frames. No image library needed."""
    import shutil
    ff = shutil.which("ffmpeg") or "/home/yahya/.hermes/tools/ffmpeg-9.0.1-linux-x64/bin/ffmpeg"
    cmd = [ff, "-loglevel", "error", "-i", path, "-vf", f"scale={size}:{size}",
           "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = size * size
    count = 0
    while count < n:
        buf = p.stdout.read(frame_bytes)
        if len(buf) < frame_bytes:
            break
        a = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) / 255.0
        yield a, None
        count += 1
    p.stdout.close()
    p.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint from train_vl.py")
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "fashion", "pathmnist", "video"])
    ap.add_argument("--video", default="")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="camera1.mp4",
                    help="MUST end in .mp4 -- without an extension ffmpeg cannot infer the container, "
                         "the pipe dies, and the failure appears as BrokenPipeError at write time")
    ap.add_argument("--task", default="vqa1", choices=["vqa1", "vqa2", "vqa3"],
                    help="which VL task the model was trained on. REQUIRED to match, because each "
                         "task has its own question set and answer vocabulary")
    ap.add_argument("--color", action="store_true")
    ap.add_argument("--device", default="gpu")
    a = ap.parse_args()

    print("=" * 100)
    print("VIRTUAL CAMERA — live frames through the model, its own labels drawn back on")
    print("=" * 100)
    T.set_device(a.device)
    net, meta = load_vision(a.model, device=a.device)
    print(f"  model   : {os.path.basename(a.model)}  kind={meta['kind']}  "
          f"concepts={len(meta['concepts'])}  d={meta['D']}")
    print(f"  source  : {a.source}")
    print(f"  output  : {os.path.join(OUT, a.out)}  {FX}x{FY} @ {a.fps}fps")

    # THE DATA PREPARATION MUST MATCH THE TASK THE MODEL WAS TRAINED ON.
    # vqa2/vqa3 have different question sets AND different answer vocabularies (vqa2 adds yes/no).
    if a.task == "vqa2":
        import vl_data2
        base = vl_data.prepare_vqa("fashion", max_train=200, color=a.color)
        d = vl_data2.prepare_vqa2(base, max_train=200)
    elif a.task == "vqa3":
        import vl_data3
        base = vl_data.prepare_vqa("fashion", max_train=200, color=a.color)
        d = vl_data3.prepare(base, "fashion", seed=0)
    else:
        d = vl_data.prepare_vqa("fashion", max_train=200, color=a.color)
    # THE CHECKPOINT'S OWN VOCABULARIES, NEVER THE DATA'S.
    # The first version used `d["avocab"]`, rebuilt from Fashion-MNIST (12 words). The vqa2 model has
    # 14 answer classes, so index 13 was out of range -- IndexError while drawing a frame. Had the
    # sizes happened to agree, the model would have printed the WRONG CLASS NAME with no error at all.
    # check_vocab_match raises on any disagreement and returns the checkpoint's words to use.
    avocab, qv = check_vocab_match(meta, d.get("avocab"), d.get("qvocab"))
    concepts = [c["name"] for c in meta["concepts"]]
    print(f"  answers : {len(avocab)} {avocab[:6]}...")
    print()

    if a.source == "synthetic":
        src = synthetic_frames(a.frames)
    elif a.source == "video":
        src = video_frames(a.video, a.frames)
    else:
        src = dataset_frames(a.source, a.frames, task=a.task, color=a.color)

    vw = frameio.VideoWriter(os.path.join(OUT, a.out), FX, FY, fps=a.fps)
    rows = []
    correct = 0
    total = 0
    for i, item in enumerate(src):
        img, truth = item
        canvas = np.zeros((FY, FX, 3), dtype=np.uint8)
        canvas[:, :] = (18, 18, 22)

        # ---- the picture, drawn from the SAME array the model is about to see
        gx, gy = M, M
        pic = frameio.upscale(np.asarray(img, dtype=np.float32).reshape(28, 28), IMG_BOX)
        canvas[gy:gy + IMG_BOX, gx:gx + IMG_BOX] = pic
        frameio.draw_rect(canvas, gx - 2, gy - 2, IMG_BOX + 4, IMG_BOX + 4, (120, 120, 140), thick=2)
        # THE BLOCKINESS IS THE MODEL'S EYES, NOT A RENDERING BUG: label it honestly. The model's
        # front end is 28x28 = 784 pixels stretched ~15x; no code can show detail the model cannot
        # see. (The real fix — higher-resolution eyes — is PLAN step 6 and needs the fusion
        # redesign, because 16 patches currently matches the 16-word window.)
        frameio.draw_text(canvas, "MODEL EYES: 28x28 px (784), stretched 15x",
                          gx, gy + IMG_BOX + 8, 1, (150, 150, 170))

        # ---- the model sees that frame, with the frame's OWN question
        q = np.asarray([truth[2] if truth is not None else d["Qtr"][0]], dtype=np.int64)
        out = net.forward_arrays(np.asarray([img], dtype=np.float32), q)
        logits = out["logits"][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        pred = int(probs.argmax())
        cf = float(probs[pred])
        h = out["concepts"][0]

        px = M + IMG_BOX + 30
        py = M
        frameio.draw_text(canvas, "VIRTUAL CAMERA", px, py, 3, (0, 255, 255))
        py += 34
        frameio.draw_text(canvas, f"FRAME {i + 1}  LIVE", px, py, 2, (200, 200, 200))
        py += 30
        frameio.draw_text(canvas, "MODEL SAYS", px, py, 2, (255, 255, 255))
        py += 22
        frameio.draw_rect(canvas, px - 4, py - 4, 430, 46, (0, 60, 0))
        frameio.draw_text(canvas, avocab[pred], px, py, 4, (0, 255, 0))
        py += 56
        frameio.draw_text(canvas, f"CONFIDENCE {cf * 100:5.1F}%", px, py, 2, (255, 255, 255))
        py += 22
        frameio.draw_bar(canvas, px, py, 420, 16, cf, (0, 255, 0))
        py += 30

        if truth is not None:
            # compare ANSWER IDS, not strings: ids are what the weights index, and string comparison
            # hid the vocabulary mismatch until it surfaced as a crash while drawing a frame.
            tid, q_text = int(truth[0]), truth[1]
            tname = avocab[tid] if 0 <= tid < len(avocab) else f"<id {tid}>"
            ok = (tid == pred)
            correct += int(ok)
            total += 1
            frameio.draw_text(canvas, "TRUTH", px, py, 2, (255, 255, 255))
            py += 22
            frameio.draw_text(canvas, tname.replace(" ", "_"), px, py, 2,
                              (0, 255, 0) if ok else (255, 90, 90))
            py += 26
            frameio.draw_text(canvas, "Q " + q_text[:30], px, py, 1, (170, 170, 255))
            py += 22

        frameio.draw_text(canvas, "NAMED CONCEPTS FIRING", px, py, 2, (255, 255, 0))
        py += 22
        order = np.argsort(-h)
        shown = 0
        for k in order:
            if h[k] < 0.5 or shown >= 6:
                continue
            frameio.draw_text(canvas, f"{concepts[k][:24]} {h[k]:.2f}", px, py, 1,
                              (255, 255, 150))
            py += 15
            shown += 1
        if shown == 0:
            frameio.draw_text(canvas, "NONE OVER 0.5", px, py, 1, (160, 160, 160))

        vw.write(canvas)
        rows.append({"frame": i, "answer": avocab[pred], "conf": cf,
                     "top_concepts": [[concepts[k], float(h[k])] for k in order[:6]]})
        if (i + 1) % 25 == 0:
            print(f"    frame {i + 1}/{a.frames}  last answer {avocab[pred]} "
                  f"({cf * 100:.1f}%)")

    ok, msg = vw.close()
    print()
    print("=" * 100)
    print(f"  video written: {'OK' if ok else 'FAILED'}  {msg}")
    if total:
        print(f"  frames with a known answer: {total}   correct: {correct}  "
              f"({correct / total * 100:.1f}%)")
    json.dump({"source": a.source, "model": a.model, "frames": rows,
               "correct": correct, "total": total},
              open(os.path.join(OUT, a.out.replace(".mp4", "_log.json")), "w"), indent=1)
    print("=" * 100)


if __name__ == "__main__":
    main()
