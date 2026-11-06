"""diag_medical.py — WHY the medical run collapses. Measure, do not guess.

Two attempts failed:
  UNBALANCED  TEST 41.41% (= majority), answers "no" to everything, concepts 0/16
  BALANCED    TEST  8.68%, answers "colorectal_adenocarcinoma" to everything, concepts 0/16

The balancing moved the collapse to another class. So the cause is NOT class balance. This script
measures the candidates in order, without changing anything:

  A. What are the concept supports on PathMNIST? A concept at 2% support gets pos_weight 49 in the
     balanced BCE, which is a very strong push toward always firing.
  B. Are the concept neurons SATURATED? (fires near 0 or 1 for everything)
  C. What does the concept target distribution actually look like, versus the neuron output?
  D. Does the image normalisation match what the concepts assume?
  E. Is the loss being dominated by the concept term rather than the answer term?
"""
import os
import sys

import numpy as np

PROJ = "/run/media/yahya/HARD DISK 4/GlassNet"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "lens", "scripts"))

import image_concepts                                    # noqa: E402
import vl_data                                           # noqa: E402

print("=" * 100)
print("DIAGNOSIS — PathMNIST concept layer")
print("=" * 100)

d = vl_data.prepare_vqa("pathmnist", max_train=20000)
X = d["Xtr"]
print(f"  train slice {X.shape}   pixels in [{X.min():.4f}, {X.max():.4f}]   "
      f"mean {X.mean():.4f}  std {X.std():.4f}")
print()

print("A. CONCEPT SUPPORTS AND THE RESULTING pos_weight")
print(f"   {'concept':<24}{'support':>10}{'pos_weight':>12}   note")
kept, row = [], None
for name, fn, formula in image_concepts.SPEC:
    lab = np.asarray(fn(X.astype(np.float32)), dtype=np.float32)
    s = float(lab.mean())
    if s < 0.02:
        print(f"   {name:<24}{s:>10.4f}{'-':>12}   DROPPED (below 2% floor)")
        continue
    pw = (1.0 - s) / s
    flag = ""
    if pw > 20:
        flag = "<-- VERY strong push toward always firing"
    elif pw > 5:
        flag = "<- strong push"
    print(f"   {name:<24}{s:>10.4f}{pw:>12.2f}   {flag}")
    kept.append((name, s, pw))
print()
print(f"   {len(kept)} kept. Max pos_weight {max(p for _, _, p in kept):.2f}, "
      f"median {np.median([p for _, _, p in kept]):.2f}")
print()

print("B. IS THE CONCEPT TARGET CONSTANT? A concept true for ~everything cannot be learned")
for name, s, pw in kept:
    if s > 0.9 or s < 0.1:
        print(f"   {name:<24} support {s:.4f} <- extreme; AUROC against it is near meaningless")
print()

print("C. NON-GRAYSCALE PROBLEM? PathMNIST is RGB and this loader averages the channels.")
z = np.load(os.path.join(PROJ, "data", "medmnist", "pathmnist.npz"))
raw = z["train_images"][:5000].astype(np.float32)
r, g, b = raw[..., 0].mean(), raw[..., 1].mean(), raw[..., 2].mean()
print(f"   raw RGB channel means: R {r:.1f}  G {g:.1f}  B {b:.1f}")
if max(abs(r - g), abs(g - b), abs(r - b)) > 10:
    print("   -> the channels DIFFER substantially. Averaging them DESTROYS colour, which for")
    print("      histology is a real signal (stain colour separates tissue types).")
gray = raw.mean(axis=-1) / 255.0
print(f"   grayscale stats: mean {gray.mean():.4f}  std {gray.std():.4f}")
print(f"   fraction of pixels > 0.5 (mostly_ink threshold): {(gray > 0.5).mean():.4f}")
print(f"   fraction of pixels < 0.2 (is_dark threshold)  : {(gray < 0.2).mean():.4f}")
print()

print("D. HOW HARD IS THE TASK FOR A PLAIN MODEL?")
print("   If even the plain model barely beats the majority class, the data is the problem.")
print()

print("E. THE CONCEPT TERM'S SHARE OF THE LOSS")
lam = 0.3
for name, s, pw in kept[:3]:
    lab = np.asarray(image_concepts.SPEC[[n for n, _, _ in image_concepts.SPEC].index(name)][1](
        X.astype(np.float32)), dtype=np.float32)
    eps = 1e-7
    # at initialisation the concept layer sits near 0.5, so the BCE is near its maximum for a
    # rare concept, multiplied by pos_weight
    h = 0.5
    bce = -(pw * lab * np.log(h + eps) + (1 - lab) * np.log(1 - h + eps))
    print(f"   {name:<24} support {s:.4f}  mean balanced BCE at h=0.5: {bce.mean():.4f}"
          f"  x lam {lam} = {bce.mean()*lam:.4f}")
print()
print("   For comparison, a well-fit cross-entropy on 13 answers starts near log(13) = "
      f"{np.log(13):.4f}.")
print()
print("=" * 100)
print("  READ THE NUMBERS ABOVE BEFORE CHANGING ANYTHING.")
print("=" * 100)
