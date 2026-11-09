import os

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
OUT = os.environ.get("GLASSNET_ROOT", ".")
BG, FG, ACC, MUT = "#0e1116", "#e6edf3", "#3fb950", "#8b949e"
plt.rcParams.update({"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,
    "text.color":FG,"axes.labelcolor":FG,"xtick.color":MUT,"ytick.color":MUT,
    "axes.edgecolor":"#30363d","font.size":10})

def finish(fig, name):
    fig.tight_layout(); fig.savefig(OUT+name, dpi=150, facecolor=BG); plt.close(fig)
    print("wrote", name)

# 1) measured results
cfg = ["d=256\n2M win\nbatch 4096","d=256\n2M win\nbatch 1024","d=256\nfull corpus","d=512\nfull corpus"]
val = [17.64, 24.20, 35.26, 37.37]
fig, ax = plt.subplots(figsize=(7,3.6))
bars = ax.bar(cfg, val, color=[MUT,MUT,ACC,ACC], width=.6)
for b,v in zip(bars,val): ax.text(b.get_x()+b.get_width()/2, v+0.7, f"{v:.2f}%", ha="center", color=FG, fontsize=11, weight="bold")
ax.set_ylabel("held-out next-token accuracy"); ax.set_ylim(0,44)
ax.set_title("Measured results - every bar is a real run", color=FG, fontsize=12, weight="bold")
ax.grid(axis="y", color="#21262d"); ax.set_axisbelow(True)
finish(fig, "results.png")

# 2) our optimiser vs plain SGD, matched sweep, 3 seeds (lens/runs/OPT-VS-SGD-TABLE.txt)
fig, ax = plt.subplots(figsize=(6,3.6))
xs = [0,1]
means = [28.59, 27.81]
seeds = [[29.32,29.03,27.43],[27.99,27.68,27.76]]
for i,(m,ss) in enumerate(zip(means,seeds)):
    ax.bar(i, m, color=ACC if i==0 else MUT, width=.5)
    ax.scatter([i]*len(ss), ss, color=FG, zorder=3, s=28)
    ax.text(i, m+0.6, f"{m:.2f}%", ha="center", color=FG, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels(["our optimiser","plain SGD (best setting)"]); ax.set_ylim(0,33)
ax.set_ylabel("held-out accuracy, assistant-masked"); ax.set_title("Our optimiser vs SGD - matched sweep, 3 seeds", color=FG, fontsize=12, weight="bold")
ax.grid(axis="y", color="#21262d"); ax.set_axisbelow(True)
finish(fig, "optimiser_vs_sgd.png")

# 3) per-layer cost
names = ["embedding","position","compute step x32","concepts","answer","direct read","loss"]
ms    = [0.0053,0.0021,0.0027,0.0008,0.0273,0.0239,0.0825]
# the compute core runs once per position, 32 positions, so its per-token cost is 32 x 0.0027 = 0.0864.
# An earlier chart added it once and published 0.1446 ms; counted correctly the stack is 0.2283 ms.
TOTAL_MS = round(sum(ms[:2]) + ms[2]*32 + sum(ms[3:]), 4)
ms = [ms[0], ms[1], round(ms[2]*32, 4)] + ms[3:]
params= [3072512,8192,131328,16448,780130,3072512,0]
fig, ax = plt.subplots(figsize=(7,3.6))
ax.barh(names, ms, color=ACC)
for i,v in enumerate(ms): ax.text(v+0.001, i, f"{v:.4f} ms", va="center", color=FG, fontsize=9)
ax.set_xlabel("ms per token (d=256, window 32, float32)"); ax.invert_yaxis()
ax.set_title(f"Where the time goes - {TOTAL_MS:.4f} ms/token, core counted 32x", color=FG, fontsize=12, weight="bold")
ax.grid(axis="x", color="#21262d"); ax.set_axisbelow(True)
finish(fig, "layer_cost.png")

# 4) data is the only lever that moved held-out
d = ["2k windows","20k","200k","2M"]
v = [4.7,11.5,19.5,27.0]
fig, ax = plt.subplots(figsize=(6,3.6))
ax.plot(d, v, marker="o", color=ACC, linewidth=2.2, markersize=7)
for x,y in zip(d,v): ax.annotate(f"{y:.1f}%", (x,y), textcoords="offset points", xytext=(0,10), ha="center", color=FG)
ax.set_ylabel("held-out accuracy"); ax.set_ylim(0,32)
ax.set_title("Seven other levers failed; more distinct data did not", color=FG, fontsize=12, weight="bold")
ax.grid(color="#21262d"); ax.set_axisbelow(True)
finish(fig, "data_curve.png")

# 5) ablations: what lost and by how much
ab = ["attention\nvs none","our architecture\nvs plain","carry path\n(arith.)"]
delta = [-5.41,-1.73,-4.0]
fig, ax = plt.subplots(figsize=(6,3.2))
ax.bar(ab, delta, color="#f85149", width=.55)
for i,v in enumerate(delta): ax.text(i, v-0.45, f"{v:+.2f} pts", ha="center", color=FG, weight="bold")
ax.axhline(0, color=MUT, linewidth=1); ax.set_ylim(-7,1.2)
ax.set_ylabel("change in held-out accuracy"); ax.set_title("Published negative results", color=FG, fontsize=12, weight="bold")
ax.grid(axis="y", color="#21262d"); ax.set_axisbelow(True)
finish(fig, "ablations.png")

# 6) architecture diagram
fig, ax = plt.subplots(figsize=(8,4.4)); ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis("off")
boxes = [("tokens\n32-token window",1.15,"#1f6feb"),("embedding\n12,002 x 256\ntied to answer",3.5,"#238636"),
         ("concepts\nnamed neurons\naudited",5.9,"#8957e5"),("answer\n+ direct mix\n12,002 classes",8.1,"#bf8700")]
for label,x,c in boxes:
    ax.add_patch(FancyBboxPatch((x-0.85,2.4),1.7,1.5, boxstyle="round,pad=0.12", linewidth=2, edgecolor=c, facecolor="#161b22"))
    ax.text(x,3.15,label,ha="center",va="center",fontsize=10,color=FG)
for (_,x1,_),(_,x2,_) in zip(boxes,boxes[1:]):
    ax.add_patch(FancyArrowPatch((x1+0.9,3.15),(x2-0.9,3.15), arrowstyle="-|>", mutation_scale=18, color=MUT, linewidth=1.6))
ax.text(5,5.3,"GlassNet: four inspectable stages, no attention, no hidden path", ha="center", fontsize=13, weight="bold", color=FG)
ax.text(5,1.6,"5,908,686 parameters (d=256, K=36)   |   0.2283 ms per token   |   float32", ha="center", fontsize=10, color=MUT)
finish(fig, "architecture.png")
