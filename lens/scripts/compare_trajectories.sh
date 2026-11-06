#!/bin/bash
# Compare training trajectories: is the SMALL model on the merged corpus undertrained, plateaued, or
# diverging? Prints the same epochs side by side so the shapes can be read directly.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
unset PYTHONPATH PYTHONHOME
.venv-gpu/bin/python3 -u - <<'PY'
import json, os, glob
runs = ["sb_S0_d256_noattn", "sb_S1_d256_attn", "t11_talk41", "t12_splitfix_d512", "t12_d512_full"]
for r in runs:
    p = f"lens/runs/chat/{r}_log.json"
    if not os.path.exists(p):
        print(f"{r:<24} (no log yet)")
        continue
    j = json.load(open(p))
    h = j.get("history") or []
    if not h:
        print(f"{r:<24} (empty history)")
        continue
    a = j.get("args", {})
    corpus = (a.get("data_dir") or "?").split("/")[-1]
    print(f"{r:<24} corpus={corpus:<18} params={j.get('params')} epochs={len(h)} "
          f"d={a.get('d')} attn={a.get('attention')} lam_concept={a.get('lam_concept')}")
    picks = [1, 8, 24, 48, 72, 96]
    row = {r["epoch"]: r for r in h}
    for e in picks:
        if e in row:
            x = row[e]
            print(f"    ep{e:>3}  train_loss {x['train_loss']:>7.3f}  train_acc {x['train_acc']*100:>5.1f}%"
                  f"  val_ppl {x['val_ppl']:>10.1f}  val_acc {x['val_acc']*100:>5.2f}%  lr {x['lr']:.3f}")
    if len(h) > 96:
        x = h[-1]
        print(f"    ep{x['epoch']:>3}  train_loss {x['train_loss']:>7.3f}  train_acc {x['train_acc']*100:>5.1f}%"
              f"  val_ppl {x['val_ppl']:>10.1f}  val_acc {x['val_acc']*100:>5.2f}%  lr {x['lr']:.3f}")
    print()
PY