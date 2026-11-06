#!/bin/bash
# recon_rungs2.sh — do rung 1 (Fashion-MNIST) and rung 3 (Banking77) have a
# concept meter at all, and are their models/checkpoints still on disk?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== loop audit.md format (iter001, iter014, iter018) ==="
for f in runs/loop/iter001/audit.md runs/loop/iter014/audit.md runs/loop/iter018/audit.md; do
  echo "--- $f"
  head -8 "$f" 2>/dev/null
  echo
done
echo "=== does any loop audit mention a concept meter (AUROC per neuron)? ==="
grep -lE "AUROC|auroc" runs/loop/*/audit.md 2>/dev/null | head
echo
echo "=== which rung does each loop iter belong to? ==="
grep -hoE "rung ?[0-9]|Rung ?[0-9]|Fashion-MNIST|PathMNIST|Banking77" runs/loop/*/hypothesis.md 2>/dev/null | sort | uniq -c | sort -rn | head
echo
echo "=== phase2 (Fashion-MNIST readable) artifacts ==="
ls -1 runs/phase2/ | head -20
echo
echo "=== any Fashion-MNIST / phase2 model checkpoints? ==="
find runs/phase2 runs/loop -maxdepth 2 \( -name "*.npz" -o -name "*.npy" \) 2>/dev/null | head -12