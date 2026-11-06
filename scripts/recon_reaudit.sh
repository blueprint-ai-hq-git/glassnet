#!/bin/bash
# recon_reaudit.sh — what can actually be re-audited under a firing floor?
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
echo "=== T2 checkpoints available (runs/t3/ckpt) ==="
ls -1 runs/t3/ckpt 2>/dev/null | wc -l
ls -1 runs/t3/ckpt 2>/dev/null | sed -E 's/_seed[0-9]+\.npz//' | sort -u | head -40
echo
echo "=== a T2 audit row: what fields exist? ==="
/home/yahya/.hermes/hermes-agent/venv/bin/python3 - <<'EOF'
import json, glob, os
p = "runs/t3/audit_K32_lr05_pa_lc10.json"
a = json.load(open(p))
s = [k for k in a if k.startswith("seed")][0]
print("row fields:", sorted(a[s]["rows"][0].keys()))
print("has firing rate?", "firing" in a[s]["rows"][0])
print()
print("=== runs/t2/measure_audit.json ===")
m = json.load(open("runs/t2/measure_audit.json"))
print("keys:", sorted(m.keys())[:15] if isinstance(m, dict) else type(m))
EOF
echo
echo "=== phase2 audit (Fashion-MNIST readable arch) head ==="
head -25 runs/phase2/audit.md 2>/dev/null
echo
echo "=== where is the rung-3 (Banking77) audit? ==="
grep -rln "Banking77\|banking77" --include="*.md" . 2>/dev/null | head -8