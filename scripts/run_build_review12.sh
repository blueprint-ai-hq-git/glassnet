#!/bin/bash
# run_build_review12.sh — build GLASSNET-REVIEW-12.md, verify it, copy to /home/yahya/.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY=python3
DEST="/home/yahya/GLASSNET-REVIEW-12.md"

echo "=== building REVIEW-12 ==="
"$PY" scripts/build_review12.py || exit 1

echo
echo "=== verifying ==="
"$PY" - <<'EOF'
import hashlib, re
p = "/run/media/yahya/HARD DISK 4/GlassNet/GLASSNET-REVIEW-12.md"
d = open(p).read()
checks = [
    ("fences even", d.count("```") % 2 == 0),
    ("no empty code blocks", len(re.findall(r"```[a-z]*\n```", d)) == 0),
    ("carries STEP 0", "STEP 0" in d and "for such a one as" in d),
    ("carries the you, case", "you," in d and "KING_What_you,_RICHARD" in d),
    ("carries the B rule", "2x pooled std" in d or "2x pooled" in d),
    ("carries the grid", "80 runs" in d),
    ("carries the prediction", "will NOT be met" in d),
    ("carries both queued items", "NAME PRECISION" in d and "MEANINGFUL TARGETS" in d),
    ("carries the 5 ideas", "Multi-word construction targets" in d and "Speaker-name targets" in d),
    ("says parity is never a win", "never a win" in d),
    ("has real code", d.count("sha256[:16]") >= 4),
    ("records A/A2/A3", "NOT MET" in d and "A3 = MET" in d),
]
ok = True
for n, p_ in checks:
    print(f"  {'OK  ' if p_ else 'FAIL'} {n}")
    ok = ok and p_
print(f"  sha256[:16] = {hashlib.sha256(d.encode()).hexdigest()[:16]}")
print(f"  lines = {len(d.splitlines())}, bytes = {len(d.encode())}")
raise SystemExit(0 if ok else 1)
EOF
[ $? -ne 0 ] && { echo "!!! verification failed"; exit 1; }

cp -f GLASSNET-REVIEW-12.md "$DEST" || exit 1
cmp -s GLASSNET-REVIEW-12.md "$DEST" && echo "  copy byte-identical" \
  || { echo "!!! copy differs"; exit 1; }
ls -l "$DEST"