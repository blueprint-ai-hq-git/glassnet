#!/bin/bash
# run_build_review11.sh — build GLASSNET-REVIEW-11.md, verify it, copy to /home/yahya/.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/Hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY=python3
DEST="/home/yahya/GLASSNET-REVIEW-11.md"

echo "=== building REVIEW-11 ==="
"$PY" scripts/build_review11.py || exit 1

echo
echo "=== verifying ==="
"$PY" - <<'EOF'
import hashlib, re
p = "/run/media/yahya/HARD DISK 4/GlassNet/GLASSNET-REVIEW-11.md"
d = open(p).read()
checks = [
    ("fences even", d.count("```") % 2 == 0),
    ("no empty code blocks", len(re.findall(r"```[a-z]*\n```", d)) == 0),
    ("has the verdict", "TARGET A3 = MET" in d or "A3-1 taught concepts pass" in d),
    ("states taught-not-discovered", "TAUGHT, NOT DISCOVERED" in d),
    ("states the parity caveat", "Parity is never a win" in d),
    ("carries the hold-out", "0/4" in d),
    ("has real code", d.count("sha256[:16]") >= 4),
    ("bedtime note present", "going to bed" in d and "no reply is needed" in d),
    ("A and A2 still NOT MET", "NOT MET" in d),
]
ok = True
for name, passed in checks:
    print(f"  {'OK  ' if passed else 'FAIL'} {name}")
    ok = ok and passed
print(f"  sha256[:16] = {hashlib.sha256(d.encode()).hexdigest()[:16]}")
print(f"  lines = {len(d.splitlines())}, bytes = {len(d.encode())}")
raise SystemExit(0 if ok else 1)
EOF
[ $? -ne 0 ] && { echo "!!! verification failed"; exit 1; }

cp -f GLASSNET-REVIEW-11.md "$DEST" || exit 1
cmp -s GLASSNET-REVIEW-11.md "$DEST" && echo "  copy to $DEST byte-identical" \
  || { echo "!!! copy differs"; exit 1; }
ls -l "$DEST"