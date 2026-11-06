#!/bin/bash
# run_build_review10.sh — build GLASSNET-REVIEW-10.md and verify it, then copy it to
# /home/yahya/ for sending (the copy is checked byte-for-byte).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/run/media/yahya/Hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY=python3
DEST="/home/yahya/GLASSNET-REVIEW-10.md"

echo "=== building REVIEW-10 ==="
"$PY" scripts/build_review10.py || exit 1

echo
echo "=== verifying ==="
"$PY" - <<'EOF'
import hashlib, os, re
p = "/run/media/yahya/HARD DISK 4/GlassNet/GLASSNET-REVIEW-10.md"
d = open(p).read()
ok = True
checks = []
f = d.count("```")
checks.append(("fences even", f % 2 == 0, f))
e = len(re.findall(r"```[a-z]*\n```", d))
checks.append(("no empty code blocks", e == 0, e))
checks.append(("has the verdict", "NOT MET" in d, ""))
checks.append(("states the wrong prediction", "NOT CONFIRMED" in d, ""))
checks.append(("has real code", "sha256[:16]" in d, d.count("sha256[:16]")))
checks.append(("carries the hold-out", "0/8" in d or "0 of 8" in d, ""))
checks.append(("carries the diagnosis", "FIRING=16" in d, ""))
checks.append(("Target A unchanged", "NOT MET" in d, ""))
for name, passed, extra in checks:
    print(f"  {'OK  ' if passed else 'FAIL'} {name} {extra}")
    ok = ok and passed
h = hashlib.sha256(d.encode()).hexdigest()[:16]
print(f"  sha256[:16] = {h}")
print(f"  lines = {len(d.splitlines())}, bytes = {len(d.encode())}")
raise SystemExit(0 if ok else 1)
EOF
RC=$?
[ "$RC" -ne 0 ] && { echo "!!! verification failed"; exit 1; }

cp -f GLASSNET-REVIEW-10.md "$DEST" || exit 1
if cmp -s GLASSNET-REVIEW-10.md "$DEST"; then
  echo "  copy to $DEST is byte-identical"
else
  echo "!!! copy differs"; exit 1
fi
ls -l "$DEST"