#!/bin/bash
# run_build_handoff.sh — build GLASSNET-HANDOFF.md + GLASSNET-HANDOFF.zip for the reviewer.
# The ZIP carries docs + code + result JSONs + logs. It deliberately EXCLUDES the
# checkpoints (87 MB) because every one of them is reproducible from the scripts.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
PY="/home/yahya/.hermes/hermes-agent/venv/bin/python3"
[ -x "$PY" ] || PY=python3
DEST=/home/yahya

echo "=== 1. build the handoff document ==="
"$PY" scripts/build_handoff.py || exit 1

echo
echo "=== 2. verify the document ==="
"$PY" - <<'EOF'
import hashlib, re
p = "/run/media/yahya/HARD DISK 4/GlassNet/GLASSNET-HANDOFF.md"
d = open(p).read()
checks = [
    ("fences even", d.count("```") % 2 == 0),
    ("no empty code blocks", len(re.findall(r"```[a-z]*\n```", d)) == 0),
    ("records A/A2/A3", "NOT MET" in d and "MET" in d),
    ("says parity is not a win", "PARITY IS NOT A WIN" in d),
    ("carries A3 numbers", "80.0%" in d and "0/4" in d),
    ("carries the A2 diagnosis", "33.3%" in d),
    ("says taught-not-discovered", "TAUGHT, NOT DISCOVERED" in d),
    ("carries the readability finding", "for such a one as" in d),
    ("carries the B prediction", "TARGET B WILL NOT BE MET" in d),
    ("states the no-license rule", "Do not choose or write a license" in d),
    ("states test-set rule", "Test set at most once per milestone" in d),
]
ok = True
for n, p_ in checks:
    print(f"  {'OK  ' if p_ else 'FAIL'} {n}")
    ok = ok and p_
print(f"  sha256[:16] = {hashlib.sha256(d.encode()).hexdigest()[:16]}")
print(f"  lines = {len(d.splitlines())}")
raise SystemExit(0 if ok else 1)
EOF
[ $? -ne 0 ] && { echo "!!! verification failed"; exit 1; }

echo
echo "=== 3. build the bundle (docs + code + results, NO checkpoints) ==="
ZIP="$DEST/GLASSNET-HANDOFF.zip"
rm -f "$ZIP"
zip -q -r "$ZIP" \
  *.md \
  scripts/*.py scripts/*.sh \
  glassnet/*.py \
  runs/t3/*.json runs/t3/*.log runs/phase2/*.md runs/phase2/*.json \
  -x "runs/t3/ckpt/*" -x "runs/t3/ckpt_preB/*" -x "*.pyc" -x "__pycache__/*" \
  2>/dev/null
[ -f "$ZIP" ] || { echo "!!! zip failed"; exit 1; }
echo "  files in bundle : $(unzip -l "$ZIP" | tail -1 | awk '{print $2}')"
echo "  size            : $(du -h "$ZIP" | cut -f1)"
echo "  checkpoints excluded? $(unzip -l "$ZIP" | grep -c ckpt) ckpt entries (expect 0)"
echo
echo "  contents (top level):"
unzip -l "$ZIP" | awk 'NR>3 && NF>=4 {print "    " $4}' | head -20

echo
echo "=== 4. copy the doc next to the bundle ==="
cp -f GLASSNET-HANDOFF.md "$DEST/GLASSNET-HANDOFF.md" || exit 1
cmp -s GLASSNET-HANDOFF.md "$DEST/GLASSNET-HANDOFF.md" \
  && echo "  copy byte-identical" || { echo "!!! copy differs"; exit 1; }
ls -l "$DEST/GLASSNET-HANDOFF.md" "$ZIP"