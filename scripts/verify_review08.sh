#!/bin/bash
# verify_review08.sh — structure check before sending.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
F=GLASSNET-REVIEW-08.md
echo "=== sections ==="
grep -nE "^#{1,3} " "$F"
echo
echo "=== code fences (must be even) ==="
grep -c '^```' "$F"
echo
echo "=== empty code blocks (must be 0) ==="
/home/yahya/.hermes/hermes-agent/venv/bin/python3 - <<'EOF'
lines = open("GLASSNET-REVIEW-08.md").read().splitlines()
inb = False; n = 0; empty = 0
for i, l in enumerate(lines):
    if l.startswith("```"):
        if inb:
            if n <= 1:
                empty += 1; print("  EMPTY block ending line", i + 1)
            inb = False; n = 0
        else:
            inb = True; n = 0
    elif inb:
        n += 1
print("  empty blocks:", empty)
EOF
echo
echo "=== key strings present? ==="
for s in "min_firing" "below_floor" "caught_by_floor_only" "LIVE PROBE" "firing_rate" "_fired_matrix" "PPMI" "k-means"; do
  printf "  %-22s %s\n" "$s" "$(grep -c "$s" "$F")"
done
echo
echo "=== copy to home and verify identity ==="
cp "$F" /home/yahya/GLASSNET-REVIEW-08.md
sha256sum "$F" /home/yahya/GLASSNET-REVIEW-08.md | awk '{print $1}' | uniq -c
ls -la /home/yahya/GLASSNET-REVIEW-08.md | awk '{print $5, $9}'