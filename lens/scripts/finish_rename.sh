#!/bin/bash
# finish_rename.sh — move the real files into lens/ with explicit paths.
# The previous attempt used a guard (`if [ ! -d lens ]`) that my own script defeated by
# being written into lens/ first. Explicit moves cannot fail that way.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1

mkdir -p lens/csrc lens/scripts lens/tests
mv -v glasscore/csrc/glasscore.c   lens/csrc/kern.c
mv -v glasscore/csrc/Makefile      lens/csrc/Makefile
mv -v glasscore/_capi.py           lens/_capi.py
mv -v glasscore/scripts/build_glasscore.sh lens/scripts/build_lens.sh
mv -v glasscore/libglasscore.so    lens/liblens_old.so     # kept, not deleted

rmdir glasscore/csrc glasscore/scripts glasscore 2>/dev/null

# fix the names the files reference
sed -i 's/glasscore\.c/kern.c/g; s/libglasscore\.so/liblens.so/g; s/glasscore/lens/g' \
  lens/csrc/Makefile lens/scripts/build_lens.sh lens/_capi.py
sed -i 's/lens\/csrc\/kern\.c/kern.c/g' lens/csrc/Makefile

echo "=== layout ==="
find lens -type f | sort
echo
echo "=== build ==="
bash lens/scripts/build_lens.sh 2>&1 | tail -6