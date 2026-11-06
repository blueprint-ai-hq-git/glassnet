#!/bin/bash
# rename_to_lens.sh — rename the rebuild from glasscore/ to lens/.
#
# Naming rationale (owner's instruction: names must be useful, memorable, stable, easy):
#   lens  -- one syllable, universally known, and it MEANS something for this project:
#            a lens is glass (transparency) and a lens is for seeing (explain()).
#            PyTorch's name works the same way: short, known, meaningful.
#   Parts stay plain and obvious, exactly as torch.nn / torch.optim do:
#            lens/kern.c, lens/cuda/, lens/tensor.py, lens/concepts.py
# Nothing is deleted: this is code written minutes ago. The OLD engine (glassnet/) and every
# prior run stay exactly where they are as reference.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1

if [ -d glasscore ] && [ ! -d lens ]; then
  mv glasscore lens
  echo "renamed glasscore/ -> lens/"
fi

# the build script and Makefile reference the old output name
sed -i 's/libglasscore\.so/liblens.so/g' lens/csrc/Makefile lens/scripts/build_glasscore.sh 2>/dev/null
sed -i 's/glasscore/lens/g' lens/_capi.py lens/scripts/build_glasscore.sh 2>/dev/null
mv lens/scripts/build_glasscore.sh lens/scripts/build_lens.sh 2>/dev/null

echo "=== new layout ==="
find lens -type f | sort
echo
echo "=== rebuild under the new name ==="
bash lens/scripts/build_lens.sh 2>&1 | tail -6