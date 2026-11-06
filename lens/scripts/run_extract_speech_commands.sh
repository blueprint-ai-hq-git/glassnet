#!/bin/bash
# Extract Speech Commands v0.02 (2.4 GB, CC BY 4.0) and inspect its layout, plus the existing audio
# line's data format so the real corpus can be wired into the same pipeline.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
if [ ! -d data/external/speech_commands_raw/extracted ]; then
  mkdir -p data/external/speech_commands_raw/extracted
  echo "extracting $(date +%H:%M)"
  tar -xzf data/external/speech_commands_raw/speech_commands_v0.02.tar.gz \
      -C data/external/speech_commands_raw/extracted
  echo "extracted $(date +%H:%M)"
fi
echo "=== word folders:"
ls data/external/speech_commands_raw/extracted | head -40
echo "=== counts (top folders):"
for w in yes no up down left right stop go; do
  n=$(ls data/external/speech_commands_raw/extracted/$w 2>/dev/null | wc -l)
  echo "   $w: $n clips"
done
echo "=== total wavs:"; find data/external/speech_commands_raw/extracted -name "*.wav" | wc -l
echo "=== existing audio line data:"
ls -la data/audio/ | head -8
du -sh data/audio 2>/dev/null
echo "=== audio scripts:"; ls lens/scripts/ | grep -i audio | head -6