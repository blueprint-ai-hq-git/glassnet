#!/bin/bash
# Speech Commands v0.02 from Google's official archive (CC BY 4.0). HF's repo has only a loading
# script, so this is the real source. ~2.4 GB. Runs in background; resume-capable (-C -).
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
mkdir -p data/external/speech_commands_raw
cd data/external/speech_commands_raw
URL=http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
echo "start $(date +%H:%M:%S)"
curl -L -C - --retry 3 --retry-delay 5 -o speech_commands_v0.02.tar.gz "$URL"
echo "curl rc=$?"
ls -la speech_commands_v0.02.tar.gz
sha256sum speech_commands_v0.02.tar.gz | head -1