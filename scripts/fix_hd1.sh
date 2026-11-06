#!/bin/bash
# fix_hd1.sh — wrapper: run the root payload for HARD DISK 1 via pkexec.
# pkexec raises a GUI password prompt on the desktop; that is the expected step.
cd "/run/media/yahya/HARD DISK 4/GlassNet" || exit 1
pkexec bash "/run/media/yahya/HARD DISK 4/GlassNet/scripts/hd1_root.sh" 2>&1
echo "wrapper exit=$?"