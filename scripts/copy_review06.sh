#!/bin/bash
# copy_review06.sh — copy the review doc to the home dir for Telegram delivery.
SRC="/run/media/yahya/HARD DISK 4/GlassNet/GLASSNET-REVIEW-06.md"
DST="/home/yahya/GLASSNET-REVIEW-06.md"
cp "$SRC" "$DST"
ls -la "$DST"
echo "--- sha256 match check ---"
sha256sum "$SRC" "$DST" | awk '{print $1}' | sort -u | wc -l
echo "(1 = identical)"
echo "--- line count ---"
wc -l "$DST"