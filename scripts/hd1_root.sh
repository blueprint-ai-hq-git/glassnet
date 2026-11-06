#!/bin/bash
# hd1_root.sh — runs as ROOT via pkexec. Standard NTFS dirty-flag recovery for
# HARD DISK 1 (/dev/sdd2), following the storage-management skill's protocol:
#   ntfsfix -> mountpoint -> ntfs3 force mount -> permanent fstab entry.
# Non-destructive: fstab is backed up first; ntfsfix only empties $LogFile and
# resets flags, it does not touch user data.
set -u
DEV=/dev/sdd2
MP="/run/media/yahya/HARD DISK 1"

echo "=== 0. back up fstab ==="
cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d-%H%M%S)"
ls -1 /etc/fstab.bak.* | tail -2

echo "=== 1. ntfsfix (safe: clears the journal, resets flags) ==="
if command -v ntfsfix >/dev/null 2>&1; then
  ntfsfix "$DEV" 2>&1 | tail -8
else
  echo "ntfsfix NOT INSTALLED - skipping (not required for the force mount)"
fi

echo "=== 2. mountpoint ==="
mkdir -p "$MP"
chmod 777 "$MP"
ls -ld "$MP"

echo "=== 3. mount with force (udisks cannot pass force; direct mount can) ==="
if mountpoint -q "$MP"; then
  echo "already mounted"
else
  mount -t ntfs3 -o force,uid=1000,gid=1000 "$DEV" "$MP" && echo "MOUNT OK" || echo "MOUNT FAILED"
fi

echo "=== 4. permanent fstab entry (same pattern as HARD DISK 3 and 4) ==="
# Spaces in the mountpoint MUST be escaped as \040 or mount gives a parse error.
# printf's format string eats backslashes, so \\040 emits the literal \040.
UUID="$(blkid -s UUID -o value "$DEV")"
if [ -n "$UUID" ] && ! grep -q "$UUID" /etc/fstab; then
  printf 'UUID=%s /run/media/yahya/HARD\\040DISK\\0401 ntfs3 force,nofail,uid=1000,gid=1000 0 0\n' "$UUID" >> /etc/fstab
  echo "fstab entry added for UUID $UUID"
else
  echo "fstab already had this UUID (or UUID empty)"
fi
systemctl daemon-reload 2>/dev/null || true

echo "=== 5. verify ==="
lsblk -o NAME,LABEL,FSTYPE,MOUNTPOINT | grep -E "sdd[0-9]"
echo "--- fstab tail ---"
tail -3 /etc/fstab
echo "--- contents of the drive ---"
ls "$MP" 2>&1 | head -12
echo "--- df ---"
df -h "$MP" 2>&1 | tail -1