#!/bin/bash
# kill_bypass.sh — the E section allocates ~7.8 GB (N x K x V). Stop it.
pkill -f t3_bypass.py
sleep 2
echo "remaining t3_bypass procs:"
ps -eo pid,cmd | grep "[t]3_bypass" | head -5
echo "(empty = stopped)"
free -g | head -2