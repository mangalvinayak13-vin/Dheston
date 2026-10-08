#!/bin/bash
# Keep COMPARISON_1.0_vs_2.0.md current: regenerate it every 20 minutes (a line is added to COMPARISON_LOG.md whenever the results change).
#   nohup bash watch_comparison.sh > logs/watch_comparison.log 2>&1 &
cd "$(dirname "$0")" || exit 1
while true; do
  python3 update_comparison.py > /dev/null 2>> logs/watch_comparison.err
  sleep 1200
done
