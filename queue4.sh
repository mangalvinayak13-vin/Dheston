#!/bin/bash
# Re-sequenced follow-up runs. Waits for the Double Heston (no jumps) walk-forward that queue3 started, then the memory-prior runs (each paired
# with a plain-shrinkage control: same strength, prior = the previous window's estimate), then Heston, then the ablations.
#   bash queue4.sh > logs/queue4.log 2>&1 &
cd "$(dirname "$0")" || exit 1
while pgrep -f "run_walk.py --model dh --tag main" > /dev/null; do sleep 30; done
sleep 10

walk() {  # walk <model> <tag> [extra run_walk args]
  m=$1; tag=$2; shift 2
  echo "=== $(date '+%H:%M') walk $m $tag $*"
  python3 run_walk.py --model "$m" --tag "$tag" --workers 8 --chunks 12 --burn 40 "$@" | tail -4
  python3 hybrids.py --model "$m" --tag "$tag" | tail -12
}

python3 hybrids.py --model dh --tag main | tail -12
walk dhj mp10 --memory-prior 10
walk dhj p10 --prior 10
walk dhj mp30 --memory-prior 30
walk dhj mq05 --memory-prior 10 --mem-q 0.05 --mem-L 5
walk heston main
walk dhj w80 --window 80
walk dhj r20 --refit 20
walk dhj w20 --window 20
walk dhj r5 --refit 5
walk dhj p5 --prior 5
walk dhj p20 --prior 20
walk dhj w120 --window 120
echo "=== $(date '+%H:%M') queue4 finished"
