#!/bin/bash
# Re-sequenced follow-up runs (replaces the tail of queue1.sh): waits for the jump-rate-as-state walk-forward that queue1 had started,
# then runs the wide-bounds test first (it matters for the 2020 crash), then the model comparison, then the ablations.
#   bash queue3.sh > logs/queue3.log 2>&1 &
cd "$(dirname "$0")" || exit 1
while pgrep -f "run_walk.py --model dhjs --tag main" > /dev/null; do sleep 30; done
sleep 10

walk() {  # walk <model> <tag> [extra run_walk args]
  m=$1; tag=$2; shift 2
  echo "=== $(date '+%H:%M') walk $m $tag $*"
  python3 run_walk.py --model "$m" --tag "$tag" --workers 8 --chunks 12 --burn 40 "$@" | tail -4
  python3 hybrids.py --model "$m" --tag "$tag" | tail -12
}

python3 hybrids.py --model dhjs --tag main | tail -12
echo "=== $(date '+%H:%M') wide bounds"
DH2_WIDE=1 python3 run_walk.py --model dhj --tag wb --workers 8 --chunks 12 --burn 40 | tail -4
DH2_WIDE=1 python3 hybrids.py --model dhj --tag wb | tail -12
walk dh main
walk heston main
walk dhj w80 --window 80
walk dhj r20 --refit 20
walk dhj w20 --window 20
walk dhj r5 --refit 5
walk dhj p5 --prior 5
walk dhj p20 --prior 20
walk dhj w120 --window 120
echo "=== $(date '+%H:%M') queue3 finished"
