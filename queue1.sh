#!/bin/bash
# Follow-up walk-forward runs, one after another (one heavy job at a time: 4 fast + 6 slow cores). Waits for the main run first.
#   bash queue1.sh > logs/queue1.log 2>&1 &
cd "$(dirname "$0")" || exit 1
mkdir -p logs
while [ ! -f data/results/hyb_dhj_main.pkl ]; do sleep 30; done
sleep 20

walk() {  # walk <model> <tag> [extra run_walk args]
  m=$1; tag=$2; shift 2
  echo "=== $(date '+%H:%M') walk $m $tag $*"
  python3 run_walk.py --model "$m" --tag "$tag" --workers 8 --chunks 12 --burn 40 "$@" | tail -4
  python3 hybrids.py --model "$m" --tag "$tag" | tail -12
}

# the jump SIZE for the jump-rate-as-state model is fixed from the development years only
python3 fit_jump_shape.py --tag main
walk dhjs main
walk dh main
walk heston main
walk dhj w80 --window 80
walk dhj r20 --refit 20
walk dhj w20 --window 20
walk dhj r5 --refit 5
walk dhj p5 --prior 5
walk dhj p20 --prior 20
walk dhj w120 --window 120
echo "=== $(date '+%H:%M') queue1 finished"
