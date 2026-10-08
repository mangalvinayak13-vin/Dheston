#!/bin/bash
# Pattern-memory tests. Single-process, so they run alongside the walk-forward queue. Each base predictor is tuned on the development years
# (before 2022) and scored once on 2022 onward. Writes data/results/memory_<base>.json and, when all are done, memory_final.json.
#   bash queue2.sh > logs/queue2.log 2>&1 &
cd "$(dirname "$0")" || exit 1
mkdir -p logs
while [ ! -f data/results/hyb_dhj_main.pkl ]; do sleep 30; done
sleep 30
for base in hyb:dhj:main:carry_res smile_scaled; do
  echo "=== $(date '+%H:%M') memory on $base"
  python3 eval_memory.py --base "$base" --grid wide | tail -6
done
python3 - <<'PY'
import json, glob
out = {p.split("memory_")[-1][:-5]: json.load(open(p)) for p in sorted(glob.glob("data/results/memory_*.json")) if not p.endswith("final.json")}
json.dump(out, open("data/results/memory_final.json", "w"), indent=2)
print("memory_final.json written for", list(out))
PY
echo "=== $(date '+%H:%M') queue2 finished"
