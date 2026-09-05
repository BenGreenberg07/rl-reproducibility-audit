#!/bin/bash
# Polls until all 320 PILOT4 result files exist, then runs the full analysis
# pipeline (audit, small-N reliability, power analysis, figures, paper tables)
# for the extended 4-environment dataset. Meant to be launched once after the
# PILOT4 sweep is started and left running in the background.
cd /Users/ben/Documents/newWayne/rl-reproducibility-audit
source rlvenv/bin/activate

TOTAL=320
while true; do
  N=$(ls results/*.npz 2>/dev/null | wc -l | tr -d ' ')
  echo "$(date '+%Y-%m-%d %H:%M:%S') progress: $N/$TOTAL"
  if [ "$N" -ge "$TOTAL" ]; then
    break
  fi
  # also bail out if the sweep process has died without finishing (crash),
  # so this doesn't spin forever
  if ! pgrep -f "sweep.py --preset PILOT4" > /dev/null; then
    echo "WARNING: sweep.py is not running and only $N/$TOTAL results exist. Stopping monitor without running analysis."
    exit 1
  fi
  sleep 120
done

echo "=== Sweep complete at $(date). Running full analysis pipeline for PILOT4. ==="
python analyze.py --preset PILOT4
python smalln_reliability.py --preset PILOT4
python power_analysis.py --preset PILOT4
python figures.py --preset PILOT4
python generate_paper_tables.py --preset PILOT4
echo "=== Analysis pipeline complete at $(date). ==="
