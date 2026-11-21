#!/bin/bash
# Runs every phase repeatedly while the dataset is being extended; after DATASET_FINAL exists, one last full pass + robustness.
cd "$(dirname "$0")"
F='^llama_\|^ggml\|^load\|^print_info\|^\s*$\|^clip_\|^mtmd\|^init_\|^alloc\|^warmup\|^main:'
echo "=== START $(date -u +%H:%M:%S)" >> progress.log
[ -f results/reuse_check.json ] || { python3 check_reuse.py 2>/dev/null | grep -v "$F" >> run_check.log; echo "=== DONE check_reuse $(date -u +%H:%M:%S)" >> progress.log; }
while true; do
  FINAL=0; [ -f DATASET_FINAL ] && FINAL=1
  python3 build_v2.py >> build.log 2>&1 || { echo "=== BUILD FAILED $(date -u +%H:%M:%S)" >> progress.log; sleep 120; continue; }
  for ph in j1_text j1_mm j1_hol j2 gen j1_gen j2_gen pair; do
    python3 run2.py $ph 2>/dev/null | grep -v "$F" >> run_$ph.log
    echo "=== DONE $ph final=$FINAL $(date -u +%H:%M:%S)" >> progress.log
    if [ $FINAL -eq 0 ] && [ -f DATASET_FINAL ]; then break; fi
  done
  if [ $FINAL -eq 1 ]; then break; fi
done
python3 run2.py rob 2>/dev/null | grep -v "$F" >> run_rob.log; echo "=== DONE rob $(date -u +%H:%M:%S)" >> progress.log
echo "CHAINDONE $(date -u +%H:%M:%S)" >> progress.log
