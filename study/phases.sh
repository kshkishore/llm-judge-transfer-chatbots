#!/bin/bash
# Runs the named run2.py phases in order (each phase resumes from its log), e.g. ./phases.sh j1_hol gen
cd "$(dirname "$0")"
F='^llama_\|^ggml\|^load\|^print_info\|^\s*$\|^clip_\|^mtmd\|^init_\|^alloc\|^warmup\|^main:'
for ph in "$@"; do
  echo "=== START $ph $(date -u +%H:%M:%S)" >> progress.log
  python3 run2.py $ph 2>/dev/null | grep -v "$F" >> run_$ph.log
  echo "=== DONE $ph $(date -u +%H:%M:%S)" >> progress.log
done
echo "BATCHDONE $* $(date -u +%H:%M:%S)" >> progress.log
