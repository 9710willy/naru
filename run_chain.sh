#!/bin/bash
cd /Users/willee1/work/scroll
echo "### A: oracle n=12 scroll @ SONNET (does a stronger model close the gap?)"
python3 bench.py --split oracle -n 12 --arms scroll --model claude-sonnet-5 \
  --judge-model claude-haiku-4-5-20251001 --workers 3 --tag v6_sonnet
echo
echo "### B: _s n=20 both arms @ HAIKU (headline, correct accounting)"
python3 bench.py --split s -n 20 --arms full,scroll --workers 3 --max-turns 10 --tag v7_s
echo "### CHAIN DONE"
