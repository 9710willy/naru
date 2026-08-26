#!/bin/bash
cd /Users/willee1/work/scroll
echo "### E: _s n=12 both arms @ SONNET (clean headline: no flake, no leak)"
python3 bench.py --split s -n 12 --arms full,scroll --model claude-sonnet-5 \
  --judge-model claude-haiku-4-5-20251001 --workers 3 --max-turns 10 --tag v10_s_sonnet
echo "### CHAIN3 DONE"
