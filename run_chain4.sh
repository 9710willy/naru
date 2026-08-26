#!/bin/bash
cd /Users/willee1/work/scroll
echo "### F: rubric ABLATION - oracle n=12 haiku, --no-rubric (isolates v8's regression)"
python3 bench.py --split oracle -n 12 --arms scroll --workers 3 --no-rubric --tag v11_norubric
echo
echo "### G: _s n=24 both arms @ HAIKU, all fixes (does the headline hold at 2x n, cheap model?)"
python3 bench.py --split s -n 24 --arms full,scroll --workers 3 --max-turns 10 --tag v12_s_n24
echo "### CHAIN4 DONE"
