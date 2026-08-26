#!/bin/bash
cd /Users/willee1/work/scroll
while pgrep -f "bench.py" >/dev/null; do sleep 20; done
echo "### C: oracle n=12 HAIKU + rubric + identity fix (vs v3 83.3%)"
python3 bench.py --split oracle -n 12 --arms scroll --workers 3 --tag v8_haiku
echo
echo "### D: oracle n=12 SONNET + rubric + identity fix (valid model compare)"
python3 bench.py --split oracle -n 12 --arms scroll --model claude-sonnet-5 \
  --judge-model claude-haiku-4-5-20251001 --workers 3 --tag v9_sonnet
echo "### CHAIN2 DONE"
