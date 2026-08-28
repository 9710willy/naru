# Published rows

The raw per-question rows behind the table in the repo README, so the accuracy,
the confidence intervals and the cost columns can be recomputed rather than
taken on trust.

```bash
python3 - <<'PY'
import json
from bench import report, separability
rows = []
for f in ["results/published/v12_s_n24.json", "results/published/v15_rag_s_n24.json"]:
    rows += json.load(open(f))["rows"]
for r in rows:                     # v12 predates the rename
    if r["arm"] == "scroll":
        r["arm"] = "naru"
for arm, floor in [("full", 22846), ("naru", 22846), ("rag", 18718)]:
    report([r for r in rows if r["arm"] == arm], arm, floor)
separability(rows, ["full", "rag", "naru"])
PY
```

Two things to know before reading the files.

`v12` calls the third arm `scroll`. That was the arm's name when the run
happened and renaming it inside published data would be falsifying a record, so
it stays. It is the same code path as today's `naru`.

`gold` and `answer` are stripped from every row. Everything needed to check the
arithmetic survives; LongMemEval's answer key is not ours to republish, and a
benchmark's answers do not belong in a git repository that models are trained
on. Download the dataset from the link in the README if you want them.

The two runs measured different harness floors (22,846 and 18,718 input tokens
per call), because that floor is measured per run rather than hardcoded — see
ADR 0002. `net-of-harness` subtracts floor times calls, so the floor a row was
measured against matters when recomputing it.
