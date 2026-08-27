#!/usr/bin/env python3
"""Re-grade saved runs with the current judge.

Isolates the judge's contribution to run-to-run flips: the answers are frozen,
so any change in verdicts comes from the judge alone. Writes <tag>_regraded.json
next to the input so noise.py can compare regraded replicates.

    python3 regrade.py results/v10_*.json results/v13_*.json
"""

import json
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backend import HAIKU, get_backend
from bench import judge, load


def main(paths, model=HAIKU, workers=6):
    if not paths:
        sys.exit("usage: regrade.py results/*.json")

    # question text is not stored in results rows; recover it from the dataset
    questions = {}
    for split in ("oracle", "s"):
        try:
            for q in load(split):
                questions[q["question_id"]] = q["question"]
        except SystemExit:
            pass

    out_paths = []
    for p in paths:
        p = pathlib.Path(p)
        d = json.loads(p.read_text())
        rows = d["rows"]

        def one(r):
            be = get_backend(model)
            q = {"question": questions.get(r["qid"], ""), "answer": r["gold"]}
            new = judge(q, r.get("answer") or "", be)
            return r, new, be.usage.cost_usd

        changed, cost = 0, 0.0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for r, new, c in ex.map(one, rows):
                cost += c
                if bool(r["correct"]) != new:
                    changed += 1
                r["correct"] = new

        outp = p.with_name(p.stem + "_regraded.json")
        outp.write_text(json.dumps(d, indent=1))
        out_paths.append(str(outp))
        for arm in sorted({r["arm"] for r in rows}):
            a = [r for r in rows if r["arm"] == arm]
            print(
                f"  {p.stem:26} {arm:7} {100 * sum(r['correct'] for r in a) / len(a):5.1f}%"
                f"  ({sum(r['correct'] for r in a)}/{len(a)})"
            )
        print(
            f"  {'':26} {changed} verdict(s) changed vs the original grading "
            f"| ${cost:.2f}\n"
        )

    print("regraded files:")
    for o in out_paths:
        print(f"  {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
