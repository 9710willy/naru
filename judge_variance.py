#!/usr/bin/env python3
"""Separate judge variance from generation variance.

A flipped verdict between two benchmark runs has two possible causes:
  generation — the agent produced a different answer
  judging    — the SAME answer text was graded differently

Only the second is a harness defect. This re-judges answers already saved in a
results file, N times each, so the judge's own instability is measured directly
instead of inferred.

    python3 judge_variance.py results/v10_*.json --repeats 5
"""

import argparse
import json
import pathlib
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backend import HAIKU, Backend
from bench import judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--judge-model", default=HAIKU)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    rows = []
    for p in a.results:
        d = json.loads(pathlib.Path(p).read_text())
        for r in d["rows"]:
            if str(r.get("answer", "")).strip():
                rows.append(r)
    # de-dup identical (question, answer) pairs across files
    seen, uniq = set(), []
    for r in rows:
        key = (r["qid"], r["arm"], str(r["answer"])[:200])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    print(f"re-judging {len(uniq)} saved answers x{a.repeats} ({a.judge_model})\n")

    def one(r):
        q = {"question": "", "answer": r["gold"]}
        # the saved rows carry gold + answer; the question text is not saved,
        # so grade on gold-vs-candidate, which is what JUDGE_SYSTEM compares
        be = Backend(model=a.judge_model)
        votes = [judge(q, r["answer"], be) for _ in range(a.repeats)]
        return r, votes, be.usage.cost_usd

    unstable, cost = [], 0.0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r, votes, c in ex.map(one, uniq):
            cost += c
            c1 = Counter(votes)
            if len(c1) > 1:
                unstable.append((r, votes))
                mark = "UNSTABLE"
            else:
                mark = "stable  "
            print(
                f"  {mark} {r['arm']:6} {r['qid'][:14]:14} "
                f"{'/'.join('T' if v else 'F' for v in votes)}  "
                f"gold={str(r['gold'])[:38]}"
            )

    n = len(uniq)
    print()
    print("=" * 68)
    print(
        f"judge instability: {len(unstable)}/{n} answers "
        f"({100 * len(unstable) / max(n, 1):.0f}%) got different verdicts on "
        f"IDENTICAL text"
    )
    print(f"judge cost: ${cost:.2f}")
    if unstable:
        print("\nthe unstable ones:")
        for r, votes in unstable:
            print(f"  {r['qid'][:14]:14} {sum(votes)}/{len(votes)} correct")
            print(f"    gold: {str(r['gold'])[:90]}")
            print(f"    ans : {str(r['answer'])[:120]}")
    print("=" * 68)
    print("Any nonzero instability here is harness noise, not model behaviour.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
