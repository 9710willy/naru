#!/usr/bin/env python3
"""Compare replicate runs to establish a noise floor.

An accuracy difference between two arms means nothing until you know how much
a single arm moves when you just run it again. This reports, per arm:

  spread   — max minus min accuracy across replicates
  flips    — questions whose verdict changed between replicates

Flips are the honest measure: two runs can land on the same accuracy while
disagreeing on half the questions.

    python3 noise.py results/v10_*.json results/v13_*.json
"""

import json
import pathlib
import sys
from itertools import combinations


def load(path):
    d = json.loads(pathlib.Path(path).read_text())
    by_arm = {}
    for r in d["rows"]:
        by_arm.setdefault(r["arm"], {})[r["qid"]] = bool(r["correct"])
    return by_arm


def main(paths):
    if len(paths) < 2:
        sys.exit("need at least two result files to measure run-to-run movement")
    runs = [(pathlib.Path(p).stem, load(p)) for p in paths]
    arms = sorted(set().union(*(set(r[1]) for r in runs)))

    print(f"replicates: {len(runs)}")
    for name, _ in runs:
        print(f"  {name}")
    print()

    floors = {}
    for arm in arms:
        present = [(n, r[arm]) for n, r in runs if arm in r]
        if len(present) < 2:
            print(f"{arm}: only one replicate, cannot measure\n")
            continue

        accs = []
        for name, verdicts in present:
            acc = 100 * sum(verdicts.values()) / len(verdicts)
            accs.append(acc)
            print(
                f"  {arm:7} {name:24} {acc:5.1f}%  "
                f"({sum(verdicts.values())}/{len(verdicts)})"
            )

        spread = max(accs) - min(accs)
        max_flips = 0
        for (na, a), (nb, b) in combinations(present, 2):
            shared = set(a) & set(b)
            flipped = sorted(q for q in shared if a[q] != b[q])
            pct = 100 * len(flipped) / len(shared) if shared else 0
            max_flips = max(max_flips, pct)
            print(
                f"  {'':7} {na} vs {nb}: {len(flipped)}/{len(shared)} questions "
                f"flipped ({pct:.0f}%)"
            )
            for q in flipped:
                print(f"  {'':9} ~ {q}")
        # Observed spread is a LOWER bound on the noise, not an estimate of it:
        # flips in opposite directions cancel, so two genuinely unstable runs
        # can report identical accuracy. The flip rate bounds how far a single
        # run could have moved had the flips not cancelled.
        floors[arm] = (spread, max_flips)
        print(
            f"  {arm:7} observed spread {spread:.1f} pts | flip rate "
            f"{max_flips:.0f}% -> up to {max_flips:.1f} pts of movement had "
            f"flips not cancelled\n"
        )

    if floors:
        obs = max(f[0] for f in floors.values())
        pot = max(f[1] for f in floors.values())
        print("=" * 68)
        print(f"observed spread across replicates : {obs:.1f} points")
        print(f"flip-implied movement bound       : {pot:.1f} points")
        print()
        print(f"Use the LARGER. A between-arm gap under {pot:.1f} points is not")
        print("distinguishable from run-to-run instability at this sample size.")
        if len(runs) < 3:
            print(
                f"Only {len(runs)} replicates: the spread is one sample, not a "
                "distribution."
            )
        print("=" * 68)
    return 0


def demo():
    """Self-check on synthetic replicates — no API calls."""
    import tempfile

    d = pathlib.Path(tempfile.mkdtemp())

    def write(name, arm_verdicts):
        rows = [
            {"arm": a, "qid": q, "correct": c}
            for a, qs in arm_verdicts.items()
            for q, c in qs.items()
        ]
        p = d / f"{name}.json"
        p.write_text(json.dumps({"rows": rows}))
        return str(p)

    # identical runs => zero spread, zero flips
    same = {"scroll": {"q1": True, "q2": True, "q3": False, "q4": True}}
    a, b = write("runA", same), write("runB", same)
    assert main([a, b]) == 0

    # two flips in OPPOSITE directions cancel in accuracy but must still show
    r1 = {"scroll": {"q1": True, "q2": False, "q3": True, "q4": True}}
    r2 = {"scroll": {"q1": False, "q2": True, "q3": True, "q4": True}}
    c, e = write("runC", r1), write("runE", r2)
    runs = [("c", load(c)), ("e", load(e))]
    accs = [100 * sum(v["scroll"].values()) / 4 for _, v in runs]
    assert accs[0] == accs[1], "accuracy should be identical here"
    flipped = [q for q in r1["scroll"] if r1["scroll"][q] != r2["scroll"][q]]
    assert len(flipped) == 2, flipped
    print("ok — noise checks passed (equal accuracy still reports 2 flips)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        sys.exit(main(sys.argv[1:]))
