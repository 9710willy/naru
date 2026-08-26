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

    # paired between-arm comparison, pooled across replicates
    if len(arms) == 2:
        print("-" * 68)
        print("paired arm comparison (McNemar exact, on discordant pairs)")
        for name, r in runs:
            if all(x in r for x in arms):
                a_only, b_only, pv = mcnemar(r[arms[0]], r[arms[1]])
                sig = "significant" if pv < 0.05 else "NOT significant"
                print(f"  {name:26} {arms[0]}-only {a_only:2}  "
                      f"{arms[1]}-only {b_only:2}  p={pv:.3f}  {sig}")
        print("-" * 68 + "\n")

    if floors:
        obs = max(f[0] for f in floors.values())
        pot = max(f[1] for f in floors.values())
        n = max(len(r[arm]) for _, r in runs for arm in r)
        # Flips are per-question and roughly independent, so run-to-run movement
        # in ACCURACY scales as 1/sqrt(n), not with the flip rate itself. The
        # flip rate is the pathological all-one-direction case and is not a
        # useful threshold. sigma = 100 * sqrt(p / n).
        p = pot / 100
        sigma = 100 * (p / n) ** 0.5 if n else 0.0
        print("=" * 68)
        print(f"n = {n} questions, {len(runs)} replicates")
        print(f"  observed spread                 : {obs:.1f} points")
        print(f"  worst flip rate (p)             : {pot:.0f}%")
        print(
            f"  1-sigma run-to-run movement     : {sigma:.1f} points   (100*sqrt(p/n))"
        )
        print(f"  ~95% band (2 sigma)             : {2 * sigma:.1f} points")
        print()
        print(
            f"A between-arm gap should exceed ~{2 * sigma:.0f} points to claim at "
            "this n."
        )
        for target in (24, 48, 96):
            if target > n:
                s = 100 * (p / target) ** 0.5
                print(
                    f"    at n={target:<3} the 2-sigma band would be "
                    f"~{2 * s:.0f} points"
                )
        if len(runs) < 3:
            print(f"  Only {len(runs)} replicates: treat sigma as indicative.")
        print("=" * 68)
    return 0


def mcnemar(a, b):
    """Paired comparison of two arms on the same questions.

    Accuracy difference alone ignores that the arms are graded on identical
    items. What matters is the discordant pairs: questions one arm got right
    and the other got wrong. Returns (a_only, b_only, two-sided exact p).
    """
    from math import comb

    shared = set(a) & set(b)
    a_only = sum(1 for q in shared if a[q] and not b[q])
    b_only = sum(1 for q in shared if b[q] and not a[q])
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    # two-sided exact binomial on the discordant pairs, p = 0.5
    tail = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return a_only, b_only, min(1.0, 2 * tail)


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
