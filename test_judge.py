#!/usr/bin/env python3
"""Judge regression tests.

Cases taken from real replicate-run disagreements. The judge marked the second
of two answers wrong when they differed only in punctuation, which is how
harness noise entered reported accuracy.

    python3 test_judge.py            # runs the cases (costs a few cheap calls)
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from backend import HAIKU, Backend
from bench import judge

# (question, gold, candidate, expected)
CASES = [
    # The observed failure: both say "30 days", differ only in punctuation.
    (
        "How many days passed between the Sunday mass at St. Mary's and the Ash "
        "Wednesday service?",
        "30 days. 31 days (including the last day) is also acceptable.",
        "30 days passed between the Sunday mass at St. Mary's Church on January "
        "2nd and the Ash Wednesday service at the same church on February 1st.",
        True,
    ),
    (
        "How many days passed between the Sunday mass at St. Mary's and the Ash "
        "Wednesday service?",
        "30 days. 31 days (including the last day) is also acceptable.",
        "30 days passed between the Sunday mass at St. Mary's Church (January "
        "2nd) and the Ash Wednesday service at the same church (February 1st).",
        True,
    ),
    # the gold's stated alternative must also pass
    (
        "How many days passed?",
        "30 days. 31 days (including the last day) is also acceptable.",
        "31 days.",
        True,
    ),
    # short factoids, formatted differently
    ("What is my cat's name?", "Luna", "Your cat's name is Luna.", True),
    ("How much did each mug cost?", "$12", "Each mug cost 12 dollars.", True),
    (
        "What move came after 27. Kg2 Bd5+?",
        "28. Kg3",
        "After 27. Kg2 Bd5+, you played 28. Kg3.",
        True,
    ),
    ("How many new postcards?", "25", "You've added 25 new postcards.", True),
    # genuinely wrong: different value
    ("How many days passed?", "30 days.", "6 days.", False),
    ("How much did each mug cost?", "$12", "You spent $60 in total.", False),
    # abstention when the fact exists is wrong
    (
        "What is my cat's name?",
        "Luna",
        "I don't have any record of your cat's name.",
        False,
    ),
    # topic without the fact is wrong
    (
        "How long did the asylum application take?",
        "over a year",
        "You mentioned an asylum application in one of our conversations.",
        False,
    ),
    # descriptive gold, correctly paraphrased
    (
        "What should I serve for dinner with my homegrown ingredients?",
        "The user would prefer dinner suggestions that incorporate their "
        "homegrown cherry tomatoes and herbs like basil and mint.",
        "Serve something built around what you've grown — your cherry tomatoes "
        "with basil and mint, like a caprese-style spread.",
        True,
    ),
]


def main():
    be = Backend(model=HAIKU)
    bad = []
    for i, (q, gold, cand, want) in enumerate(CASES, 1):
        got = judge({"question": q, "answer": gold}, cand, be)
        ok = got == want
        if not ok:
            bad.append((i, gold, cand, want, got))
        print(
            f"  {'ok  ' if ok else 'FAIL'} case {i:2}  want="
            f"{'CORRECT' if want else 'WRONG':7} got="
            f"{'CORRECT' if got else 'WRONG':7} gold={gold[:34]}"
        )
    print(f"\n{len(CASES) - len(bad)}/{len(CASES)} passed | {be.usage}")
    for i, gold, cand, want, got in bad:
        print(f"\ncase {i} disagreed:")
        print(f"  gold: {gold}")
        print(f"  cand: {cand}")
        print(f"  want {want}, got {got}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
