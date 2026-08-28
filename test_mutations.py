#!/usr/bin/env python3
"""Do the self-checks actually catch anything?

Every module here ships a `demo()` that prints "ok". That proves the code runs.
It does not prove the assertions have teeth, and four of this repo's did not:
the two separability scenarios shared one output buffer so neither assertion
was bound to the case that had to produce it, the p-value assertion compared a
tuple slot to itself, and two guards were asserted against re-typed copies of
themselves rather than against the functions main() calls. Each of those stayed
green while the bug it named was live.

So: break the code on purpose, one edit at a time, and require the self-check
to fail. A mutation that survives is a check that is decorative.

    python3 test_mutations.py

Anchors are exact source strings and a refactor will break them. That is
intended — a missing anchor fails loudly and asks you to confirm the check
still catches the bug, rather than silently testing nothing.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent
COPY = ("ms.py", "kernel.py", "eviction.py", "agent.py", "backend.py", "noise.py")

# (name, file, find, replace) or (name, file, find, replace, only_if)
# only_if is a Python expression: when it is false the mutation cannot
# change behaviour here and is reported n/a rather than counted a survivor.
MUTATIONS = [
    (
        "rag falls through to full",
        "bench.py",
        'if arm == "rag":\n        ms, _ = ingest(q, build_index=False)',
        "if False:\n        ms, _ = ingest(q, build_index=False)",
    ),
    (
        "errored runs score as wrong answers",
        "bench.py",
        'if not r.get("errors") and not r.get("judge_errors")',
        "if True",
    ),
    (
        "unmeasured judge errors print as zero",
        "bench.py",
        'if all("judge_errors" in r for r in rows)',
        "if True",
    ),
    (
        "no Bonferroni across the arm pairs",
        "bench.py",
        "alpha = 0.05 / pairs",
        "alpha = 0.05",
    ),
    (
        "separability verdict inverted",
        "bench.py",
        'f"REAL at p<{alpha:.3g}"\n            if p < alpha',
        'f"REAL at p<{alpha:.3g}"\n            if p >= alpha',
    ),
    (
        "gap names the losing arm",
        "bench.py",
        "gap = 100 * (only_b - only_a) / shared",
        "gap = 100 * (only_a - only_b) / shared",
    ),
    (
        "main() stops validating --arms",
        "bench.py",
        "unknown = unknown_arms(arms)",
        "unknown = []",
    ),
    (
        "--rag-k accepts SQLite's 'no limit'",
        "bench.py",
        "if a.rag_k < 1:",
        "if False:",
    ),
    (
        "run config records the backend's arguments",
        "bench.py",
        'return (shlex.split(cmd) or ["claude-cli"])[0] if cmd else "claude-cli"',
        'return cmd if cmd else "claude-cli"',
    ),
    (
        "rag hits keep BM25 rank order",
        "bench.py",
        'sorted(hits, key=lambda h: h["seq"])',
        "hits",
    ),
    (
        "FTS5 operators reach the query from question text",
        "bench.py",
        "t.lower() if t in _FTS_OPS else t for t in question.split()",
        "t for t in question.split()",
    ),
    (
        "McNemar p-value becomes one-sided",
        "noise.py",
        "return a_only, b_only, min(1.0, 2 * tail)",
        "return a_only, b_only, min(1.0, tail)",
    ),
    (
        "a refused rlimit is reported as applied",
        "kernel.py",
        'applied[name] = f"NOT APPLIED: {type(e).__name__}"',
        "applied[name] = want",
        # Linux grants RLIMIT_AS, so the except branch never runs there and the
        # mutation is a no-op. macOS refuses it, which is the whole reason the
        # branch exists.
        'sys.platform == "darwin"',
    ),
    (
        "a crashed child is reported as a timeout",
        "kernel.py",
        'f"exceeded {self.timeout}s wall clock"\n            if hung',
        'f"exceeded {self.timeout}s wall clock"\n            if not hung',
    ),
    (
        "the child opens the log writable",
        "kernel.py",
        'ns["ms"] = MemorySurface.open_readonly(db).readonly()',
        'ns["ms"] = MemorySurface(db).readonly()',
    ),
    (
        "the child runs without -I, so CWD is importable",
        "kernel.py",
        '            "-I",\n            "-c",',
        '            "-c",',
    ),
    (
        "an in-memory log is accepted for a child process",
        "kernel.py",
        'if db == ":memory:":',
        "if False:",
    ),
    (
        "prune preview forgets PRUNE_KEEP",
        "ms.py",
        (
            '" WHERE created_at IS NOT NULL AND created_at < ?" + PRUNE_KEEP,\n'
            "            (before_iso,),\n        ).fetchone()"
        ),
        (
            '" WHERE created_at IS NOT NULL AND created_at < ?",\n'
            "            (before_iso,),\n        ).fetchone()"
        ),
    ),
]


def run_mutated(target, find, replace):
    """Apply one edit in a throwaway copy and return the self-check's exit code."""
    body = (REPO / target).read_text()
    if find not in body:
        raise AssertionError(
            f"anchor not found in {target} — the code moved. Re-confirm this "
            f"mutation still describes a real bug, then update the anchor:\n{find}"
        )
    work = pathlib.Path(tempfile.mkdtemp())
    for name in (*COPY, "bench.py"):
        shutil.copy(REPO / name, work / name)
    published = REPO / "results" / "published"
    if published.is_dir():
        (work / "results" / "published").mkdir(parents=True)
        for f in published.glob("*.json"):
            shutil.copy(f, work / "results" / "published" / f.name)
    (work / target).write_text(body.replace(find, replace))
    # each invariant belongs to the module whose demo asserts it
    cmd = {
        "ms.py": ["ms.py"],
        "kernel.py": ["kernel.py"],
    }.get(target, ["bench.py", "--selfcheck"])
    return subprocess.run(
        [sys.executable, *cmd], cwd=work, capture_output=True, text=True, check=False
    ).returncode


def main():
    survivors, skipped = [], []
    for mutation in MUTATIONS:
        name, target, find, replace = mutation[:4]
        only_if = mutation[4] if len(mutation) > 4 else None
        if only_if and not eval(only_if):
            print(f"  {'n/a':9} {name}  ({only_if})")
            skipped.append(name)
            continue
        caught = run_mutated(target, find, replace) != 0
        print(f"  {'caught' if caught else 'SURVIVED':9} {name}")
        if not caught:
            survivors.append(name)
    n = len(MUTATIONS) - len(skipped)
    if survivors:
        print(
            f"\n{len(survivors)} of {n} mutations survived — those checks are decorative:"
        )
        for s in survivors:
            print(f"  - {s}")
        return 1
    tail = f" ({len(skipped)} n/a on {sys.platform})" if skipped else ""
    print(f"\nok — {n}/{n} mutations caught{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
