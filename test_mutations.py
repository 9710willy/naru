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
COPY = (
    "ms.py",
    "kernel.py",
    "eviction.py",
    "agent.py",
    "backend.py",
    "noise.py",
    "metrics.py",
)

# (name, file, find, replace) or (name, file, find, replace, only_if)
# only_if is a Python expression: when it is false the mutation cannot
# change behaviour here and is reported n/a rather than counted a survivor.
MUTATIONS = [
    (
        "rag falls through to full",
        "bench.py",
        'if arm == "rag":\n        ms, _ = ingest(q, build_index=False, db=":memory:")',
        'if False:\n        ms, _ = ingest(q, build_index=False, db=":memory:")',
    ),
    (
        "rag creates a sandbox file log",
        "bench.py",
        'ms, _ = ingest(q, build_index=False, db=":memory:")',
        "ms, _ = ingest(q, build_index=False)",
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
        "noise keeps errored rows",
        "noise.py",
        "if backend or judge:",
        "if False:",
    ),
    (
        "noise hides exclusion counts",
        "noise.py",
        'f"  {name} | excluded {excluded[\'rows\']} error row(s)"',
        'f"  {name} | excluded {0} error row(s)"',
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
        "sandbox stderr becomes a blocking pipe",
        "kernel.py",
        "stderr=self._stderr,",
        "stderr=subprocess.PIPE,",
    ),
    (
        "a first prose reply is banked as the answer",
        "agent.py",
        "if turn == 0 and turn < max_turns - 1:",
        "if False:",
    ),
    (
        "agent.py never takes the sandbox branch",
        "agent.py",
        'if os.environ.get("NARU_KERNEL") == "sandbox" and path and path != ":memory:":',
        "if False:",
    ),
    (
        "run_naru leaves its kernel open",
        "agent.py",
        "        kernel.close()",
        "        pass",
    ),
    (
        "agent trace uses a local sequence",
        "agent.py",
        "reply_seq = ms.append(",
        "reply_seq = 0\n            ms.append(",
    ),
    (
        "invalid headline state reaches the log",
        "agent.py",
        "or not isinstance(verified, list)",
        "or False",
    ),
    (
        "headline state is never persisted",
        "agent.py",
        'kind="agent_state",',
        'kind="agent_state_off",',
    ),
    (
        "default search exposes agent trace",
        "ms.py",
        "return f\"({column} IS NULL OR {column} NOT GLOB 'agent_*')\"",
        "return \"1\"",
    ),
    (
        "session-scoped recovery includes other rows",
        "ms.py",
        'where.append("session_id IS ?")\n            params.append(session_id)',
        "pass",
    ),
    (
        "source range accepts another run",
        "ms.py",
        'if {r["seq"] for r in rows} != {source_seq_lo, source_seq_hi}:',
        "if False:",
    ),
    (
        "externalized trace pointer loses its session",
        "ms.py",
        'f"-> ms.expand({seq}{scope})]"',
        'f"-> ms.expand({seq})]"',
    ),
    (
        "folded payload loses its trace session",
        "eviction.py",
        "fold_payloads(older, recovery_session)",
        "fold_payloads(older)",
    ),
    (
        "callback dispatch is unguarded in the parent",
        "kernel.py",
        "            try:\n                fn(*args, **kwargs)\n            except Exception as e:",
        "            if True:\n                fn(*args, **kwargs)\n            except Exception as e:",
    ),
    (
        "the log path stays in the child's environment",
        "kernel.py",
        'db = os.environ.pop("NARU_KERNEL_DB", None)',
        'db = os.environ.get("NARU_KERNEL_DB")',
    ),
    (
        "submit_answer no longer stops the cell",
        "kernel.py",
        '            if _n == "submit_answer":\n                raise _Done()',
        '            if False:\n                raise _Done()',
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
    (
        "prune deletes promoted provenance",
        "ms.py",
        "curated.source_run_id = conversation_history.session_id",
        "0",
    ),
    (
        "prune keeps dropped provenance",
        "ms.py",
        "curated.promoted = 1",
        "curated.promoted <> 0",
    ),
    (
        "promoted provenance has no range index",
        "ms.py",
        "CREATE INDEX IF NOT EXISTS ix_promoted_sources",
        "CREATE INDEX IF NOT EXISTS ix_promoted_sources_off",
    ),
    (
        "store identity ignores database incarnation",
        "ms.py",
        'f"{location}\\0{store_uuid}".encode()',
        "location.encode()",
    ),
    (
        "blob gc deletes live payloads",
        "ms.py",
        "if not path.is_file() or str(path) in live:",
        "if not path.is_file():",
    ),
    (
        "outline has no session index",
        "ms.py",
        "CREATE INDEX IF NOT EXISTS ix_session_id",
        "CREATE INDEX IF NOT EXISTS ix_session_id_off",
    ),
    (
        "show receipt ignores store identity",
        "metrics.py",
        'and e.get("store") == store_id',
        "and True",
    ),
    (
        "show receipt ignores run identity",
        "metrics.py",
        'and e.get("run") == run_id',
        "and True",
    ),
    (
        "legacy show receipt authorizes a claim",
        "metrics.py",
        'e.get("v") == 2',
        "True",
    ),
    (
        "separate endpoint receipts cover a span",
        "metrics.py",
        'and e["lo"] <= lo <= hi <= e["hi"]',
        'and (e["lo"] <= lo <= e["hi"] or e["lo"] <= hi <= e["hi"])',
    ),
    (
        "empty show records evidence",
        "naru.py",
        (
            "        if rows:\n"
            "            metrics.record_show(ms.store_id, run, rows[0].seq, rows[-1].seq)"
        ),
        (
            "        metrics.record_show(ms.store_id, run, "
            "rows[0].seq if rows else lo, "
            "rows[-1].seq if rows else (hi if hi is not None else lo))"
        ),
    ),
    (
        "Codex repeats unchanged Naru context",
        "naru.py",
        'if event_name == "UserPromptSubmit" and _codex_seen(ms, session_id) == doc_hash:',
        'if event_name == "UserPromptSubmit" and False:',
    ),
    (
        "Codex hook state reaches normal search",
        "naru.py",
        'kind="agent_state",\n        session_id=session_id,\n        agent_id="codex",',
        'kind="tool_result",\n        session_id=session_id,\n        agent_id="codex",',
    ),
    (
        "Codex refresh watches only the highest promoted seq",
        "naru.py",
        "doc_hash = hashlib.sha256(doc.encode()).hexdigest()",
        "doc_hash = str(doc_seq)",
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
        "agent.py": ["agent.py"],
        "eviction.py": ["eviction.py"],
        "naru.py": ["naru.py", "--selfcheck"],
        "noise.py": ["noise.py", "--selfcheck"],
        "metrics.py": ["metrics.py", "--selfcheck"],
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
