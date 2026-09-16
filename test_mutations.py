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

import ast
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent
SELF_CHECK_TIMEOUT = 30
COPY = (
    "ms.py",
    "kernel.py",
    "eviction.py",
    "agent.py",
    "backend.py",
    "noise.py",
    "metrics.py",
    "naru.py",
    "hook_spill.py",
    "curation_probe.py",
    "regrade.py",
    "beam.py",
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
        "benchmark jobs share process stdout",
        "bench.py",
        "from concurrent.futures import ProcessPoolExecutor, as_completed",
        (
            "from concurrent.futures import ThreadPoolExecutor as "
            "ProcessPoolExecutor, as_completed"
        ),
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
        "judge accepts a verdict prefix",
        "bench.py",
        'if v == "CORRECT":',
        'if v.startswith("CORRECT"):',
    ),
    (
        "saved answers truncate before the judge boundary",
        "bench.py",
        "                else candidate",
        "                else candidate[:400]",
    ),
    (
        "regrade keeps stale judge errors",
        "regrade.py",
        'r["judge_errors"] = usage.errors',
        'r["judge_errors"] = r.get("judge_errors", 0)',
    ),
    (
        "regrade accepts legacy truncated answers",
        "regrade.py",
        "if old_format < RESULT_FORMAT and any(",
        "if False and any(",
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
        "failed arms relax the planned threshold",
        "bench.py",
        "alpha = 0.05 / pairs",
        "alpha = 0.05 / len(list(combinations(verdicts, 2)))",
    ),
    (
        "duplicate result rows overwrite each other",
        "bench.py",
        'if len(qids) != len(set(qids)):',
        "if False:",
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
        "main() accepts duplicate arms",
        "bench.py",
        'if len(set(arms)) != len(arms):\n        sys.exit("--arms must not contain duplicates")',
        "if False:\n        sys.exit(\"--arms must not contain duplicates\")",
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
        "prompt estimates omit the system text",
        "backend.py",
        'supplied = f"{system}\\n\\n{p}" if system else p',
        "supplied = p",
    ),
    (
        "failed floor calls report a measured value",
        "backend.py",
        "if not result or b.usage.errors or not b.usage.calls:",
        "if not b.usage.calls:",
    ),
    (
        "later failures discard known answer usage",
        "bench.py",
        '"billed_input": be.usage.billed_input,',
        '"billed_input": 0,',
    ),
    (
        "BEAM drops estimated prompt totals",
        "beam.py",
        'row.get("prompt_tokens_estimated", 0)',
        'row.get("peak_view_tokens", 0)',
    ),
    (
        "rag hits keep BM25 rank order",
        "bench.py",
        'sorted(hits, key=lambda h: h["seq"])',
        "hits",
    ),
    (
        "rsm packer exceeds its token budget",
        "bench.py",
        'if est("\\n\\n".join((*groups, group))) > budget:',
        "if False:",
    ),
    (
        "rsm runs without an embedder",
        "bench.py",
        'if "rsm" in arms:',
        "if False:",
    ),
    (
        "zero centroid reaches division",
        "bench.py",
        "    if not norm:\n        return None",
        "    if False:\n        return None",
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
        "kernel digest copies the full resident list",
        "kernel.py",
        "items = v[:200] if isinstance(v, (list, tuple)) else []",
        "items = list(v)[:200] if isinstance(v, (list, tuple)) else []",
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
        "agent trace omits the system prompt",
        "agent.py",
        '"prompt_tokens": est(f"{system}\\n\\n{prompt}"),',
        '"prompt_tokens": est(prompt),',
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
        "trace index pointer loses its session",
        "agent.py",
        '                "session_id": run_id,',
        '                "session_id": None,',
    ),
    (
        "callback dispatch is unguarded in the parent",
        "kernel.py",
        (
            "            try:\n"
            "                fn(*args, **kwargs)\n"
            "            except Exception as e:\n"
            "                # The arguments come from model-authored code. Dispatching them\n"
            "                # unguarded let `headline(1,2,3,4,5)` raise a live TypeError out\n"
            "                # of run() and into the harness — the one thing a child process\n"
            "                # is here to prevent.\n"
            "                err = err or f\"{name}(): {type(e).__name__}: {e}\""
        ),
        "            fn(*args, **kwargs)",
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
        "prune does not reserve the writer",
        "ms.py",
        '        self.db.execute("BEGIN IMMEDIATE")\n        try:\n            doomed =',
        "        try:\n            doomed =",
    ),
    (
        "blob gc does not reserve the writer",
        "ms.py",
        '        self.db.execute("BEGIN IMMEDIATE")\n        try:\n            if not self.blobs.is_dir():',
        "        try:\n            if not self.blobs.is_dir():",
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
        "legacy store metadata is rejected",
        "ms.py",
        'if "key" in meta_cols:',
        "if False:",
    ),
    (
        "an existing store gets a no-op metadata write",
        "ms.py",
        "if row is None:",
        "if True:",
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
        "show receipts never link back to spills",
        "metrics.py",
        'and shown["lo"] <= spill["seq"] <= shown["hi"]',
        "and False",
    ),
    (
        "spill metrics omit store identity",
        "hook_spill.py",
        "        store=ms.store_id,",
        "        store=None,",
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
        "inject treats an unreadable file as empty",
        "naru.py",
        "except FileNotFoundError:",
        "except (OSError, UnicodeDecodeError):",
    ),
    (
        "Codex repeats unchanged Naru context",
        "naru.py",
        'if event_name == "UserPromptSubmit" and _codex_seen(ms, session_id) == doc_hash:',
        'if event_name == "UserPromptSubmit" and False:',
    ),
    (
        "Codex delivery state bypasses its typed table",
        "naru.py",
        "    remember_context_delivery(\n",
        "    ms.append(\n",
    ),
    (
        "legacy Codex delivery state is ignored",
        "naru.py",
        (
            "    rows = ms.sql_query(\n"
            "        \"SELECT content, created_at FROM conversation_history\"\n"
            "        \" WHERE kind='agent_state' AND agent_id='codex' AND session_id=?\"\n"
            "        \" ORDER BY seq DESC LIMIT 1\",\n"
            "        (session_id,),\n"
            "    )"
        ),
        "    rows = []",
    ),
    (
        "Codex delivery metrics omit the harness",
        "naru.py",
        '        harness="codex",',
        '        harness="legacy",',
    ),
    (
        "Codex refresh watches only the highest promoted seq",
        "naru.py",
        "doc_hash = hashlib.sha256(doc.encode()).hexdigest()",
        "doc_hash = str(doc_seq)",
    ),
    (
        "paired probe omits approved context",
        "curation_probe.py",
        (
            "    system = (\n"
            "        BASE_SYSTEM if arm == \"plain\" else "
            "f\"{BASE_SYSTEM}\\n\\n{_codex_context(doc)}\"\n"
            "    )"
        ),
        "    system = BASE_SYSTEM",
    ),
    (
        "paired probe ignores forbidden answer text",
        "curation_probe.py",
        'return {"passed": not missing and not present, "missing": missing, "present": present}',
        'return {"passed": not missing, "missing": missing, "present": present}',
    ),
    (
        "paired probe accepts duplicate case ids",
        "curation_probe.py",
        "    if len(ids) != len(set(ids)):",
        "    if False:",
    ),
    (
        "paired probe scores failed model calls",
        "curation_probe.py",
        (
            "            if not row[\"plain\"][\"usage\"][\"errors\"]\n"
            "            and not row[\"naru\"][\"usage\"][\"errors\"]"
        ),
        "            if True",
    ),
]


def selfcheck_command(target):
    """Return the command that owns invariants in target."""
    return {
        "ms.py": ["ms.py"],
        "kernel.py": ["kernel.py"],
        "agent.py": ["agent.py"],
        "eviction.py": ["eviction.py"],
        "naru.py": ["naru.py", "--selfcheck"],
        "noise.py": ["noise.py", "--selfcheck"],
        "metrics.py": ["metrics.py", "--selfcheck"],
        "hook_spill.py": ["hook_spill.py", "--selfcheck"],
        "curation_probe.py": ["curation_probe.py", "--selfcheck"],
        "backend.py": ["backend.py", "--selfcheck"],
        "regrade.py": ["regrade.py", "--selfcheck"],
        "beam.py": ["beam.py", "--selfcheck"],
    }.get(target, ["bench.py", "--selfcheck"])


def run_selfcheck(target, cwd):
    """Run one self-check with an outer deadline."""
    try:
        return subprocess.run(
            [sys.executable, *selfcheck_command(target)],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=SELF_CHECK_TIMEOUT,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            error.cmd,
            124,
            stdout="",
            stderr=f"timed out after {SELF_CHECK_TIMEOUT}s\n",
        )


def run_mutated(target, find, replace):
    """Apply one edit in a throwaway copy and return the self-check result."""
    body = (REPO / target).read_text()
    if find not in body:
        raise AssertionError(
            f"anchor not found in {target} — the code moved. Re-confirm this "
            f"mutation still describes a real bug, then update the anchor:\n{find}"
        )
    with tempfile.TemporaryDirectory() as work_dir:
        work = pathlib.Path(work_dir)
        for name in (*COPY, "bench.py"):
            shutil.copy(REPO / name, work / name)
        published = REPO / "results" / "published"
        if published.is_dir():
            (work / "results" / "published").mkdir(parents=True)
            for f in published.glob("*.json"):
                shutil.copy(f, work / "results" / "published" / f.name)
        mutated = body.replace(find, replace)
        try:
            ast.parse(mutated, filename=target)
        except SyntaxError as error:
            raise AssertionError(
                f"mutation creates invalid Python in {target}: {error}"
            ) from error
        (work / target).write_text(mutated)
        return run_selfcheck(target, work)


def mutation_caught(result):
    """Return whether a valid self-check rejected the mutation."""
    if result.returncode == 124 or result.returncode < 0:
        raise RuntimeError(
            f"self-check infrastructure failed with exit {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    return result.returncode != 0


def main():
    try:
        mutation_caught(subprocess.CompletedProcess([], 124, "", "outer timeout"))
    except RuntimeError:
        pass
    else:
        raise AssertionError("an outer timeout counted as a caught mutation")

    for target in dict.fromkeys(m[1] for m in MUTATIONS):
        result = run_selfcheck(target, REPO)
        if result.returncode:
            print(
                f"baseline failed for {target}, exit {result.returncode}",
                file=sys.stderr,
            )
            print(result.stdout, end="", file=sys.stderr)
            print(result.stderr, end="", file=sys.stderr)
            return 1

    survivors, skipped, failures = [], [], []
    for mutation in MUTATIONS:
        name, target, find, replace = mutation[:4]
        only_if = mutation[4] if len(mutation) > 4 else None
        if only_if and not eval(only_if):
            print(f"  {'n/a':9} {name}  ({only_if})")
            skipped.append(name)
            continue
        result = run_mutated(target, find, replace)
        try:
            caught = mutation_caught(result)
        except RuntimeError as error:
            print(f"  {'FAILED':9} {name}  ({error})")
            failures.append(name)
            continue
        print(f"  {'caught' if caught else 'SURVIVED':9} {name}")
        if not caught:
            survivors.append(name)
    n = len(MUTATIONS) - len(skipped)
    if failures:
        print(f"\n{len(failures)} mutation run(s) had infrastructure failures:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
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
