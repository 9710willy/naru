"""LongMemEval harness: ingest → answer → judge → score.

Benchmark: https://arxiv.org/abs/2410.10813

Runs four arms over the same data so the comparison is controlled:
  full     — the whole history stuffed into one prompt (the usual approach)
  rag      — top-k BM25 hits pasted in, one call, no kernel (the control)
  naru     — history in the Session Environment, model writes code to reach it
  rsm      — grouped dense retrieval over derived Event Log chunks

Reports accuracy, tokens billed, and cost for each.
"""

import argparse
import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

from agent import LONGMEMEVAL_RUBRIC, run_naru
from backend import (
    BACKEND_CHOICES,
    HAIKU,
    Usage,
    backend_fingerprint,
    default_model_for_backend,
    get_backend,
    measure_floor,
    resolve_backend,
)
from context_policy import CONTEXT_POLICIES
from eviction import est, rollup
from jev import CommandJev, host_stdio_evaluator
from ms import MemorySurface

# One owner for the p-value ADR 0006 publishes. noise.py had it first and
# imports nothing local, so the dependency runs this way round.
from noise import mcnemar

DATA = pathlib.Path(__file__).parent / "data"
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")
# One owner for the arm names: main() validates against it, demo() asserts on it.
ARMS = ("full", "rag", "naru", "rsm")
RESULT_FORMAT = 3


class _ParentExecutor:
    """Future-compatible thread pool for parent-owned host MCP calls."""

    def __init__(self, max_workers=None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return self._executor.__exit__(exc_type, exc_value, traceback)

    def submit(self, function, *args, **kwargs) -> Future:
        return self._executor.submit(function, *args, **kwargs)


def backend_label(cmd):
    """Provenance for a run: the program that answered, never its arguments.

    NARU_BACKEND may be any command reading a prompt on stdin, so it can carry
    a credential (`sh -c 'curl -H "Authorization: Bearer ..."'`).
    results/published/ is committed to a public history, where rotating a key
    that already shipped does not undo it. argv[0] is all the provenance the
    field is for — telling a `cat` run apart from a real one. No configured
    command means the provider resolver chose automatically.
    """
    return (shlex.split(cmd) or ["auto"])[0] if cmd else "auto"


def telemetry_id(value):
    """Stable short identifier for telemetry without copying task text."""
    if value is None:
        return None
    return hashlib.sha256(str(value).encode()).hexdigest()[:16]


def _worker_pid(_):
    """Return the worker process ID for the offline executor check."""
    return os.getpid()


def embedder_argv():
    """Validate the optional embedding command before paid work starts."""
    cmd = os.environ.get("NARU_EMBED")
    if not cmd:
        raise ValueError(
            "rsm needs NARU_EMBED: a command that reads JSON on stdin"
        )
    try:
        argv = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"invalid NARU_EMBED command: {e}") from e
    if not argv:
        raise ValueError("NARU_EMBED is empty; expected a command to run")
    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(f"NARU_EMBED command not found: {argv[0]!r}")
    return argv


def _normalized(vector, where):
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"NARU_EMBED {where} must be a non-empty array")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in vector):
        raise ValueError(f"NARU_EMBED {where} has a nonnumeric value")
    if any(not math.isfinite(x) for x in vector):
        raise ValueError(f"NARU_EMBED {where} has a non-finite value")
    norm = math.sqrt(sum(x * x for x in vector))
    if not norm:
        raise ValueError(f"NARU_EMBED {where} is a zero vector")
    return tuple(x / norm for x in vector)


def embed_vectors(argv, texts):
    """Run the embedding boundary once and return normalized vectors."""
    try:
        p = subprocess.run(
            argv,
            input=json.dumps({"texts": texts}),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        raise RuntimeError(f"NARU_EMBED could not start: {e}") from e
    if p.returncode:
        raise RuntimeError(f"NARU_EMBED exit {p.returncode}: {p.stderr.strip()[:300]}")
    try:
        response = json.loads(p.stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"NARU_EMBED returned invalid JSON: {e.msg}") from e
    if not isinstance(response, dict) or set(response) != {"vectors"}:
        raise ValueError("NARU_EMBED response must contain only a vectors array")
    vectors = response["vectors"]
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        got = len(vectors) if isinstance(vectors, list) else "non-array"
        raise ValueError(f"NARU_EMBED returned {got} vectors for {len(texts)} texts")
    out = [_normalized(v, f"vector {i}") for i, v in enumerate(vectors)]
    if len({len(v) for v in out}) != 1:
        raise ValueError("NARU_EMBED vectors have mismatched dimensions")
    return out


def unknown_arms(arms):
    """Arm names main() will refuse. A function so the self-check can exercise
    the real predicate — asserting on a re-typed copy of it passes even when
    main()'s validation has been deleted."""
    return [x for x in arms if x not in ARMS]


def iso(d):
    """'2023/04/10 (Mon) 17:50' -> '2023-04-10T17:50'."""
    m = DATE_RE.search(d or "")
    if not m:
        return "1970-01-01T00:00"
    t = re.search(r"(\d{2}:\d{2})", (d or "")[m.end() :])
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{t.group(1) if t else '00:00'}"


def load(split, n=None, seed=0, qtype=None):
    path = DATA / f"longmemeval_{split}.json"
    if not path.exists():
        sys.exit(f"missing {path} — download it first (see README)")
    qs = json.load(open(path))
    if qtype:
        qs = [q for q in qs if q["question_type"] == qtype]
    # deterministic stratified slice: round-robin over question types
    by = {}
    for q in qs:
        by.setdefault(q["question_type"], []).append(q)
    for v in by.values():
        v.sort(key=lambda x: x["question_id"])
    order, keys = [], sorted(by)
    while any(by[k] for k in keys):
        for k in keys:
            if by[k]:
                order.append(by[k].pop(0))
    return order[:n] if n else order


def sessions(q):
    """The question's haystack sessions in chronological order.

    ONE source of truth for ordering. Both arms must see the same history in
    the same order or every comparison in this harness is silently invalid.
    """
    return sorted(
        zip(q["haystack_dates"], q["haystack_session_ids"], q["haystack_sessions"]),
        key=lambda s: iso(s[0]),
    )


def _log_path():
    """Where one question's Event Log lives, and why it is usually nowhere."""
    if os.environ.get("NARU_KERNEL") != "sandbox":
        return ":memory:"
    return str(pathlib.Path(tempfile.mkdtemp(prefix="naru-bench-")) / "log.db")


def ingest(q, build_index=True, db=None):
    """Build the Event Log for one question, session by session.

    Section 3.3: "we ingest each conversation history into Scroll session by
    session, in chronological order. At each session boundary, the raw context
    is cleared, and only Scroll's internal state (the eviction index and the
    Event Log) is carried forward."

    Returns (ms, index). The index is the tiered eviction index built as each
    session's raw context is cleared — the landmarks the agent starts with,
    rather than starting blind. Passing build_index=False reproduces the
    earlier behaviour, for ablation.
    """
    # NARU_KERNEL=sandbox runs cells in a child process, and a child cannot
    # open an in-memory database. Without a file the sandbox was unreachable
    # from the only entry point that runs the agent, so it was a feature with
    # no caller. The default stays in-memory and fast.
    ms = MemorySurface(_log_path() if db is None else db)
    index = []
    for i, (date, sid, turns) in enumerate(sessions(q), 1):
        stamp = iso(date)
        lo = hi = None
        first_user = None
        for t in turns:
            role = t.get("role", "user")
            # session/date tag inline so lexical search can hit it too
            body = f"[Session {i} | {stamp[:10]}] {role}: {t.get('content', '')}"
            seq = ms.append(
                role,
                body,
                kind="context_msg" if role == "user" else "model_turn",
                session_id=sid,
                created_at=stamp,
            )
            lo = seq if lo is None else lo
            hi = seq
            if first_user is None and role == "user":
                first_user = t.get("content", "")

        if build_index and lo is not None:
            # Session boundary: the raw context is cleared and its landmark
            # enters the eviction index, anchored to the exact seq span.
            index.append([]) if not index else None
            index[0].append(
                {
                    "lo": lo,
                    "hi": hi,
                    "headline": f"session {i} | {stamp[:10]} | "
                    f"{first_user.strip()[:70] if first_user else ''}",
                }
            )
            rollup(index, 4)
    return ms, index


def discard_log(ms):
    """Close and remove one file-backed benchmark Event Log."""
    path = ms.path
    ms.close()
    if path != ":memory:":
        shutil.rmtree(pathlib.Path(path).parent, ignore_errors=True)


def history_text(q):
    """Flat transcript for the full-context arm."""
    out = []
    for i, (date, _sid, turns) in enumerate(sessions(q), 1):
        out.append(f"\n=== Session {i} | {iso(date)[:10]} ===")
        for t in turns:
            out.append(f"{t.get('role', 'user')}: {t.get('content', '')}")
    return "\n".join(out)


FULL_SYSTEM = """\
You answer a question about a long conversation history between a user and an
assistant. Base your answer only on that history.

Treat role 'user' turns as evidence of the user's own facts and preferences; an
assistant suggestion is not evidence the user adopted it. When a fact changed
over time, use the most recent user evidence. Preserve exact numbers, units,
names and dates. Never invent specifics. If the history does not state the
fact, say so plainly.

Reply with the answer only — no preamble, no reasoning."""

JUDGE_SYSTEM = """\
You grade a candidate answer against a gold answer for a question about a
conversation history. You are checking ONE thing: does the candidate state the
gold fact?

CORRECT when the candidate states the gold fact, even if:
  - worded differently, reordered, or paraphrased
  - punctuated or formatted differently ("on January 2nd" vs "(January 2nd)")
  - surrounded by extra correct context, reasoning, or supporting detail
  - it gives one of several values the gold answer marks acceptable. A gold
    answer may list alternatives, e.g. "30 days. 31 days (including the last
    day) is also acceptable." — then EITHER value is CORRECT.
  - the unit or currency is written differently ($12 vs 12 dollars)

WRONG when the candidate:
  - omits the gold fact, or states a different value for it
  - contradicts the gold fact
  - refuses, or says the information is absent — UNLESS the gold answer also
    says it is absent
  - only names the topic without giving the asked-for fact

ABSENCE GOLD. Some gold answers say the information was never provided, e.g.
"You did not mention this information. You mentioned your cat Luna but not your
hamster." For these the gold FACT is only the absence. A candidate is CORRECT as
soon as it conveys that the asked-for thing was never mentioned. It does NOT
need to reproduce the near-miss detail (the cat, the other role, the related
item) — that is explanation, not the fact. A candidate that instead supplies a
made-up value is WRONG.

Grade the FACT, not the prose. Do not require the candidate's wording to
resemble the gold answer's wording.

Reply with exactly one word: CORRECT or WRONG."""


def judge(q, response, backend, votes=3):
    """Grade one answer. Majority of `votes` independent gradings.

    A single grading is unstable on paraphrase: two answers differing only in
    punctuation were graded differently in replicate runs, which put harness
    noise straight into the reported accuracy. Judge calls are small and cheap,
    so voting is the cheapest available variance reduction.
    """
    if not response or not response.strip():
        return False
    candidate = response.strip()[:2000]
    p = (
        f"Question: {q['question']}\n\nGold answer: {q['answer']}\n\n"
        f"Candidate answer: {candidate}\n\nVerdict:"
    )
    yes = no = 0
    for i in range(votes):
        v = (
            backend(p, system=JUDGE_SYSTEM, nudge="(Reply with exactly one word.)")
            .strip()
            .upper()
        )
        if v == "CORRECT":
            yes += 1
        elif v == "WRONG":
            no += 1
        else:
            backend.usage.errors += 1
        # early exit once the outcome cannot change
        if yes > votes // 2 or no > votes // 2:
            break
    return yes > (votes // 2)


# ms._to_match passes these through as FTS5 boolean operators, which the
# interactive CLI wants and a benchmark question does not.
_FTS_OPS = ("AND", "OR", "NOT")


def rag_context(ms, question, k):
    """Top-k BM25 hits for the question, back in chronological order.

    Each row already carries its own `[Session i | date] role:` prefix from
    ingest(), so the hits need no further framing to be readable.

    The question is dataset text, not a query. One containing a bare uppercase
    AND/OR/NOT raised `fts5: syntax error`, which one()'s handler turned into a
    forfeited question for THIS arm while full and naru answered it normally —
    a one-sided accuracy penalty on the control arm, from punctuation.
    """
    query = " ".join(
        t.lower() if t in _FTS_OPS else t for t in question.split()
    ).strip()
    if not query:
        return ""
    hits = ms.search(query, k=k)
    return "\n".join(h["content"] for h in sorted(hits, key=lambda h: h["seq"]))


@dataclasses.dataclass
class RsmAtom:
    members: list
    centroid: tuple


@dataclasses.dataclass
class RsmPack:
    context: str
    stats: dict


def _cosine(left, right):
    return sum(a * b for a, b in zip(left, right))


def _centroid(members):
    values = [sum(v) / len(members) for v in zip(*(m["vector"] for m in members))]
    norm = math.sqrt(sum(v * v for v in values))
    if not norm:
        return None
    return tuple(v / norm for v in values)


def rsm_pack(ms, question, k, tau, chunk_turns, budget, argv=None, vectorize=None):
    """Select bounded atom members and address them in the Event Log."""
    rows = ms.expand(1, sys.maxsize)
    members = [
        {
            "seqs": tuple(row["seq"] for row in rows[i : i + chunk_turns]),
            "text": "\n".join(row["content"] for row in rows[i : i + chunk_turns]),
        }
        for i in range(0, len(rows), chunk_turns)
    ]
    texts = [m["text"] for m in members] + [question]
    started = time.monotonic()
    vectors = vectorize(texts) if vectorize is not None else embed_vectors(argv, texts)
    embedding_seconds = time.monotonic() - started
    if len(vectors) != len(texts):
        raise ValueError("RSM embedder returned the wrong number of vectors")
    for member, vector in zip(members, vectors[:-1]):
        member["vector"] = vector
    question_vector = vectors[-1]

    started = time.monotonic()
    atoms = []
    for member in members:
        if not atoms:
            atoms.append(RsmAtom([member], member["vector"]))
            continue
        atom = max(atoms, key=lambda a: _cosine(member["vector"], a.centroid))
        centroid_score = _cosine(member["vector"], atom.centroid)
        member_score = max(_cosine(member["vector"], m["vector"]) for m in atom.members)
        if max(centroid_score, member_score) >= tau:
            merged = _centroid([*atom.members, member])
            if merged is None:
                atoms.append(RsmAtom([member], member["vector"]))
                continue
            atom.members.append(member)
            atom.centroid = merged
        else:
            atoms.append(RsmAtom([member], member["vector"]))
    indexing_seconds = time.monotonic() - started

    groups = []
    selected_members = 0

    def add_group(rank, chosen):
        nonlocal selected_members
        groups.append(
            f"=== Atom {rank} ===\n" + "\n".join(member["text"] for member in chosen)
        )
        selected_members += len(chosen)

    def finish():
        context = "\n\n".join(groups)
        return RsmPack(
            context,
            {
                "rsm_embedding_input_tokens": sum(est(text) for text in texts),
                "rsm_atoms": len(atoms),
                "rsm_members": len(members),
                "rsm_selected_members": selected_members,
                "rsm_embedding_seconds": round(embedding_seconds, 3),
                "rsm_indexing_seconds": round(indexing_seconds, 3),
                "rsm_context_tokens": est(context) if context else 0,
            },
        )

    for rank, atom in enumerate(
        sorted(atoms, key=lambda a: _cosine(question_vector, a.centroid), reverse=True)[:k],
        1,
    ):
        chosen = []
        for member in sorted(
            atom.members,
            key=lambda m: _cosine(question_vector, m["vector"]),
            reverse=True,
        ):
            selected = sorted((*chosen, member), key=lambda m: m["seqs"])
            group = f"=== Atom {rank} ===\n" + "\n".join(m["text"] for m in selected)
            if est("\n\n".join((*groups, group))) > budget:
                if chosen:
                    add_group(rank, chosen)
                return finish()
            chosen = selected
        if chosen:
            add_group(rank, chosen)
    return finish()


def rsm_context(q, question, k, tau, chunk_turns, budget, argv=None, vectorize=None):
    """Build one bounded RSM context from derived Event Log chunks."""
    ms, _ = ingest(q, build_index=False, db=":memory:")
    try:
        pack = rsm_pack(
            ms,
            question,
            k,
            tau,
            chunk_turns,
            budget,
            argv=argv,
            vectorize=vectorize,
        )
        return pack.context, pack.stats
    finally:
        ms.close()


def _question_prompt(q, body):
    return (
        f"{body}\n\n=== Question (asked {q.get('question_date', '')}) ===\n"
        f"{q['question']}"
    )


def build_prompt(
    q,
    arm,
    rag_k=8,
):
    """The prompt for a single-call arm. Pure: no backend, no judge, no clock.

    Split out of one() so the arm dispatch is reachable from the self-check.
    Inline, breaking it answered every rag question from the full 124k-token
    history while the result row, the report and the published JSON all still
    said "rag" — turning the headline comparison into full-vs-full.

    `rag` uses FULL_SYSTEM, the same system prompt as `full`, on purpose: the
    two single-call arms must differ in exactly one variable, which is what
    goes in the prompt.
    """
    if arm == "rag":
        ms, _ = ingest(q, build_index=False, db=":memory:")
        try:
            body = rag_context(ms, q["question"], rag_k)
        finally:
            ms.close()
    elif arm == "full":
        body = history_text(q)
    else:
        raise ValueError(f"not a direct prompt arm: {arm}")
    return _question_prompt(q, body)


def _result_status(be, judge_usage, error, judge_failure, correct=False):
    """Return task status fields without treating harness errors as misses."""
    errors = be.usage.errors + int(error is not None and not judge_failure)
    judge_errors = (judge_usage.errors if judge_usage else 0) + int(
        error is not None and judge_failure
    )
    if error is not None:
        failure_class = "judge" if judge_failure else "harness"
    elif be.usage.errors:
        failure_class = "model"
    elif judge_usage is not None and judge_usage.errors:
        failure_class = "judge"
    else:
        failure_class = None
    return {
        "errors": errors,
        "judge_errors": judge_errors,
        "task_success": bool(correct and not errors and not judge_errors),
        "failure_class": failure_class,
    }


def _context_result_fields(context_stats, jev_bridge):
    """Return bounded Jev and context telemetry for one result row."""
    usage = getattr(jev_bridge, "usage", None)
    return {
        "context_policy": context_stats["context_policy"],
        "context_considered_tokens": context_stats.get(
            "context_considered_tokens", 0
        ),
        "context_selected_tokens": context_stats.get("context_selected_tokens", 0),
        "context_dropped_tokens": context_stats.get("context_dropped_tokens", 0),
        "context_peak_selected_tokens": context_stats.get(
            "context_peak_selected_tokens", 0
        ),
        "context_selected_blocks": context_stats.get("context_selected_blocks", []),
        "context_omitted_blocks": context_stats.get("context_omitted_blocks", []),
        "context_jev_calls": context_stats.get("context_jev_calls", 0),
        "context_jev_attempts": getattr(
            usage, "attempts", context_stats.get("context_jev_calls", 0)
        ),
        "context_jev_input_tokens": context_stats.get(
            "context_jev_input_tokens", 0
        ),
        "context_jev_cache_read_tokens": context_stats.get(
            "context_jev_cache_read_tokens", 0
        ),
        "context_jev_cache_write_tokens": context_stats.get(
            "context_jev_cache_write_tokens", 0
        ),
        "context_jev_output_tokens": context_stats.get(
            "context_jev_output_tokens", 0
        ),
        "context_jev_latency_ms": round(
            context_stats.get("context_jev_latency_ms", 0.0), 3
        ),
        "context_jev_startup_latency_ms": round(
            getattr(usage, "startup_latency_ms", 0.0), 3
        ),
        "context_jev_provider_latency_ms": round(
            getattr(usage, "provider_latency_ms", 0.0), 3
        ),
        "context_jev_transport_overhead_ms": round(
            max(
                0.0,
                getattr(usage, "latency_ms", 0.0)
                - getattr(usage, "provider_latency_ms", 0.0),
            ),
            3,
        ),
        "context_jev_transport": getattr(usage, "transport", None),
        "context_jev_errors": context_stats.get("context_jev_errors", 0),
        "context_fallbacks": context_stats.get("context_fallbacks", 0),
        "context_jev_unavailable": context_stats.get(
            "context_jev_unavailable", 0
        ),
        "tool_call_count": context_stats.get("context_tool_calls", 0),
        "tool_result_tokens": context_stats.get("context_tool_result_tokens", 0),
        "model_switch": False,
        "subagent_call_count": 0,
        "cost_includes_jev": not bool(context_stats.get("context_jev_calls")),
        "jev_provider_usage": dict(getattr(usage, "native_usage", {})),
        "jev_usage": (
            usage.as_dict()
            if usage is not None and hasattr(usage, "as_dict")
            else {}
        ),
    }


def one(
    q,
    arm,
    model,
    judge_model,
    max_turns,
    budget,
    verbose,
    rubric=True,
    no_index=False,
    rag_k=8,
    rsm_k=6,
    rsm_tau=0.85,
    rsm_chunk_turns=5,
    rsm_budget=4000,
    rsm_argv=None,
    trace=None,
    context_policy="deterministic",
    jev_command=None,
    jev_evaluator=None,
):
    """Run a single question through one arm. Returns a result record."""
    be = get_backend(model)
    t0 = time.time()
    context_stats = {
        "context_policy": context_policy if arm == "naru" else "not_applicable"
    }
    jev_bridge = None
    owns_jev_bridge = False
    if arm == "naru" and context_policy in ("shadow", "jev"):
        if jev_evaluator is not None:
            jev_bridge = jev_evaluator
        elif jev_command or os.environ.get("NARU_JEV"):
            configured = jev_command or os.environ.get("NARU_JEV")
            try:
                jev_bridge = CommandJev(configured)
                owns_jev_bridge = True
            except (ValueError, FileNotFoundError):
                context_stats["context_jev_unavailable"] = 1
                context_stats["context_fallbacks"] = 1
        else:
            context_stats["context_jev_unavailable"] = 1

    def result(
        answer, turns, peak, correct=False, judge_backend=None, error=None,
        judge_failure=False,
    ):
        candidate = (answer or "").strip()[:2000]
        judge_usage = judge_backend.usage if judge_backend is not None else None
        model_usage = be.usage.normalized()
        judge_normalized = (
            judge_usage.normalized()
            if judge_usage is not None and hasattr(judge_usage, "normalized")
            else {}
        )
        model_cost = (
            round(be.usage.cost_usd, 4) if be.usage.cost_measured else None
        )
        judge_cost = (
            round(judge_usage.cost_usd, 4)
            if judge_usage is not None and judge_usage.cost_measured
            else None
        )
        cost_measured = be.usage.cost_measured and (
            judge_usage is None or judge_usage.cost_measured
        )
        status = _result_status(
            be, judge_usage, error, judge_failure, correct=correct
        )
        row = {
            "task_id": telemetry_id(q["question_id"]),
            "session_id": context_stats.get("context_session_id") if arm == "naru" else None,
            "turn_id": None,
            "model": model,
            "qid": q["question_id"],
            "type": q["question_type"],
            "arm": arm,
            "backend": getattr(be, "label", model),
            "judge_backend": (
                getattr(judge_backend, "label", judge_model)
                if judge_backend is not None
                else None
            ),
            "correct": correct,
            "gold": q["answer"],
            "answer": (
                f"HARNESS: {error}" if error is not None and not judge_failure
                else candidate
            ),
            "turns": turns,
            "peak_view_tokens": peak,
            "prompt_tokens_estimated": be.usage.prompt_tokens_estimated,
            "peak_prompt_tokens_estimated": be.usage.peak_prompt_tokens_estimated,
            "backend_calls": be.usage.attempts,
            "seconds": round(time.time() - t0, 1),
            "billed_input": be.usage.billed_input,
            "fresh_input": be.usage.input_tokens,
            "cache_creation": be.usage.cache_creation,
            "cache_read": be.usage.cache_read,
            "output": be.usage.output_tokens,
            "cost": model_cost,
            "judge_cost": judge_cost,
            "errors": status["errors"],
            "judge_errors": status["judge_errors"],
            "call_retries": be.usage.call_retries,
            "empty_retries": be.usage.empty_retries,
            "latency_ms": round((time.time() - t0) * 1000, 3),
            "pricing_version": os.environ.get("NARU_PRICING_VERSION") or None,
            "task_cost_usd": (
                round(
                    be.usage.cost_usd
                    + (judge_usage.cost_usd if judge_usage else 0),
                    4,
                )
                if cost_measured
                else None
            ),
            "provider_usage": dict(getattr(be.usage, "native_usage", {})),
            "judge_provider_usage": dict(
                getattr(judge_usage, "native_usage", {})
            ),
        }
        row.update(status)
        row.update(_context_result_fields(context_stats, jev_bridge))
        row.update(model_usage)
        if judge_normalized:
            row["judge_usage_normalized"] = judge_normalized
        if trace is not None:
            row["trace"] = trace
        if error is not None and judge_failure:
            row["judge_error"] = f"HARNESS: {error}"
        row.update(rsm)
        if owns_jev_bridge and jev_bridge is not None:
            jev_bridge.close()
        for backend_instance in (be, judge_backend):
            close = getattr(backend_instance, "close", None)
            if callable(close):
                close()
        return row

    rsm = {}
    try:
        if arm == "naru":
            ms, index = ingest(q, build_index=not no_index)
            try:
                ans, turns, peak = run_naru(
                    ms,
                    q["question"],
                    be,
                    question_date=q.get("question_date"),
                    max_turns=max_turns,
                    budget=budget,
                    verbose=verbose,
                    trace=trace,
                    rubric=LONGMEMEVAL_RUBRIC if rubric else None,
                    index=index,
                    context_mode=context_policy,
                    context_evaluator=jev_bridge,
                    context_stats=context_stats,
                )
            finally:
                discard_log(ms)
        elif arm == "rsm":
            body, rsm = rsm_context(
                q,
                q["question"],
                rsm_k,
                rsm_tau,
                rsm_chunk_turns,
                rsm_budget,
                argv=rsm_argv,
            )
            prompt = _question_prompt(q, body)
            ans, turns, peak = be(prompt, system=FULL_SYSTEM), 1, 0
        else:
            prompt = build_prompt(q, arm, rag_k)
            ans, turns, peak = be(prompt, system=FULL_SYSTEM), 1, 0
    except Exception as error:
        return result("", 0, 0, error=error)

    candidate = (ans or "").strip()[:2000]
    try:
        jb = get_backend(judge_model)
        ok = judge(q, candidate, jb)
    except Exception as error:
        return result(
            candidate, turns, peak, judge_backend=locals().get("jb"),
            error=error, judge_failure=True,
        )
    return result(candidate, turns, peak, correct=ok, judge_backend=jb)


def wilson(k, n, z=1.96):
    """95% confidence interval on a proportion, Wilson score.

    Not the textbook normal approximation: at n=24 with p near 0.8 that one
    produces a bound above 1.0, which reads as a measurement and isn't. Wilson
    stays inside [0,1] at every n this harness can afford to run.
    """
    if not n:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def report(rows, label, floor, measured=True):
    """Print one arm's results. `measured` is the backend's own reports_tokens:
    a generic pipe never touches the token counters, so billed-in and cost are
    reported as not measurable rather than as zero."""
    if not rows:
        return
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / n
    # A question whose run lost a call is scored wrong above, because that is
    # what the row says. It is not evidence about the arm, and a multi-turn arm
    # meets a per-call failure rate proportionally more often. Both numbers are
    # printed: the gap between them is the flakiness tax.
    ok_rows = [r for r in rows if not r.get("errors") and not r.get("judge_errors")]
    lost = n - len(ok_rows)
    bi = sum(r["billed_input"] for r in rows)
    # floor is None when the backend reports no usage at all. Subtracting 0 and
    # printing the result would present "not measured" as a measurement.
    net = (
        None
        if floor is None
        else sum(max(0, r["billed_input"] - floor * r["turns"]) for r in rows)
    )
    cost_measured = all(
        isinstance(r.get("cost"), (int, float))
        and not isinstance(r.get("cost"), bool)
        and isinstance(r.get("judge_cost"), (int, float))
        and not isinstance(r.get("judge_cost"), bool)
        for r in rows
    )
    model_cost = sum(r["cost"] for r in rows) if cost_measured else 0
    judge_cost = sum(r["judge_cost"] for r in rows) if cost_measured else 0
    cost = model_cost + judge_cost
    successful = sum(
        bool(
            r.get(
                "task_success",
                r.get("correct") and not r.get("errors") and not r.get("judge_errors"),
            )
        )
        for r in rows
    )
    bar = "#" * round(acc * 28) + "." * (28 - round(acc * 28))
    lo, hi = wilson(n_correct, n)
    # The interval is printed on the same line as the accuracy on purpose. A
    # bare percentage invites a reader to compare two arms that overlap.
    print(
        f"\n  {label:8} {bar} {acc * 100:5.1f}%  ({n_correct}/{n})"
        f"  95% CI {lo * 100:.0f}-{hi * 100:.0f}%"
    )
    if lost and ok_rows:  # every row lost leaves nothing to rescore
        k2 = sum(r["correct"] for r in ok_rows)
        n2 = len(ok_rows)
        l2, h2 = wilson(k2, n2)
        print(
            f"           excluding {lost} harness-lost question(s): "
            f"{100 * k2 / n2:.1f}%  ({k2}/{n2})  95% CI {l2 * 100:.0f}-{h2 * 100:.0f}%"
        )
    if measured:
        print(
            f"           billed-in {bi / n:>9,.0f}/q   net-of-harness "
            + ("not measurable" if net is None else f"{net / n:>9,.0f}/q")
        )
    else:
        print("           billed-in  not measurable   net-of-harness not measurable")
    calls = sum(r.get("backend_calls", r["turns"]) for r in rows) / n
    prompt_total = sum(
        r.get("prompt_tokens_estimated", r["peak_view_tokens"]) for r in rows
    ) / n
    prompt_peak = sum(
        r.get("peak_prompt_tokens_estimated", r["peak_view_tokens"]) for r in rows
    ) / n
    print(
        f"           out {sum(r['output'] for r in rows) / n:>7,.0f}/q   "
        f"turns {sum(r['turns'] for r in rows) / n:>4.1f}   calls {calls:>4.1f}   "
        + (
            f"${cost:.2f} total"
            if measured and cost_measured
            else "cost not measurable"
        )
    )
    print(
        f"           prompt-est total {prompt_total:>8,.0f}t/q   "
        f"peak {prompt_peak:>8,.0f}t/q"
    )
    if label == "naru":
        print(
            f"           dynamic-view peak "
            f"{sum(r['peak_view_tokens'] for r in rows) / n:>6,.0f}t/q"
        )
    if measured and cost_measured:
        # The arm's own dollars, and the cache share that makes a token ratio
        # and a money ratio disagree. Both are published columns; printing
        # only a combined total left them hand-computed and unreproducible.
        cr = sum(r.get("cache_read", 0) for r in rows)
        print(
            f"           model ${model_cost / n:.4f}/q   "
            f"judge ${judge_cost / n:.4f}/q   "
            f"cache-read {100 * cr / max(1, bi):.0f}% of billed input"
        )
        task_costs = [r.get("task_cost_usd") for r in rows]
        if successful and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in task_costs
        ):
            task_total = sum(task_costs)
            suffix = (
                " (Jev cost omitted; wall overhead measured)"
                if any(r.get("context_jev_calls") for r in rows)
                else ""
            )
            print(
                f"           task cost / successful task "
                f"${task_total / successful:.4f}{suffix}  ({successful}/{n} successful)"
            )
        elif successful:
            print("           task cost / successful task not measurable")
        else:
            print("           task cost / successful task not measurable (0 successes)")
    if label == "rsm":
        selected = sum(r.get("rsm_selected_members", 0) for r in rows) / n
        source = sum(r.get("rsm_context_tokens", 0) for r in rows) / n
        print(
            f"           atoms embed-in {sum(r.get('rsm_embedding_input_tokens', 0) for r in rows) / n:.0f}t/q  "
            f"atoms {sum(r.get('rsm_atoms', 0) for r in rows) / n:.1f}  "
            f"members {sum(r.get('rsm_members', 0) for r in rows) / n:.1f}  "
            f"selected {selected:.1f}  "
            f"embed {sum(r.get('rsm_embedding_seconds', 0) for r in rows) / n:.3f}s/q  "
            f"index {sum(r.get('rsm_indexing_seconds', 0) for r in rows) / n:.3f}s/q  "
            f"source {source:.0f}t/q"
        )
        print("           model cost excludes embedding-provider cost")
    if label == "naru":
        selected = sum(r.get("context_selected_tokens", 0) for r in rows) / n
        dropped = sum(r.get("context_dropped_tokens", 0) for r in rows) / n
        jev_calls = sum(r.get("context_jev_calls", 0) for r in rows)
        jev_fallbacks = sum(r.get("context_fallbacks", 0) for r in rows)
        jev_latency = sum(r.get("context_jev_latency_ms", 0) for r in rows)
        jev_startup = sum(
            r.get("context_jev_startup_latency_ms", 0) for r in rows
        )
        print(
            f"           context selected {selected:,.0f}t/q  "
            f"dropped {dropped:,.0f}t/q  Jev calls {jev_calls}  "
            f"fallbacks {jev_fallbacks}  "
            f"Jev wall {jev_latency:,.1f}ms  startup {jev_startup:,.1f}ms"
        )
    errs = sum(r["errors"] for r in rows)
    cretries = sum(r.get("call_retries", 0) for r in rows)
    # Rows written before judge_errors existed have no such key. Summing them
    # to 0 would print "0 judge errors" for a run where nobody counted, which
    # is ADR 0002's mistake in miniature: not measured rendered as measured.
    jerrs = (
        sum(r["judge_errors"] for r in rows)
        if all("judge_errors" in r for r in rows)
        else None
    )
    retries = sum(r.get("empty_retries", 0) for r in rows)
    if errs or retries or jerrs or cretries:
        judge_part = "judge errors not recorded" if jerrs is None else f"{jerrs} judge errors"
        print(
            f"           {errs} backend errors, {judge_part}, "
            f"{retries} empty-reply retries, {cretries} call retries"
        )
    by = {}
    for r in rows:
        by.setdefault(r["type"], []).append(r["correct"])
    print(
        "           "
        + "  ".join(
            f"{k.replace('single-session-', 'ss-')[:18]} {sum(v)}/{len(v)}"
            for k, v in sorted(by.items())
        )
    )


def separability(rows, arms):
    """State which arm differences are real and which are this run's luck.

    CLAUDE.md says to report a noise floor rather than imply a result. That
    rule lived only in prose, so every run needed a human to remember it. It
    is a print statement now.

    This answers "is this gap real on these questions". noise.py owns the
    different question of how far a rerun moves, and needs replicates for it.
    """
    # A row whose run errored has correct=False, which is indistinguishable
    # from a wrong answer. McNemar reads only the discordant pairs, so one
    # contaminated question moves p hard: on the published run full-vs-rag is
    # 1-vs-6 (p=0.125, "not separable"); had that single full win been a CLI
    # timeout it is 0-vs-6 (p=0.031) and the harness prints a significance
    # claim manufactured by a hung subprocess. Drop them from the pairing —
    # the shared-key intersection then removes each dropped question from
    # both arms, which is what a paired test requires.
    if len(set(arms)) != len(arms):
        raise ValueError("comparison arms must be unique")
    pairs = len(list(combinations(arms, 2)))
    if not pairs:
        return
    verdicts, dropped = {}, {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        qids = [r["qid"] for r in rs]
        if len(qids) != len(set(qids)):
            raise ValueError(f"duplicate result rows for arm {arm}")
        v = {
            r["qid"]: bool(r["correct"])
            for r in rs
            if not r.get("errors") and not r.get("judge_errors")
        }
        dropped[arm] = len(rs) - len(v)
        if v:
            verdicts[arm] = v
    if len(verdicts) < 2:
        return
    # Three arms means three tests. At an uncorrected 0.05 each, at least one
    # pair reads REAL in ~6% of runs where nothing separates, against ~2% for
    # a single pair. The verdict is the harness's published claim, so it is
    # the number that has to be honest.
    alpha = 0.05 / pairs
    note = f", Bonferroni for {pairs} planned pairs" if pairs > 1 else ""
    print(
        f"\n  separability — paired McNemar on the questions the arms"
        f" disagree on{note}"
    )
    if any(dropped.values()):
        drops = ", ".join(f"{a} {d}" for a, d in dropped.items() if d)
        print(f"    dropped from the pairing (run errored): {drops}")
    # verdicts is built by iterating arms, so its key order is already the
    # filtered arm list — and unlike a list it cannot yield `full vs full`.
    for a, b in combinations(verdicts, 2):
        va, vb = verdicts[a], verdicts[b]
        shared = len(va.keys() & vb.keys())
        if not shared:
            # results/published/README.md documents comparing across two
            # loaded result files. Two different splits share no question
            # ids, and a traceback is a worse answer than saying so.
            print(f"    {a:5} vs {b:5}   no shared questions — not comparable")
            continue
        only_a, only_b, p = mcnemar(va, vb)
        # Questions both arms got right cancel in the subtraction, so the
        # gap over the shared set is exactly the difference of the two
        # disagreement counts mcnemar already computed.
        gap = 100 * (only_b - only_a) / shared
        mark = (
            f"REAL at p<{alpha:.3g}"
            if p < alpha
            else "not separable — this run's luck"
        )
        ahead = b if gap > 0 else a
        print(
            f"    {a:5} vs {b:5} {abs(gap):5.1f} pts to {ahead:5}  "
            f"{a} only {only_a}, {b} only {only_b}   p={p:.3f}   {mark}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oracle", choices=["oracle", "s", "m"])
    ap.add_argument("-n", type=int, default=12)
    # CLAUDE.md: rag is the control and must never be quietly dropped, so it is
    # in the default. A two-arm run has to be asked for.
    ap.add_argument("--arms", default="full,rag,naru")
    ap.add_argument(
        "--rag-k",
        type=int,
        default=8,
        help="hits the rag arm pastes into the prompt (~2.5k tokens at 8)",
    )
    ap.add_argument("--rsm-k", type=int, default=6, help="atoms the rsm arm retrieves")
    ap.add_argument(
        "--rsm-tau",
        type=float,
        default=0.85,
        help="RSM merge threshold; paper BGE value, calibrate other embedding spaces",
    )
    ap.add_argument(
        "--rsm-chunk-turns",
        type=int,
        default=5,
        help="Event Log turns per RSM member",
    )
    ap.add_argument(
        "--rsm-budget", type=int, default=4000, help="RSM packed-context token budget"
    )
    ap.add_argument(
        "--model",
        default=None,
        help="answerer model; defaults to the resolved provider's configured model",
    )
    ap.add_argument(
        "--judge-model",
        default=None,
        help="judge model; defaults to the answerer model",
    )
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qtype", default=None)
    ap.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default="auto",
        help="main and judge runtime; auto prefers Codex, then Claude",
    )
    ap.add_argument(
        "--backend-command",
        default=None,
        help="stdin provider command when --backend command is selected",
    )
    ap.add_argument(
        "--harness-floor",
        type=int,
        default=None,
        help="CLI input-token overhead per call; measured if omitted",
    )
    ap.add_argument("--tag", default="run")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--no-rubric", action="store_true", help="ablate the per-dataset layout rubric"
    )
    ap.add_argument(
        "--no-index",
        action="store_true",
        help="ablate the ingestion-time eviction index (CLAUDE.md, ADR 0003)",
    )
    ap.add_argument(
        "--context-policy",
        choices=CONTEXT_POLICIES,
        default=os.environ.get("NARU_CONTEXT_POLICY", "deterministic"),
        help=(
            "Naru mutable-context policy; Jev modes require --jev-command, "
            "NARU_JEV, or --jev-host-stdio"
        ),
    )
    ap.add_argument(
        "--jev-command",
        default=None,
        help="JSON stdin/stdout adapter for mcp__jev__evaluate",
    )
    ap.add_argument(
        "--jev-host-stdio",
        action="store_true",
        help="broker Jev requests through the host MCP binding on stdin",
    )
    a = ap.parse_args()
    if a.backend_command and a.backend != "command":
        sys.exit("--backend-command requires --backend command")
    try:
        backend_kind, backend_command = resolve_backend(
            a.backend, a.backend_command, model=a.model
        )
    except (ValueError, FileNotFoundError) as error:
        sys.exit(str(error))
    # Workers inherit one resolved provider. Keep the environment bridge for
    # existing callers, while the resolver gives the CLI a general boundary.
    os.environ["NARU_BACKEND"] = backend_command or backend_kind
    a.model = a.model or default_model_for_backend(backend_kind)
    a.judge_model = a.judge_model or a.model
    if a.jev_host_stdio:
        if a.context_policy not in ("shadow", "jev"):
            sys.exit("--jev-host-stdio requires --context-policy shadow or jev")
        if a.jev_command or os.environ.get("NARU_JEV"):
            sys.exit(
                "--jev-host-stdio cannot be combined with --jev-command or NARU_JEV"
            )
        print(
            "host MCP Jev mode: model jobs use a parent thread pool; Jev requests "
            "are serialized",
            file=sys.stderr,
        )

    # A typo'd arm would otherwise run as `full` and quietly corrupt the run.
    # Checked before the data split is read or any paid model call is made, so
    # a typo costs nothing.
    arms = a.arms.split(",")
    unknown = unknown_arms(arms)
    if unknown:
        sys.exit(f"unknown arm(s): {unknown} — pick from {', '.join(ARMS)}")
    if a.context_policy not in CONTEXT_POLICIES:
        sys.exit(
            f"unknown context policy {a.context_policy!r} — pick from "
            f"{', '.join(CONTEXT_POLICIES)}"
        )
    if len(set(arms)) != len(arms):
        sys.exit("--arms must not contain duplicates")
    # SQLite reads a negative LIMIT as NO limit, so --rag-k -1 pastes the whole
    # history and the rag arm silently becomes a second full arm at ~50x the
    # cost, still labelled rag. 0 is the mirror: empty context, every answer
    # wrong, nothing in the output naming why.
    if a.rag_k < 1:
        sys.exit(f"--rag-k must be >= 1, got {a.rag_k}")
    if a.rsm_k < 1:
        sys.exit(f"--rsm-k must be >= 1, got {a.rsm_k}")
    if not -1 <= a.rsm_tau <= 1:
        sys.exit(f"--rsm-tau must be between -1 and 1, got {a.rsm_tau}")
    if a.rsm_chunk_turns < 1:
        sys.exit(f"--rsm-chunk-turns must be >= 1, got {a.rsm_chunk_turns}")
    if a.rsm_budget < 1:
        sys.exit(f"--rsm-budget must be >= 1, got {a.rsm_budget}")
    if a.context_policy != "deterministic" and not (
        a.jev_command or os.environ.get("NARU_JEV") or a.jev_host_stdio
    ):
        print(
            f"context policy {a.context_policy!r} requested without a Jev adapter; "
            "Naru will use deterministic fallback"
        )
    rsm_argv = None
    if "rsm" in arms:
        try:
            rsm_argv = embedder_argv()
        except (ValueError, FileNotFoundError) as e:
            sys.exit(str(e))

    probe_backend = get_backend(a.model)
    try:
        measured = probe_backend.reports_tokens
    finally:
        probe_backend.close()
    if a.harness_floor is None:
        a.harness_floor = measure_floor(a.model)
        if a.harness_floor is None:
            print(
                "harness floor NOT measurable with this backend — token columns "
                "are marked not measurable and net-of-harness is omitted"
            )
        else:
            print(f"measured harness floor: {a.harness_floor:,} input tok/call")

    qs = load(a.split, a.n, qtype=a.qtype)
    print(
        f"LongMemEval-{a.split}  n={len(qs)}  arms={arms}  "
        f"backend={backend_kind}  model={a.model}  "
        f"judge={a.judge_model}  budget={a.budget}t  max_turns={a.max_turns}"
    )
    avg_hist = sum(est(history_text(q)) for q in qs) / len(qs)
    print(f"avg history per question: {avg_hist:,.0f} tokens")

    jobs = [(q, arm) for arm in arms for q in qs]
    rows = []
    host_jev = host_stdio_evaluator() if a.jev_host_stdio else None
    executor = _ParentExecutor if a.jev_host_stdio else ProcessPoolExecutor
    with executor(max_workers=a.workers) as ex:
        futs = {
            ex.submit(
                one,
                q,
                arm,
                a.model,
                a.judge_model,
                a.max_turns,
                a.budget,
                a.verbose,
                # named: three of the last four are bools and a silent
                # transposition here would corrupt a paid run.
                rubric=not a.no_rubric,
                no_index=a.no_index,
                rag_k=a.rag_k,
                rsm_k=a.rsm_k,
                rsm_tau=a.rsm_tau,
                rsm_chunk_turns=a.rsm_chunk_turns,
                rsm_budget=a.rsm_budget,
                rsm_argv=rsm_argv,
                context_policy=a.context_policy,
                jev_command=a.jev_command,
                jev_evaluator=(
                    host_jev
                    if a.jev_host_stdio
                    and arm == "naru"
                    and a.context_policy in ("shadow", "jev")
                    else None
                ),
            ): (q, arm)
            for q, arm in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                q, arm = futs[f]
                r = {
                    "task_id": telemetry_id(q["question_id"]),
                    "session_id": None,
                    "turn_id": None,
                    "model": a.model,
                    "qid": q["question_id"],
                    "type": q["question_type"],
                    "arm": arm,
                    "backend": backend_label(os.environ.get("NARU_BACKEND")),
                    "judge_backend": backend_label(
                        os.environ.get("NARU_BACKEND")
                    ),
                    "correct": False,
                    "gold": q["answer"],
                    "answer": f"HARNESS: {e}",
                    "turns": 0,
                    "peak_view_tokens": 0,
                    "prompt_tokens_estimated": 0,
                    "peak_prompt_tokens_estimated": 0,
                    "backend_calls": 0,
                    "seconds": 0,
                    "billed_input": 0,
                    "fresh_input": 0,
                    "cache_creation": 0,
                    "cache_read": 0,
                    "output": 0,
                    "cost": 0,
                    "judge_cost": 0,
                    "errors": 1,
                    "judge_errors": 0,
                    "call_retries": 0,
                    "empty_retries": 0,
                    "context_policy": a.context_policy if arm == "naru" else "not_applicable",
                    "context_considered_tokens": 0,
                    "context_selected_tokens": 0,
                    "context_dropped_tokens": 0,
                    "context_peak_selected_tokens": 0,
                    "context_selected_blocks": [],
                    "context_omitted_blocks": [],
                    "context_jev_calls": 0,
                    "context_jev_attempts": 0,
                    "context_jev_input_tokens": 0,
                    "context_jev_cache_read_tokens": 0,
                    "context_jev_cache_write_tokens": 0,
                    "context_jev_output_tokens": 0,
                    "context_jev_latency_ms": 0,
                    "context_jev_startup_latency_ms": 0,
                    "context_jev_provider_latency_ms": 0,
                    "context_jev_transport_overhead_ms": 0,
                    "context_jev_transport": None,
                    "context_jev_errors": 0,
                    "context_fallbacks": 0,
                    "context_jev_unavailable": 0,
                    "tool_call_count": 0,
                    "tool_result_tokens": 0,
                    "model_switch": False,
                    "subagent_call_count": 0,
                    "latency_ms": 0,
                    "task_success": False,
                    "failure_class": "harness",
                    "pricing_version": os.environ.get("NARU_PRICING_VERSION") or None,
                    "task_cost_usd": 0,
                    "cost_includes_jev": True,
                    "provider_usage": {},
                    "judge_provider_usage": {},
                    "jev_provider_usage": {},
                    "jev_usage": {},
                }
                if arm == "rsm":
                    r.update(
                        rsm_embedding_input_tokens=0,
                        rsm_atoms=0,
                        rsm_members=0,
                        rsm_selected_members=0,
                        rsm_embedding_seconds=0,
                        rsm_indexing_seconds=0,
                        rsm_context_tokens=0,
                    )
            rows.append(r)
            mark = "+" if r["correct"] else "-"
            print(
                f"\r  [{i}/{len(jobs)}] {mark} {r['arm']:6} {r['qid'][:22]:22}",
                end="",
                flush=True,
            )
    if host_jev is not None:
        host_jev.close()
    print()

    for arm in arms:
        report([r for r in rows if r["arm"] == arm], arm, a.harness_floor, measured)
    separability(rows, arms)

    out = DATA.parent / "results" / f"{a.tag}_{a.split}_n{len(qs)}.json"
    out.parent.mkdir(exist_ok=True)
    # vars(a) records the requested settings. Stamp the provider that actually
    # answered, and keep it separate from the model name for auditability.
    cfg = dict(vars(a))
    cfg["result_format"] = RESULT_FORMAT
    cfg["resolved_backend"] = backend_kind
    cfg["backend"] = backend_label(os.environ.get("NARU_BACKEND"))
    cfg["backend_fingerprint"] = backend_fingerprint(
        backend_kind, backend_command
    )
    cfg["judge_resolved_backend"] = backend_kind
    cfg["judge_backend"] = backend_label(os.environ.get("NARU_BACKEND"))
    cfg["judge_backend_fingerprint"] = backend_fingerprint(
        backend_kind, backend_command
    )
    # Command arguments can carry credentials. Keep only the safe executable
    # label in result metadata and the one-way fingerprint above.
    cfg["backend_command"] = (
        backend_label(backend_command) if backend_command else None
    )
    cfg["jev_command"] = backend_label(
        a.jev_command or os.environ.get("NARU_JEV")
    ) if (a.jev_command or os.environ.get("NARU_JEV")) else None
    cfg["rsm_embedder"] = rsm_argv[0] if rsm_argv else None
    cfg["tokens_measured"] = measured
    cfg["pricing_version"] = os.environ.get("NARU_PRICING_VERSION") or None
    cfg["prompt_tokens_estimated"] = True
    json.dump({"config": cfg, "rows": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


def demo():
    """Offline self-check. No API calls, no results written.

    Reads results/published/, which is committed, to check the README's
    published claims against the rows they came from. It does not touch
    data/, which is gitignored and 277MB.

    bench.py was the one module without one, which is how `rag` could have
    silently run as `full`.
    """
    # Wilson, not the normal approximation: at 24/24 the textbook interval
    # reaches past 1.0 and prints a bound that cannot happen.
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(24, 24)[1] <= 1.0
    assert wilson(0, 24)[0] >= 0.0
    lo24, hi24 = wilson(19, 24)
    lo96, hi96 = wilson(76, 96)
    assert (hi96 - lo96) < (hi24 - lo24), "more questions must narrow the interval"

    assert iso("2023/04/10 (Mon) 17:50") == "2023-04-10T17:50"
    assert iso(None) == "1970-01-01T00:00", "a missing date must not crash ingest"

    class VerdictBackend:
        def __init__(self, verdict):
            self.verdict = verdict
            self.usage = type("Usage", (), {"errors": 0})()

        def __call__(self, prompt, system=None, nudge=None):
            return self.verdict

    judge_q = {"question": "q", "answer": "a"}
    malformed = VerdictBackend("CORRECTED")
    assert not judge(judge_q, "a", malformed, votes=1)
    assert malformed.usage.errors == 1
    exact = VerdictBackend(" correct \n")
    assert judge(judge_q, "a", exact, votes=1)

    # rag_context returns hits in LOG order, not BM25 rank order. The last row
    # repeats the term most, so BM25 ranks it first and the sort must move it
    # back to the end — otherwise the arm feeds the model a scrambled history.
    ms = MemorySurface(":memory:")
    for i, body in enumerate(
        [
            "[Session 1 | 2023-01-01] user: I bought a kayak",
            "[Session 2 | 2023-02-01] user: the kayak leaks",
            "[Session 3 | 2023-03-01] user: kayak kayak kayak, I sold the kayak",
        ],
        1,
    ):
        ms.append("user", body, kind="context_msg", session_id=f"s{i}",
                  created_at=f"2023-0{i}-01T00:00")
    # a question is dataset text, not a query: bare AND/OR/NOT are FTS5
    # operators and used to raise, forfeiting the question for this arm alone
    for hostile in ("kayak AND NOT leaks", "kayak OR sold", "", "   "):
        rag_context(ms, hostile, 3)
    assert "kayak" in rag_context(ms, "kayak AND NOT leaks", 3)
    assert rag_context(ms, "   ", 3) == ""

    ranked = [h["seq"] for h in ms.search("kayak", k=3)]
    assert ranked[0] == 3, f"expected BM25 to rank seq 3 first, got {ranked}"
    ctx = rag_context(ms, "kayak", 3)
    assert ctx.index("Session 1") < ctx.index("Session 2") < ctx.index("Session 3")

    # separability must call the published run a tie, and must call a blowout
    # separable — a function that only ever says "not separable" is not a check.
    def rows_for(arm, k, n):
        return [
            {"arm": arm, "qid": f"q{i}", "correct": i < k, "type": "t"}
            for i in range(n)
        ]

    def sep_out(rows, arms):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            separability(rows, arms)
        return buf.getvalue()

    # One buffer per scenario, and each assert pinned to the scenario that has
    # to produce it. Sharing one buffer let the PAIR of calls satisfy both
    # asserts, so swapping the two branches of `mark` — which reports the
    # published tie as a significant result — passed this check.
    both = ["full", "naru"]
    tie = sep_out(rows_for("full", 16, 24) + rows_for("naru", 19, 24), both)
    blowout = sep_out(rows_for("full", 2, 24) + rows_for("naru", 23, 24), both)
    assert "not separable" in tie and "REAL" not in tie, tie
    # pin the gap AND its direction: it is derived from the McNemar counts
    # rather than recomputed, so a swapped subtraction prints the wrong arm as
    # the winner and every other assertion here would wave it through.
    assert "12.5 pts to naru" in tie, tie
    assert "REAL at p<0.05" in blowout and "not separable" not in blowout, blowout
    # paired beats independent intervals: 16/24 vs 21/24 overlap as Wilson
    # intervals, but disagree 1-vs-6 when paired. Losing that is why the test
    # changed.
    # The claim this whole test change rests on, exercised rather than
    # asserted in a comment: 16/24 and 21/24 overlap as independent intervals
    # while the paired test sees a 1-vs-6 split.
    assert wilson(16, 24)[1] > wilson(21, 24)[0], "intervals do overlap"
    paired = {f"q{i}": i < 16 for i in range(24)}
    other = {f"q{i}": 0 < i <= 21 for i in range(24)}
    assert mcnemar(paired, other) == (1, 6, 0.125), mcnemar(paired, other)
    agree = {f"q{i}": i < 16 for i in range(24)}
    better = {f"q{i}": i < 16 or i >= 22 for i in range(24)}
    # The p must be a literal. Comparing the slot to itself is a tautology
    # that a one-sided p-value passes. 2 discordant pairs, both one way.
    assert mcnemar(agree, better) == (0, 2, 0.5)
    assert mcnemar(agree, agree)[2] == 1.0, "identical arms cannot differ"
    # A question whose run errored must leave the pairing, not score as wrong.
    # Scored wrong it is a discordant pair and moves p; dropped it is removed
    # from both arms, which is what the paired test requires.
    # q16 is a discordant pair: full got it wrong, naru right. Erroring it
    # must remove it from the counts, not leave it scored as a full loss.
    contaminated = rows_for("full", 16, 24) + rows_for("naru", 19, 24)
    for r in contaminated:
        if r["arm"] == "full" and r["qid"] == "q16":
            r["errors"] = 1
    out = sep_out(contaminated, both)
    assert "dropped from the pairing (run errored): full 1" in out, out
    # and the dropped question must really leave the counts, not just be named
    clean = sep_out(rows_for("full", 16, 24) + rows_for("naru", 19, 24), both)
    assert "full only 0, naru only 3" in clean, clean
    assert "full only 0, naru only 2" in out, out
    # Three arms means three tests, so the threshold must tighten — pinned on
    # the split that turns on it. 6 discordant pairs all one way is p=0.031:
    # REAL against an uncorrected 0.05, not separable against 0.05/3.
    two_arm = sep_out(rows_for("full", 18, 24) + rows_for("naru", 24, 24), both)
    assert "REAL at p<0.05" in two_arm, two_arm
    three = (
        rows_for("full", 18, 24) + rows_for("rag", 18, 24) + rows_for("naru", 24, 24)
    )
    out3 = sep_out(three, ["full", "rag", "naru"])
    assert "Bonferroni for 3 planned pairs" in out3, out3
    assert "REAL" not in out3, out3
    assert "p=0.031" in out3, out3
    failed_arm = rows_for("full", 18, 24) + rows_for("naru", 24, 24)
    failed_arm += [
        {
            "arm": "rag", "qid": f"q{i}", "correct": False,
            "type": "t", "errors": 1,
        }
        for i in range(24)
    ]
    failed_out = sep_out(failed_arm, ["full", "rag", "naru"])
    assert "Bonferroni for 3 planned pairs" in failed_out, failed_out
    assert "REAL" not in failed_out and "p=0.031" in failed_out, failed_out
    try:
        separability(rows_for("full", 1, 2), ["full", "full"])
    except ValueError as error:
        assert "unique" in str(error), error
    else:
        raise AssertionError("duplicate comparison arms were accepted")
    duplicate_rows = rows_for("full", 1, 2) + rows_for("naru", 1, 2)
    duplicate_rows.append(dict(duplicate_rows[0]))
    try:
        separability(duplicate_rows, both)
    except ValueError as error:
        assert "duplicate result rows" in str(error), error
    else:
        raise AssertionError("duplicate result rows were accepted")
    # A judge failure returns "" for every vote, which reads as WRONG. It must
    # leave the pairing too, not just a hard backend error.
    jrows = rows_for("full", 16, 24) + rows_for("naru", 19, 24)
    for r in jrows:
        if r["arm"] == "full" and r["qid"] == "q16":
            r["judge_errors"] = 1
    assert "full 1" in sep_out(jrows, both)
    # one arm alone has nothing to compare against and must print nothing
    assert sep_out(rows_for("rag", 5, 24), ["rag"]) == ""

    # The arm dispatch itself, which used to be reachable only through a paid
    # call: break it and every rag question is answered from the full history
    # while the row, the report and the published JSON still say "rag".
    synth = {
        "question": "what did I say about kayaks",
        "question_date": "2023/04/10 (Mon) 17:50",
        "answer": "-",
        "question_id": "synthetic",
        "question_type": "t",
        "haystack_dates": ["2023/01/01 (Sun) 10:00", "2023/02/01 (Wed) 10:00"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "I bought a kayak"}],
            [{"role": "user", "content": "unrelated tarragon and bicycles"}],
        ],
    }
    full_p = build_prompt(synth, "full")
    old_backend = os.environ.get("NARU_BACKEND")
    os.environ["NARU_BACKEND"] = "cat"
    try:
        saved = one(synth, "full", HAIKU, HAIKU, 1, 6000, False)
    finally:
        if old_backend is None:
            os.environ.pop("NARU_BACKEND", None)
        else:
            os.environ["NARU_BACKEND"] = old_backend
    expected_answer = (FULL_SYSTEM + "\n\n" + full_p).strip()[:2000]
    assert len(expected_answer) > 400 and saved["answer"] == expected_answer
    expected_prompt = est(FULL_SYSTEM + "\n\n" + full_p)
    assert saved["prompt_tokens_estimated"] == expected_prompt
    assert saved["peak_prompt_tokens_estimated"] == expected_prompt
    assert saved["backend_calls"] == 1 and saved["peak_view_tokens"] == 0
    assert saved["task_id"] == "b3cc0475bb78a502" and saved["turn_id"] is None
    assert saved["model"] == HAIKU and saved["context_policy"] == "not_applicable"
    assert saved["backend"] == "cat" and saved["judge_backend"] == "cat"
    assert saved["context_input_tokens"] == expected_prompt
    assert saved["context_input_tokens_source"] == "estimated_prompt_chars_per_4"
    assert "provider_usage" in saved and "failure_class" in saved

    class KnownUsageBackend:
        def __init__(self):
            self.usage = Usage(
                attempts=1, calls=1, input_tokens=7,
                prompt_tokens_estimated=9, peak_prompt_tokens_estimated=9,
            )

        def __call__(self, prompt, system=None, nudge=None):
            return "saved answer"

    from unittest.mock import patch

    known = KnownUsageBackend()
    with patch.object(
        sys.modules[__name__], "get_backend",
        side_effect=[known, RuntimeError("judge unavailable")],
    ):
        failed_judge = one(synth, "full", HAIKU, HAIKU, 1, 6000, False)
    assert failed_judge["answer"] == "saved answer"
    assert failed_judge["billed_input"] == 7
    assert failed_judge["backend_calls"] == 1
    assert failed_judge["errors"] == 0 and failed_judge["judge_errors"] == 1
    old_kernel = os.environ.get("NARU_KERNEL")
    old_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory() as temp_root:
        tempfile.tempdir = temp_root
        os.environ["NARU_KERNEL"] = "sandbox"
        try:
            rag_p = build_prompt(synth, "rag", rag_k=1)
            leftovers = list(pathlib.Path(temp_root).glob("naru-bench-*"))
        finally:
            tempfile.tempdir = old_tempdir
            if old_kernel is None:
                os.environ.pop("NARU_KERNEL", None)
            else:
                os.environ["NARU_KERNEL"] = old_kernel
    assert not leftovers, f"rag left benchmark logs: {leftovers}"
    assert "kayak" in rag_p and "kayak" in full_p
    assert "tarragon" in full_p, "the full arm must carry the whole history"
    assert "tarragon" not in rag_p, "the rag arm must carry retrieved hits only"
    assert len(rag_p) < len(full_p)

    with ProcessPoolExecutor(max_workers=2) as ex:
        worker_pids = list(ex.map(_worker_pid, range(4)))
    assert all(pid != os.getpid() for pid in worker_pids), worker_pids
    with _ParentExecutor(max_workers=4) as ex:
        parent_future = ex.submit(lambda: "parent result")
    assert parent_future.result() == "parent result"

    # RSM groups chronological Event Log chunks by atom. The query ranks the
    # later alpha chunk first, but the packer must restore seq order inside
    # the selected atom and omit the unrelated atom.
    rsm_synth = {
        **synth,
        "question": "which alpha detail matters",
        "haystack_dates": [
            "2023/01/01 (Sun) 10:00",
            "2023/02/01 (Wed) 10:00",
            "2023/03/01 (Wed) 10:00",
        ],
        "haystack_session_ids": ["s1", "s2", "s3"],
        "haystack_sessions": [
            [{"role": "user", "content": "first alpha detail"}],
            [{"role": "user", "content": "unrelated boats detail"}],
            [{"role": "user", "content": "later alpha detail"}],
        ],
    }

    def fake_embed(texts):
        vectors = []
        for text in texts:
            if text == rsm_synth["question"] or "later alpha" in text:
                vectors.append((1.0, 0.0))
            elif "first alpha" in text:
                vectors.append((0.8, 0.6))
            else:
                vectors.append((0.0, 1.0))
        return vectors

    rsm_ctx, rsm_stats = rsm_context(
        rsm_synth,
        rsm_synth["question"],
        k=1,
        tau=0.7,
        chunk_turns=1,
        budget=200,
        vectorize=fake_embed,
    )
    assert "unrelated boats" not in rsm_ctx, rsm_ctx
    assert rsm_ctx.count("=== Atom") == 1, rsm_ctx
    assert rsm_ctx.index("first alpha") < rsm_ctx.index("later alpha"), rsm_ctx
    assert rsm_stats["rsm_atoms"] == 2 and rsm_stats["rsm_members"] == 3

    limited, _ = rsm_context(
        rsm_synth,
        rsm_synth["question"],
        k=2,
        tau=0.7,
        chunk_turns=1,
        budget=35,
        vectorize=fake_embed,
    )
    assert est(limited) <= 35, (est(limited), limited)
    # Opposite normalized vectors cancel to zero. Leave the current atom
    # unchanged and start a new atom instead of dividing by zero.
    opposite = {
        **rsm_synth,
        "haystack_dates": rsm_synth["haystack_dates"][:2],
        "haystack_session_ids": rsm_synth["haystack_session_ids"][:2],
        "haystack_sessions": rsm_synth["haystack_sessions"][:2],
    }
    _, zero_stats = rsm_context(
        opposite,
        opposite["question"],
        k=2,
        tau=-1,
        chunk_turns=1,
        budget=200,
        vectorize=lambda _: [(1.0, 0.0), (-1.0, 0.0), (1.0, 0.0)],
    )
    assert zero_stats["rsm_atoms"] == 2, zero_stats
    try:
        build_prompt(rsm_synth, "rsm")
    except ValueError as error:
        assert "direct prompt arm" in str(error), error
    else:
        raise AssertionError("the unused RSM prompt path survived")

    # The external boundary rejects bad JSON without a network request.
    bad_embed = [sys.executable, "-c", "print('{bad json')"]
    try:
        embed_vectors(bad_embed, ["one"])
    except ValueError as e:
        assert "invalid JSON" in str(e), e
    else:
        raise AssertionError("malformed NARU_EMBED response was accepted")

    # a run predating judge_errors must say so, not report zero of them
    def rep_out(rows):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(rows, "x", 0)
        return buf.getvalue()

    old_rows = rows_for("x", 1, 2)
    for r in old_rows:
        r.update(billed_input=1, output=1, turns=1, peak_view_tokens=1,
                 cost=0.0, judge_cost=0.0, errors=1)
    assert "judge errors not recorded" in rep_out(old_rows), rep_out(old_rows)
    # every row lost: nothing to rescore, and it must not divide by zero
    assert "excluding" not in rep_out(old_rows), rep_out(old_rows)
    # one of three lost: the excluding line appears and rescores the rest
    mixed = rows_for("x", 2, 3)
    for r in mixed:
        r.update(billed_input=1, output=1, turns=1, peak_view_tokens=1,
                 cost=0.0, judge_cost=0.0, errors=0, judge_errors=0)
    mixed[2]["errors"] = 1          # the wrong one was a harness loss
    out_mixed = rep_out(mixed)
    assert "excluding 1 harness-lost question(s): 100.0%  (2/2)" in out_mixed, out_mixed
    for r in old_rows:
        r["judge_errors"] = 0
    assert "0 judge errors" in rep_out(old_rows), rep_out(old_rows)

    # The published run, read from the rows rather than from literals retyped
    # here. Republish a different run and this fails, which is the point: the
    # README's claims and the data that produced them cannot drift apart.
    pub = sorted((DATA.parent / "results" / "published").glob("*.json"))
    assert pub, "results/published/ is committed; a missing one is a broken checkout"
    # Grouped by (model, n), because the two published runs use different
    # models and summing them together compares nothing.
    expected = {
        ("claude-haiku-4-5-20251001", 24): {
            "full": (16, 24),
            "naru": (19, 24),
            "rag": (21, 24),
        },
        ("claude-sonnet-5", 12): {
            "full": (8, 12),
            "naru": (11, 12),
            "rag": (9, 12),
        },
        ("claude-sonnet-5", 96): {
            "full": (64, 96),
            "naru": (72, 96),
            "rag": (59, 96),
        },
    }
    runs = {}
    for f in pub:
        d = json.loads(f.read_text())
        key = (d["config"]["model"], d["config"]["n"])
        for r in d["rows"]:
            if r["arm"] == "scroll":  # v12 predates the rename
                r["arm"] = "naru"
        runs[key] = runs.get(key, []) + d["rows"]
    assert set(runs) == set(expected), (sorted(runs), sorted(expected))
    for key, rows_ in runs.items():
        tally = {}
        for r in rows_:
            k_, n_ = tally.get(r["arm"], (0, 0))
            tally[r["arm"]] = (k_ + bool(r["correct"]), n_ + 1)
        assert tally == expected[key], (key, tally)
        # "read the accuracy column as a tie": every interval overlaps every
        # other, and no pair separates once the threshold is corrected.
        for x, y in combinations(tally, 2):
            assert wilson(*tally[x])[1] > wilson(*tally[y])[0], (key, x, y)
            assert wilson(*tally[y])[1] > wilson(*tally[x])[0], (key, x, y)
        assert "REAL" not in sep_out(rows_, ["full", "rag", "naru"]), key
    # the ranking reverses between the two models, which is the finding
    haiku = expected[("claude-haiku-4-5-20251001", 24)]
    sonnet = expected[("claude-sonnet-5", 12)]
    assert haiku["rag"][0] / 24 > haiku["naru"][0] / 24, "rag led on Haiku"
    assert sonnet["naru"][0] / 12 > sonnet["rag"][0] / 12, "naru led on Sonnet"

    # NARU_BACKEND can hold a credential and results/published/ is committed,
    # so the recorded provenance must be argv[0] and nothing after it.
    leaky = "sh -c 'curl -H \"Authorization: Bearer sk-secret\"'"
    assert backend_label(leaky) == "sh", backend_label(leaky)
    assert "sk-secret" not in backend_label(leaky)
    assert backend_label(None) == "auto"
    assert backend_label("   ") == "auto", "blank must not IndexError"
    assert backend_label("cat") == "cat"
    # arms sharing no question ids must say so rather than divide by zero
    disjoint = rows_for("full", 2, 3) + [
        {"arm": "rag", "qid": "elsewhere", "correct": True, "type": "t"}
    ]
    assert "not comparable" in sep_out(disjoint, ["full", "rag"])

    # A typo'd arm must never fall through to `full` and corrupt a paid run.
    assert unknown_arms(["full", "rag", "nauru"]) == ["nauru"]
    assert unknown_arms(list(ARMS)) == []
    # Assert on main()'s wiring, not only the predicate: deleting the call in
    # main() leaves every in-process assertion above green. Reachable offline
    # only because the check now runs before the backend and the data file.
    r = subprocess.run(
        [sys.executable, __file__, "--arms", "nauru"],
        capture_output=True,
        text=True,
        check=False,  # a nonzero exit is the thing being asserted
        env={**os.environ, "NARU_BACKEND": "cat"},
    )
    assert r.returncode != 0 and "nauru" in r.stderr, (r.returncode, r.stderr)
    r = subprocess.run(
        [sys.executable, __file__, "--arms", "full,full"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NARU_BACKEND": "definitely-not-a-real-backend"},
    )
    assert r.returncode != 0 and "duplicates" in r.stderr, (r.returncode, r.stderr)
    # same guard shape, same place: a negative k is SQLite's "no limit"
    r = subprocess.run(
        [sys.executable, __file__, "--rag-k", "-1"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NARU_BACKEND": "cat"},
    )
    assert r.returncode != 0 and "rag-k" in r.stderr, (r.returncode, r.stderr)
    embed_env = {**os.environ, "NARU_BACKEND": "cat"}
    embed_env.pop("NARU_EMBED", None)
    r = subprocess.run(
        [sys.executable, __file__, "--arms", "rsm"],
        capture_output=True,
        text=True,
        check=False,
        env=embed_env,
    )
    assert r.returncode != 0 and "NARU_EMBED" in r.stderr, (r.returncode, r.stderr)
    r = subprocess.run(
        [sys.executable, __file__, "--arms", "naru+atoms"],
        capture_output=True,
        text=True,
        check=False,
        env={**embed_env, "NARU_BACKEND": "definitely-not-a-real-backend"},
    )
    assert r.returncode != 0 and "unknown arm" in r.stderr, (r.returncode, r.stderr)
    command_env = dict(os.environ)
    command_env.pop("NARU_BACKEND", None)
    r = subprocess.run(
        [
            sys.executable,
            __file__,
            "--backend",
            "command",
            "--backend-command",
            "cat",
            "--arms",
            "nauru",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=command_env,
    )
    assert r.returncode != 0 and "unknown arm" in r.stderr, (r.returncode, r.stderr)
    r = subprocess.run(
        [sys.executable, __file__, "--backend", "command", "--arms", "nauru"],
        capture_output=True,
        text=True,
        check=False,
        env=command_env,
    )
    assert r.returncode != 0 and "requires --backend-command" in r.stderr, (
        r.returncode,
        r.stderr,
    )

    print(
        "ok — bench checks passed "
        f"(19/24 is {100 * wilson(19, 24)[0]:.0f}-{100 * wilson(19, 24)[1]:.0f}%, "
        "overlapping 16/24)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        main()
