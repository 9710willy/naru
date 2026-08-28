"""LongMemEval harness: ingest → answer → judge → score.

Runs three arms over the same data so the comparison is controlled:
  full     — the whole history stuffed into one prompt (the usual approach)
  rag      — top-k BM25 hits pasted in, one call, no kernel (the control)
  naru     — history in the Session Environment, model writes code to reach it

Reports accuracy, tokens billed, and cost for each.
"""

import argparse
import contextlib
import io
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

from agent import LONGMEMEVAL_RUBRIC, run_naru
from backend import HAIKU, get_backend, measure_floor
from eviction import est, rollup
from ms import MemorySurface

# One owner for the p-value ADR 0006 publishes. noise.py had it first and
# imports nothing local, so the dependency runs this way round.
from noise import mcnemar

DATA = pathlib.Path(__file__).parent / "data"
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")
# One owner for the arm names: main() validates against it, demo() asserts on it.
ARMS = ("full", "rag", "naru")


def backend_label(cmd):
    """Provenance for a run: the program that answered, never its arguments.

    NARU_BACKEND is documented as any command reading a prompt on stdin, so it
    can carry a credential (`sh -c 'curl -H "Authorization: Bearer ..."'`).
    results/published/ is committed to a public history, where rotating a key
    that already shipped does not undo it. argv[0] is all the provenance the
    field is for — telling a `cat` run apart from a real one.
    """
    return (shlex.split(cmd) or ["claude-cli"])[0] if cmd else "claude-cli"


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


def ingest(q, build_index=True):
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
    ms = MemorySurface(":memory:")
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
    p = (
        f"Question: {q['question']}\n\nGold answer: {q['answer']}\n\n"
        f"Candidate answer: {response.strip()[:2000]}\n\nVerdict:"
    )
    yes = 0
    for i in range(votes):
        v = (
            backend(p, system=JUDGE_SYSTEM, nudge="(Reply with exactly one word.)")
            .strip()
            .upper()
        )
        yes += v.startswith("CORRECT")
        # early exit once the outcome cannot change
        if yes > votes // 2 or (i + 1 - yes) > votes // 2:
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


def build_prompt(q, arm, rag_k=8):
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
        ms, _ = ingest(q, build_index=False)
        body = rag_context(ms, q["question"], rag_k)
    else:
        body = history_text(q)
    return (
        f"{body}\n\n=== Question (asked {q.get('question_date', '')}) ===\n"
        f"{q['question']}"
    )


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
):
    """Run a single question through one arm. Returns a result record."""
    be = get_backend(model)
    t0 = time.time()

    if arm == "naru":
        ms, index = ingest(q, build_index=not no_index)
        ans, turns, peak = run_naru(
            ms,
            q["question"],
            be,
            question_date=q.get("question_date"),
            max_turns=max_turns,
            budget=budget,
            verbose=verbose,
            rubric=LONGMEMEVAL_RUBRIC if rubric else None,
            index=index,
        )
    else:
        prompt = build_prompt(q, arm, rag_k)
        ans, turns, peak = be(prompt, system=FULL_SYSTEM), 1, est(prompt)

    elapsed = time.time() - t0
    jb = get_backend(judge_model)
    ok = judge(q, ans, jb)

    return {
        "qid": q["question_id"],
        "type": q["question_type"],
        "arm": arm,
        "correct": ok,
        "gold": q["answer"],
        "answer": (ans or "")[:400],
        "turns": turns,
        "peak_view_tokens": peak,
        "seconds": round(elapsed, 1),
        "billed_input": be.usage.billed_input,
        "fresh_input": be.usage.input_tokens,
        # cache creation bills ~1.25x base, cache reads ~0.1x — the full-context
        # arm writes a fresh 124k history per question and never reuses it,
        # which is where its cost actually goes.
        "cache_creation": be.usage.cache_creation,
        "cache_read": be.usage.cache_read,
        "output": be.usage.output_tokens,
        "cost": round(be.usage.cost_usd, 4),
        "judge_cost": round(jb.usage.cost_usd, 4),
        "errors": be.usage.errors,
        # A judge that times out returns "" for every vote, which reads as
        # WRONG. Counted separately, it is the only call whose failure is
        # indistinguishable from a real negative result.
        "judge_errors": jb.usage.errors,
        "empty_retries": be.usage.empty_retries,
    }


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
    zeros that would otherwise read as measurements."""
    if not rows:
        return
    n = len(rows)
    n_correct = sum(r["correct"] for r in rows)
    acc = n_correct / n
    bi = sum(r["billed_input"] for r in rows)
    # floor is None when the backend reports no usage at all. Subtracting 0 and
    # printing the result would present "not measured" as a measurement.
    net = (
        None
        if floor is None
        else sum(max(0, r["billed_input"] - floor * r["turns"]) for r in rows)
    )
    model_cost = sum(r["cost"] for r in rows)
    judge_cost = sum(r["judge_cost"] for r in rows)
    cost = model_cost + judge_cost
    bar = "#" * round(acc * 28) + "." * (28 - round(acc * 28))
    lo, hi = wilson(n_correct, n)
    # The interval is printed on the same line as the accuracy on purpose. A
    # bare percentage invites a reader to compare two arms that overlap.
    print(
        f"\n  {label:8} {bar} {acc * 100:5.1f}%  ({n_correct}/{n})"
        f"  95% CI {lo * 100:.0f}-{hi * 100:.0f}%"
    )
    if measured:
        print(
            f"           billed-in {bi / n:>9,.0f}/q   net-of-harness "
            + ("not measurable" if net is None else f"{net / n:>9,.0f}/q")
        )
    else:
        print("           billed-in  not measurable   net-of-harness not measurable")
    print(
        f"           out {sum(r['output'] for r in rows) / n:>7,.0f}/q   "
        f"turns {sum(r['turns'] for r in rows) / n:>4.1f}   "
        f"view {sum(r['peak_view_tokens'] for r in rows) / n:>6,.0f}t   "
        + (f"${cost:.2f} total" if measured else "cost not measurable")
    )
    if measured:
        # The arm's own dollars, and the cache share that makes a token ratio
        # and a money ratio disagree. Both are published columns; printing
        # only a combined total left them hand-computed and unreproducible.
        cr = sum(r.get("cache_read", 0) for r in rows)
        print(
            f"           model ${model_cost / n:.4f}/q   "
            f"judge ${judge_cost / n:.4f}/q   "
            f"cache-read {100 * cr / max(1, bi):.0f}% of billed input"
        )
    errs = sum(r["errors"] for r in rows)
    # Rows written before judge_errors existed have no such key. Summing them
    # to 0 would print "0 judge errors" for a run where nobody counted, which
    # is ADR 0002's mistake in miniature: not measured rendered as measured.
    jerrs = (
        sum(r["judge_errors"] for r in rows)
        if all("judge_errors" in r for r in rows)
        else None
    )
    retries = sum(r.get("empty_retries", 0) for r in rows)
    if errs or retries or jerrs:
        judge_part = "judge errors not recorded" if jerrs is None else f"{jerrs} judge errors"
        print(
            f"           {errs} backend errors, {judge_part}, "
            f"{retries} empty-reply retries"
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
    verdicts, dropped = {}, {}
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        v = {
            r["qid"]: bool(r["correct"])
            for r in rs
            if not r.get("errors") and not r.get("judge_errors")
        }
        if v:
            verdicts[arm] = v
            dropped[arm] = len(rs) - len(v)
    if len(verdicts) < 2:
        return
    pairs = len(list(combinations(verdicts, 2)))
    # Three arms means three tests. At an uncorrected 0.05 each, at least one
    # pair reads REAL in ~6% of runs where nothing separates, against ~2% for
    # a single pair. The verdict is the harness's published claim, so it is
    # the number that has to be honest.
    alpha = 0.05 / pairs
    note = f", Bonferroni for {pairs} pairs" if pairs > 1 else ""
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
    ap.add_argument("--model", default=HAIKU)
    ap.add_argument("--judge-model", default=HAIKU)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qtype", default=None)
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
    a = ap.parse_args()

    # A typo'd arm would otherwise run as `full` and quietly corrupt the run.
    # Checked before the backend is touched and before a 277MB split is read,
    # so a typo costs nothing.
    arms = a.arms.split(",")
    unknown = unknown_arms(arms)
    if unknown:
        sys.exit(f"unknown arm(s): {unknown} — pick from {', '.join(ARMS)}")
    # SQLite reads a negative LIMIT as NO limit, so --rag-k -1 pastes the whole
    # history and the rag arm silently becomes a second full arm at ~50x the
    # cost, still labelled rag. 0 is the mirror: empty context, every answer
    # wrong, nothing in the output naming why.
    if a.rag_k < 1:
        sys.exit(f"--rag-k must be >= 1, got {a.rag_k}")

    measured = get_backend(a.model).reports_tokens
    if a.harness_floor is None:
        a.harness_floor = measure_floor(a.model)
        if a.harness_floor is None:
            print(
                "harness floor NOT measurable with this backend — token columns "
                "will read as zero and net-of-harness is omitted, not reported as 0"
            )
        else:
            print(f"measured harness floor: {a.harness_floor:,} input tok/call")

    qs = load(a.split, a.n, qtype=a.qtype)
    print(
        f"LongMemEval-{a.split}  n={len(qs)}  arms={arms}  model={a.model}  "
        f"judge={a.judge_model}  budget={a.budget}t  max_turns={a.max_turns}"
    )
    avg_hist = sum(est(history_text(q)) for q in qs) / len(qs)
    print(f"avg history per question: {avg_hist:,.0f} tokens")

    jobs = [(q, arm) for arm in arms for q in qs]
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
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
            ): (q, arm)
            for q, arm in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            try:
                r = f.result()
            except Exception as e:
                q, arm = futs[f]
                r = {
                    "qid": q["question_id"],
                    "type": q["question_type"],
                    "arm": arm,
                    "correct": False,
                    "gold": q["answer"],
                    "answer": f"HARNESS: {e}",
                    "turns": 0,
                    "peak_view_tokens": 0,
                    "seconds": 0,
                    "billed_input": 0,
                    "fresh_input": 0,
                    "output": 0,
                    "cost": 0,
                    "judge_cost": 0,
                    "errors": 1,
                }
            rows.append(r)
            mark = "+" if r["correct"] else "-"
            print(
                f"\r  [{i}/{len(jobs)}] {mark} {r['arm']:6} {r['qid'][:22]:22}",
                end="",
                flush=True,
            )
    print()

    for arm in arms:
        report([r for r in rows if r["arm"] == arm], arm, a.harness_floor, measured)
    separability(rows, arms)

    out = DATA.parent / "results" / f"{a.tag}_{a.split}_n{len(qs)}.json"
    out.parent.mkdir(exist_ok=True)
    # vars(a) records --model even when NARU_BACKEND replaced it, which made a
    # `NARU_BACKEND=cat` run byte-identical to a real Haiku run that cost
    # nothing. Stamp what actually answered, and whether the numbers are real.
    cfg = dict(vars(a))
    cfg["backend"] = backend_label(os.environ.get("NARU_BACKEND"))
    cfg["tokens_measured"] = measured
    json.dump({"config": cfg, "rows": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


def demo():
    """Offline self-check. No data file, no API calls, no results written.

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
    # The published run. If this ever stops overlapping, the README's "read the
    # accuracy column as a tie" has become false and must be rewritten.
    assert wilson(19, 24)[0] < wilson(16, 24)[1]

    assert iso("2023/04/10 (Mon) 17:50") == "2023-04-10T17:50"
    assert iso(None) == "1970-01-01T00:00", "a missing date must not crash ingest"

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
    assert "Bonferroni for 3 pairs" in out3, out3
    assert "REAL" not in out3, out3
    assert "p=0.031" in out3, out3
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
    rag_p = build_prompt(synth, "rag", rag_k=1)
    assert "kayak" in rag_p and "kayak" in full_p
    assert "tarragon" in full_p, "the full arm must carry the whole history"
    assert "tarragon" not in rag_p, "the rag arm must carry retrieved hits only"
    assert len(rag_p) < len(full_p)

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
    for r in old_rows:
        r["judge_errors"] = 0
    assert "0 judge errors" in rep_out(old_rows), rep_out(old_rows)

    # NARU_BACKEND can hold a credential and results/published/ is committed,
    # so the recorded provenance must be argv[0] and nothing after it.
    leaky = "sh -c 'curl -H \"Authorization: Bearer sk-secret\"'"
    assert backend_label(leaky) == "sh", backend_label(leaky)
    assert "sk-secret" not in backend_label(leaky)
    assert backend_label(None) == "claude-cli"
    assert backend_label("   ") == "claude-cli", "blank must not IndexError"
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
    # same guard shape, same place: a negative k is SQLite's "no limit"
    r = subprocess.run(
        [sys.executable, __file__, "--rag-k", "-1"],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "NARU_BACKEND": "cat"},
    )
    assert r.returncode != 0 and "rag-k" in r.stderr, (r.returncode, r.stderr)

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
