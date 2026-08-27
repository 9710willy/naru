"""LongMemEval harness: ingest → answer → judge → score.

Runs two arms over the same data so the comparison is controlled:
  full     — the whole history stuffed into one prompt (the usual approach)
  naru   — history in the Session Environment, model writes code to reach it

Reports accuracy, tokens billed, and cost for each.
"""

import argparse
import json
import pathlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent import LONGMEMEVAL_RUBRIC, run_naru
from backend import HAIKU, get_backend, measure_floor
from eviction import est, rollup
from ms import MemorySurface

DATA = pathlib.Path(__file__).parent / "data"
DATE_RE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


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
        v = backend(p, system=JUDGE_SYSTEM).strip().upper()
        yes += v.startswith("CORRECT")
        # early exit once the outcome cannot change
        if yes > votes // 2 or (i + 1 - yes) > votes // 2:
            break
    return yes > (votes // 2)


def one(q, arm, model, judge_model, max_turns, budget, verbose, rubric=True,
        no_index=False):
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
        hist = history_text(q)
        prompt = (
            f"{hist}\n\n=== Question (asked {q.get('question_date', '')}) ===\n"
            f"{q['question']}"
        )
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
        "empty_retries": be.usage.empty_retries,
    }


def report(rows, label, floor):
    if not rows:
        return
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n
    bi = sum(r["billed_input"] for r in rows)
    net = sum(max(0, r["billed_input"] - floor * r["turns"]) for r in rows)
    cost = sum(r["cost"] + r["judge_cost"] for r in rows)
    bar = "#" * round(acc * 28) + "." * (28 - round(acc * 28))
    print(
        f"\n  {label:8} {bar} {acc * 100:5.1f}%  ({sum(r['correct'] for r in rows)}/{n})"
    )
    print(f"           billed-in {bi / n:>9,.0f}/q   net-of-harness {net / n:>9,.0f}/q")
    print(
        f"           out {sum(r['output'] for r in rows) / n:>7,.0f}/q   "
        f"turns {sum(r['turns'] for r in rows) / n:>4.1f}   "
        f"view {sum(r['peak_view_tokens'] for r in rows) / n:>6,.0f}t   "
        f"${cost:.2f} total"
    )
    errs = sum(r["errors"] for r in rows)
    retries = sum(r.get("empty_retries", 0) for r in rows)
    if errs or retries:
        print(f"           {errs} backend errors, {retries} empty-reply retries")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="oracle", choices=["oracle", "s", "m"])
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--arms", default="full,naru")
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
    a = ap.parse_args()

    if a.harness_floor is None:
        a.harness_floor = measure_floor(a.model)
        print(f"measured harness floor: {a.harness_floor:,} input tok/call")

    qs = load(a.split, a.n, qtype=a.qtype)
    arms = a.arms.split(",")
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
                not a.no_rubric,
                a.no_index,
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
        report([r for r in rows if r["arm"] == arm], arm, a.harness_floor)

    out = DATA.parent / "results" / f"{a.tag}_{a.split}_n{len(qs)}.json"
    out.parent.mkdir(exist_ok=True)
    json.dump({"config": vars(a), "rows": rows}, open(out, "w"), indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
