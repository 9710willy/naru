"""Run Naru against the official BEAM chat benchmark."""

import argparse
import json
import pathlib
import shutil
import tempfile
from datetime import datetime

from backend import HAIKU
from agent import run_naru
from bench import ingest, one


BEAM_CHATS = pathlib.Path("/private/tmp/beam/chats/100K")


def _date(value):
    """BEAM's March-15-2024 -> the date shape bench.py already accepts."""
    try:
        return datetime.strptime(value, "%B-%d-%Y").strftime("%Y/%m/%d")
    except (TypeError, ValueError):
        return "1970/01/01"


def _turns(batch):
    """Flatten BEAM's list of short conversations into one session."""
    return [
        {"role": turn.get("role", "user"), "content": turn.get("content", "")}
        for exchange in batch.get("turns", [])
        for turn in exchange
    ]


def _batch_date(batch):
    """Use the turn anchor BEAM gives precedence over a batch label."""
    for exchange in batch.get("turns", []):
        for turn in exchange:
            if turn.get("time_anchor"):
                return _date(turn["time_anchor"])
    return _date(batch.get("time_anchor"))


def map_chat(chat_id, chats, questions):
    """Return LongMemEval-shaped records for one BEAM chat directory."""
    dates = [_batch_date(batch) for batch in chats]
    sessions = [_turns(batch) for batch in chats]
    ids = [f"beam-{chat_id}-{batch.get('batch_number', i)}" for i, batch in enumerate(chats, 1)]
    out = []
    for kind in sorted(questions):
        for i, probe in enumerate(questions[kind], 1):
            out.append(
                {
                    "question_id": f"beam-{chat_id}-{kind}-{i}",
                    "question_type": kind,
                    "question": probe["question"],
                    "answer": "\n".join(probe["rubric"]),
                    "question_date": dates[-1] if dates else "1970/01/01",
                    "haystack_dates": dates,
                    "haystack_session_ids": ids,
                    "haystack_sessions": sessions,
                }
            )
    return out


def load(chat_root=BEAM_CHATS, n=None, qtype=None):
    """Load a deterministic, round-robin slice over BEAM question types."""
    root = pathlib.Path(chat_root)
    if not root.exists():
        raise FileNotFoundError(f"missing {root}; clone the official BEAM repo first")
    grouped = {}
    for folder in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        chat = folder / "chat.json"
        probes = folder / "probing_questions" / "probing_questions.json"
        if not chat.exists() or not probes.exists():
            continue
        for row in map_chat(folder.name, json.load(chat.open()), json.load(probes.open())):
            if qtype is None or row["question_type"] == qtype:
                grouped.setdefault(row["question_type"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["question_id"])
    rows = []
    while any(grouped.values()) and (n is None or len(rows) < n):
        for kind in sorted(grouped):
            if grouped[kind]:
                rows.append(grouped[kind].pop(0))
                if n is not None and len(rows) == n:
                    break
    return rows


def summary(rows):
    """Small JSON-safe totals. Per-row data stays available for audit."""
    n = len(rows)
    return {
        "questions": n,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / n if n else 0.0,
        "mean_peak_view_tokens": sum(row["peak_view_tokens"] for row in rows) / n if n else 0.0,
        "billed_input": sum(row["billed_input"] for row in rows),
        "fresh_input": sum(row["fresh_input"] for row in rows),
        "cost": round(sum(row["cost"] + row["judge_cost"] for row in rows), 4),
        "errors": sum(row["errors"] + row.get("judge_errors", 0) for row in rows),
    }


def run(args):
    rows = []
    for q in load(args.beam_dir, args.n, args.qtype):
        row = one(
            q,
            "naru",
            args.model,
            args.judge_model,
            args.max_turns,
            args.budget,
            args.verbose,
        )
        rows.append(row)
        print(f"{'+' if row['correct'] else '-'} {row['qid']}")
    result = {
        "config": {
            "beam_dir": str(args.beam_dir),
            "n": args.n,
            "qtype": args.qtype,
            "model": args.model,
            "judge_model": args.judge_model,
            "max_turns": args.max_turns,
            "budget": args.budget,
        },
        "summary": summary(rows),
        "rows": rows,
    }
    print(json.dumps(result["summary"], sort_keys=True))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
REPLAY = (
    "headline(task='map sessions', verified=[], next_action='open first session', status='working')\nprint(ms.outline())",
    "headline(task='read first session', verified=['session map opened'], next_action='inspect evidence', status='working')\nprint(ms.expand(1)[0].content[:200])",
    "headline(task='inspect evidence', verified=['first session read'], next_action='check later rows', status='working')\nprint(ms.sql_query('SELECT seq, content FROM conversation_history ORDER BY seq LIMIT 1'))",
    "headline(task='check later rows', verified=['one evidence row checked'], next_action='prepare answer', status='working')\nprint('evidence checked')",
    "headline(task='prepare answer', verified=['evidence checked'], next_action='answer', status='working')\nprint('ready')",
    "headline(task='answer', verified=['evidence checked'], next_action='finish', status='done')\nsubmit_answer('replay complete')",
)


def replay(args):
    rows = []
    for q in load(args.beam_dir, args.n, args.qtype):
        ms, index = ingest(q)
        try:
            trace = []
            step = iter(REPLAY)

            def backend(prompt, system=None, nudge=None):
                return "```python\n" + next(step) + "\n```"

            answer, turns, view = run_naru(
                ms, q["question"], backend, question_date=q["question_date"],
                max_turns=len(REPLAY), budget=args.budget, trace=trace, index=index,
            )
            prompts = [item["prompt_tokens"] for item in trace]
            row = {
                "qid": q["question_id"], "type": q["question_type"], "answer": answer,
                "turns": turns, "view_tokens": view, "prompt_tokens": prompts,
                "total_prompt_tokens": sum(prompts), "peak_prompt_tokens": max(prompts),
            }
            rows.append(row)
            print(f"{row['qid']}: " + ", ".join(map(str, prompts)))
        finally:
            path = ms.path
            ms.close()
            if path != ":memory:":
                shutil.rmtree(pathlib.Path(path).parent, ignore_errors=True)
    n = len(rows)
    result = {
        "config": {"beam_dir": str(args.beam_dir), "n": args.n, "qtype": args.qtype,
                   "budget": args.budget, "replay_turns": len(REPLAY)},
        "summary": {
            "questions": n,
            "mean_total_prompt_tokens": sum(row["total_prompt_tokens"] for row in rows) / n if n else 0.0,
            "mean_peak_prompt_tokens": sum(row["peak_prompt_tokens"] for row in rows) / n if n else 0.0,
            "mean_view_tokens": sum(row["view_tokens"] for row in rows) / n if n else 0.0,
            "mean_turns": sum(row["turns"] for row in rows) / n if n else 0.0,
        },
        "rows": rows,
    }
    print(json.dumps(result["summary"], sort_keys=True))
    if args.out:
        pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {args.out}")
    return result


def selfcheck():
    """Offline fixture check: no BEAM clone, model, or network required."""
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        folder = root / "1" / "probing_questions"
        folder.mkdir(parents=True)
        (folder.parent / "chat.json").write_text(
            json.dumps(
                [
                    {
                        "batch_number": 1,
                        "time_anchor": "April-25-2024",
                        "turns": [[{"role": "user", "content": "needle", "time_anchor": "March-15-2024"}]],
                    }
                ]
            )
        )
        (folder / "probing_questions.json").write_text(
            json.dumps(
                {
                    "alpha": [{"question": "where?", "ideal_response": "wrong", "rubric": ["needle"]}],
                    "beta": [{"question": "what?", "ideal_answer": "needle", "rubric": ["needle", "there"]}],
                }
            )
        )
        rows = load(root, n=2)
        empty = replay(argparse.Namespace(
            beam_dir=root, n=2, qtype="missing", budget=6000, out=None
        ))
    assert [row["question_type"] for row in rows] == ["alpha", "beta"]
    assert rows[0]["haystack_dates"] == ["2024/03/15"]
    assert rows[0]["haystack_sessions"][0][0]["content"] == "needle"
    assert rows[0]["answer"] == "needle"
    assert rows[1]["answer"] == "needle\nthere"
    assert all(empty["summary"][key] == 0.0 for key in (
        "mean_total_prompt_tokens", "mean_peak_prompt_tokens",
        "mean_view_tokens", "mean_turns",
    ))
    assert summary([])["accuracy"] == 0.0
    assert len(REPLAY) == 6 and all("verified=" in step for step in REPLAY)
    print("ok — BEAM adapter self-check passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam-dir", type=pathlib.Path, default=BEAM_CHATS)
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--qtype")
    ap.add_argument("--model", default=HAIKU)
    ap.add_argument("--judge-model", default=HAIKU)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--budget", type=int, default=6000)
    ap.add_argument("--out")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--replay", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck()
    elif args.n < 1:
        ap.error("-n must be >= 1")
    elif args.replay:
        replay(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
