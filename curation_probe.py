"""Compare the same organic task with and without approved Naru context.

Cases are JSONL. Each row has an id, kind, prompt, and literal answer checks:

    {"id":"port","kind":"fact","prompt":"Which port do I use?",
     "must_include":["port 7443"],"must_exclude":["port 443"]}

Use kind ``fact`` when the answer needs approved context. Use kind ``control``
when it does not. Candidate prompts receive no arm names or scoring text.
"""

import argparse
import hashlib
import json
import pathlib
import tempfile
import time
from dataclasses import dataclass

from backend import HAIKU, Usage, get_backend
from ms import DEFAULT_DB, MemorySurface
from naru import _codex_context
from noise import mcnemar

BASE_SYSTEM = (
    "Follow the user's request. Use supplied standing context when it applies. "
    "Return only the requested answer."
)


@dataclass(frozen=True)
class Case:
    id: str
    kind: str
    prompt: str
    must_include: tuple
    must_exclude: tuple

    @classmethod
    def parse(cls, raw, line):
        if not isinstance(raw, dict):
            raise TypeError(f"line {line}: expected a JSON object")
        case_id = raw.get("id")
        kind = raw.get("kind")
        prompt = raw.get("prompt")
        include = raw.get("must_include", [])
        exclude = raw.get("must_exclude", [])
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"line {line}: id must be a non-empty string")
        if kind not in ("fact", "control"):
            raise ValueError(f"line {line}: kind must be fact or control")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {line}: prompt must be a non-empty string")
        if not all(
            isinstance(values, list)
            and all(isinstance(value, str) and value for value in values)
            for values in (include, exclude)
        ):
            raise ValueError(f"line {line}: answer checks must be string lists")
        if not include and not exclude:
            raise ValueError(f"line {line}: add at least one answer check")
        return cls(case_id, kind, prompt, tuple(include), tuple(exclude))


def load_cases(path):
    cases = []
    for line, text in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        if text.strip():
            cases.append(Case.parse(json.loads(text), line))
    if not cases:
        raise ValueError("case file is empty")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return cases


def score(case, answer):
    body = answer.casefold()
    missing = [value for value in case.must_include if value.casefold() not in body]
    present = [value for value in case.must_exclude if value.casefold() in body]
    return {"passed": not missing and not present, "missing": missing, "present": present}


def usage_dict(usage):
    measured = getattr(usage, "cost_measured", True)
    return {
        "attempts": usage.attempts,
        "calls": usage.calls,
        "billed_input": usage.billed_input,
        "output_tokens": usage.output_tokens,
        "cost_usd": usage.cost_usd if measured else None,
        "cost_measured": measured,
        "prompt_tokens_estimated": usage.prompt_tokens_estimated,
        "peak_prompt_tokens_estimated": usage.peak_prompt_tokens_estimated,
        "errors": usage.errors,
    }


def run_arm(case, arm, doc, model, backend_factory):
    backend = backend_factory(model)
    system = (
        BASE_SYSTEM if arm == "plain" else f"{BASE_SYSTEM}\n\n{_codex_context(doc)}"
    )
    started = time.monotonic()
    answer = backend(case.prompt, system=system)
    return {
        "answer": answer,
        "score": score(case, answer),
        "seconds": time.monotonic() - started,
        "backend": backend.label,
        "usage": usage_dict(backend.usage),
    }


def summarize(rows):
    def cost_total(selected, arm):
        costs = [row[arm]["usage"]["cost_usd"] for row in selected]
        return None if any(value is None for value in costs) else sum(costs)

    out = {}
    for kind in ("fact", "control", "all"):
        selected = (
            rows if kind == "all" else [row for row in rows if row["kind"] == kind]
        )
        paired = [
            row
            for row in selected
            if not row["plain"]["usage"]["errors"]
            and not row["naru"]["usage"]["errors"]
        ]
        plain = {row["id"]: row["plain"]["score"]["passed"] for row in paired}
        naru = {row["id"]: row["naru"]["score"]["passed"] for row in paired}
        plain_only, naru_only, p = mcnemar(plain, naru)
        out[kind] = {
            "cases": len(selected),
            "paired": len(paired),
            "plain_pass": sum(plain.values()),
            "naru_pass": sum(naru.values()),
            "plain_only": plain_only,
            "naru_only": naru_only,
            "mcnemar_p": p,
            "plain_errors": sum(row["plain"]["usage"]["errors"] for row in selected),
            "naru_errors": sum(row["naru"]["usage"]["errors"] for row in selected),
            "plain_seconds": sum(row["plain"]["seconds"] for row in selected),
            "naru_seconds": sum(row["naru"]["seconds"] for row in selected),
            "plain_cost_usd": cost_total(selected, "plain"),
            "naru_cost_usd": cost_total(selected, "naru"),
            "plain_prompt_tokens_estimated": sum(
                row["plain"]["usage"]["prompt_tokens_estimated"] for row in selected
            ),
            "naru_prompt_tokens_estimated": sum(
                row["naru"]["usage"]["prompt_tokens_estimated"] for row in selected
            ),
        }
    out["harmful_carryover"] = out["control"]["plain_only"]
    return out


def run_probe(cases, doc, doc_seq, model=HAIKU, backend_factory=get_backend):
    rows = []
    for case in cases:
        order_bit = int(hashlib.sha256(case.id.encode()).hexdigest(), 16) % 2
        first = "plain" if order_bit else "naru"
        arms = (first, "naru" if first == "plain" else "plain")
        row = {
            "id": case.id,
            "kind": case.kind,
            "prompt": case.prompt,
            "must_include": list(case.must_include),
            "must_exclude": list(case.must_exclude),
            "order": list(arms),
        }
        for arm in arms:
            row[arm] = run_arm(case, arm, doc, model, backend_factory)
        rows.append(row)
        print(
            f"{case.id}: plain={'pass' if row['plain']['score']['passed'] else 'fail'} "
            f"naru={'pass' if row['naru']['score']['passed'] else 'fail'}"
        )
    return {
        "config": {
            "model": model,
            "doc_seq": doc_seq,
            "doc_sha256": hashlib.sha256(doc.encode()).hexdigest(),
        },
        "summary": summarize(rows),
        "rows": rows,
    }


def print_summary(result):
    for kind in ("fact", "control", "all"):
        row = result["summary"][kind]
        print(
            f"{kind}: plain {row['plain_pass']}/{row['paired']}  "
            f"naru {row['naru_pass']}/{row['paired']}  "
            f"plain-only {row['plain_only']}  naru-only {row['naru_only']}  "
            f"p={row['mcnemar_p']:.3f}"
        )
        if row["plain_errors"] or row["naru_errors"]:
            print(
                f"  excluded errors: plain {row['plain_errors']}  "
                f"naru {row['naru_errors']}"
            )
    print(f"harmful carryover: {result['summary']['harmful_carryover']}")
    totals = result["summary"]["all"]
    plain_cost = (
        "unknown"
        if totals["plain_cost_usd"] is None
        else f"${totals['plain_cost_usd']:.4f}"
    )
    naru_cost = (
        "unknown"
        if totals["naru_cost_usd"] is None
        else f"${totals['naru_cost_usd']:.4f}"
    )
    print(
        "resources: "
        f"plain {totals['plain_prompt_tokens_estimated']:,} estimated prompt tokens, "
        f"{plain_cost}, {totals['plain_seconds']:.1f}s  "
        f"naru {totals['naru_prompt_tokens_estimated']:,} estimated prompt tokens, "
        f"{naru_cost}, {totals['naru_seconds']:.1f}s"
    )


def selfcheck():
    calls = []

    class Fake:
        label = "fake"

        def __init__(self):
            self.usage = Usage()

        def __call__(self, prompt, system=None):
            calls.append((prompt, system))
            self.usage.attempts = self.usage.calls = 1
            if prompt == "Which port do I use?":
                return "port 7443" if "Service port is 7443" in system else "port 443"
            return "blue"

    factory = lambda _model: Fake()
    cases = [
        Case(
            "port", "fact", "Which port do I use?", ("port 7443",), ("port 443",)
        ),
        Case("color", "control", "Reply with blue.", ("blue",), ("red",)),
    ]
    doc = "# naru · seq 7\n\n## Decisions\n- Service port is 7443.\n"
    result = run_probe(cases, doc, 7, model="fake", backend_factory=factory)
    assert result["summary"]["fact"]["naru_only"] == 1
    assert result["summary"]["control"]["plain_pass"] == 1
    assert result["summary"]["control"]["naru_pass"] == 1
    assert result["summary"]["harmful_carryover"] == 0
    broken = json.loads(json.dumps(result["rows"][0]))
    broken["id"] = "broken"
    broken["plain"]["usage"]["errors"] = 1
    checked = summarize([*result["rows"], broken])
    assert checked["all"]["cases"] == 3 and checked["all"]["paired"] == 2
    assert checked["all"]["plain_errors"] == 1
    unsafe = Case("unsafe", "control", "p", ("blue",), ("red",))
    assert not score(unsafe, "blue and red")["passed"]
    assert len(calls) == 4
    assert result["rows"][0]["must_include"] == ["port 7443"]
    assert sorted(result["rows"][0]["order"]) == ["naru", "plain"]
    assert {prompt for prompt, _ in calls} == {case.prompt for case in cases}
    assert sum("Naru live context follows" in system for _, system in calls) == 2
    try:
        Case.parse({"id": "x", "kind": "fact", "prompt": "p"}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted a case without answer checks")
    with tempfile.TemporaryDirectory() as tmp:
        duplicate = pathlib.Path(tmp) / "cases.jsonl"
        row = {"id": "x", "kind": "control", "prompt": "p", "must_include": ["x"]}
        duplicate.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
        try:
            load_cases(duplicate)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted duplicate case ids")
    print("ok — paired curation checks passed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="?")
    parser.add_argument("--db", type=pathlib.Path, default=DEFAULT_DB)
    parser.add_argument("--model", default=HAIKU)
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--selfcheck", action="store_true")
    args = parser.parse_args()
    if args.selfcheck:
        selfcheck()
        return
    if not args.cases:
        parser.error("cases is required")
    cases = load_cases(args.cases)
    surface = MemorySurface.open_readonly(str(args.db))
    try:
        result = run_probe(cases, surface.doc(), surface.doc_version(), args.model)
    finally:
        surface.close()
    print_summary(result)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
