#!/usr/bin/env python3
"""Compare paired Naru runs without exposing benchmark content.

The benchmark already runs one policy at a time. This module makes the first
controlled proof reproducible by comparing two saved runs on the exact same
question IDs. It reports task success, failures, latency, context movement,
and Jev overhead. It never prints answers, gold text, prompts, or raw IDs.

Examples:

    python3 experiment.py \
      --baseline results/naru-deterministic_oracle_n12.json \
      --candidate results/naru-jev_oracle_n12.json

If Jev was used, pass Jev input and output rates in USD per million tokens and
a pricing version before the comparator will call total cost measurable. Cache
rates are optional. If Jev reports cache tokens without a supplied cache rate,
the cost result remains unknown. The built-in profile records the supplied
`$0.042/MTok` input rate and free output, without guessing cache prices.
"""

import argparse
import json
import math
import pathlib
import sys

from bench import wilson
from noise import mcnemar


FORMAT = 1
POLICIES = {"deterministic", "shadow", "jev"}
PRICE_FIELDS = (
    "input_usd_per_million",
    "cache_read_usd_per_million",
    "cache_write_usd_per_million",
    "output_usd_per_million",
)
REQUIRED_PRICE_FIELDS = (
    "input_usd_per_million",
    "output_usd_per_million",
)
JEV_PRICING_PROFILES = {
    "typesafe-jev-input-0.042-free-output": {
        "version": "typesafe-jev-input-0.042-free-output",
        "input_usd_per_million": 0.042,
        # The supplied Jev rate covers input tokens. Cache-specific rates are
        # intentionally unknown until Jev reports a cache category and its
        # billing semantics are verified.
        "cache_read_usd_per_million": None,
        "cache_write_usd_per_million": None,
        "output_usd_per_million": 0.0,
    }
}
CONFIG_FIELDS = (
    "split",
    "qtype",
    "model",
    "judge_model",
    "resolved_backend",
    "backend",
    "backend_fingerprint",
    "judge_resolved_backend",
    "judge_backend",
    "judge_backend_fingerprint",
    "budget",
    "max_turns",
    "harness_floor",
    "workers",
    "no_rubric",
    "no_index",
    "rag_k",
    "rsm_k",
    "rsm_tau",
    "rsm_chunk_turns",
    "rsm_budget",
    "tokens_measured",
    "pricing_version",
    "context_policy",
)
COMPATIBILITY_FIELDS = tuple(
    field
    for field in CONFIG_FIELDS
    if field not in ("context_policy", "tokens_measured", "jev_command")
)


class ExperimentError(ValueError):
    """The two files cannot support a controlled comparison."""


def _number(value, default=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return value if math.isfinite(value) else default


def _positive_number(value, name):
    value = _number(value, -1)
    if value < 0:
        raise ExperimentError(f"{name} must be a non-negative number")
    return float(value)


def _safe_config(config):
    """Keep run metadata useful without echoing command lines or paths."""
    return {key: config[key] for key in CONFIG_FIELDS if key in config}


def _run_label(arm, config):
    """Name a run without echoing a caller-chosen filename."""
    policy = config.get("context_policy")
    suffix = f"/{policy}" if isinstance(policy, str) else ""
    return f"{arm}{suffix}"


def load_run(path, arm):
    """Load one arm and reject duplicate or missing task rows."""
    path = pathlib.Path(path)
    try:
        document = json.loads(path.read_text())
    except OSError as error:
        raise ExperimentError(f"cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise ExperimentError(f"invalid JSON in {path}: {error.msg}") from error
    if not isinstance(document, dict) or not isinstance(document.get("rows"), list):
        raise ExperimentError(f"{path} must contain a rows array")
    config = document.get("config")
    if not isinstance(config, dict):
        config = {}
    rows = {}
    for row in document["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("qid"), str):
            raise ExperimentError(f"{path} contains a row without a string qid")
        if row.get("arm") != arm:
            continue
        qid = row["qid"]
        if qid in rows:
            raise ExperimentError(f"{path} has duplicate result rows for {arm}")
        rows[qid] = row
    if not rows:
        raise ExperimentError(f"{path} has no rows for arm {arm!r}")
    return {
        "path": path,
        "arm": arm,
        "config": config,
        "rows": rows,
        "label": _run_label(arm, config),
    }


def _check_compatible(baseline, candidate):
    """Validate fixed variables before treating the rows as a pair."""
    left = baseline["rows"]
    right = candidate["rows"]
    if set(left) != set(right):
        raise ExperimentError(
            "baseline and candidate must contain the same question set"
        )
    for key in COMPATIBILITY_FIELDS:
        left_config = baseline["config"]
        right_config = candidate["config"]
        if key not in left_config or key not in right_config:
            raise ExperimentError(
                f"controlled variable {key!r} is missing from one run"
            )
        if left_config[key] != right_config[key]:
            raise ExperimentError(f"controlled variable {key!r} differs")
    for run in (baseline, candidate):
        policy = run["config"].get("context_policy")
        if policy is not None and policy not in POLICIES:
            raise ExperimentError(f"unknown context policy in {run['path']}: {policy!r}")


def _task_success(row):
    if "task_success" in row:
        return bool(row["task_success"])
    return bool(
        row.get("correct")
        and not row.get("errors")
        and not row.get("judge_errors")
    )


def _has_error(row):
    return bool(row.get("errors") or row.get("judge_errors"))


def _mean(rows, key):
    values = [_number(row.get(key), None) for row in rows]
    values = [value for value in values if value is not None]
    return round(sum(values) / len(values), 6) if values else None


def _sum(rows, key):
    values = [_number(row.get(key), None) for row in rows]
    if any(value is None for value in values):
        return None
    return round(sum(values), 6)


def _rate(count, total):
    return round(count / total, 6) if total else None


def _cost_measured(run, rows):
    """Only trust costs when the benchmark says provider usage was measured."""
    return bool(run["config"].get("tokens_measured") is True) and all(
        isinstance(row.get("task_cost_usd"), (int, float))
        and not isinstance(row.get("task_cost_usd"), bool)
        for row in rows
    )


def _jev_usage(row):
    usage = row.get("jev_usage")
    if isinstance(usage, dict) and usage.get("tokens_measured") is False:
        return None
    expected_calls = _number(row.get("context_jev_calls"), 0)
    if isinstance(usage, dict) and (
        _number(usage.get("errors"), 0)
        or (
            expected_calls
            and _number(usage.get("calls"), 0) < expected_calls
        )
    ):
        return None
    if not isinstance(usage, dict):
        if expected_calls and (
            "context_jev_input_tokens" not in row
        ):
            return None
        usage = {}
    return {
        "input_tokens": max(
            0,
            _number(
                usage.get("input_tokens", row.get("context_jev_input_tokens", 0))
            ),
        ),
        "cache_read_tokens": max(
            0,
            _number(
                usage.get(
                    "cache_read_tokens",
                    row.get("context_jev_cache_read_tokens", 0),
                )
            ),
        ),
        "cache_write_tokens": max(
            0,
            _number(
                usage.get(
                    "cache_write_tokens",
                    row.get("context_jev_cache_write_tokens", 0),
                )
            ),
        ),
        "output_tokens": max(
            0,
            _number(
                usage.get("output_tokens", row.get("context_jev_output_tokens", 0))
            ),
        ),
    }


def _jev_cost(row, prices):
    if prices is None:
        return None
    usage = _jev_usage(row)
    if usage is None:
        return None
    total = 0.0
    for key, field in (
        ("input_tokens", "input_usd_per_million"),
        ("cache_read_tokens", "cache_read_usd_per_million"),
        ("cache_write_tokens", "cache_write_usd_per_million"),
        ("output_tokens", "output_usd_per_million"),
    ):
        tokens = usage[key]
        rate = prices.get(field)
        if rate is None:
            if tokens:
                return None
            continue
        total += tokens * rate / 1_000_000
    return total


def _validate_prices(prices):
    if prices is None:
        return None
    if not isinstance(prices, dict) or not isinstance(prices.get("version"), str):
        raise ExperimentError("Jev pricing needs a string version")
    clean = {"version": prices["version"]}
    for field in PRICE_FIELDS:
        value = prices.get(field)
        if value is None and field not in REQUIRED_PRICE_FIELDS:
            clean[field] = None
        else:
            clean[field] = _positive_number(value, field)
    return clean


def _effective_cost(row, model_cost_measured, prices):
    if not model_cost_measured:
        return None
    model_cost = _number(row.get("task_cost_usd"), None)
    if model_cost is None:
        return None
    calls = _number(row.get("context_jev_calls"), 0)
    if calls and not row.get("cost_includes_jev", False):
        jev_cost = _jev_cost(row, prices)
        if jev_cost is None:
            return None
        model_cost += jev_cost
    return model_cost


def _jev_total_cost(rows, prices):
    if prices is None:
        return None
    costs = [
        _jev_cost(row, prices)
        for row in rows
        if _number(row.get("context_jev_calls"), 0)
    ]
    if any(cost is None for cost in costs):
        return None
    return round(sum(costs), 6)


def _run_metrics(run, prices):
    rows = list(run["rows"].values())
    success = sum(_task_success(row) for row in rows)
    errors = sum(_has_error(row) for row in rows)
    model_cost_measured = _cost_measured(run, rows)
    effective_costs = [
        _effective_cost(row, model_cost_measured, prices) for row in rows
    ]
    effective_cost_measured = all(cost is not None for cost in effective_costs)
    total_cost = round(sum(effective_costs), 6) if effective_cost_measured else None
    success_cost = (
        round(
            sum(
                cost
                for row, cost in zip(rows, effective_costs)
                if _task_success(row)
            )
            / success,
            6,
        )
        if effective_cost_measured and success
        else None
    )
    jev_calls = _sum(rows, "context_jev_calls")
    return {
        "questions": len(rows),
        "task_successes": success,
        "task_success_rate": _rate(success, len(rows)),
        "task_success_ci95": (
            [round(value, 6) for value in wilson(success, len(rows))]
            if rows
            else None
        ),
        "error_rows": errors,
        "error_rate": _rate(errors, len(rows)),
        "latency_ms_mean": _mean(rows, "latency_ms"),
        "task_cost_usd_total": total_cost,
        "task_cost_usd_per_success": success_cost,
        "cost_status": "measured" if effective_cost_measured else "unknown",
        "context_selected_tokens_mean": _mean(rows, "context_selected_tokens"),
        "context_dropped_tokens_mean": _mean(rows, "context_dropped_tokens"),
        "context_considered_tokens_mean": _mean(rows, "context_considered_tokens"),
        "jev_calls": jev_calls,
        "jev_input_tokens": _sum(rows, "context_jev_input_tokens"),
        "jev_cache_read_tokens": _sum(rows, "context_jev_cache_read_tokens"),
        "jev_cache_write_tokens": _sum(rows, "context_jev_cache_write_tokens"),
        "jev_output_tokens": _sum(rows, "context_jev_output_tokens"),
        "jev_wall_latency_ms": _sum(rows, "context_jev_latency_ms"),
        "jev_startup_latency_ms": _sum(rows, "context_jev_startup_latency_ms"),
        "jev_transport_overhead_ms": _sum(
            rows, "context_jev_transport_overhead_ms"
        ),
        "jev_cost_usd_total": _jev_total_cost(rows, prices),
    }


def _delta_percent(candidate, baseline, key):
    left = baseline.get(key)
    right = candidate.get(key)
    if left is None or right is None or left == 0:
        return None
    return round(100 * (right - left) / left, 6)


def _adoption(baseline, candidate, max_quality_drop, max_failure_increase, max_latency_increase):
    """Apply explicit guardrails and never infer missing cost evidence."""
    reasons = []
    quality_delta = None
    failure_delta = None
    latency_delta = _delta_percent(candidate, baseline, "latency_ms_mean")
    if (
        baseline.get("task_success_rate") is not None
        and candidate.get("task_success_rate") is not None
    ):
        quality_delta = round(
            100
            * (candidate["task_success_rate"] - baseline["task_success_rate"]),
            6,
        )
        if quality_delta < -max_quality_drop:
            reasons.append("task success dropped beyond the quality guardrail")
    if baseline.get("error_rate") is not None and candidate.get("error_rate") is not None:
        failure_delta = round(
            100 * (candidate["error_rate"] - baseline["error_rate"]), 6
        )
        if failure_delta > max_failure_increase:
            reasons.append("error rate increased beyond the failure guardrail")
    if latency_delta is None:
        reasons.append("latency is not measured for both runs")
    elif latency_delta > max_latency_increase:
        reasons.append("latency increased beyond the latency guardrail")

    base_cost = baseline.get("task_cost_usd_per_success")
    candidate_cost = candidate.get("task_cost_usd_per_success")
    if base_cost is None or candidate_cost is None:
        reasons.append("cost per successful task is unknown")
        decision = "defer"
    elif candidate_cost >= base_cost:
        reasons.append("candidate is not cheaper per successful task")
        decision = "no-go"
    elif reasons:
        decision = "no-go"
    else:
        decision = "go"
    return {
        "decision": decision,
        "quality_delta_points": quality_delta,
        "failure_delta_points": failure_delta,
        "latency_delta_percent": latency_delta,
        "reasons": reasons,
        "guardrails": {
            "max_quality_drop_points": max_quality_drop,
            "max_failure_increase_points": max_failure_increase,
            "max_latency_increase_percent": max_latency_increase,
        },
    }


def compare_runs(
    baseline_path,
    candidate_path,
    baseline_arm="naru",
    candidate_arm="naru",
    prices=None,
    max_quality_drop=0.0,
    max_failure_increase=0.0,
    max_latency_increase=10.0,
):
    """Return a privacy-safe paired comparison suitable for JSON or prose."""
    prices = _validate_prices(prices)
    for value, name in (
        (max_quality_drop, "max_quality_drop"),
        (max_failure_increase, "max_failure_increase"),
        (max_latency_increase, "max_latency_increase"),
    ):
        _positive_number(value, name)
    baseline = load_run(baseline_path, baseline_arm)
    candidate = load_run(candidate_path, candidate_arm)
    _check_compatible(baseline, candidate)
    shared_usable = {
        qid: (baseline["rows"][qid], candidate["rows"][qid])
        for qid in baseline["rows"]
        if not _has_error(baseline["rows"][qid])
        and not _has_error(candidate["rows"][qid])
    }
    left_verdicts = {
        qid: _task_success(rows[0]) for qid, rows in shared_usable.items()
    }
    right_verdicts = {
        qid: _task_success(rows[1]) for qid, rows in shared_usable.items()
    }
    left_only, right_only, p_value = mcnemar(left_verdicts, right_verdicts)
    baseline_metrics = _run_metrics(baseline, prices)
    candidate_metrics = _run_metrics(candidate, prices)
    adoption = _adoption(
        baseline_metrics,
        candidate_metrics,
        max_quality_drop,
        max_failure_increase,
        max_latency_increase,
    )
    return {
        "format": FORMAT,
        "baseline": {
            "label": baseline["label"],
            "arm": baseline_arm,
            "config": _safe_config(baseline["config"]),
            "metrics": baseline_metrics,
        },
        "candidate": {
            "label": candidate["label"],
            "arm": candidate_arm,
            "config": _safe_config(candidate["config"]),
            "metrics": candidate_metrics,
        },
        "paired": {
            "questions": len(baseline["rows"]),
            "usable_questions": len(shared_usable),
            "baseline_only_successes": left_only,
            "candidate_only_successes": right_only,
            "mcnemar_p_value": round(p_value, 6),
        },
        "adoption": adoption,
        "pricing": (
            {"status": "provided", "version": prices["version"]}
            if prices is not None
            else {"status": "unknown", "version": None}
        ),
    }


def _fmt(value, suffix=""):
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return f"{value:,.3f}{suffix}"
    return f"{value:,}{suffix}"


def _evidence(value):
    return "measured" if value is not None else "unknown"


def render(result):
    """Render concise labeled evidence for a human review."""
    baseline = result["baseline"]
    candidate = result["candidate"]
    bm = baseline["metrics"]
    cm = candidate["metrics"]
    paired = result["paired"]
    adoption = result["adoption"]
    lines = [
        "paired context experiment",
        f"  baseline  {baseline['label']}",
        f"  candidate {candidate['label']}",
        f"  [measured] paired questions {paired['questions']:,}"
        f"; usable {paired['usable_questions']:,}",
        f"  [measured] task success  baseline {bm['task_successes']}/{bm['questions']}"
        f" ({100 * bm['task_success_rate']:.1f}%)"
        f"; candidate {cm['task_successes']}/{cm['questions']}"
        f" ({100 * cm['task_success_rate']:.1f}%)",
        f"  [{_evidence(bm['latency_ms_mean'])}] mean latency  baseline {_fmt(bm['latency_ms_mean'], 'ms')}"
        f"; candidate {_fmt(cm['latency_ms_mean'], 'ms')}",
        f"  [{_evidence(bm['context_selected_tokens_mean'])}] context selected  baseline {_fmt(bm['context_selected_tokens_mean'], 't/q')}"
        f"; candidate {_fmt(cm['context_selected_tokens_mean'], 't/q')}",
        f"  [{_evidence(cm['jev_calls'])}] Jev calls {_fmt(cm['jev_calls'])}; wall {_fmt(cm['jev_wall_latency_ms'], 'ms')}"
        f"; startup {_fmt(cm['jev_startup_latency_ms'], 'ms')}"
        f"; transport {_fmt(cm['jev_transport_overhead_ms'], 'ms')}",
        f"  [{result['pricing']['status']}] Jev pricing"
        + (
            f" version {result['pricing']['version']}"
            if result["pricing"]["version"]
            else ""
        ),
        f"  [baseline {bm['cost_status']}; candidate {cm['cost_status']}]"
        f" cost/success  baseline {_fmt(bm['task_cost_usd_per_success'], '$')}"
        f"; candidate {_fmt(cm['task_cost_usd_per_success'], '$')}",
        f"  [measured] paired McNemar p={paired['mcnemar_p_value']:.6f}"
        f"; baseline-only {paired['baseline_only_successes']}"
        f"; candidate-only {paired['candidate_only_successes']}",
        f"  [inferred] adoption {adoption['decision']}",
    ]
    for reason in adoption["reasons"]:
        lines.append(f"    {reason}")
    if not adoption["reasons"]:
        lines.append("    all configured guardrails passed")
    return "\n".join(lines)


def _prices(args):
    supplied = [getattr(args, field) for field in PRICE_FIELDS]
    if args.jev_pricing_profile:
        if any(value is not None for value in supplied) or args.jev_pricing_version:
            raise ExperimentError(
                "choose --jev-pricing-profile or explicit Jev rates, not both"
            )
        return dict(JEV_PRICING_PROFILES[args.jev_pricing_profile])
    if not any(value is not None for value in supplied):
        if args.jev_pricing_version:
            raise ExperimentError("Jev pricing version requires Jev rates")
        return None
    for field in REQUIRED_PRICE_FIELDS:
        if getattr(args, field) is None:
            raise ExperimentError(
                "provide Jev input and output rates; cache rates are optional"
            )
    if not args.jev_pricing_version:
        raise ExperimentError("Jev rates require --jev-pricing-version")
    return {
        field: (
            None
            if getattr(args, field) is None
            else _positive_number(getattr(args, field), field)
        )
        for field in PRICE_FIELDS
    } | {"version": args.jev_pricing_version}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline-arm", default="naru")
    parser.add_argument("--candidate-arm", default="naru")
    parser.add_argument("--max-quality-drop-points", type=float, default=0.0)
    parser.add_argument("--max-failure-increase-points", type=float, default=0.0)
    parser.add_argument("--max-latency-increase-percent", type=float, default=10.0)
    parser.add_argument(
        "--jev-pricing-profile",
        choices=tuple(JEV_PRICING_PROFILES),
        help="named Jev pricing profile with explicit cache-rate semantics",
    )
    parser.add_argument("--jev-pricing-version")
    for field, label in (
        ("input_usd_per_million", "input"),
        ("cache_read_usd_per_million", "cache-read"),
        ("cache_write_usd_per_million", "cache-write"),
        ("output_usd_per_million", "output"),
    ):
        parser.add_argument(
            f"--jev-{label}-usd-per-million",
            dest=field,
            type=float,
            help=f"Jev {label} price in USD per million tokens",
        )
    parser.add_argument("--json", action="store_true", help="emit the safe report as JSON")
    args = parser.parse_args(argv)
    try:
        prices = _prices(args)
        for value, name in (
            (args.max_quality_drop_points, "--max-quality-drop-points"),
            (args.max_failure_increase_points, "--max-failure-increase-points"),
            (args.max_latency_increase_percent, "--max-latency-increase-percent"),
        ):
            _positive_number(value, name)
        result = compare_runs(
            args.baseline,
            args.candidate,
            baseline_arm=args.baseline_arm,
            candidate_arm=args.candidate_arm,
            prices=prices,
            max_quality_drop=args.max_quality_drop_points,
            max_failure_increase=args.max_failure_increase_points,
            max_latency_increase=args.max_latency_increase_percent,
        )
    except ExperimentError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=1) if args.json else render(result))
    return 0


def demo():
    """Offline checks for pairing, pricing, guardrails, and privacy."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        common = {
            "split": "oracle",
            "qtype": None,
            "model": "test-model",
            "judge_model": "test-judge",
            "resolved_backend": "test",
            "backend": "test",
            "backend_fingerprint": "answerer",
            "judge_resolved_backend": "test",
            "judge_backend": "test",
            "judge_backend_fingerprint": "judge",
            "budget": 100,
            "max_turns": 3,
            "harness_floor": 100,
            "workers": 1,
            "no_rubric": False,
            "no_index": False,
            "rag_k": 8,
            "rsm_k": 6,
            "rsm_tau": 0.85,
            "rsm_chunk_turns": 5,
            "rsm_budget": 4000,
            "tokens_measured": True,
            "pricing_version": None,
        }
        base = {
            "config": {**common, "context_policy": "deterministic"},
            "rows": [
                {
                    "qid": "secret-question-1",
                    "arm": "naru",
                    "task_success": True,
                    "latency_ms": 100,
                    "task_cost_usd": 0.10,
                    "context_selected_tokens": 80,
                },
                {
                    "qid": "secret-question-2",
                    "arm": "naru",
                    "task_success": False,
                    "latency_ms": 120,
                    "task_cost_usd": 0.20,
                    "context_selected_tokens": 80,
                },
                {
                    "qid": "secret-question-3",
                    "arm": "naru",
                    "task_success": True,
                    "latency_ms": 100,
                    "task_cost_usd": 0.20,
                    "context_selected_tokens": 80,
                },
            ],
        }
        candidate = {
            "config": {**common, "context_policy": "jev"},
            "rows": [
                {
                    "qid": "secret-question-1",
                    "arm": "naru",
                    "task_success": True,
                    "latency_ms": 105,
                    "task_cost_usd": 0.05,
                    "context_selected_tokens": 40,
                    "context_jev_calls": 1,
                    "context_jev_input_tokens": 2,
                    "context_jev_output_tokens": 1,
                    "cost_includes_jev": False,
                },
                {
                    "qid": "secret-question-2",
                    "arm": "naru",
                    "task_success": False,
                    "latency_ms": 115,
                    "task_cost_usd": 0.10,
                    "context_selected_tokens": 40,
                    "context_jev_calls": 1,
                    "context_jev_input_tokens": 2,
                    "context_jev_output_tokens": 1,
                    "cost_includes_jev": False,
                },
                {
                    "qid": "secret-question-3",
                    "arm": "naru",
                    "task_success": True,
                    "latency_ms": 105,
                    "task_cost_usd": 0.05,
                    "context_selected_tokens": 40,
                    "context_jev_calls": 1,
                    "context_jev_input_tokens": 2,
                    "context_jev_output_tokens": 1,
                    "cost_includes_jev": False,
                },
            ],
        }
        base_path = root / "baseline.json"
        candidate_path = root / "candidate.json"
        base_path.write_text(json.dumps(base))
        candidate_path.write_text(json.dumps(candidate))
        prices = {
            "input_usd_per_million": 1.0,
            "cache_read_usd_per_million": 1.0,
            "cache_write_usd_per_million": 1.0,
            "output_usd_per_million": 1.0,
            "version": "test-prices",
        }
        profile = JEV_PRICING_PROFILES["typesafe-jev-input-0.042-free-output"]
        assert _jev_cost(
            {"context_jev_input_tokens": 1_000_000, "context_jev_output_tokens": 1},
            profile,
        ) == 0.042
        assert _jev_cost(
            {"context_jev_cache_read_tokens": 1}, profile
        ) is None
        assert _jev_cost(
            {
                "context_jev_calls": 1,
                "jev_usage": {"tokens_measured": False},
            },
            profile,
        ) is None
        assert _jev_cost(
            {
                "context_jev_calls": 1,
                "jev_usage": {"calls": 0, "errors": 1},
            },
            profile,
        ) is None
        result = compare_runs(base_path, candidate_path, prices=prices)
        assert result["adoption"]["decision"] == "go", result
        assert result["candidate"]["metrics"]["context_selected_tokens_mean"] == 40
        assert result["candidate"]["metrics"]["jev_cost_usd_total"] == 0.000009
        assert (
            result["candidate"]["metrics"]["task_cost_usd_per_success"]
            == 0.050003
        )
        assert "secret-question-1" not in json.dumps(result)
        unknown = compare_runs(base_path, candidate_path)
        assert unknown["adoption"]["decision"] == "defer", unknown
        assert unknown["pricing"]["status"] == "unknown"
        unmeasured = json.loads(base_path.read_text())
        unmeasured["config"]["tokens_measured"] = False
        unmeasured_path = root / "unmeasured.json"
        unmeasured_path.write_text(json.dumps(unmeasured))
        not_measured = compare_runs(unmeasured_path, candidate_path, prices=prices)
        assert not_measured["adoption"]["decision"] == "defer", not_measured
        assert not_measured["baseline"]["metrics"]["cost_status"] == "unknown"
        mismatch = root / "mismatch.json"
        mismatch.write_text(json.dumps({**candidate, "config": {**candidate["config"], "budget": 101}}))
        try:
            compare_runs(base_path, mismatch)
        except ExperimentError as error:
            assert "budget" in str(error)
        else:
            raise AssertionError("incompatible runs were accepted")
        provider_mismatch = root / "provider-mismatch.json"
        provider_mismatch.write_text(
            json.dumps(
                {
                    **candidate,
                    "config": {
                        **candidate["config"],
                        "resolved_backend": "other",
                    },
                }
            )
        )
        try:
            compare_runs(base_path, provider_mismatch)
        except ExperimentError as error:
            assert "resolved_backend" in str(error)
        else:
            raise AssertionError("different providers were accepted")
        mismatch_set = json.loads(candidate_path.read_text())
        mismatch_set["rows"][0]["qid"] = "different-question"
        mismatch_set_path = root / "mismatch-set.json"
        mismatch_set_path.write_text(json.dumps(mismatch_set))
        try:
            compare_runs(base_path, mismatch_set_path)
        except ExperimentError as error:
            assert "same question set" in str(error)
        else:
            raise AssertionError("different question sets were accepted")
    print("ok — experiment checks passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selfcheck"]:
        demo()
    else:
        sys.exit(main())
