"""Bounded selection of the mutable context shown to the Naru model.

The stable system prompt and resident kernel digest stay outside this policy.
The policy owns only the per-turn indexes, state, and latest observation. Its
default is deterministic and its Jev path is a narrow, opt-in checkpoint: Jev
chooses among named blocks, while this module still enforces the token budget.

No block body is sent to Jev.  The evaluator receives the goal, next action,
and non-sensitive block metadata, so a malformed or unavailable evaluator can
fall back without making context delivery depend on a model call.
"""

import json
from dataclasses import dataclass, replace

from eviction import est

CONTEXT_POLICIES = ("deterministic", "shadow", "jev")
JEV_CHECKPOINTS = frozenset(("budget_pressure", "large_tool_result"))
MIN_JEV_CONFIDENCE = 0.80
JEV_INPUT_TOKEN_BUDGET = 800
_CHOICES = frozenset(("needed", "useful", "omit", "no_match"))


@dataclass(frozen=True)
class ContextBlock:
    """One named piece of mutable context.

    ``recovery`` is a short handle that must survive clipping of a large
    observation.  It is deliberately separate from ``content`` so callers
    cannot accidentally treat a recovery pointer as part of the payload.
    """

    block_id: str
    source_type: str
    content: str
    provenance: str = ""
    required: bool = False
    priority: int = 0
    recovery: str = ""
    budgeted: bool = True

    def __post_init__(self):
        if not isinstance(self.block_id, str) or not self.block_id.strip():
            raise ValueError("context block id must be non-empty")
        if not isinstance(self.source_type, str) or not self.source_type.strip():
            raise ValueError("context block source_type must be non-empty")
        if not isinstance(self.content, str):
            raise TypeError("context block content must be text")
        if not isinstance(self.provenance, str) or not isinstance(self.recovery, str):
            raise TypeError("context block metadata must be text")

    def render(self):
        body = self.content.rstrip()
        if self.recovery:
            body += "\n-> " + self.recovery
        return body

    @property
    def token_count(self):
        return est(self.render())

    def metadata(self):
        """Return the Jev-safe description, never the block body."""
        return {
            "id": self.block_id[:80],
            "source_type": self.source_type[:40],
            "token_count": self.token_count,
            "provenance": self.provenance[:120],
            "required": self.required,
            "budgeted": self.budgeted,
        }

    def clipped(self, max_tokens):
        """Clip content while retaining a recovery handle when one exists."""
        if max_tokens < 1:
            raise ValueError("a context block needs at least one token")
        if self.token_count <= max_tokens:
            return self
        room = max(1, max_tokens * 4)
        content = self.content
        if self.recovery:
            room = max(1, room - len("\n-> " + self.recovery))
        clipped = replace(self, content=content[:room].rstrip())
        while clipped.token_count > max_tokens and len(clipped.content) > 1:
            clipped = replace(clipped, content=clipped.content[:-1].rstrip())
        if clipped.token_count > max_tokens:
            raise ValueError("context block cannot fit its token budget")
        return clipped


@dataclass(frozen=True)
class ContextSelection:
    """The budget-enforced result of one context decision."""

    blocks: tuple
    considered_tokens: int
    selected_tokens: int
    dropped_tokens: int
    policy: str
    checkpoint: str
    jev_invoked: bool = False
    jev_applied: bool = False
    jev_confident_choices: int = 0
    jev_input_tokens: int = 0
    jev_cache_read_tokens: int = 0
    jev_cache_write_tokens: int = 0
    jev_output_tokens: int = 0
    jev_latency_ms: float = 0.0
    jev_errors: int = 0
    fallback_reason: str = ""
    choices: tuple = ()
    all_block_ids: tuple = ()

    @property
    def text(self):
        return "\n\n".join(block.render() for block in self.blocks)

    def as_dict(self):
        selected = {block.block_id for block in self.blocks}
        return {
            "policy": self.policy,
            "checkpoint": self.checkpoint,
            "considered_tokens": self.considered_tokens,
            "selected_tokens": self.selected_tokens,
            "dropped_tokens": self.dropped_tokens,
            "selected_blocks": [block.block_id for block in self.blocks],
            "omitted_blocks": [
                block_id for block_id in self.all_block_ids if block_id not in selected
            ],
            "jev_invoked": self.jev_invoked,
            "jev_applied": self.jev_applied,
            "jev_confident_choices": self.jev_confident_choices,
            "jev_input_tokens": self.jev_input_tokens,
            "jev_cache_read_tokens": self.jev_cache_read_tokens,
            "jev_cache_write_tokens": self.jev_cache_write_tokens,
            "jev_output_tokens": self.jev_output_tokens,
            "jev_latency_ms": round(self.jev_latency_ms, 3),
            "jev_errors": self.jev_errors,
            "fallback_reason": self.fallback_reason,
            "choices": dict(self.choices),
        }


def _joined(items):
    return "\n\n".join(item.render() for item in items)


def _fit(blocks, budget):
    """Keep required blocks and add optional blocks by descending priority."""
    if budget < 1:
        raise ValueError("context budget must be positive")
    selected = [
        (i, block) for i, block in enumerate(blocks) if not block.budgeted
    ]
    required = [
        (i, block)
        for i, block in enumerate(blocks)
        if block.budgeted and block.required
    ]
    optional = [
        (i, block)
        for i, block in enumerate(blocks)
        if block.budgeted and not block.required
    ]

    def budgeted_items(items):
        return [item for _, item in items if item.budgeted]

    for position, block in required:
        candidate = block
        current = budgeted_items(selected)
        if est(_joined(current + [candidate])) > budget:
            remaining = max(1, budget - est(_joined(current)))
            candidate = block.clipped(remaining)
            while (
                est(_joined(budgeted_items(selected) + [candidate])) > budget
                and remaining > 1
            ):
                remaining -= 1
                candidate = block.clipped(remaining)
        if est(_joined(budgeted_items(selected) + [candidate])) > budget:
            raise ValueError(f"required context block cannot fit: {block.block_id}")
        selected.append((position, candidate))

    for position, block in sorted(optional, key=lambda pair: (-pair[1].priority, pair[0])):
        if est(_joined(budgeted_items(selected) + [block])) <= budget:
            selected.append((position, block))
    selected.sort(key=lambda pair: pair[0])
    return tuple(block for _, block in selected)


def _question_map(blocks):
    questions = {}
    for block in blocks:
        if block.required or not block.budgeted:
            continue
        questions[block.block_id] = {
            "type": "choice",
            "instructions": (
                f"For context block {block.block_id!r}, decide whether it is needed, "
                "useful, or safe to omit for the current goal and next action."
            ),
            "criteria": {
                "needed": "Necessary to answer or safely continue.",
                "useful": "Relevant if budget allows.",
                "omit": "Not needed for the current task.",
                "no_match": "No meaningful relation to the current task.",
            },
        }
    return questions


def jev_state(goal, next_action, blocks):
    """Build the metadata-only state sent to an evaluator."""
    return {
        "goal": str(goal)[:800],
        "next_action": str(next_action)[:500],
        "blocks": [block.metadata() for block in blocks],
    }


def _jev_request_tokens(goal, next_action, blocks, questions):
    """Estimate the complete Jev input, including typed question rubrics."""
    request = {
        "state": jev_state(goal, next_action, blocks),
        "questions": questions,
    }
    return est(json.dumps(request, sort_keys=True, separators=(",", ":")))


def _parse_jev(response, questions):
    if not isinstance(response, dict):
        raise ValueError("Jev response must be an object")
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise ValueError("Jev response must contain typed answers")
    parsed = {}
    for block_id in questions:
        answer = answers.get(block_id)
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"Jev answer is not a choice: {block_id}")
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        if choice not in _CHOICES:
            raise ValueError(f"Jev returned an unknown choice: {block_id}")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"Jev confidence is not numeric: {block_id}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Jev confidence is outside [0,1]: {block_id}")
        parsed[block_id] = {"choice": choice, "confidence": float(confidence)}
    return parsed


def _usage(response):
    usage = response.get("usage", {}) if isinstance(response, dict) else {}
    if not isinstance(usage, dict):
        return 0, 0, 0, 0, 0.0
    return (
        int(usage.get("input_tokens", 0) or 0),
        int(
            usage.get(
                "cache_read_tokens",
                usage.get("cache_read_input_tokens", 0),
            )
            or 0
        ),
        int(
            usage.get(
                "cache_write_tokens",
                usage.get("cache_creation_input_tokens", 0),
            )
            or 0
        ),
        int(usage.get("output_tokens", 0) or 0),
        float(usage.get("latency_ms", 0) or 0),
    )


def select_context(
    blocks,
    budget,
    goal="",
    next_action="",
    mode="deterministic",
    evaluator=None,
    checkpoint="turn",
):
    """Select a bounded context tail, optionally consulting Jev once.

    Jev is invoked only at a named checkpoint under pressure.  Its answer can
    reorder or omit optional blocks in ``jev`` mode; ``shadow`` records the
    same answer but applies the deterministic policy.  Any invalid response,
    evaluator exception, or missing evaluator uses the deterministic result.
    """
    if mode not in CONTEXT_POLICIES:
        raise ValueError(f"unknown context policy: {mode}")
    blocks = tuple(blocks)
    if len({block.block_id for block in blocks}) != len(blocks):
        raise ValueError("context block ids must be unique")
    budgeted = [block for block in blocks if block.budgeted]
    considered_tokens = est(_joined(budgeted)) if budgeted else 0
    questions = _question_map(blocks)
    pressure = considered_tokens > budget or checkpoint == "large_tool_result"
    jev_request_tokens = _jev_request_tokens(goal, next_action, blocks, questions)
    should_invoke = (
        mode in ("shadow", "jev")
        and checkpoint in JEV_CHECKPOINTS
        and pressure
        and evaluator is not None
        and bool(questions)
        and jev_request_tokens <= JEV_INPUT_TOKEN_BUDGET
    )
    choices = {}
    jev_invoked = False
    jev_applied = False
    confident = 0
    input_tokens = cache_read_tokens = cache_write_tokens = output_tokens = 0
    latency_ms = 0.0
    jev_errors = 0
    fallback_reason = ""
    candidates = list(blocks)

    if (
        mode in ("shadow", "jev")
        and checkpoint in JEV_CHECKPOINTS
        and pressure
        and bool(questions)
        and evaluator is None
    ):
        fallback_reason = "evaluator_unavailable"
    elif (
        mode in ("shadow", "jev")
        and checkpoint in JEV_CHECKPOINTS
        and pressure
        and bool(questions)
        and evaluator is not None
        and jev_request_tokens > JEV_INPUT_TOKEN_BUDGET
    ):
        fallback_reason = "jev_input_too_large"

    if should_invoke:
        jev_invoked = True
        try:
            response = evaluator(jev_state(goal, next_action, blocks), questions)
            choices = _parse_jev(response, questions)
            (
                input_tokens,
                cache_read_tokens,
                cache_write_tokens,
                output_tokens,
                latency_ms,
            ) = _usage(response)
            confident = sum(
                1 for answer in choices.values()
                if answer["confidence"] >= MIN_JEV_CONFIDENCE
            )
            if mode == "jev":
                ranked = []
                for position, block in enumerate(candidates):
                    answer = choices.get(block.block_id)
                    if answer and answer["confidence"] >= MIN_JEV_CONFIDENCE:
                        choice = answer["choice"]
                        if choice in ("omit", "no_match") and not block.required:
                            continue
                        boost = 100 if choice == "needed" else 10 if choice == "useful" else 0
                        block = replace(
                            block,
                            priority=block.priority + boost,
                            required=block.required or choice == "needed",
                        )
                    ranked.append((position, block))
                candidates = [block for _, block in ranked]
                jev_applied = True
        except Exception as error:
            jev_errors = 1
            fallback_reason = f"{type(error).__name__}"

    selected = _fit(candidates, budget)
    selected_budgeted = [block for block in selected if block.budgeted]
    selected_tokens = est(_joined(selected_budgeted)) if selected_budgeted else 0
    selection = ContextSelection(
        blocks=selected,
        considered_tokens=considered_tokens,
        selected_tokens=selected_tokens,
        dropped_tokens=max(0, considered_tokens - selected_tokens),
        policy=mode,
        checkpoint=checkpoint,
        jev_invoked=jev_invoked,
        jev_applied=jev_applied,
        jev_confident_choices=confident,
        jev_input_tokens=input_tokens,
        jev_cache_read_tokens=cache_read_tokens,
        jev_cache_write_tokens=cache_write_tokens,
        jev_output_tokens=output_tokens,
        jev_latency_ms=latency_ms,
        jev_errors=jev_errors,
        fallback_reason=fallback_reason,
        choices=tuple(sorted(choices.items())),
        all_block_ids=tuple(block.block_id for block in blocks),
    )
    return selection


def update_stats(stats, selection, tool_result_tokens=0):
    """Accumulate numeric, non-content telemetry for one selection."""
    stats["context_decisions"] = stats.get("context_decisions", 0) + 1
    stats["context_considered_tokens"] = (
        stats.get("context_considered_tokens", 0) + selection.considered_tokens
    )
    stats["context_selected_tokens"] = (
        stats.get("context_selected_tokens", 0) + selection.selected_tokens
    )
    stats["context_dropped_tokens"] = (
        stats.get("context_dropped_tokens", 0) + selection.dropped_tokens
    )
    stats["context_peak_selected_tokens"] = max(
        stats.get("context_peak_selected_tokens", 0), selection.selected_tokens
    )
    stats["context_tool_result_tokens"] = (
        stats.get("context_tool_result_tokens", 0) + tool_result_tokens
    )
    if selection.jev_invoked:
        stats["context_jev_calls"] = stats.get("context_jev_calls", 0) + 1
    stats["context_jev_input_tokens"] = (
        stats.get("context_jev_input_tokens", 0) + selection.jev_input_tokens
    )
    stats["context_jev_cache_read_tokens"] = (
        stats.get("context_jev_cache_read_tokens", 0)
        + selection.jev_cache_read_tokens
    )
    stats["context_jev_cache_write_tokens"] = (
        stats.get("context_jev_cache_write_tokens", 0)
        + selection.jev_cache_write_tokens
    )
    stats["context_jev_output_tokens"] = (
        stats.get("context_jev_output_tokens", 0) + selection.jev_output_tokens
    )
    stats["context_jev_latency_ms"] = (
        stats.get("context_jev_latency_ms", 0.0) + selection.jev_latency_ms
    )
    stats["context_jev_errors"] = (
        stats.get("context_jev_errors", 0) + selection.jev_errors
    )
    if selection.fallback_reason:
        stats["context_fallbacks"] = stats.get("context_fallbacks", 0) + 1
    stats["context_selected_blocks"] = [
        block.block_id for block in selection.blocks
    ]
    stats["context_omitted_blocks"] = selection.as_dict()["omitted_blocks"]


def demo():
    """Offline checks for deterministic fitting and Jev fallback semantics."""
    recovery = "ms.expand(41, session_id='run')"
    blocks = [
        ContextBlock("source", "index", "SOURCE " + "s" * 80, priority=1),
        ContextBlock("resident", "kernel", "RESIDENT", required=True, priority=5),
        ContextBlock("current", "state", "CURRENT", required=True, priority=6),
        ContextBlock(
            "latest",
            "observation",
            "SECRET-OBSERVATION-" + "x" * 500,
            required=True,
            recovery=recovery,
        ),
    ]
    state_seen = {}

    def evaluator(state, questions):
        state_seen.update(state=state, questions=questions)
        return {
            "answers": {
                key: {"type": "choice", "choice": "omit", "confidence": 0.95}
                for key in questions
            },
            "usage": {
                "input_tokens": 11,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
                "output_tokens": 7,
                "latency_ms": 2.5,
            },
        }

    deterministic = select_context(
        blocks, 80, goal="find a fact", next_action="answer", checkpoint="budget_pressure"
    )
    assert "resident" in [b.block_id for b in deterministic.blocks]
    assert "current" in [b.block_id for b in deterministic.blocks]
    assert recovery in deterministic.text, deterministic.text
    assert est(deterministic.text) <= 80

    jev = select_context(
        blocks,
        80,
        goal="find a fact",
        next_action="answer",
        mode="jev",
        evaluator=evaluator,
        checkpoint="budget_pressure",
    )
    assert jev.jev_invoked and jev.jev_applied
    assert (
        jev.jev_input_tokens == 11
        and jev.jev_cache_read_tokens == 3
        and jev.jev_cache_write_tokens == 2
        and jev.jev_output_tokens == 7
    )
    assert "source" not in [b.block_id for b in jev.blocks]
    assert recovery in jev.text
    assert "SECRET-OBSERVATION" not in state_seen["state"]
    assert all("content" not in block for block in state_seen["state"]["blocks"])

    low_blocks = [
        ContextBlock("current", "state", "C", required=True),
        ContextBlock("latest", "observation", "L", required=True, recovery="recover()"),
        ContextBlock("source", "index", "SOURCE", priority=1),
        ContextBlock("filler", "index", "F" * 200),
    ]

    def low_confidence(_state, questions):
        return {
            "answers": {
                key: {"type": "choice", "choice": "omit", "confidence": 0.79}
                for key in questions
            }
        }

    low = select_context(
        low_blocks,
        30,
        mode="jev",
        evaluator=low_confidence,
        checkpoint="budget_pressure",
    )
    assert "source" in [b.block_id for b in low.blocks]
    assert "filler" not in [b.block_id for b in low.blocks]

    def needed(_state, questions):
        return {
            "answers": {
                key: {"type": "choice", "choice": "needed", "confidence": 1.0}
                for key in questions
            }
        }

    needed_selection = select_context(
        [
            ContextBlock("current", "state", "CURRENT", required=True),
            ContextBlock("evidence", "index", "EVIDENCE " + "e" * 200),
        ],
        12,
        goal="use the evidence",
        next_action="answer",
        mode="jev",
        evaluator=needed,
        checkpoint="budget_pressure",
    )
    assert "evidence" in [b.block_id for b in needed_selection.blocks]

    oversized = [
        ContextBlock(
            f"block-{i}",
            "tool-result",
            "x" * 200,
            provenance="p" * 120,
        )
        for i in range(16)
    ]
    oversized_calls = []

    def oversized_evaluator(state, questions):
        oversized_calls.append((state, questions))
        return {"answers": {}}

    bounded = select_context(
        oversized,
        20,
        goal="find a fact",
        next_action="answer",
        mode="jev",
        evaluator=oversized_evaluator,
        checkpoint="budget_pressure",
    )
    assert not oversized_calls
    assert not bounded.jev_invoked
    assert bounded.fallback_reason == "jev_input_too_large"

    stable = select_context(
        [
            ContextBlock("trace", "trace-index", "T" * 500, budgeted=False),
            ContextBlock("tail", "state", "TAIL", required=True),
        ],
        10,
    )
    assert "trace" in [b.block_id for b in stable.blocks]
    assert stable.selected_tokens <= 10

    shadow = select_context(
        blocks,
        80,
        goal="find a fact",
        next_action="answer",
        mode="shadow",
        evaluator=evaluator,
        checkpoint="budget_pressure",
    )
    assert shadow.jev_invoked and not shadow.jev_applied

    bad = select_context(
        blocks,
        80,
        mode="jev",
        evaluator=lambda _state, _questions: {"answers": {}},
        checkpoint="budget_pressure",
    )
    assert bad.jev_errors == 1 and bad.fallback_reason == "ValueError"
    stats = {}
    update_stats(stats, jev, tool_result_tokens=13)
    assert stats["context_jev_calls"] == 1 and stats["context_tool_result_tokens"] == 13
    print("ok — context policy checks passed")


if __name__ == "__main__":
    demo()
