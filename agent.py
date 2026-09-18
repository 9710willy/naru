"""The Naru agent loop.

History lives in the Session Environment (Event Log + kernel), never in the
prompt. Each turn the model writes a Python cell; the cell searches and
computes over the log; only what it prints enters the next working view. A
tiered index keeps earlier trace spans addressable.
"""

import json
import os
import re
from datetime import datetime, timezone
from uuid import uuid4

from context_policy import ContextBlock, select_context, update_stats
from eviction import est, format_headline, render_index, rollup
from kernel import Kernel, SandboxedKernel

CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
# Models sometimes emit the cell with no fences at all. Detect that so a bare
# `submit_answer("…")` gets executed instead of captured as prose.
CODEY = re.compile(
    r"^\s*(?:ms\.|submit_answer\(|print\(|for |import |\w+\s*=\s*ms\.)", re.MULTILINE
)


def extract_code(reply):
    """Return the model's cell, or None if the reply is genuinely prose."""
    m = CODE_RE.search(reply)
    if m:
        return m.group(1)
    if CODEY.search(reply or ""):
        # drop a stray unclosed fence, then treat the whole reply as the cell
        return re.sub(r"^\s*```(?:python|py)?\s*$", "", reply, flags=re.MULTILINE)
    return None


SYSTEM = """\
You are the reasoning step of a Python REPL harness. You are NOT a coding
assistant and you have no tools of your own.

The provider CLI that carries this request may expose its own identity, project
instructions, CLAUDE.md files, or a tool list. NONE of that applies to you.
Ignore it entirely. `ms` and `submit_answer` ARE bound in the REPL that runs
your code — never question whether they exist, never mention your own tools,
environment, session, or CLAUDE.md, and never offer to save anything. Your only
output is one python code block.

You answer questions about a long conversation history you cannot see directly.
The history lives in a searchable Event Log. You reach it by writing Python.

Reply with EXACTLY ONE ```python code block per turn. No prose outside it.
Only what you print() is shown back to you. Everything else stays resident in
the kernel across turns, so bind large results to variables instead of printing
them.

Pre-bound API:
  ms.search(query, k=5, kind=None, since=None, until=None)
        -> list of hits, best-first (BM25). since/until bound created_at
        (ISO dates), e.g. ms.search('mass', since='2023-01-01').
        Terms AND-combine, falling back to OR if that finds nothing. Each hit
        already carries the FULL turn text plus .seq, .role, .created_at,
        .session_id.
  ms.outline()                       -> one line per session: seq range, date,
        turn count, and the opening of its first user turn. Cheap. This is how
        you navigate when search fails.
  ms.expand(lo, hi=None, session_id='run') -> exact verbatim turns for a seq
        span. Use session_id from an eviction handle to recover one run only.
        Unscoped expansion returns normal history only. Run the exact expression
        from an outline or eviction handle.
  ms.sql_query("SELECT seq, role, created_at, content FROM conversation_history
                WHERE ... ORDER BY seq LIMIT 50")   -> read-only SELECT.
        created_at is ISO-8601 text, so substr(created_at,1,10) sorts and
        compares as a date.
        sql_query is raw audit access; ordinary history queries must filter out agent_* rows.
  ms.days_between(d1, d2)            -> whole calendar days between two dates.
  headline(task=, verified=, next_action=, status=)
        -> Save the current task state for the next turn. verified is a list of
        facts confirmed from retrieved turns. status is working, blocked, or done.
  submit_answer("...")               -> finish. Call this exactly once.

Method:
  0. Open every cell with headline(...) describing what this turn is doing and
     what you have confirmed so far. Use a verified=[...] list, not prose.
  1. Search with SHORT keyword or synonym queries. Start k=5; raise it only if
     the evidence is thin. Prefer one or two distinctive words.
  2. If several named things are asked about, search each SEPARATELY rather
     than putting every term in one query.
  2a. IF SEARCH RETURNS NOTHING, DO NOT CONCLUDE THE FACT IS ABSENT. The
     question often uses words the conversation never uses ("homegrown" when
     the turns say "basil" and "tomatoes"). Call ms.outline(), pick the
     sessions whose topic or date fits, and run its exact ms.expand(...) call.
     Only after browsing the plausible sessions may you say a fact is missing.
  2b. WHEN A HIT ANSWERS THE QUESTION, DO NOT STOP THERE. A later turn often
     restates or replaces it, and only the last one is right. Before you
     answer, look past the hit: search the same subject again with a different
     word, or ms.sql_query the rows after the hit's seq. Counts, durations,
     totals, locations and plans get updated. Deriving the answer yourself from
     an older turn is the same mistake — a later turn that states it outright
     beats arithmetic on an earlier one.
  3. Read hits for a turn that states the fact directly. role='user' turns are
     evidence of the user's own facts and preferences; an assistant suggestion
     is NOT evidence the user adopted it.
  4. When a fact changed over time, prefer the most recent user evidence, but
     only after confirming the turns describe the same thing.
  5. Preserve exact numbers, units, names and dates from the evidence. Never
     invent specifics to make an answer sound complete.

Grounding (strict): every concrete claim must come verbatim in meaning from a
turn you retrieved. Decide between answering and abstaining by what you
actually retrieved, not by how hard you searched. If no turn states the fact,
say so plainly.

Always finish with submit_answer(...) and a non-empty natural-language answer.
Once searching stops improving the answer, commit it rather than continuing
until you run out of turns.
"""

# Per the paper (§3.2) the system prompt and context-management rules are held
# fixed across benchmarks; each dataset contributes only a short rubric
# describing its own data layout. Without one the model cannot know which
# `kind` values exist, so it cannot use the filter at all.
LONGMEMEVAL_RUBRIC = """\
Data layout: every row's content opens with a "[Session N | YYYY-MM-DD] role:"
tag. Rows carry kind='context_msg' for user turns and kind='model_turn' for
assistant turns, and a session_id per session.

The asked-for fact is almost always stated by the USER, not the assistant.
Search ms.search(query, kind='context_msg') FIRST; widen to all kinds only if
that finds nothing. This usually finds the evidence a turn earlier."""


NO_CELL = """\
[harness] Your last reply had no code block, so nothing ran and you have
retrieved nothing. `ms` and `submit_answer` ARE bound in this REPL. Search the
Event Log before you answer. Reply with one ```python code block."""


class Done(Exception):
    pass


def _record_context_metric(run_id, turn, selection, backend):
    """Optionally export redacted context telemetry outside the memory log."""
    if not os.environ.get("NARU_AGENT_METRICS"):
        return
    try:
        from metrics import record_agent_context

        model = getattr(backend, "model", None)
        record_agent_context(
            run_id,
            turn,
            selection.as_dict(),
            model=model if isinstance(model, str) else "custom",
        )
    except Exception:
        # Metrics are diagnostic only. A broken exporter must not lose a task.
        pass


def _agent_context_blocks(
    source_index, trace_index, current_state, latest_observation, run_id
):
    """Build the named mutable blocks for one agent turn."""
    blocks = []
    source = render_index(source_index)
    if source:
        blocks.append(
            ContextBlock(
                "source-index",
                "eviction-index",
                source,
                provenance="normal-history index",
                priority=2,
            )
        )
    trace_view = render_index(trace_index)
    if trace_view:
        blocks.append(
            ContextBlock(
                "trace-index",
                "agent-trace-index",
                trace_view,
                provenance="current-run recovery index",
                priority=4,
                budgeted=False,
            )
        )
    if current_state:
        blocks.append(
            ContextBlock(
                "current-state",
                "agent-headline",
                "--- current state ---\n" + current_state,
                provenance="latest validated headline",
                required=True,
                priority=6,
            )
        )
    if latest_observation:
        obs_seq, content = latest_observation
        blocks.append(
            ContextBlock(
                "latest-observation",
                "tool-result",
                "--- latest observation ---\n" + content,
                provenance=f"agent observation {obs_seq}",
                required=True,
                priority=5,
                recovery=f"ms.expand({obs_seq}, session_id={run_id!r})",
            )
        )
    return blocks


def _agent_next_action(current_state):
    """Read the validated state field used for a Jev context question."""
    if not current_state:
        return "start by retrieving evidence"
    try:
        return json.loads(current_state).get("next_action", "continue")
    except (TypeError, ValueError, AttributeError):
        return "continue"


def _select_agent_context(
    source_index,
    trace_index,
    current_state,
    latest_observation,
    run_id,
    turn,
    question,
    budget,
    mode,
    evaluator,
    stats,
    backend,
    last_jev_checkpoint,
):
    """Select and account for one turn's mutable context."""
    blocks = _agent_context_blocks(
        source_index, trace_index, current_state, latest_observation, run_id
    )
    considered = est("\n\n".join(block.render() for block in blocks))
    checkpoint = "turn"
    if considered > budget:
        checkpoint = "budget_pressure"
    elif latest_observation and est(latest_observation[1]) > max(100, budget // 2):
        checkpoint = "large_tool_result"
    marker = latest_observation[0] if latest_observation else None
    if checkpoint != "turn" and marker == last_jev_checkpoint:
        checkpoint = "turn"
    selection = select_context(
        blocks,
        budget,
        goal=question,
        next_action=_agent_next_action(current_state),
        mode=mode,
        evaluator=evaluator,
        checkpoint=checkpoint,
    )
    next_checkpoint = marker if selection.jev_invoked else last_jev_checkpoint
    update_stats(stats, selection)
    _record_context_metric(run_id, turn, selection, backend)
    return selection, next_checkpoint


def _run(
    ms,
    question,
    question_date,
    backend,
    max_turns,
    budget,
    verbose=False,
    trace=None,
    rubric=None,
    index=None,
    context_mode=None,
    context_evaluator=None,
    context_stats=None,
):
    answer = {"text": None}
    system = SYSTEM + ("\n\n" + rubric if rubric else "")
    run_id = uuid4().hex

    def record_answer(text):
        """The side effect alone. A sandboxed kernel's child raises its own
        stop signal locally, so replaying Done in the parent would throw out of
        run() instead of ending a cell that already ended."""
        answer["text"] = str(text)

    def submit_answer(text):
        record_answer(text)
        raise Done()

    landmark = {"text": None}
    pending_states = []

    def headline(task=None, verified=None, next_action=None, status=None):
        if (
            not isinstance(task, str)
            or not task.strip()
            or not isinstance(next_action, str)
            or not next_action.strip()
            or not isinstance(verified, list)
            or any(not isinstance(item, str) or not item.strip() for item in verified)
            or not isinstance(status, str)
            or status not in ("working", "blocked", "done")
        ):
            raise ValueError("headline needs task, verified, next_action, and status")
        state = {
            "task": task.strip(),
            "verified": [item.strip() for item in verified],
            "next_action": next_action.strip(),
            "status": status,
        }
        content = json.dumps(state, sort_keys=True, separators=(",", ":"))
        recover = f"ms.expand(9223372036854775807, session_id={run_id!r})"
        if est("--- current state ---\n" + content
            + "\n\n--- latest observation ---\n-> " + recover
        ) > budget:
            raise ValueError("headline state exceeds the working-view budget")
        pending_states.append(content)
        landmark["text"] = format_headline(task, verified, next_action, status)

    # Section 2.2: the Event Log is read-only from the kernel.
    # NARU_KERNEL=sandbox runs cells in a child process, which survives a
    # runaway loop or a crash. It needs the log by path, so it is opt-in rather
    # than default: bench.py ingests into an in-memory database that no child
    # can open. See ADR 0007.
    path = getattr(ms, "path", None)
    if os.environ.get("NARU_KERNEL") == "sandbox" and path and path != ":memory:":
        kernel = SandboxedKernel(
            db=path,
            callbacks={"submit_answer": record_answer, "headline": headline},
        )
    else:
        kernel = Kernel(
            ms=ms.readonly() if hasattr(ms, "readonly") else ms,
            submit_answer=submit_answer,
            headline=headline,
        )
    source_index = [] if index is None else index
    trace_index = []
    current_state = None
    latest_observation = None
    turns_used = 0
    peak = 0
    context_stats = {} if context_stats is None else context_stats
    context_stats.setdefault("context_policy", context_mode)
    context_stats.setdefault("context_session_id", run_id)
    last_jev_checkpoint = object()

    header = (
        f"Question (asked {question_date}): {question}"
        if question_date
        else f"Question: {question}"
    )

    def record_trace(spans):
        if not trace_index:
            trace_index.append([])
        trace_index[0].append(
            {
                "lo": min(seq for seq, _ in spans),
                "hi": max(seq for seq, _ in spans),
                "headline": "; ".join(headline for _, headline in spans[:2])[:90],
                "session_id": run_id,
            }
        )
        rollup(trace_index, 4)

    try:
        for turn in range(max_turns):
            turns_used = turn + 1
            parts = [header]
            selection, last_jev_checkpoint = _select_agent_context(
                source_index,
                trace_index,
                current_state,
                latest_observation,
                run_id,
                turns_used,
                question,
                budget,
                context_mode,
                context_evaluator,
                context_stats,
                backend,
                last_jev_checkpoint,
            )
            # The resident digest is a stable kernel summary, not part of the
            # mutable working-view budget. It stays visible while the policy
            # bounds indexes, state, and the latest observation below it.
            parts.append(kernel.digest())
            if selection.text:
                parts.append(selection.text)
            peak = max(peak, selection.selected_tokens)
            if turn == max_turns - 1:
                parts.append("LAST TURN. You must call submit_answer(...) now.")
            prompt = "\n\n".join(parts)

            # The retry nudge lives here, not in backend: a code-block reminder
            # is right for a turn that must emit Python and wrong for a judge
            # that must emit one word.
            reply = backend(
                prompt, system=system, nudge="(Reply with one ```python code block.)"
            )
            reply_seq = ms.append(
                "agent",
                reply,
                kind="agent_reply",
                session_id=run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                payload=reply,
            )
            code = extract_code(reply)
            if trace is not None:
                trace.append(
                    {
                        "turn": turns_used,
                        "prompt_tokens": est(f"{system}\n\n{prompt}"),
                        "context": selection.as_dict(),
                        "code": code,
                        "reply_if_no_code": None if code else reply,
                    }
                )
            if code is None:
                if turn == 0 and turn < max_turns - 1:
                    correction_seq = ms.append(
                        "tool",
                        NO_CELL,
                        kind="agent_observation",
                        session_id=run_id,
                        created_at=datetime.now(timezone.utc).isoformat(),
                        payload=NO_CELL,
                    )
                    latest_observation = (correction_seq, NO_CELL)
                    if trace is not None:
                        trace[-1].update(
                            obs=NO_CELL, reply_seq=reply_seq,
                            obs_seq=correction_seq, state_seq=None,
                        )
                    record_trace([(reply_seq, "no code"), (correction_seq, "no code")])
                    continue
                if reply.strip():
                    answer["text"] = reply.strip()
                break

            spans = [(reply_seq, f"exec {reply_seq}")]
            context_stats["context_tool_calls"] = (
                context_stats.get("context_tool_calls", 0) + 1
            )

            landmark["text"] = None
            pending_states.clear()
            out, err = kernel.run(code)
            if landmark["text"]:
                spans[0] = (reply_seq, f"[{reply_seq}] {landmark['text']}")
            if answer["text"] is not None:
                obs = out.strip() or "(submitted)"
            else:
                obs = out.strip() or "(nothing printed)"
            if err and answer["text"] is None:
                obs = f"{obs}\nERROR: {err}" if out.strip() else f"ERROR: {err}"
            obs_seq = ms.append(
                "tool",
                obs,
                kind="agent_observation",
                session_id=run_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                payload=obs,
            )
            stored = ms.sql_query(
                "SELECT content FROM conversation_history WHERE seq=?", (obs_seq,)
            )[0].content
            latest_observation = (obs_seq, stored)
            context_stats["context_tool_result_tokens"] = (
                context_stats.get("context_tool_result_tokens", 0) + est(stored)
            )
            spans.append((obs_seq, f"obs {obs_seq}: {stored[:50]}"))
            state_seq = None
            for content in pending_states:
                state_seq = ms.append(
                    "agent",
                    content,
                    kind="agent_state",
                    session_id=run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_run_id=run_id,
                    source_seq_lo=reply_seq,
                    source_seq_hi=obs_seq,
                )
                current_state = content
                spans.append((state_seq, "state"))
            if trace is not None:
                trace[-1]["obs"] = obs
                trace[-1]["state_seq"] = state_seq
                trace[-1]["reply_seq"] = reply_seq
                trace[-1]["obs_seq"] = obs_seq
            record_trace(spans)
            if answer["text"] is not None:
                break

            if verbose:
                print(
                    f"  turn {turn + 1}: {est(code)}t code -> {est(obs)}t obs"
                    f"{' ERR' if err else ''}"
                )

        return answer["text"], turns_used, peak
    finally:
        # A sandboxed kernel owns a child process and its SQLite
        # connection. Without this a 96-question run finishes holding
        # 96 idle children; the in-process Kernel's close() is a no-op
        # so the call site needs no branch.
        kernel.close()


def run_naru(
    ms,
    question,
    backend,
    question_date=None,
    max_turns=8,
    budget=6000,
    verbose=False,
    trace=None,
    rubric=None,
    index=None,
    context_mode=None,
    context_evaluator=None,
    context_stats=None,
):
    """Answer one question over an already-ingested Event Log.

    Pass `trace=[]` to collect each turn's cell, printed observation and prompt
    size — the only way to see where turns are actually spent.
    """
    mode = context_mode or os.environ.get("NARU_CONTEXT_POLICY") or "deterministic"
    evaluator = context_evaluator
    stats = context_stats if context_stats is not None else {}
    owned_evaluator = False
    if evaluator is None and mode in ("shadow", "jev"):
        command = os.environ.get("NARU_JEV")
        if command:
            try:
                from jev import CommandJev

                evaluator = CommandJev(command)
                owned_evaluator = True
            except (ValueError, FileNotFoundError):
                stats["context_jev_unavailable"] = 1
                stats["context_fallbacks"] = stats.get("context_fallbacks", 0) + 1
        else:
            stats["context_jev_unavailable"] = 1
    try:
        try:
            return _run(
                ms,
                question,
                question_date,
                backend,
                max_turns,
                budget,
                verbose,
                trace,
                rubric,
                index,
                mode,
                evaluator,
                stats,
            )
        except Done:
            return None, max_turns, 0
    finally:
        if owned_evaluator and evaluator is not None:
            evaluator.close()


def demo():
    """Offline check with a scripted fake backend — no API calls, no key."""
    from ms import MemorySurface

    # regression: unfenced cells must execute, prose must not
    assert extract_code("```python\nprint(1)\n```").strip() == "print(1)"
    assert extract_code('submit_answer("hi")').strip() == 'submit_answer("hi")'
    assert extract_code('hits = ms.search("x")').strip() == 'hits = ms.search("x")'
    assert extract_code("The answer is 38 subjects.") is None
    assert extract_code("") is None

    ms = MemorySurface(":memory:")
    ms.append(
        "user",
        "I drive a blue Subaru Outback",
        kind="context_msg",
        created_at="2023-01-05T10:00:00",
    )
    ms.append(
        "assistant", "Nice car!", kind="model_turn", created_at="2023-01-05T10:01:00"
    )
    for i in range(40):  # bulk filler so eviction has to fire
        ms.append(
            "user",
            f"unrelated chatter number {i} " + "z" * 300,
            kind="context_msg",
            created_at="2023-02-01T10:00:00",
        )

    script = [
        "```python\nhits = ms.search('Subaru')\nprint(hits)\n```",
        "```python\nprint(ms.expand(1)[0].content)\n```",
        "```python\nbulk = ms.search('chatter', k=40)\nprint(len(bulk))\n```",
        "```python\nsubmit_answer('A blue Subaru Outback')\n```",
    ]
    calls = {"n": 0, "prompts": []}

    def fake(prompt, system=None, nudge=None):
        calls["prompts"].append(prompt)
        r = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return r

    ans, turns, peak = run_naru(
        ms,
        "What car do I drive?",
        fake,
        question_date="2023-03-01",
        max_turns=6,
        budget=400,
    )

    assert ans == "A blue Subaru Outback", ans
    assert turns == 4, turns

    # the model saw the question, and the view stayed bounded
    assert "What car do I drive?" in calls["prompts"][0]
    last = calls["prompts"][-1]

    # THE core claim: a 40-row search result never entered the prompt. It is
    # resident in the kernel; only its length was printed. The view therefore
    # stays so small that eviction never even needs to fire here.
    assert "bulk: list[40]" in last, "kernel should hold the bulk result"
    assert "unrelated chatter number 39" not in last, "bulk result leaked into context"
    assert est(last) < 400, f"working view too big: {est(last)}t"

    jev_surface = MemorySurface(":memory:")
    jev_prompt = []
    jev_seen = {}
    jev_stats = {}

    def fake_jev(state, questions):
        jev_seen.update(state=state, questions=questions)
        return {
            "answers": {
                key: {"type": "choice", "choice": "omit", "confidence": 0.95}
                for key in questions
            },
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    def jev_backend(prompt, system=None, nudge=None):
        jev_prompt.append(prompt)
        return "```python\nsubmit_answer('jev done')\n```"

    jev_answer, _, _ = run_naru(
        jev_surface,
        "Jev context test",
        jev_backend,
        max_turns=1,
        budget=20,
        index=[[{"lo": 1, "hi": 2, "headline": "secret source body", "session_id": "s"}]],
        context_mode="jev",
        context_evaluator=fake_jev,
        context_stats=jev_stats,
    )
    assert jev_answer == "jev done"
    assert jev_stats["context_jev_calls"] == 1
    assert "secret source body" not in jev_prompt[0]
    assert "content" not in json.dumps(jev_seen["state"])
    jev_surface.close()

    refusal = [
        "I don't have a record of that conversation and no search tool here.",
        "I still do not know.",
    ]
    r = {"i": 0, "prompts": []}

    def fake3(prompt, system=None, nudge=None):
        r["prompts"].append(prompt)
        out = refusal[min(r["i"], len(refusal) - 1)]
        r["i"] += 1
        return out

    refusal_trace = []
    ans3, turns3, _ = run_naru(
        ms, "What car do I drive?", fake3, max_turns=6, trace=refusal_trace
    )
    assert ans3 == "I still do not know.", ans3
    assert turns3 == 2, turns3
    assert "no code block" in r["prompts"][1], r["prompts"][1][:300]
    assert refusal_trace[0]["obs"] == NO_CELL and refusal_trace[0]["reply_seq"]
    assert refusal_trace[0]["obs_seq"] and refusal_trace[0]["state_seq"] is None
    assert refusal_trace[0]["prompt_tokens"] == est(
        f"{SYSTEM}\n\n{r['prompts'][0]}"
    )
    assert ms.sql_query(
        "SELECT content FROM conversation_history WHERE kind='agent_observation' "
        "AND content=?",
        (NO_CELL,),
    )[0].content == NO_CELL

    after = ["```python\nprint(ms.search('Subaru')[0].content)\n```",
             "You drive a blue Subaru Outback."]
    a = {"i": 0}

    def fake4(prompt, system=None, nudge=None):
        out = after[min(a["i"], len(after) - 1)]
        a["i"] += 1
        return out

    ans4, _, _ = run_naru(ms, "What car do I drive?", fake4, max_turns=6)
    assert ans4 == "You drive a blue Subaru Outback.", ans4

    # Eviction still fires when printed output genuinely overflows the budget.
    noisy = ["```python\nheadline(task='probe %d', verified=['found nothing yet'],"
             " next_action='widen search', status='working')\n"
             "print('L%d ' + 'q'*2000)\n```" % (i, i) for i in range(6)]
    noisy.append("```python\nsubmit_answer('done')\n```")
    n = {"i": 0, "prompts": []}

    def fake2(prompt, system=None, nudge=None):
        n["prompts"].append(prompt)
        r = noisy[min(n["i"], len(noisy) - 1)]
        n["i"] += 1
        return r

    ans2, _, _ = run_naru(ms, "noisy?", fake2, max_turns=8, budget=500)
    assert ans2 == "done", ans2
    over = n["prompts"][-1]
    assert "evicted" in over and "ms.expand" in over, over[:300]
    # the paper's landmark shape, authored by the model, must reach the index
    assert "task=probe" in over, f"model headline missing from index: {over[:400]}"
    assert "status=working" in over, over[:400]
    assert est(over) < 1500, f"eviction failed to bound view: {est(over)}t"

    states = MemorySurface(":memory:")
    state_script = [
        "```python\nheadline(task='first', verified=[], next_action='search', status='working')\n"
        "headline(task='second', verified=['blue Subaru'], next_action='answer', status='working')\n"
        "print('LATEST-OBSERVATION')\n```",
        "```python\nsubmit_answer('done')\n```",
    ]
    sp = {"i": 0, "prompts": []}

    def state_backend(prompt, system=None, nudge=None):
        sp["prompts"].append(prompt)
        reply = state_script[sp["i"]]
        sp["i"] += 1
        return reply

    assert run_naru(states, "state test", state_backend, max_turns=3, budget=200)[0] == "done"
    saved = states.sql_query(
        "SELECT * FROM conversation_history WHERE kind='agent_state' ORDER BY seq"
    )
    assert len(saved) == 2, saved
    assert json.loads(saved[-1].content) == {
        "task": "second", "verified": ["blue Subaru"], "next_action": "answer", "status": "working"
    }
    assert all(row.source_run_id == row.session_id for row in saved)
    assert [row.seq for row in states.expand(
        saved[-1].source_seq_lo, saved[-1].source_seq_hi, session_id=saved[-1].source_run_id
    )] == [1, 2]
    assert saved[-1].content in sp["prompts"][1], sp["prompts"][1]
    assert "LATEST-OBSERVATION" in sp["prompts"][1], sp["prompts"][1]
    assert '"task":"first"' not in sp["prompts"][1], sp["prompts"][1]

    invalid = MemorySurface(":memory:")
    bad = ["```python\nheadline(task='bad', verified='fact', next_action='x', status='working')\n```"]
    assert run_naru(invalid, "bad state", lambda *args, **kwargs: bad[0], max_turns=1)[0] is None
    assert invalid.sql_query("SELECT * FROM conversation_history WHERE kind='agent_state'") == []
    assert "ERROR: ValueError" in invalid.sql_query(
        "SELECT content FROM conversation_history WHERE kind='agent_observation'"
    )[0].content

    near = MemorySurface(":memory:")
    near_script = [
        "```python\nheadline(task='x', verified=[], next_action='continue', status='working')\n"
        "print('z' * 5000)\n```",
        "```python\nsubmit_answer('done')\n```",
    ]
    np = {"i": 0, "prompts": []}

    def near_backend(prompt, system=None, nudge=None):
        np["prompts"].append(prompt)
        reply = near_script[np["i"]]
        np["i"] += 1
        return reply

    near_answer, _, near_peak = run_naru(
        near, "near budget", near_backend, max_turns=3, budget=50
    )
    near_view = "--- current state ---" + np["prompts"][1].split(
        "--- current state ---", 1
    )[1]
    assert near_answer == "done" and near_peak <= 50, near_peak
    assert est(near_view) <= 50 and near_view.rstrip().endswith("')"), near_view

    recovery = MemorySurface(":memory:")
    recovery.append(
        "user",
        "OLD-HISTORY-MUST-NOT-APPEAR",
        kind="context_msg",
        created_at="2023-01-01T00:00:00",
    )
    append = recovery.append
    interleaved = {"done": False}

    def append_trace(*args, **kwargs):
        if kwargs.get("kind") == "agent_observation" and not interleaved["done"]:
            append(
                "tool",
                "UNRELATED-BETWEEN-TRACE-ROWS",
                kind="tool_result",
                created_at="2023-01-01T00:00:00",
            )
            interleaved["done"] = True
        return append(*args, **kwargs)

    recovery.append = append_trace
    recovery_script = [
        "```python\nheadline(task='recover trace', verified=[], next_action='finish', status='working')\n"
        "print('RECOVERY-OBS ' + 'x' * 5000)\n```",
        "```python\nprint('FINAL-OBS')\nsubmit_answer('trace done')\n```",
    ]
    rp = {"i": 0, "prompts": []}

    def recovery_backend(prompt, system=None, nudge=None):
        rp["prompts"].append(prompt)
        reply = recovery_script[min(rp["i"], len(recovery_script) - 1)]
        rp["i"] += 1
        return reply

    try:
        recovery_answer, _, _ = run_naru(
            recovery, "recover trace", recovery_backend, max_turns=3, budget=100,
        )
    finally:
        recovery.append = append
    assert recovery_answer == "trace done", recovery_answer
    handle = re.search(
        r"ms\.expand\((\d+), (\d+), session_id='([0-9a-f]+)'\)", rp["prompts"][1]
    )
    assert handle, rp["prompts"][1]
    lo, hi, session_id = int(handle[1]), int(handle[2]), handle[3]
    recovered = recovery.expand(lo, hi, session_id=session_id)
    assert (lo, hi) == (2, 5), (lo, hi)
    assert [row.seq for row in recovered] == [2, 4, 5], recovered
    assert any("headline(task='recover trace'" in row.content for row in recovered)
    assert any("RECOVERY-OBS" in row.content for row in recovered)
    assert all("UNRELATED-BETWEEN-TRACE-ROWS" not in row.content for row in recovered)
    assert max(len(row.content) for row in recovered) > 4000, "payload was not recovered"
    assert rp["prompts"][1].rstrip().endswith(
        f"ms.expand(4, session_id={session_id!r})"
    ), "latest observation did not end in its scoped recovery call"
    assert recovery.expand(6, 7, session_id=session_id)[1].content == "FINAL-OBS"

    # NARU_KERNEL=sandbox, end to end. Nothing exercised this branch: the
    # demo above builds MemorySurface(":memory:"), so the selection at the top
    # of _run() always fell through to the in-process kernel and the sandbox
    # path shipped with no offline check at all.
    import pathlib
    import tempfile

    sandbox_tmp = tempfile.TemporaryDirectory()
    sandbox_dir = pathlib.Path(sandbox_tmp.name)
    fms = MemorySurface(str(sandbox_dir / "log.db"))
    fms.append(
        "user", "I drive a blue Subaru Outback", kind="context_msg",
        session_id="s1", created_at="2023-01-01T00:00",
    )
    sandbox_script = [
        # The pid is the only assertion that can tell WHERE the cell ran.
        # Asserting on the answer alone passes when the branch falls through
        # to the in-process kernel, which returns exactly the same string.
        "```python\nimport os\nprint('PID', os.getpid())\n```",
        "```python\nprint(ms.search('Subaru')[0]['content'])\n```",
        (
            "```python\nheadline(task='probe', verified=[], next_action='answer', status='working')\n"
            "submit_answer('A blue Subaru Outback')\n```"
        ),
    ]
    sn = {"n": 0, "prompts": []}

    def sandbox_backend(prompt, system=None, nudge=None):
        sn["prompts"].append(prompt)
        r = sandbox_script[min(sn["n"], len(sandbox_script) - 1)]
        sn["n"] += 1
        return r

    prev = os.environ.get("NARU_KERNEL")
    os.environ["NARU_KERNEL"] = "sandbox"
    closed = []
    close = SandboxedKernel.close

    def tracked_close(kernel):
        closed.append(kernel._proc)
        return close(kernel)

    SandboxedKernel.close = tracked_close
    try:
        sans, _sturns, _ = run_naru(
            fms, "What car do I drive?", sandbox_backend, max_turns=5, budget=800
        )
    finally:
        SandboxedKernel.close = close
        if prev is None:
            os.environ.pop("NARU_KERNEL", None)
        else:
            os.environ["NARU_KERNEL"] = prev
    assert sans == "A blue Subaru Outback", sans
    seen = [p for p in sn["prompts"] if "PID " in p]
    assert seen, "the pid cell never came back"
    assert f"PID {os.getpid()}" not in seen[-1], (
        "the cell ran in THIS process — the sandbox branch was not taken"
    )
    assert closed and closed[0].poll() is not None, "run_naru did not close its kernel"
    for surface in (ms, states, invalid, near, recovery, fms):
        surface.close()

    print(
        f"ok — agent checks passed: 4 turns, final view {est(last)}t, "
        f"40-row result stayed in kernel; eviction bounds a noisy run to "
        f"{est(over)}t with recovery pointers"
    )


if __name__ == "__main__":
    demo()
