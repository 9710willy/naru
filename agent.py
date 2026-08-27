"""The Naru agent loop.

History lives in the Session Environment (Event Log + kernel), never in the
prompt. Each turn the model writes a Python cell; the cell searches and
computes over the log; only what it prints enters the next working view. When
the view exceeds budget, Algorithm 1 evicts spans to a tiered index that keeps
them addressable.
"""

import re

from eviction import Block, est, evict, format_headline, render_index
from kernel import Kernel

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

The `claude` CLI that carries this request may expose its own identity, project
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
  ms.expand(lo, hi=None)             -> exact verbatim turns for a seq span.
        Only needed to read neighbours of a hit; a hit is already complete.
  ms.sql_query("SELECT seq, role, created_at, content FROM conversation_history
                WHERE ... ORDER BY seq LIMIT 50")   -> read-only SELECT.
        created_at is ISO-8601 text, so substr(created_at,1,10) sorts and
        compares as a date.
  ms.days_between(d1, d2)            -> whole calendar days between two dates.
  headline(task=, state=, next_action=, status=)
        -> Record a landmark for THIS turn. Call it once per turn, first. It is
        bound to this turn's address, so when the turn is later evicted from
        your view you can still see what it was for and jump back to it by
        address. Keep each field under ~10 words.
  submit_answer("...")               -> finish. Call this exactly once.

Method:
  0. Open every cell with headline(...) describing what this turn is doing and
     what you have confirmed so far. That landmark is what lets you navigate
     back to an evicted turn by position instead of by remembering its wording.
  1. Search with SHORT keyword or synonym queries. Start k=5; raise it only if
     the evidence is thin. Prefer one or two distinctive words.
  2. If several named things are asked about, search each SEPARATELY rather
     than putting every term in one query.
  2a. IF SEARCH RETURNS NOTHING, DO NOT CONCLUDE THE FACT IS ABSENT. The
     question often uses words the conversation never uses ("homegrown" when
     the turns say "basil" and "tomatoes"). Call ms.outline(), pick the
     sessions whose topic or date fits, and ms.expand(lo, hi) to read them.
     Only after browsing the plausible sessions may you say a fact is missing.
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


class Done(Exception):
    pass


def _run(
    ms,
    question,
    question_date,
    backend,
    max_turns,
    budget,
    protect_tail,
    verbose=False,
    trace=None,
    rubric=None,
    index=None,
):
    answer = {"text": None}
    system = SYSTEM + ("\n\n" + rubric if rubric else "")

    def submit_answer(text):
        answer["text"] = str(text)
        raise Done()

    landmark = {"text": None}

    def headline(task=None, state=None, next_action=None, status=None):
        """Model-authored landmark for the current turn (paper section 2.4)."""
        landmark["text"] = format_headline(task, state, next_action, status)

    # Section 2.2: the Event Log is read-only from the kernel.
    kernel = Kernel(
        ms=ms.readonly() if hasattr(ms, "readonly") else ms,
        submit_answer=submit_answer,
        headline=headline,
    )
    # Section 3.3: the eviction index built during ingestion is carried
    # forward; the raw context starts empty.
    view = []
    index = [] if index is None else index
    seq = 0
    turns_used = 0

    header = (
        f"Question (asked {question_date}): {question}"
        if question_date
        else f"Question: {question}"
    )

    for turn in range(max_turns):
        turns_used = turn + 1
        parts = [header]
        idx = render_index(index)
        if idx:
            parts.append(idx)
        parts.append(kernel.digest())
        if view:
            parts.append("--- working view ---")
            parts += [b.text for b in view]
        if turn == max_turns - 1:
            parts.append("LAST TURN. You must call submit_answer(...) now.")
        prompt = "\n\n".join(parts)

        # The retry nudge lives here, not in backend: a code-block reminder
        # is right for a turn that must emit Python and wrong for a judge
        # that must emit one word.
        reply = backend(
            prompt, system=system, nudge="(Reply with one ```python code block.)"
        )
        code = extract_code(reply)
        if trace is not None:
            trace.append(
                {
                    "turn": turns_used,
                    "prompt_tokens": est(prompt),
                    "code": code,
                    "reply_if_no_code": None if code else reply,
                }
            )
        if code is None:
            # Genuine prose, no code. Take it as the final answer.
            if reply.strip():
                answer["text"] = reply.strip()
            break

        seq += 1
        exec_block = Block(seq, "exec", f"[exec {seq}]\n{code.strip()}",
                           headline=f"exec {seq}")
        view.append(exec_block)

        landmark["text"] = None
        out, err = kernel.run(code)
        # Bind the landmark at append time, to the address this turn was given.
        if landmark["text"]:
            exec_block.headline = f"[{seq}] {landmark['text']}"
        if answer["text"] is not None:
            if trace is not None:
                trace[-1]["obs"] = "(submitted)"
            break

        seq += 1
        obs = out.strip() or "(nothing printed)"
        if err:
            obs = f"{obs}\nERROR: {err}" if out.strip() else f"ERROR: {err}"
        if trace is not None:
            trace[-1]["obs"] = obs
        view.append(
            Block(
                seq,
                "obs",
                f"[obs {seq}]\n{obs}",
                headline=f"obs {seq}: {obs[:50]}",
                is_payload=True,
            )
        )

        if verbose:
            print(
                f"  turn {turn + 1}: {est(code)}t code -> {est(obs)}t obs"
                f"{' ERR' if err else ''}"
            )

        view, index = evict(view, index, budget=budget, protect_tail=protect_tail)

    peak = sum(b.tokens() for b in view)
    return answer["text"], turns_used, peak


def run_naru(
    ms,
    question,
    backend,
    question_date=None,
    max_turns=8,
    budget=6000,
    protect_tail=4,
    verbose=False,
    trace=None,
    rubric=None,
    index=None,
):
    """Answer one question over an already-ingested Event Log.

    Pass `trace=[]` to collect each turn's cell, printed observation and prompt
    size — the only way to see where turns are actually spent.
    """
    try:
        return _run(
            ms,
            question,
            question_date,
            backend,
            max_turns,
            budget,
            protect_tail,
            verbose,
            trace,
            rubric,
            index,
        )
    except Done:
        return None, max_turns, 0


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
        protect_tail=2,
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

    # Eviction still fires when printed output genuinely overflows the budget.
    noisy = ["```python\nheadline(task='probe %d', state='found nothing yet',"
             " next_action='widen search', status='working')\n"
             "print('L%d ' + 'q'*2000)\n```" % (i, i) for i in range(6)]
    noisy.append("```python\nsubmit_answer('done')\n```")
    n = {"i": 0, "prompts": []}

    def fake2(prompt, system=None, nudge=None):
        n["prompts"].append(prompt)
        r = noisy[min(n["i"], len(noisy) - 1)]
        n["i"] += 1
        return r

    ans2, _, _ = run_naru(
        ms, "noisy?", fake2, max_turns=8, budget=500, protect_tail=2
    )
    assert ans2 == "done", ans2
    over = n["prompts"][-1]
    assert "evicted" in over and "ms.expand" in over, over[:300]
    # the paper's landmark shape, authored by the model, must reach the index
    assert "task=probe" in over, f"model headline missing from index: {over[:400]}"
    assert "status=working" in over, over[:400]
    assert est(over) < 1500, f"eviction failed to bound view: {est(over)}t"

    print(
        f"ok — agent checks passed: 4 turns, final view {est(last)}t, "
        f"40-row result stayed in kernel; eviction bounds a noisy run to "
        f"{est(over)}t with recovery pointers"
    )


if __name__ == "__main__":
    demo()
