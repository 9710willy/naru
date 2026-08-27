#!/usr/bin/env python3
"""PostToolUse hook: spill oversized tool output into the Naru Event Log.

Claude Code re-sends the whole conversation every turn, so one large tool
result is paid for on every later turn and occupies the context window for the
rest of the session. This hook intercepts the result at the harness boundary,
stores it verbatim in a searchable Event Log, and replaces it in context with a
bounded preview plus a retrieval handle.

Nothing is lost: the full text stays addressable by `seq`.

Enable per-project in .claude/settings.json:

    {"hooks": {"PostToolUse": [{"matcher": "Bash|Read",
      "hooks": [{"type": "command",
                 "command": "python3 /path/to/naru/hook_spill.py"}]}]}}

The replacement MUST match the tool's own output schema or Claude Code discards
it and keeps the original, so we mutate the text field in place inside the
response object rather than returning a bare string.
"""

import json
import os
import pathlib
import sys
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import metrics
from ms import DEFAULT_DB

THRESHOLD = int(os.environ.get("NARU_SPILL_THRESHOLD", "2000"))  # chars
KEEP = int(os.environ.get("NARU_SPILL_PREVIEW", "600"))
SIGNPOST_EVERY = 40  # one signpost line per N lines of spilled text
DB = pathlib.Path(os.environ.get("NARU_SPILL_DB", DEFAULT_DB))

# Text-bearing fields, in the order tools tend to use them.
TEXT_FIELDS = ("stdout", "content", "output", "text", "result", "stderr")


def find_text(resp):
    """Return (container, key, text) for the largest text field, or None."""
    if isinstance(resp, str):
        return (None, None, resp)
    if not isinstance(resp, dict):
        return None
    best = None
    for k in TEXT_FIELDS:
        v = resp.get(k)
        if isinstance(v, str) and (best is None or len(v) > len(best[2])):
            best = (resp, k, v)
    return best


def outline(text):
    """Line-numbered signposts so the model can see what it can no longer read
    and jump to the part it needs."""
    lines = text.splitlines()
    marks = [
        f"  {i + 1:>6}  {lines[i].strip()[:70]}"
        for i in range(0, len(lines), SIGNPOST_EVERY)
        if lines[i].strip()
    ]
    return "\n".join(marks[:12])


def main():
    try:
        raw = sys.stdin.read()
        event = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return 0  # not our business; pass through untouched

    # NARU_SPILL_DEBUG=<path> records the real event shape. Claude Code's
    # per-tool response schemas are not documented, and a replacement that does
    # not match the tool's schema is discarded silently, so the shape has to be
    # observed rather than assumed.
    dbg = os.environ.get("NARU_SPILL_DEBUG")
    if dbg:
        try:
            with open(dbg, "a") as fh:
                fh.write(raw[:20000] + "\n---\n")
        except OSError:
            pass

    resp = event.get("tool_response")
    found = find_text(resp)
    if not found:
        return 0
    container, key, text = found
    if len(text) <= THRESHOLD:
        # Recorded too: without the skips there is no way to tell later whether
        # the threshold is leaving savings on the table.
        metrics.record("hook", tool=event.get("tool_name"), chars=len(text),
                       spilled=False)
        return 0  # small enough to keep inline

    # Store verbatim, addressable by seq.
    try:
        from ms import MemorySurface

        DB.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        ms = MemorySurface(str(DB))
        seq = ms.append(
            "tool",
            text,
            kind="tool_result",
            session_id=str(event.get("session_id", "?"))[:16]
            + " "
            + str(event.get("tool_name", "tool")),
            created_at=datetime.now().isoformat(timespec="seconds"),
            # No payload=: the content column already holds the full text.
            # Passing it would store the same bytes twice and leave the copy in
            # a temp directory macOS may purge, breaking recovery.
        )
    except Exception as e:  # never break the user's tool call over a spill
        metrics.record("error", tool=event.get("tool_name"), chars=len(text),
                       msg=f"{type(e).__name__}: {e}")
        print(f"naru spill failed, output left inline: {e}", file=sys.stderr)
        return 0

    lines = len(text.splitlines())
    replacement = (
        f"{text[:KEEP]}\n"
        f"\n[naru: {len(text):,} chars / {lines:,} lines spilled to the Event "
        f"Log as seq {seq}. Showing the first {KEEP} chars.]\n"
        f"[signposts]\n{outline(text)}\n"
        f"[recover: `naru show {seq}` for the full text, or "
        f'`naru search "<terms>"` to find a part of it]'
    )

    if container is None:
        updated = replacement
    else:
        updated = dict(container)
        updated[key] = replacement

    metrics.record("hook", tool=event.get("tool_name"), chars=len(text),
                   kept=len(replacement), spilled=True, seq=seq)

    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "updatedToolOutput": updated,
            }
        },
        sys.stdout,
    )
    return 0


def demo():
    """Self-check: exercises the hook contract without Claude Code."""
    import subprocess
    import tempfile

    global DB
    DB = pathlib.Path(tempfile.mkdtemp()) / "s.db"
    me = [sys.executable, str(pathlib.Path(__file__).resolve())]
    # Isolate metrics too: a self-check must not write into the user's
    # real observability store.
    env = {**os.environ, "NARU_SPILL_DB": str(DB),
           "NARU_METRICS": str(DB.parent / "m.jsonl")}

    def run(payload):
        p = subprocess.run(
            me, input=json.dumps(payload), capture_output=True, text=True, env=env
        )
        assert p.returncode == 0, p.stderr
        return json.loads(p.stdout) if p.stdout.strip() else None

    # small output passes through untouched (no stdout at all)
    assert run({"tool_name": "Bash", "tool_response": {"stdout": "hi"}}) is None

    # large output is replaced, and the response SHAPE is preserved
    big = "\n".join(f"line {i} " + "x" * 60 for i in range(400))
    out = run(
        {
            "tool_name": "Bash",
            "session_id": "abc",
            "tool_response": {"stdout": big, "stderr": "", "interrupted": False},
        }
    )
    upd = out["hookSpecificOutput"]["updatedToolOutput"]
    assert out["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert set(upd) == {"stdout", "stderr", "interrupted"}, upd.keys()
    assert upd["interrupted"] is False, "non-text fields must survive"
    assert len(upd["stdout"]) < len(big) / 4, "replacement should be much smaller"
    assert "seq 1" in upd["stdout"] and "naru show 1" in upd["stdout"]
    assert "line 0" in upd["stdout"], "preview should show the head"
    assert "line 200" in upd["stdout"], "signposts should reach the middle"

    # the full text really is recoverable, verbatim
    from ms import MemorySurface

    ms = MemorySurface(str(DB))
    assert ms.expand(1)[0].content == big, "spilled text not recovered verbatim"
    assert ms.search("line", k=3), "spilled text not searchable"

    # a bare-string response is handled too
    out2 = run({"tool_name": "X", "tool_response": big})
    assert isinstance(out2["hookSpecificOutput"]["updatedToolOutput"], str)

    # malformed stdin must never break the tool call
    p = subprocess.run(me, input="not json", capture_output=True, text=True, env=env)
    assert p.returncode == 0 and not p.stdout.strip()

    saved = len(big) - len(upd["stdout"])
    print(
        f"ok — hook checks passed ({len(big):,} chars -> "
        f"{len(upd['stdout']):,}, {saved / len(big) * 100:.0f}% removed from "
        f"context, full text recoverable at seq 1)"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        sys.exit(main())
