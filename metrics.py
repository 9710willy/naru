#!/usr/bin/env python3
"""Append-only metrics for the live spill hook.

The Event Log already records every spill, so it answers "what did we store".
It structurally cannot answer the two questions worth revisiting:

  - outputs that did NOT spill      -> is the threshold set right?
  - whether recoveries get used     -> does the handle earn its place?

Neither leaves a row behind. So one compact JSONL line per hook invocation and
per recovery, appended, never read on the hot path.

Recording must never break a tool call: every failure here is swallowed.
"""

import json
import os
import pathlib
import sys
from datetime import datetime

PATH = pathlib.Path(
    os.environ.get("NARU_METRICS", pathlib.Path.home() / ".naru" / "metrics.jsonl")
)
MAX_BYTES = int(os.environ.get("NARU_METRICS_MAX", 2_000_000))  # ~20k events


def record(event, **fields):
    """Append one event. Silent on any failure — this is never load-bearing."""
    try:
        PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # No trimming here. The hook runs as one process per tool call and
        # Claude Code fires them in parallel, so a read-modify-write races and
        # drops whatever another process appended in between — NEW events, not
        # old ones. `naru prune` owns the trim; this path only appends.
        line = {"t": datetime.now().isoformat(timespec="seconds"), "e": event}
        line.update(fields)
        with PATH.open("a") as fh:
            fh.write(json.dumps(line, separators=(",", ":")) + "\n")
    except Exception:
        pass


def read(days=None):
    """Load events, optionally only those newer than `days` ago."""
    if not PATH.exists():
        return []
    cutoff = None
    if days is not None:
        from datetime import timedelta

        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    out = []
    for raw in PATH.read_text(errors="replace").splitlines():
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        if cutoff and d.get("t", "") < cutoff:
            continue
        out.append(d)
    return out


def record_show(store_id, run_id, lo, hi):
    """Record one printed span with the store and run that own it."""
    record("show", v=2, store=store_id, run=run_id, lo=lo, hi=hi)


def opened(store_id, run_id, lo, hi, days=None):
    """True when one receipt covers the whole cited span."""
    return any(
        e.get("e") == "show"
        and e.get("v") == 2
        and e.get("store") == store_id
        and e.get("run") == run_id
        and type(e.get("lo")) is int
        and type(e.get("hi")) is int
        and e["lo"] <= lo <= hi <= e["hi"]
        for e in read(days)
    )


def _pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def _reopened_spills(events):
    """Return distinct attributable spills opened by a later show receipt."""
    shows = [
        e
        for e in events
        if e.get("e") == "show"
        and e.get("v") == 2
        and isinstance(e.get("store"), str)
        and type(e.get("lo")) is int
        and type(e.get("hi")) is int
    ]
    spills = [e for e in events if e.get("e") == "hook" and e.get("spilled")]
    attributable = [
        e
        for e in spills
        if isinstance(e.get("store"), str) and type(e.get("seq")) is int
    ]
    reopened = {
        (spill["store"], spill["seq"])
        for spill in attributable
        if any(
            shown["store"] == spill["store"]
            and shown["lo"] <= spill["seq"] <= shown["hi"]
            for shown in shows
        )
    }
    return len(reopened), len(attributable), len(spills) - len(attributable)


def _recovery_line(reopened, attributable, legacy_spills):
    line = f"  spilled rows reopened {reopened:>5,}/{attributable:,} attributable"
    if legacy_spills:
        return line + f"  ({legacy_spills:,} legacy spill(s) cannot be linked)"
    if attributable and not reopened:
        return line + "   <- no recorded spill was reopened"
    return line


def _grouped_lines(calls, deliveries):
    by_harness = {}
    for event in calls:
        harness = event.get("harness", "legacy")
        by_harness.setdefault(harness, [0, 0])
        by_harness[harness][0] += 1
        by_harness[harness][1] += bool(event.get("spilled"))
    lines = []
    if by_harness:
        lines.append(
            "  spill hooks by harness: "
            + "  ".join(
                f"{h}={s}/{n}" for h, (n, s) in sorted(by_harness.items())
            )
        )
    if deliveries:
        counts = {}
        for event in deliveries:
            harness = event.get("harness", "legacy")
            counts[harness] = counts.get(harness, 0) + 1
        lines.append(
            "  context deliveries: "
            + "  ".join(f"{h}={n}" for h, n in sorted(counts.items()))
        )
    return lines


def report(days=None, threshold=None):
    """Human summary. Returns the lines so callers can test it."""
    ev = read(days)
    if not ev:
        return [f"no metrics yet ({PATH})"]

    calls = [e for e in ev if e["e"] == "hook"]
    spills = [e for e in calls if e.get("spilled")]
    skips = [e for e in calls if not e.get("spilled")]
    searches = [e for e in ev if e["e"] == "search"]
    shows = [e for e in ev if e["e"] == "show"]
    deliveries = [e for e in ev if e["e"] == "context_delivery"]
    fails = [e for e in ev if e["e"] == "error"]
    reopened, attributable, legacy_spills = _reopened_spills(ev)

    saved = sum(e.get("chars", 0) - e.get("kept", 0) for e in spills)
    L = []
    win = f"last {days}d" if days else "all time"
    L.append(f"naru observability ({win})   {PATH}")
    L.append("")
    L.append(f"  hook invocations   {len(calls):>7,}")
    L.append(
        f"    spilled          {len(spills):>7,}"
        f"  ({100 * len(spills) / max(len(calls), 1):.0f}% of calls)"
    )
    L.append(f"    under threshold  {len(skips):>7,}")
    L.append(f"  tokens kept out of context ~{saved // 4:>10,}")
    if spills:
        sizes = [e.get("chars", 0) for e in spills]
        L.append(
            f"    spilled size  p50 {_pct(sizes, 50):>8,}  "
            f"p90 {_pct(sizes, 90):>8,}  max {max(sizes):>8,} chars"
        )
    if skips:
        sizes = [e.get("chars", 0) for e in skips]
        L.append(
            f"    skipped size  p50 {_pct(sizes, 50):>8,}  "
            f"p90 {_pct(sizes, 90):>8,}  max {max(sizes):>8,} chars"
        )
        # Threshold advice: if many skipped outputs sit just under the line,
        # the threshold is leaving real savings on the table.
        if threshold:
            near = [s for s in sizes if s > threshold * 0.6]
            if near:
                L.append(
                    f"    {len(near)} skipped output(s) above "
                    f"{int(threshold * 0.6):,} chars — consider lowering "
                    f"NARU_SPILL_THRESHOLD"
                )
    L.append(f"  searches           {len(searches):>7,}")
    L.append(f"  show commands       {len(shows):>7,}")
    L.append(_recovery_line(reopened, attributable, legacy_spills))
    if fails:
        L.append(f"  ERRORS             {len(fails):>7,}")
        for e in fails[-3:]:
            L.append(f"    {e.get('t', '')} {str(e.get('msg'))[:80]}")

    by_tool = {}
    for e in calls:
        by_tool.setdefault(e.get("tool", "?"), [0, 0])
        by_tool[e.get("tool", "?")][0] += 1
        by_tool[e.get("tool", "?")][1] += 1 if e.get("spilled") else 0
    if by_tool:
        L.append("")
        L.append(
            "  by tool: "
            + "  ".join(f"{t}={s}/{n}" for t, (n, s) in sorted(by_tool.items()))
        )
    L.extend(_grouped_lines(calls, deliveries))
    return L


def demo():
    import tempfile

    global PATH
    tmp = tempfile.TemporaryDirectory()
    PATH = pathlib.Path(tmp.name) / "m.jsonl"

    assert report() == [f"no metrics yet ({PATH})"]

    record("hook", harness="claude-code", tool="Bash", chars=500, spilled=False)
    record(
        "hook", harness="claude-code", tool="Bash", chars=30000, kept=1300,
        spilled=True, store="store-a", run="run-a", seq=1,
    )
    record("hook", harness="claude-code", tool="Bash", chars=7000, spilled=False)
    record("hook", tool="X", chars=20000, kept=1000, spilled=True, seq=50)
    record_show("store-a", "run-a", 1, 3)
    record_show("store-a", "run-a", 10, 10)
    record_show("store-a", "run-a", 12, 12)
    record("show", store="store-a", run="run-a", lo=20, hi=22)
    record("search", q="needle", hits=1)
    record("context_delivery", harness="codex", hook="SessionStart", doc_seq=4)
    record("error", msg="disk full")

    ev = read()
    assert len(ev) == 11, ev
    assert opened("store-a", "run-a", 1, 3)
    assert not opened("store-b", "run-a", 1, 3)
    assert not opened("store-a", "run-b", 1, 3)
    assert not opened("store-a", "run-a", 10, 12), "split receipts covered a span"
    assert not opened("store-a", "run-a", 20, 22), "legacy receipt authorized a span"
    out = "\n".join(report(threshold=10000))
    assert "hook invocations         4" in out, out
    assert "spilled                2  (50% of calls)" in out, out
    assert "tokens kept out of context" in out
    assert "11,925" in out, out
    assert "searches                 1" in out, out
    assert "show commands" in out and "             4" in out, out
    assert "spilled rows reopened     1/1 attributable" in out, out
    assert "1 legacy spill(s) cannot be linked" in out, out
    assert "context deliveries: codex=1" in out, out
    assert "ERRORS" in out and "disk full" in out
    # a 7,000-char skip is >60% of a 10,000 threshold -> should advise lowering
    assert "consider lowering" in out, out
    assert "Bash=1/3" in out and "X=1/1" in out, out
    assert "claude-code=1/3" in out and "legacy=1/1" in out, out

    # never raises, even with an unwritable path
    PATH = pathlib.Path("/nonexistent-dir-xyz/m.jsonl")
    record("hook", tool="Bash", chars=1)  # must be silent
    assert read() == []

    print("ok — metrics checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        print("\n".join(report()))
