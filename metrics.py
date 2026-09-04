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


def opened(days=None):
    """Closed seq intervals a reader actually opened.

    `naru show LO HI` prints every row in the range, so one event covers the
    whole closed interval. Only `show` counts. `search` prints a 300-char
    preview, and a preview is not an open — that distinction is the whole
    point of the citation lock (ADR 0010).
    """
    spans = []
    for e in read(days):
        if e.get("e") != "show" or type(e.get("seq")) is not int:
            continue
        lo = e["seq"]
        hi = e.get("hi")
        spans.append((lo, hi if type(hi) is int and hi >= lo else lo))
    return spans


def _pct(vals, p):
    if not vals:
        return 0
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


def report(days=None, threshold=None):
    """Human summary. Returns the lines so callers can test it."""
    ev = read(days)
    if not ev:
        return [f"no metrics yet ({PATH})"]

    calls = [e for e in ev if e["e"] == "hook"]
    spills = [e for e in calls if e.get("spilled")]
    skips = [e for e in calls if not e.get("spilled")]
    recalls = [e for e in ev if e["e"] in ("show", "search")]
    fails = [e for e in ev if e["e"] == "error"]

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
    L.append(
        f"  recoveries used    {len(recalls):>7,}"
        + (
            "   <- nothing has been recalled; the handle may be dead weight"
            if spills and not recalls
            else ""
        )
    )
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
    return L


def demo():
    import tempfile

    global PATH
    PATH = pathlib.Path(tempfile.mkdtemp()) / "m.jsonl"

    assert report() == [f"no metrics yet ({PATH})"]

    record("hook", tool="Bash", chars=500, spilled=False)
    record("hook", tool="Bash", chars=30000, kept=1300, spilled=True, seq=1)
    record("hook", tool="Bash", chars=7000, spilled=False)
    record("show", seq=1)
    record("error", msg="disk full")

    ev = read()
    assert len(ev) == 5, ev
    out = "\n".join(report(threshold=10000))
    assert "hook invocations         3" in out, out
    assert "spilled                1  (33% of calls)" in out, out
    assert "tokens kept out of context" in out
    assert "7,175" in out, out  # (30000-1300)/4 tokens saved
    assert "recoveries used          1" in out or "recoveries used        1" in out, out
    assert "ERRORS" in out and "disk full" in out
    # a 7,000-char skip is >60% of a 10,000 threshold -> should advise lowering
    assert "consider lowering" in out, out
    assert "Bash=1/3" in out, out

    # never raises, even with an unwritable path
    PATH = pathlib.Path("/nonexistent-dir-xyz/m.jsonl")
    record("hook", tool="Bash", chars=1)  # must be silent
    assert read() == []

    print("ok — metrics checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":  # noqa: F821
        demo()
    else:
        print("\n".join(report()))
