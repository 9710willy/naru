#!/usr/bin/env python3
"""note — save and recall notes through the Scroll Event Log.

    note add "sprint planning" < notes.md    # or: note add "topic" "text"
    note search "context budget"
    note outline
    note show 12 18
    note prune [days] [--dry-run]   # default 30; deletes older rows + index
    note gc                          # remove orphaned blob dirs
    note stats [days]                # spill/recovery observability

Notes and hook-spilled tool output share ~/.scroll/log.db (override with
SCROLL_DB). Recall prints
matching lines only, never the whole store, so pulling a fact back into a
session costs a few hundred tokens instead of everything you ever wrote.
"""

import os
import pathlib
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import metrics
from ms import DEFAULT_DB, MemorySurface

DB = pathlib.Path(os.environ.get("SCROLL_NOTES", DEFAULT_DB))


def store():
    DB.parent.mkdir(parents=True, exist_ok=True)
    return MemorySurface(str(DB))


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    ms = store()

    if cmd == "add":
        if not args:
            print('need a topic: note add "sprint planning" [text]', file=sys.stderr)
            return 2
        topic = args[0]
        body = " ".join(args[1:]) if len(args) > 1 else sys.stdin.read()
        if not body.strip():
            print("nothing to save (no text and empty stdin)", file=sys.stderr)
            return 2
        now = datetime.now().isoformat(timespec="minutes")
        # One row per paragraph: search returns the relevant chunk, not the
        # whole meeting.
        chunks = [c.strip() for c in body.split("\n\n") if c.strip()] or [body.strip()]
        seqs = [
            ms.append(
                "note", c, kind="note", session_id=f"{now[:10]} {topic}", created_at=now
            )
            for c in chunks
        ]
        print(f"saved {len(seqs)} chunk(s) as seq {seqs[0]}-{seqs[-1]} -> {DB}")

    elif cmd == "search":
        if not args:
            print("need a query", file=sys.stderr)
            return 2
        k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 8
        q = (
            " ".join(args[:-1])
            if len(args) > 1 and args[-1].isdigit()
            else " ".join(args)
        )
        hits = ms.search(q, k=k)
        metrics.record("search", q=q[:60], hits=len(hits))
        if not hits:
            print("no hits. try `note outline` and `note show LO HI` to browse.")
            return 1
        for h in hits:
            print(f"[{h.seq}] {h.created_at} | {h.session_id}")
            print(f"     {h.content.strip()[:300]}")

    elif cmd == "outline":
        # Human listing: the topic is the anchor you navigate by, and
        # ms.outline() deliberately omits session labels (they are opaque
        # hashes in the benchmark data it was built for).
        rows = ms.sql_query(
            "SELECT session_id, MIN(seq) lo, MAX(seq) hi, MIN(created_at) at,"
            " COUNT(*) n FROM conversation_history GROUP BY session_id"
            " ORDER BY lo"
        )
        if not rows:
            print("(empty — nothing saved yet)")
            return 1
        for r in rows:
            print(
                f"seq {r.lo}-{r.hi} | {str(r.at)[:16]} | {r.n} chunk(s) | "
                f"{r.session_id}"
            )

    elif cmd == "show":
        if not args:
            print("need a seq: note show 12 [18]", file=sys.stderr)
            return 2
        lo = int(args[0])
        hi = int(args[1]) if len(args) > 1 else None
        metrics.record("show", seq=lo)
        for r in ms.expand(lo, hi):
            print(f"--- [{r.seq}] {r.created_at} | {r.session_id}")
            print(r.content)

    elif cmd == "prune":
        days = int(args[0]) if args and args[0].isdigit() else 30
        dry = "--dry-run" in args or "-n" in args
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        doomed = ms.sql_query(
            "SELECT COUNT(*) n, COALESCE(SUM(LENGTH(content)),0) b FROM"
            " conversation_history WHERE created_at IS NOT NULL AND"
            " created_at < ?",
            (cutoff,),
        )[0]
        kept = ms.sql_query(
            "SELECT COUNT(*) n FROM conversation_history WHERE created_at IS"
            " NULL OR created_at >= ?",
            (cutoff,),
        )[0]
        print(f"cutoff {cutoff[:10]} (older than {days} days)")
        print(f"  would remove : {doomed.n} rows, {doomed.b:,} chars")
        print(f"  would keep   : {kept.n} rows")
        if dry:
            print("  --dry-run: nothing deleted")
            return 0
        if not doomed.n:
            return 0
        removed = ms.prune(cutoff)
        print(f"  removed {removed} rows, index cleaned, database vacuumed")
        # metrics age out with the rows they describe
        kept_ev = [e for e in metrics.read() if e.get("t", "") >= cutoff]
        try:
            import json as _j
            metrics.PATH.write_text(
                "".join(_j.dumps(e, separators=(",", ":")) + "\n"
                        for e in kept_ev))
            print(f"  metrics trimmed to {len(kept_ev)} event(s)")
        except OSError:
            pass

    elif cmd == "stats":
        days = int(args[0]) if args and args[0].isdigit() else None
        thr = int(os.environ.get("SCROLL_SPILL_THRESHOLD", 10000))
        print("\n".join(metrics.report(days=days, threshold=thr)))

    elif cmd == "gc":
        # Orphaned blob directories from before the hook stopped writing them.
        import shutil
        import tempfile

        live = {
            r.payload_path
            for r in ms.sql_query(
                "SELECT payload_path FROM conversation_history"
                " WHERE payload_path IS NOT NULL"
            )
        }
        freed = n = 0
        for d in pathlib.Path(tempfile.gettempdir()).glob("scroll-blobs-*"):
            if not d.is_dir():
                continue
            if any(str(f) in live for f in d.iterdir()):
                continue  # still referenced by a row
            freed += sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            shutil.rmtree(d, ignore_errors=True)
            n += 1
        print(f"removed {n} orphaned blob dir(s), freed {freed / 1024:.0f} KB")

    else:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    return 0


def demo():
    """Self-check against a throwaway store."""
    import tempfile

    global DB
    DB = pathlib.Path(tempfile.mkdtemp()) / "t.db"
    metrics.PATH = DB.parent / "m.jsonl"   # never touch the real store

    assert (
        main(
            [
                "add",
                "scroll review",
                "Deferred the eviction index; worth 1.8 pts.\n\n"
                "Slack is the best adapter target.",
            ]
        )
        == 0
    )
    assert main(["add", "sprint planning", "Ship the Email MFE by Friday."]) == 0

    ms = store()
    # paragraphs became separate rows, so recall is chunk-level
    assert len(ms.search("eviction")) == 1, ms.search("eviction")
    assert ms.search("eviction")[0].seq == 1
    assert ms.search("Slack")[0].seq == 2, "second paragraph should be its own row"
    assert ms.search("Friday")[0].seq == 3

    # ms.outline() previews notes even though no row has role='user'
    ol = ms.outline()
    assert "Deferred the eviction" in ol, f"outline preview empty: {ol}"
    assert "Ship the Email MFE" in ol, ol

    # the CLI listing shows the topic, which is what a human navigates by
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["outline"]) == 0
    listing = buf.getvalue()
    assert "scroll review" in listing and "sprint planning" in listing, listing

    # verbatim recovery by address
    assert "1.8 pts" in ms.expand(1)[0].content

    # unknown command and empty input are rejected, not silently accepted
    assert main(["bogus"]) == 2
    assert main(["add"]) == 2

    # prune removes old rows AND their FTS entries; recent rows survive
    old = ms.append("tool", "ANCIENTMARKER payload", kind="tool_result",
                    created_at="2000-01-01T00:00:00")
    assert ms.search("ANCIENTMARKER"), "should be findable before prune"
    removed = ms.prune("2001-01-01T00:00:00")
    assert removed == 1, removed
    assert ms.search("ANCIENTMARKER") == [], "FTS index not cleaned by prune"
    assert ms.expand(old) == [], "row survived prune"
    assert ms.search("eviction"), "prune must not touch recent rows"

    # a missing blob must not break recovery — content column is authoritative
    import tempfile as _tf
    ms2 = MemorySurface(str(pathlib.Path(_tf.mkdtemp()) / "b.db"))
    big = "FULLTEXT " + "z" * 5000
    s2 = ms2.append("tool", big, created_at="2026-01-01T00:00", payload=big)
    row = ms2.sql_query("SELECT payload_path FROM conversation_history WHERE seq=?",
                        (s2,))[0]
    pathlib.Path(row.payload_path).unlink()          # simulate a /var/folders purge
    got = ms2.expand(s2)[0].content
    assert got.startswith("FULLTEXT"), "recovery broke when the blob vanished"

    print("ok — note checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        sys.exit(main(sys.argv[1:]))
