#!/usr/bin/env python3
"""naru — one curated doc, shared across sessions, models and harnesses.

Agents propose claims and skills. You decide. Only what you promote reaches the doc, which is
why the doc stays small while the log grows without bound.

    naru claim "<text>" [--key k] [--by agent]  # an agent proposes a fact
    naru skill "<procedure>" [--key k] [--by agent]
    naru inbox                                  # decide: promote / drop / skip
    naru inject [path]                          # render the doc: stdout or file
    naru promote SEQ --yes | naru drop SEQ --yes    # decide without the prompt

`promote` and `drop` refuse to run without a terminal unless you pass --yes.
That is a speed bump so a tool loop cannot promote by accident, not a security
boundary — anything that can run this CLI can also pass --yes.

To revise a promoted item: `naru drop <old> --yes`, then propose and promote the
new one. Superseding without retiring leaves both promoted and parks the key
under ## Unresolved.

`inject <path>` splices the doc between `<!-- naru:begin -->` and
`<!-- naru:end -->` markers. It never overwrites the rest of the file, so
pointing it at a CLAUDE.md you maintain by hand is safe.

    naru add "sprint planning" < notes.md   # or: naru add "topic" "text"
    naru search "context budget"
    naru outline
    naru show 12 18 [--run ID]
    naru prune [days] [--dry-run]   # default 30; deletes older rows + index
    naru gc                          # remove orphaned blob dirs
    naru stats [days]                # spill/recovery observability

Harness-agnostic by construction: anything that can run a shell command can
both write proposals and read the doc. `inject` targets the context file the
harness already reads. The Codex plugin calls `naru codex-hook` to load the
same approved doc through Codex lifecycle hooks.

    naru inject CLAUDE.md        # Claude Code
    naru inject AGENTS.md        # Codex, DeepSeek Harness, most others
    naru inject .cursorrules     # Cursor

Notes, proposals and hook-spilled tool output share ~/.naru/log.db (override with
NARU_DB). Recall prints matching lines only, never the whole store, so pulling
a fact back into a session costs a few hundred tokens instead of everything you
ever wrote.
"""

import contextlib
import hashlib
import io
import json
import os
import pathlib
import sys
from datetime import datetime, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import metrics
from ms import BEGIN, DEFAULT_DB, END, MemorySurface

DB = pathlib.Path(os.environ.get("NARU_NOTES", DEFAULT_DB))
CODEX_STATE_SCHEMA = "naru.codex.context.v1"
CODEX_EVENTS = {"SessionStart", "UserPromptSubmit", "SubagentStart"}
CODEX_SESSION_LIMIT = 512


def store():
    # The store holds whatever a tool printed, which can include secrets
    # the agent happened to display. 0o700 so it is not world-readable.
    DB.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    return MemorySurface(str(DB))


def _opt(args, name, default=None):
    """Pull `--name value` out of `args` in place. The rest of this CLI parses
    by hand; argparse for three flags would be the bigger change."""
    if name not in args:
        return default
    i = args.index(name)
    if i + 1 == len(args) or not args[i + 1].strip() or args[i + 1].startswith("--"):
        del args[i]
        raise ValueError(f"{name} needs a value")
    val = args[i + 1]
    del args[i : i + 2]
    return val


def _splice(path, text):
    """Write the doc into `path` between markers, never over the whole file.

    `naru inject CLAUDE.md` names a file the user wrote and maintains. A plain
    write_text() silently destroys it — the documented happy path was a data
    loss bug. Existing markers are replaced in place; a file without them keeps
    everything it had and gains a marked block at the end.
    """
    block = f"{BEGIN}\n{text.rstrip()}\n{END}\n"
    p = pathlib.Path(path)
    try:
        old = p.read_text()
    except (OSError, UnicodeDecodeError):
        old = ""
    if BEGIN in old and END in old:
        head, rest = old.split(BEGIN, 1)
        new = head + block + rest.split(END, 1)[1].lstrip("\n")
    else:
        new = (old.rstrip() + "\n\n" if old.strip() else "") + block
    p.write_text(new)
    return len(block), len(old)


def _j_dumps(events):
    """Serialize metric events exactly as record() writes them."""
    import json

    return "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events)


def _apply(ms, seq, answer):
    """Act on one inbox answer. Returns the line to print.

    Split out of the input() loop so promote / drop / skip can be asserted
    without a terminal, and so the rowcount is not thrown away: another writer
    can decide the same claim while the human is still reading it.
    """
    if answer.startswith("p"):
        return "promoted" if ms.decide(seq, True) else "already decided elsewhere"
    if answer.startswith("d"):
        return "dropped" if ms.decide(seq, False) else "already decided elsewhere"
    return "skipped"


def _codex_session(event):
    session_id = event.get("session_id")
    if (
        not isinstance(session_id, str)
        or not session_id
        or len(session_id) > CODEX_SESSION_LIMIT
        or "\0" in session_id
    ):
        return None
    return f"codex:{session_id}"


def _codex_seen(ms, session_id):
    rows = ms.sql_query(
        "SELECT content FROM conversation_history"
        " WHERE kind='agent_state' AND agent_id='codex' AND session_id=?"
        " ORDER BY seq DESC LIMIT 1",
        (session_id,),
    )
    if not rows:
        return None
    try:
        state = json.loads(rows[0].content)
    except (TypeError, json.JSONDecodeError):
        return None
    if state.get("schema") != CODEX_STATE_SCHEMA:
        return None
    return state.get("doc_hash") if isinstance(state.get("doc_hash"), str) else None


def _codex_remember(ms, session_id, doc_seq, doc_hash):
    if _codex_seen(ms, session_id) == doc_hash:
        return
    ms.append(
        "agent",
        json.dumps(
            {
                "schema": CODEX_STATE_SCHEMA,
                "doc_seq": doc_seq,
                "doc_hash": doc_hash,
            },
            separators=(",", ":"),
        ),
        kind="agent_state",
        session_id=session_id,
        agent_id="codex",
        created_at=datetime.now().isoformat(timespec="seconds"),
    )


def _codex_context(doc):
    return (
        "Naru live context follows. It replaces any older Naru block from "
        "AGENTS.md.\n"
        "File a lasting fact with `naru claim \"<text>\" --key <topic> --by "
        "codex`; a person decides it with `naru inbox`. Use `naru search "
        "\"<terms>\"` to recall archived facts.\n\n"
        + doc
    )


def _codex_hook(event, ms):
    if not isinstance(event, dict) or event.get("hook_event_name") not in CODEX_EVENTS:
        return None
    session_id = _codex_session(event)
    if session_id is None:
        return None
    event_name = event["hook_event_name"]
    doc = ms.doc()
    doc_seq = ms.doc_version()
    doc_hash = hashlib.sha256(doc.encode()).hexdigest()
    if event_name == "UserPromptSubmit" and _codex_seen(ms, session_id) == doc_hash:
        return None
    if event_name != "SubagentStart":
        _codex_remember(ms, session_id, doc_seq, doc_hash)
    return {
        "hookSpecificOutput": {
            "hookEventName": event_name,
            "additionalContext": _codex_context(doc),
        }
    }


def codex_hook_main(raw=None):
    """Codex hook boundary. A Naru failure must not stop the agent."""
    try:
        event = json.loads(sys.stdin.read() if raw is None else raw)
        output = _codex_hook(event, store())
    except Exception:
        return 0
    if output:
        json.dump(output, sys.stdout)
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    if cmd == "codex-hook":
        return codex_hook_main()
    ms = store()

    if cmd == "add":
        if not args:
            print('need a topic: naru add "sprint planning" [text]', file=sys.stderr)
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

    elif cmd in ("claim", "skill"):
        try:
            key, by, base = _opt(args, "--key"), _opt(args, "--by"), _opt(args, "--base")
            run, source = _opt(args, "--run"), _opt(args, "--source")
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        text = " ".join(args).strip() or sys.stdin.read().strip()
        if not text:
            print(f'need text: naru {cmd} "<text>" [--key k]', file=sys.stderr)
            return 2
        if bool(run) != bool(source):
            print("--run and --source LO:HI must be used together", file=sys.stderr)
            return 2
        if source:
            try:
                lo, hi = (int(x) for x in source.split(":", 1))
                if lo > hi:
                    raise ValueError
            except ValueError:
                print("--source must be LO:HI with LO <= HI", file=sys.stderr)
                return 2
        else:
            lo = hi = None
        if base:
            try:
                base_seq = int(base)
            except ValueError:
                print("--base must be an integer", file=sys.stderr)
                return 2
        else:
            base_seq = ms.doc_version()
        try:
            seq = ms.append(
                "agent", text, kind=cmd, agent_id=by, topic_key=key,
                base_seq=base_seq, created_at=datetime.now().isoformat(timespec="seconds"),
                source_run_id=run, source_seq_lo=lo, source_seq_hi=hi,
            )
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        print(f"{cmd} {seq} pending — `naru inbox` to decide")

    elif cmd in ("promote", "drop"):
        args = list(args)
        yes = "--yes" in args
        if yes:
            args.remove("--yes")
        if not args or not args[0].isdigit():
            print(f"need a seq: naru {cmd} 12", file=sys.stderr)
            return 2
        # A speed bump, NOT a security boundary. An agent that can run this CLI
        # can also pass --yes, so do not read this as "only a human can
        # promote". It exists so that promotion is never something a tool loop
        # does by accident, which is a different and much weaker claim.
        if not yes and not sys.stdin.isatty():
            print(
                f"refusing to {cmd} without a terminal; pass --yes if you mean it",
                file=sys.stderr,
            )
            return 2
        if not ms.decide(int(args[0]), cmd == "promote"):
            print(
                f"seq {args[0]} is not a pending claim or skill this can {cmd}"
                " (already decided, or not a curation item)",
                file=sys.stderr,
            )
            return 1
        print(f"seq {args[0]} {'promoted' if cmd == 'promote' else 'dropped'}")

    elif cmd == "inbox":
        pend = ms.pending()
        if not pend:
            print("inbox clear")
            return 0
        version = ms.doc_version()
        for i, c in enumerate(pend, 1):
            print(
                f"[{i}/{len(pend)}] {c.kind} seq {c.seq} · {c.agent_id or '?'} · "
                f"{(c.created_at or '')[:16]}"
                + (f" · key: {c.topic_key}" if c.topic_key else "")
            )
            # Staleness cannot be prevented — the author was not wrong when
            # they started. Show it and let the human weigh it.
            if c.base_seq is not None and version > c.base_seq:
                print(
                    f"  ⚠ written against doc version {c.base_seq} — "
                    f"the doc is now at {version}"
                )
            print(f"  + {c.content.strip()}")
            if c.source_run_id:
                print(
                    f"  source: naru show {c.source_seq_lo} {c.source_seq_hi} "
                    f"--run {c.source_run_id}"
                )
            try:
                ans = input("  promote / drop / skip ? ").strip().lower()
            except EOFError:
                print("\n(no input — nothing decided)")
                return 0
            print("  " + _apply(ms, c.seq, ans) + "\n")

    elif cmd == "inject":
        # One render for every file-based harness, spliced between markers so
        # it never destroys a file it does not own.
        text = ms.doc()
        if args:
            wrote, had = _splice(args[0], text)
            print(
                f"wrote {wrote} chars into {args[0]}"
                + (f" (kept the {had} chars already there)" if had else "")
            )
        else:
            print(text, end="")

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
            print("no hits. try `naru outline` and `naru show LO HI` to browse.")
            return 1
        for h in hits:
            print(f"[{h.seq}] {h.created_at} | {h.session_id}")
            print(f"     {h.content.strip()[:300]}")

    elif cmd == "outline":
        # Human listing: the topic is the anchor you navigate by, and
        # ms.outline() deliberately omits session labels (they are opaque
        # hashes in the benchmark data it was built for).
        rows = ms.session_ranges()
        if not rows:
            print("(empty — nothing saved yet)")
            return 1
        for r in rows:
            print(
                f"seq {r.lo}-{r.hi} | {str(r.at)[:16]} | {r.n} chunk(s) | "
                f"{r.session_id}"
            )

    elif cmd == "show":
        try:
            run = _opt(args, "--run")
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        if not args:
            print("need a seq: naru show 12 [18] [--run ID]", file=sys.stderr)
            return 2
        try:
            lo = int(args[0])
            hi = int(args[1]) if len(args) > 1 else None
        except ValueError:
            print("show sequence values must be integers", file=sys.stderr)
            return 2
        metrics.record("show", seq=lo)
        for r in ms.expand(lo, hi, session_id=run) if run else ms.expand(lo, hi):
            print(f"--- [{r.seq}] {r.created_at} | {r.session_id}")
            print(r.content)

    elif cmd == "prune":
        days = int(args[0]) if args and args[0].isdigit() else 30
        dry = "--dry-run" in args or "-n" in args
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
        n_doomed, b_doomed, n_kept = ms.prune_preview(cutoff)
        print(f"cutoff {cutoff[:10]} (older than {days} days)")
        print(f"  would remove : {n_doomed} rows, {b_doomed:,} chars")
        print(f"  would keep   : {n_kept} rows (promoted claims, skills, and evidence are never aged out)")
        if dry:
            print("  --dry-run: nothing deleted")
            return 0
        if not n_doomed:
            return 0
        removed = ms.prune(cutoff)
        print(f"  removed {removed} rows, index cleaned, database vacuumed")
        # Metrics age out with the rows they describe. This is also the ONE
        # place the file is size-capped: record() used to trim in-line, which
        # raced across parallel hook processes and dropped new events.
        kept_ev = [e for e in metrics.read() if e.get("t", "") >= cutoff]
        while len(_j_dumps(kept_ev)) > metrics.MAX_BYTES:
            kept_ev = kept_ev[len(kept_ev) // 4 :]
        try:
            metrics.PATH.write_text(_j_dumps(kept_ev))
            print(f"  metrics trimmed to {len(kept_ev)} event(s)")
        except OSError:
            pass

    elif cmd == "stats":
        days = int(args[0]) if args and args[0].isdigit() else None
        # One owner for the threshold. Re-reading the env with a different
        # default made `stats` advise against a number the hook never used.
        from hook_spill import THRESHOLD

        print("\n".join(metrics.report(days=days, threshold=THRESHOLD)))

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
        for d in pathlib.Path(tempfile.gettempdir()).glob("naru-blobs-*"):
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
    metrics.PATH = DB.parent / "m.jsonl"  # never touch the real store

    # Close the door rather than hoping it is shut. Several assertions below
    # depend on stdin being at EOF; inheriting the caller's stdin meant
    # `python3 naru.py --selfcheck` — the exact pre-commit command — hung
    # forever on a terminal, at `claim` with no text and again at `inbox`.
    real_stdin = sys.stdin
    sys.stdin = io.StringIO("")
    try:
        _demo(real_stdin)
    finally:
        sys.stdin = real_stdin


def _demo(real_stdin):

    assert (
        main(
            [
                "add",
                "naru review",
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
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["outline"]) == 0
    listing = buf.getvalue()
    assert "naru review" in listing and "sprint planning" in listing, listing

    # verbatim recovery by address
    assert "1.8 pts" in ms.expand(1)[0].content

    # unknown command and empty input are rejected, not silently accepted
    assert main(["bogus"]) == 2
    assert main(["add"]) == 2

    # prune removes old rows AND their FTS entries; recent rows survive
    old = ms.append(
        "tool",
        "ANCIENTMARKER payload",
        kind="tool_result",
        created_at="2000-01-01T00:00:00",
    )
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
    row = ms2.sql_query(
        "SELECT payload_path FROM conversation_history WHERE seq=?", (s2,)
    )[0]
    pathlib.Path(row.payload_path).unlink()  # simulate a /var/folders purge
    got = ms2.expand(s2)[0].content
    assert got.startswith("FULLTEXT"), "recovery broke when the blob vanished"

    # ---- curation through the CLI -----------------------------------------
    assert (
        main(
            [
                "claim",
                "Store is SQLite, WAL on.",
                "--key",
                "store.engine",
                "--by",
                "claude-opus",
            ]
        )
        == 0
    )
    assert (
        main(["claim", "Retry budget is 5.", "--key", "bench.retries", "--by", "gpt-5"])
        == 0
    )
    assert main(["claim"]) == 2

    pend = store().pending()
    assert len(pend) == 2, pend
    assert pend[0].agent_id == "claude-opus" and pend[0].topic_key == "store.engine"
    assert pend[0].base_seq is not None, "a claim must record the doc it saw"

    # promotion refuses to run unattended without an explicit --yes
    assert main(["promote", str(pend[0].seq)]) == 2, "unattended promote allowed"
    assert main(["promote", str(pend[0].seq), "--yes"]) == 0
    assert main(["drop", str(pend[1].seq), "--yes"]) == 0
    assert store().pending() == [], "decided claims must leave the inbox"
    # promoting twice is refused, not silently re-applied
    assert main(["promote", str(pend[0].seq), "--yes"]) == 1
    assert main(["promote", "9999", "--yes"]) == 1
    assert main(["promote", "--yes"]) == 2
    # A promoted fact stays revisable: retire the old one, promote the new one.
    # Without this, superseding leaves both promoted and parks the key under
    # ## Unresolved forever, losing the fact the doc used to state.
    assert main(["drop", str(pend[0].seq), "--yes"]) == 0, (
        "promoted claim not retirable"
    )
    assert main(["promote", str(pend[0].seq), "--yes"]) == 1, (
        "a dropped claim came back"
    )
    assert (
        main(
            [
                "claim",
                "Store is SQLite, WAL on.",
                "--key",
                "store.engine",
                "--by",
                "human",
            ]
        )
        == 0
    )
    fresh = store().pending()[-1].seq
    assert main(["promote", str(fresh), "--yes"]) == 0
    assert store().conflicts() == {}, "superseding must not create a conflict"

    # _apply is the inbox's decision logic, reachable without a terminal
    ms_a = store()
    escaped = ms_a.append("agent", f"literal {BEGIN} {END}", kind="claim")
    assert _apply(ms_a, escaped, "p") == "promoted"
    assert BEGIN not in ms_a.doc() and END not in ms_a.doc()
    s = ms_a.append(
        "agent",
        "apply path",
        kind="claim",
        agent_id="t",
        created_at="2026-08-27T12:00:00",
    )
    assert _apply(ms_a, s, "skip") == "skipped"
    assert _apply(ms_a, s, "p") == "promoted"
    assert _apply(ms_a, s, "p") == "already decided elsewhere", (
        "inbox must not claim success when another writer already decided"
    )
    assert main(["drop", str(s), "--yes"]) == 0

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["inject"]) == 0
    doc = buf.getvalue()
    assert "Store is SQLite" in doc, doc
    assert "Retry budget" not in doc, "a dropped claim reached the doc"

    # File-based harnesses still use the shared marker-safe integration.
    target = DB.parent / "AGENTS.md"
    assert main(["inject", str(target)]) == 0
    assert "Store is SQLite" in target.read_text()

    # inject must NEVER destroy a file it does not own. The documented path is
    # `naru inject CLAUDE.md`, a file the user wrote.
    hand = DB.parent / "CLAUDE.md"
    hand.write_text("# My rules\n\nAlways run the tests before committing.\n")
    assert main(["inject", str(hand)]) == 0
    body = hand.read_text()
    assert "Always run the tests" in body, "inject destroyed a hand-written file"
    assert "Store is SQLite" in body, "inject did not add the doc"
    assert body.count(BEGIN) == 1 and body.count(END) == 1, body
    assert "naru: begin" in body and "naru: end" in body, body

    # re-injecting replaces the block in place, it does not stack copies
    assert main(["inject", str(hand)]) == 0
    body2 = hand.read_text()
    assert body2.count(BEGIN) == 1, "re-inject appended a second block"
    assert "Always run the tests" in body2, "re-inject lost the user's content"

    # Codex gets the approved doc at startup, then only after the doc changes.
    codex_id = "session-" + "x" * 80
    codex_run = f"codex:{codex_id}"
    start = _codex_hook(
        {"hook_event_name": "SessionStart", "session_id": codex_id}, store()
    )
    assert start["hookSpecificOutput"]["hookEventName"] == "SessionStart"
    assert "Store is SQLite" in start["hookSpecificOutput"]["additionalContext"]
    state = store().sql_query(
        "SELECT session_id, content FROM conversation_history"
        " WHERE kind='agent_state' AND agent_id='codex' ORDER BY seq DESC LIMIT 1"
    )[0]
    assert state.session_id == codex_run, "Codex session ID was truncated"
    state_content = json.loads(state.content)
    assert state_content["doc_seq"] == store().doc_version()
    assert state_content["doc_hash"] == hashlib.sha256(store().doc().encode()).hexdigest()
    assert not any(
        r.content.startswith('{"schema":"naru.codex.context.v1"')
        for r in store().search("naru", k=50)
    ), "Codex hook state leaked into normal search"
    assert _codex_hook(
        {
            "hook_event_name": "UserPromptSubmit",
            "session_id": codex_id,
            "prompt": "PRIVATE_CODEX_PROMPT",
        },
        store(),
    ) is None, "unchanged Naru context was sent twice"
    assert not store().sql_query(
        "SELECT seq FROM conversation_history WHERE content LIKE ?",
        ("%PRIVATE_CODEX_PROMPT%",),
    ), "Codex prompt was persisted"

    assert main(["claim", "Codex refresh marker.", "--key", "codex.refresh"]) == 0
    codex_claim = store().pending()[-1].seq
    assert main(["promote", str(codex_claim), "--yes"]) == 0
    assert main(["claim", "Codex high marker.", "--key", "codex.high"]) == 0
    codex_high = store().pending()[-1].seq
    assert main(["promote", str(codex_high), "--yes"]) == 0
    refresh = _codex_hook(
        {"hook_event_name": "UserPromptSubmit", "session_id": codex_id}, store()
    )
    assert "Codex refresh marker" in refresh["hookSpecificOutput"]["additionalContext"]
    assert "Codex high marker" in refresh["hookSpecificOutput"]["additionalContext"]
    assert _codex_hook(
        {"hook_event_name": "UserPromptSubmit", "session_id": codex_id}, store()
    ) is None, "refreshed Naru context was sent twice"

    assert main(["drop", str(codex_claim), "--yes"]) == 0
    retired = _codex_hook(
        {"hook_event_name": "UserPromptSubmit", "session_id": codex_id}, store()
    )
    assert "Codex refresh marker" not in retired["hookSpecificOutput"]["additionalContext"]
    assert "Codex high marker" in retired["hookSpecificOutput"]["additionalContext"]
    assert _codex_hook(
        {"hook_event_name": "UserPromptSubmit", "session_id": codex_id}, store()
    ) is None, "retired Naru context was sent twice"
    assert main(["drop", str(codex_high), "--yes"]) == 0

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert codex_hook_main(
            json.dumps(
                {"hook_event_name": "SubagentStart", "session_id": codex_id}
            )
        ) == 0
    subagent = json.loads(buf.getvalue())
    assert subagent["hookSpecificOutput"]["hookEventName"] == "SubagentStart"
    assert "Store is SQLite" in subagent["hookSpecificOutput"]["additionalContext"]
    assert "Codex high marker" not in subagent["hookSpecificOutput"]["additionalContext"]
    assert _codex_hook(
        {"hook_event_name": "PostToolUse", "session_id": codex_id}, store()
    ) is None, "unsupported Codex event produced output"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert codex_hook_main("not json") == 0
    assert not buf.getvalue(), "malformed Codex input produced output"

    # a note is not a claim: `add` must never reach the doc
    assert main(["add", "notes topic", "This line is a note, not a claim."]) == 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["inject"])
    assert "not a claim" not in buf.getvalue(), "a note leaked into the doc"

    trace_run = "trace-run"
    reply = store().append(
        "agent", "```python\\nprint('evidence')\\n```", kind="agent_reply",
        session_id=trace_run, created_at="2026-08-27T12:00:00",
    )
    observation = store().append(
        "tool", "evidence", kind="agent_observation", session_id=trace_run,
        created_at="2026-08-27T12:00:01",
    )
    assert main([
        "claim", "trace fact", "--key", "trace.fact", "--run", trace_run,
        "--source", f"{reply}:{observation}",
    ]) == 0
    trace_claim = store().pending()[-1]
    assert trace_claim.source_run_id == trace_run
    assert (trace_claim.source_seq_lo, trace_claim.source_seq_hi) == (reply, observation)
    assert main(["claim", "partial", "--run", trace_run]) == 2
    assert main(["claim", "partial", "--source", f"{reply}:{observation}"]) == 2
    assert main(["claim", "bad", "--run", trace_run, "--source", "4:nope"]) == 2
    assert main(["claim", "wrong run", "--run", "wrong", "--source", f"{reply}:{observation}"]) == 2
    assert main(["claim", "bad", "--base", "nope"]) == 2
    assert main(["claim", "missing", "--key"]) == 2
    assert main(["claim", "missing", "--key", ""]) == 2
    assert main(["skill", "missing", "--run"]) == 2
    assert main(["show", str(reply), "--run"]) == 2
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["inbox"]) == 0
    assert f"claim seq {trace_claim.seq}" in buf.getvalue()
    assert f"naru show {reply} {observation} --run {trace_run}" in buf.getvalue()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert main(["show", str(reply), str(observation), "--run", trace_run]) == 0
    assert "evidence" in buf.getvalue()

    assert main(["skill", "Search the Event Log first.", "--key", "lookup"]) == 0
    skill = store().pending()[-1]
    assert skill.kind == "skill" and "Search the Event Log" not in store().doc()
    assert main(["promote", str(skill.seq), "--yes"]) == 0
    assert "## Skills" in store().doc() and "Search the Event Log" in store().doc()

    # inbox at EOF must not hang and must not decide anything on its own
    pre = len(store().pending())
    assert main(["claim", "pending forever", "--by", "codex"]) == 0
    with contextlib.redirect_stdout(io.StringIO()):
        assert main(["inbox"]) == 0
    assert len(store().pending()) == pre + 1, "inbox decided something unprompted"

    # `claim` with no text and no stdin is an error, not a block
    assert main(["claim"]) == 2

    print("ok — naru checks passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        demo()
    else:
        sys.exit(main(sys.argv[1:]))
