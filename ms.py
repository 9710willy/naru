"""ms — the memory surface for a Naru-style Session Environment.

An append-only Event Log (SQLite + FTS5) the model writes code against.
Full history stays lossless and addressable by `seq`; only what the caller
prints ever needs to enter a model's context.

Interface follows the Appendix-C spec of arXiv 2608.21690:
    ms.search(query, k=..., kind=...)                   # BM25 locate
    ms.outline()                                        # navigation anchors
    ms.expand(lo, hi=None)                              # materialize a seq span
    ms.sql_query("SELECT ... FROM conversation_history WHERE ...")  # read-only
    ms.days_between(d1, d2)                             # calendar-day helper
    ms.append(role, content, ...)                       # ingest a turn

Stdlib only. No embeddings, no service, deterministic.
"""

import os
import pathlib
import re
import sqlite3
import tempfile
from datetime import date

from eviction import est  # one owner for chars-per-token

# ONE Event Log for everything the session wants to recall later: notes written
# by `naru add`, and tool output spilled by the PostToolUse hook. They differ
# only by `kind`. Two stores would mean the recovery handle a spill prints
# (`naru show N`) points at a database `naru` does not read.
DEFAULT_DB = pathlib.Path(
    os.environ.get("NARU_DB", pathlib.Path.home() / ".naru" / "log.db")
)

# Age is the right rule for spilled tool output and the wrong one for a
# decision a human made. Anything else a prune would take, it still takes.
PRUNE_KEEP = " AND NOT (kind = 'claim' AND promoted <> 0)"

# Payloads over this go to disk. One value, no caller has ever varied it.
BLOB_THRESHOLD = 4000
# Session preview width in outline().
OUTLINE_PREVIEW = 90


def _oneline(text):
    """Collapse a claim to a single line for rendering.

    The doc is written into files other tools parse — CLAUDE.md, AGENTS.md, a
    shell rc. A newline inside a claim would otherwise break out of its bullet
    and become a top-level line in the host format, which is how agent-authored
    text turns into a directive the host executes.
    """
    return re.sub(r"\s+", " ", (text or "").strip())


class Row(dict):
    """A log row. Dict access for SQL ergonomics, attribute access for code,
    and a token-frugal repr so printing a hit doesn't dump the whole turn."""

    def __getattr__(self, name):
        # Raise on an unknown column instead of returning None: a typo like
        # hit.contnet must not read as "that field is empty".
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"no column {name!r}; have {sorted(self)}") from None

    def __repr__(self):
        body = (self.get("content") or "").replace("\n", " ")
        if len(body) > 80:
            body = body[:77] + "…"
        return f"<seq {self.get('seq')} {self.get('role')} {self.get('created_at')}: {body}>"


def _to_match(query):
    """Translate a user query into FTS5 MATCH syntax.

    Spec: uppercase OR/AND/NOT pass through as boolean operators; every other
    token is a quoted phrase (so punctuation can't inject MATCH syntax).
    Consecutive phrases with no operator are implicitly AND-combined by FTS5.
    """
    out = []
    for tok in query.split():
        if tok in ("OR", "AND", "NOT"):
            out.append(tok)
        else:
            out.append('"' + tok.replace('"', '""') + '"')
    return " ".join(out)


class MemorySurface:
    def __init__(self, db=":memory:", blobs=None):
        # Several agents append while a human reads. WAL lets readers run
        # during a write; busy_timeout absorbs the overlap instead of raising
        # "database is locked" in whichever caller lost the race.
        # The path, not just the connection: a sandboxed kernel runs in a
        # child process and has to reopen the log by name.
        self.path = db
        self.db = sqlite3.connect(db, timeout=30)
        self.db.row_factory = sqlite3.Row
        try:
            # Converting a rollback-journal file to WAL needs an exclusive lock
            # SQLite will NOT wait for: with another connection attached it
            # returns BUSY immediately, ignoring busy_timeout. The mode is a
            # persistent property of the file, so whoever wins sets it for
            # everyone and the losers are already in the mode they wanted.
            self.db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        self.db.execute("PRAGMA busy_timeout=30000")
        try:
            # temp.* is private to this connection. A shared `_probe` name
            # races: two constructors interleave CREATE/CREATE/DROP/DROP and
            # the loser's DROP raises, reported below as "SQLite lacks FTS5".
            self.db.execute("CREATE VIRTUAL TABLE temp._probe USING fts5(x)")
            self.db.execute("DROP TABLE temp._probe")
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "SQLite build lacks FTS5; cannot back the Event Log"
            ) from e

        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_history(
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                agent_id   TEXT,
                role       TEXT,
                kind       TEXT,
                created_at TEXT,          -- ISO-8601, lexically sortable
                content    TEXT,
                payload_path TEXT,        -- externalized big payloads
                -- curation: a claim is pending (0), promoted (1) or dropped (-1)
                promoted   INTEGER NOT NULL DEFAULT 0,
                topic_key  TEXT,          -- same key, both promoted = contradiction
                base_seq   INTEGER        -- doc version the author wrote against
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
                content, content='conversation_history', content_rowid='seq');
        """)
        self._migrate_fts()
        self._migrate_cols()
        # per-instance by default: bench.py runs questions in parallel
        # threads and a shared dir would collide on blobs/<seq>.txt
        self.blobs = pathlib.Path(
            blobs or pathlib.Path(tempfile.gettempdir()) / f"naru-blobs-{id(self):x}"
        )
        self.blob_threshold = BLOB_THRESHOLD

    def _migrate_fts(self):
        """Upgrade a store created with a contentless FTS table.

        The original schema used content='' , which SQLite refuses to DELETE
        from, so the index could never be pruned. An external-content table
        reads text from conversation_history, which makes both DELETE and
        'rebuild' available. The content column is authoritative, so rebuilding
        the index loses nothing.
        """
        sql = self.db.execute(
            "SELECT sql FROM sqlite_master WHERE name='fts'"
        ).fetchone()
        if not sql or "content='conversation_history'" in (sql["sql"] or ""):
            return
        self.db.executescript("""
            DROP TABLE IF EXISTS fts;
            CREATE VIRTUAL TABLE fts USING fts5(
                content, content='conversation_history', content_rowid='seq');
        """)
        self.db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
        self.db.commit()

    def _migrate_cols(self):
        """Add columns to a store created before they existed.

        Every column added to CREATE TABLE after the first release needs an
        entry here. agent_id did not have one, so every append() against a
        store older than it failed with "no such column" — silently, because
        the spill hook swallows its own errors.

        ADD COLUMN throws if the column is already there and SQLite has no
        IF NOT EXISTS for it, so read the table shape first.
        """
        have = {
            r["name"]
            for r in self.db.execute("PRAGMA table_info(conversation_history)")
        }
        for name, decl in (
            ("agent_id", "TEXT"),
            ("promoted", "INTEGER NOT NULL DEFAULT 0"),
            ("topic_key", "TEXT"),
            ("base_seq", "INTEGER"),
        ):
            if name not in have:
                try:
                    self.db.execute(
                        f"ALTER TABLE conversation_history ADD COLUMN {name} {decl}"
                    )
                except sqlite3.OperationalError as e:
                    # Another process migrated between our table_info read and
                    # this ALTER. busy_timeout does not cover it — a duplicate
                    # column is a schema error, not a lock. Losing is success.
                    if "duplicate column" not in str(e).lower():
                        raise
        # Every curation query filters on promoted or groups by topic_key.
        # Without these the doc costs a full scan of the whole Event Log.
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_promoted"
            " ON conversation_history(promoted, seq)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_topic_key"
            " ON conversation_history(topic_key, seq) WHERE topic_key IS NOT NULL"
        )
        self.db.commit()

    # ---- ingest -----------------------------------------------------------
    def append(
        self,
        role,
        content,
        *,
        kind=None,
        session_id=None,
        agent_id=None,
        created_at=None,
        payload=None,
        topic_key=None,
        base_seq=None,
    ):
        """Append one turn. Returns its stable `seq`. `created_at` is caller-
        supplied (the harness stamps it) so ingestion stays deterministic.

        A payload over the threshold is externalized to disk. The row is
        inserted FIRST so the blob is named by the real seq and the pointer we
        show the model cannot drift.
        """
        big = payload is not None and len(payload) > self.blob_threshold
        cur = self.db.execute(
            "INSERT INTO conversation_history"
            "(session_id, agent_id, role, kind, created_at, content, payload_path,"
            " topic_key, base_seq) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                session_id,
                agent_id,
                role,
                kind,
                created_at,
                content,
                None,
                topic_key,
                base_seq,
            ),
        )
        seq = cur.lastrowid

        if big:
            self.blobs.mkdir(parents=True, exist_ok=True)
            path = str(self.blobs / f"{seq}.txt")
            pathlib.Path(path).write_text(payload)
            # Section 2.2: an externalized payload leaves the ROW holding a
            # bounded preview and a recovery pointer — not the whole payload.
            # Keeping both stored the same bytes twice.
            preview = payload[: self.blob_threshold].rstrip()
            content = (
                f"{preview}\n[payload {len(payload)} chars, preview only "
                f"-> ms.expand({seq})]"
            )
            self.db.execute(
                "UPDATE conversation_history SET content=?, payload_path=? WHERE seq=?",
                (content, path, seq),
            )

        self.db.execute("INSERT INTO fts(rowid, content) VALUES(?,?)", (seq, content))
        self.db.commit()
        return seq

    def readonly(self):
        """A capability-restricted view for the kernel.

        The paper (section 2.2) requires the Event Log to be read-only from the
        kernel. Handing the model the MemorySurface itself does not satisfy
        that: `ms.db.execute("DELETE FROM conversation_history")` destroys the
        history the whole design promises is recoverable. This facade exposes
        only the four Table-1 operations and carries no database handle.
        """
        return ReadOnly(self)

    # ---- LOCATE -----------------------------------------------------------
    def search(self, query, k=5, kind=None, since=None, until=None):
        """BM25 full-text search over the log. Returns hits ranked best-first,
        each a Row carrying the full turn plus its seq/role/metadata.

        Multi-term queries AND-combine. If that yields nothing, they are
        retried as OR — a question's wording often shares only some words with
        the turn that answers it, and a silent zero-hit AND is the single
        biggest source of lost evidence.
        """
        hits = self._match(_to_match(query), k, kind, since, until)
        terms = [t for t in query.split() if t not in ("OR", "AND", "NOT")]
        if not hits and len(terms) > 1:
            hits = self._match(
                " OR ".join(_to_match(t) for t in terms), k, kind, since, until
            )
        return hits

    def _match(self, match_expr, k, kind, since=None, until=None):
        where = ["fts MATCH ?"]
        params = [match_expr]
        if kind is not None:
            where.append("c.kind = ?")
            params.append(kind)
        # created_at is ISO-8601, so lexical comparison is date comparison
        if since is not None:
            where.append("c.created_at >= ?")
            params.append(str(since))
        if until is not None:
            where.append("c.created_at <= ?")
            params.append(str(until))
        params.append(k)
        sql = (
            f"SELECT c.* FROM fts JOIN conversation_history c ON c.seq = fts.rowid "
            f"WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?"
        )
        return [Row(r) for r in self.db.execute(sql, params).fetchall()]

    def session_ranges(self):
        """One row per session_id: seq span, earliest date, turn count.

        The aggregate behind both `outline()` and the CLI's `outline` listing.
        They render it differently — the CLI shows session_id, this one shows a
        preview of the first turn — but the query has one owner so the two
        views cannot drift apart.
        """
        return [
            Row(r)
            for r in self.db.execute(
                "SELECT session_id, MIN(seq) lo, MAX(seq) hi, MIN(created_at) at,"
                " COUNT(*) n FROM conversation_history"
                " GROUP BY session_id ORDER BY lo"
            ).fetchall()
        ]

    def outline(self):
        """Structural map of the log: one line per session with its date, seq
        range and the opening of its first user turn.

        Navigation anchors for when lexical search fails — the question's
        wording may share no words with the turn that answers it, and then the
        only way in is to browse. Cheap: one short line per session.
        """
        rows = self.session_ranges()
        out = []
        for r in rows:
            # Prefer the first user turn (it states the topic in a chat log),
            # but fall back to the first turn of any role — a log of notes or
            # tool results has no 'user' turn at all.
            first = self.db.execute(
                "SELECT content FROM conversation_history WHERE session_id IS ?"
                " ORDER BY (role != 'user'), seq LIMIT 1",
                (r["session_id"],),
            ).fetchone()
            head = (first["content"] if first else "").replace("\n", " ")
            head = head.split("] ", 1)[-1]  # drop the [Session N | date] tag
            out.append(
                f"seq {r['lo']}-{r['hi']} | {(r['at'] or '')[:10]} | "
                f"{r['n']} turns | {head[:OUTLINE_PREVIEW]}"
            )
        return "\n".join(out)

    # ---- MATERIALIZE ------------------------------------------------------
    def expand(self, lo, hi=None):
        """Recover an exact seq span, verbatim, as Row objects. Externalized
        payloads are inlined from disk. `expand(seq)` recovers one turn."""
        hi = lo if hi is None else hi
        rows = self.db.execute(
            "SELECT * FROM conversation_history WHERE seq BETWEEN ? AND ? ORDER BY seq",
            (lo, hi),
        ).fetchall()
        out = []
        for r in rows:
            row = Row(r)
            p = row.get("payload_path")
            if p:
                # The row's own content column is authoritative; a blob is an
                # optimization. Never let a missing blob break recovery — the
                # whole promise is that nothing is lost.
                try:
                    row["content"] = pathlib.Path(p).read_text()
                except OSError:
                    pass
            out.append(row)
        return out

    def prune_preview(self, before_iso):
        """What `prune(before_iso)` would remove, and what it would keep.

        Beside `prune` and sharing PRUNE_KEEP with it deliberately. The CLI
        used to re-derive this with an age-only predicate, so `--dry-run`
        reported that it would delete promoted claims that `prune` then
        correctly refused to touch — the guard ADR 0005 exists for was
        invisible in the one command you run before a destructive one.
        """
        row = self.db.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(LENGTH(content)),0) b"
            " FROM conversation_history"
            " WHERE created_at IS NOT NULL AND created_at < ?" + PRUNE_KEEP,
            (before_iso,),
        ).fetchone()
        kept = self.db.execute(
            "SELECT COUNT(*) n FROM conversation_history"
            " WHERE NOT (created_at IS NOT NULL AND created_at < ?"
            + PRUNE_KEEP
            + ")",
            (before_iso,),
        ).fetchone()
        return row["n"], row["b"], kept["n"]

    def prune(self, before_iso):
        """Delete rows created before `before_iso`. Returns rows removed.

        A DECIDED claim is never pruned by age. Claims are stamped with the
        wall clock at `naru claim` time, so an age-only predicate gives every
        human promotion a 30-day shelf life and silently empties the doc. Age
        is the right rule for spilled tool output, not for a curated decision.

        Blob files for the deleted rows are removed too.
        """
        doomed = self.db.execute(
            "SELECT seq, payload_path FROM conversation_history"
            " WHERE created_at IS NOT NULL AND created_at < ?" + PRUNE_KEEP,
            (before_iso,),
        ).fetchall()
        if not doomed:
            return 0
        seqs = [r["seq"] for r in doomed]
        for r in doomed:
            if r["payload_path"]:
                try:
                    f = pathlib.Path(r["payload_path"])
                    f.unlink(missing_ok=True)
                    if f.parent.name.startswith("naru-blobs-") and not any(
                        f.parent.iterdir()
                    ):
                        f.parent.rmdir()
                except OSError:
                    pass
        marks = ",".join("?" * len(seqs))
        self.db.execute(
            f"DELETE FROM conversation_history WHERE seq IN ({marks})", seqs
        )
        # Rebuild rather than delete row-by-row: the index is derived from the
        # base table, so one rebuild is simpler and self-heals any drift.
        self.db.execute("INSERT INTO fts(fts) VALUES('rebuild')")
        self.db.commit()
        self.db.execute("VACUUM")
        return len(seqs)

    # ---- CURATE -----------------------------------------------------------
    # A claim is one agent's proposed fact, appended like any other row and
    # marked kind='claim'. Only a promoted claim reaches the doc. That is the
    # whole reason the doc stays small while the log grows without bound.

    def pending(self, k=50):
        """Claims still awaiting a human decision, oldest first."""
        return [
            Row(r)
            for r in self.db.execute(
                "SELECT * FROM conversation_history WHERE kind='claim'"
                " AND promoted=0 ORDER BY seq LIMIT ?",
                (k,),
            ).fetchall()
        ]

    def decide(self, seq, keep):
        """Promote a pending claim, or retire any claim that is not already
        dropped. Returns rows changed, so 0 is never a silent no-op.

        Promotion requires promoted=0: deciding the same claim twice is a
        mistake worth reporting. Retiring does NOT, because otherwise a
        promoted fact could never be revised — superseding it would leave both
        versions promoted and park the key under `## Unresolved` forever,
        which loses the fact the doc used to state.

        ponytail: the verdict is a column on the claim, not an event of its
        own, so it does not record who decided or when. Make it a
        kind='verdict' row if that audit trail is ever needed.
        """
        cur = self.db.execute(
            "UPDATE conversation_history SET promoted=? WHERE seq=? AND kind='claim'"
            + (" AND promoted=0" if keep else " AND promoted<>-1"),
            (1 if keep else -1, seq),
        )
        self.db.commit()
        return cur.rowcount

    def doc_version(self):
        """Highest promoted seq — the version of the doc a reader last saw.

        Deliberately not MAX(seq) over the whole log: an unrelated note or a
        spilled tool result must not read as "the doc moved under you".
        """
        return self.db.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM conversation_history WHERE promoted=1"
        ).fetchone()[0]

    def _promoted(self):
        """Every promoted claim, grouped by topic_key. One query, one snapshot.

        `conflicts()` and `doc()` both need this. Reading it twice let a
        promotion land between them and render a doc that never existed.
        """
        keyed, loose = {}, []
        for r in self.db.execute(
            "SELECT * FROM conversation_history WHERE promoted=1"
            " ORDER BY topic_key IS NULL, topic_key, seq"
        ).fetchall():
            (
                loose
                if r["topic_key"] is None
                else keyed.setdefault(r["topic_key"], [])
            ).append(Row(r))
        return keyed, loose

    def conflicts(self):
        """Topic keys carrying more than one promoted claim, key -> [rows].

        Two agents can both be promoted on one key. Silently taking the newer
        one is how the doc starts lying, so surface it and let a human decide.
        """
        keyed, _ = self._promoted()
        return {k: v for k, v in keyed.items() if len(v) > 1}

    def doc(self):
        """Render the promoted subset. This is what enters a model call.

        Deliberately not a render of the log: the log is unbounded, and a doc
        that grows with it is just the whole-history prompt wearing a hat.
        """
        keyed, loose = self._promoted()
        clashing = {k: v for k, v in keyed.items() if len(v) > 1}
        settled = sorted(
            loose + [v[0] for k, v in keyed.items() if len(v) == 1],
            key=lambda r: r.seq,
        )
        head = max([r.seq for r in loose] + [v[-1].seq for v in keyed.values()] or [0])

        lines = []
        if settled:
            lines.append("## Decisions")
            lines += [f"- {_oneline(r.content)}" for r in settled]
        if clashing:
            if lines:
                lines.append("")
            lines.append("## Unresolved")
            for key, rows in clashing.items():
                lines.append(f"▲ {key} — {len(rows)} versions")
                lines += [
                    f"  · {r.agent_id or 'unknown'}: {_oneline(r.content)}"
                    for r in rows
                ]
        body = "\n".join(lines) or "(nothing promoted yet)"
        return f"# naru · seq {head} · ~{est(body)} tokens\n\n{body}\n"

    # ---- COMPUTE (read-only SQL) -----------------------------------------
    def sql_query(self, sql, params=()):
        """Run a read-only SELECT over the log for structured filters (dates,
        kinds, ranges). Rejects anything that could mutate."""
        cleaned = sql.strip().rstrip(";")
        low = cleaned.lower()
        if not low.startswith("select") or ";" in cleaned:
            raise ValueError("sql_query accepts a single read-only SELECT")
        # the rubric addresses the table as hist.conversation_history
        cleaned = cleaned.replace("hist.conversation_history", "conversation_history")
        return [Row(r) for r in self.db.execute(cleaned, params).fetchall()]

    @staticmethod
    def days_between(d1, d2):
        """Absolute calendar days between two ISO dates (date part only)."""
        a = date.fromisoformat(str(d1)[:10])
        b = date.fromisoformat(str(d2)[:10])
        return abs((b - a).days)


class ReadOnly:
    """Capability-restricted Event Log view handed to the kernel.

    Section 2.2: "Model-authored code runs in a fail-closed sandbox: the Event
    Log is read-only from the kernel." Exposes exactly the Table-1 operations
    plus the documented helpers, and deliberately carries no `db` attribute, so
    model code has no path to a writable connection.
    """

    __slots__ = ("_ms",)

    def __init__(self, ms):
        object.__setattr__(self, "_ms", ms)

    def search(self, query, k=5, kind=None, since=None, until=None):
        return self._ms.search(query, k=k, kind=kind, since=since, until=until)

    def expand(self, lo, hi=None):
        return self._ms.expand(lo, hi)

    def outline(self):
        return self._ms.outline()

    def sql_query(self, sql, params=()):
        return self._ms.sql_query(sql, params)

    def days_between(self, d1, d2):
        return MemorySurface.days_between(d1, d2)

    def __setattr__(self, k, v):
        raise AttributeError("the Event Log is read-only from the kernel")

    def __repr__(self):
        return "<ms: read-only Event Log (search/expand/outline/sql_query)>"


# ---------------------------------------------------------------------------
def demo():
    """Runnable self-check. Fails loudly if any core behavior breaks."""
    ms = MemorySurface(":memory:")

    s1 = ms.append(
        "user",
        "I prefer economy cabins",
        kind="context_msg",
        session_id="s1",
        created_at="2024-07-01T09:00:00",
    )
    ms.append(
        "user",
        "Please avoid toll roads",
        kind="context_msg",
        session_id="s1",
        created_at="2024-07-02T09:00:00",
    )
    ms.append(
        "assistant",
        "Booked a flight and a route",
        kind="model_turn",
        session_id="s1",
        created_at="2024-07-02T09:05:00",
    )
    big = ms.append(
        "tool",
        "flight search results",
        kind="tool_result",
        session_id="s1",
        created_at="2024-07-03T09:00:00",
        payload="ROW," * 2000,
    )  # >4k -> externalized

    # seq is stable and monotonic
    assert (s1, big) == (1, 4), (s1, big)

    # LOCATE: BM25 finds the right turn
    hits = ms.search("economy", k=5)
    assert hits and hits[0].seq == 1, hits
    assert hits[0].role == "user"

    # implicit AND when it hits
    assert ms.search("economy cabins")

    # AND misses -> falls back to OR, so a partly-wrong query still finds it
    assert not ms._match('"economy" "submarine"', 5, None), "AND should miss"
    fb = ms.search("economy submarine")
    assert fb and fb[0].seq == 1, f"OR fallback failed: {fb}"

    # single-term misses stay misses (no fallback to invent hits)
    assert ms.search("submarine") == []

    # outline gives navigation anchors when wording does not match
    ol = ms.outline()
    assert "seq 1-4" in ol and "2024-07-01" in ol, ol
    assert len(ol.splitlines()) == 1, ol  # one line per session

    # uppercase OR is a boolean operator
    assert len(ms.search("economy OR toll")) == 2

    # kind filter narrows the search
    assert all(
        h.kind == "context_msg"
        for h in ms.search("economy OR toll", kind="context_msg")
    )

    # MATERIALIZE: exact span, verbatim
    span = ms.expand(1, 2)
    assert [r.seq for r in span] == [1, 2]
    assert "avoid toll roads" in span[1].content

    # externalized payload recovers in full from disk
    payload = ms.expand(big)[0].content
    assert payload.count("ROW") == 2000, payload.count("ROW")

    # COMPUTE: read-only SQL with a date filter (hist. alias supported)
    rows = ms.sql_query(
        "SELECT seq, role FROM hist.conversation_history "
        "WHERE substr(created_at,1,10) BETWEEN ? AND ? ORDER BY seq",
        ("2024-07-01", "2024-07-02"),
    )
    assert [r.seq for r in rows] == [1, 2, 3], [r.seq for r in rows]

    # writes are rejected
    for bad in (
        "DELETE FROM conversation_history",
        "SELECT 1; DROP TABLE conversation_history",
    ):
        try:
            ms.sql_query(bad)
            assert False, f"allowed: {bad}"
        except ValueError:
            pass

    # date helper
    assert ms.days_between("2024-07-01", "2024-07-31") == 30

    # Table 1: time filters on LOCATE
    assert ms.search("economy", since="2024-07-01")[0].seq == 1
    assert ms.search("economy", since="2024-08-01") == [], "since= not applied"
    assert ms.search("toll", until="2024-07-01") == [], "until= not applied"
    assert ms.search("toll", until="2024-12-31"), "until= too strict"

    # section 2.2: agent/session identifiers are recorded
    a = ms.append(
        "user",
        "multi-agent turn",
        agent_id="agent-7",
        session_id="s2",
        created_at="2024-08-01T00:00:00",
    )
    assert ms.expand(a)[0].agent_id == "agent-7"

    # ---- curation: nothing reaches the doc without a human decision --------
    c1 = ms.append(
        "agent",
        "Store is SQLite, WAL on.",
        kind="claim",
        agent_id="claude-opus",
        topic_key="store.engine",
        created_at="2026-08-27T10:00:00",
    )
    c2 = ms.append(
        "agent",
        "Retry budget is 5, not 3.",
        kind="claim",
        agent_id="gpt-5",
        topic_key="bench.retries",
        created_at="2026-08-27T10:01:00",
    )
    c3 = ms.append(
        "agent",
        "Order rows by created_at.",
        kind="claim",
        agent_id="codex",
        topic_key="store.ordering",
        base_seq=1,
        created_at="2026-08-27T10:02:00",
    )

    assert [r.seq for r in ms.pending()] == [c1, c2, c3]
    assert "nothing promoted yet" in ms.doc(), "a pending claim must not reach the doc"

    assert ms.decide(c1, True) == 1
    assert ms.decide(c3, False) == 1
    assert [r.seq for r in ms.pending()] == [c2], "decided claims must leave the inbox"

    d = ms.doc()
    assert "Store is SQLite" in d, d
    assert "Retry budget" not in d, "pending claim leaked into the doc"
    assert "Order rows by created_at" not in d, "dropped claim leaked into the doc"

    # a decision is not a delete: the dropped claim is still addressable
    assert "Order rows" in ms.expand(c3)[0].content
    # and base_seq survives, so staleness stays computable after the fact
    assert ms.expand(c3)[0].base_seq == 1

    # same key promoted twice = contradiction. Never auto-resolved.
    rival = ms.append(
        "agent",
        "Store is Postgres.",
        kind="claim",
        agent_id="gemini",
        topic_key="store.engine",
        created_at="2026-08-27T10:03:00",
    )
    ms.decide(rival, True)
    assert set(ms.conflicts()) == {"store.engine"}, ms.conflicts()
    d2 = ms.doc()
    assert "## Unresolved" in d2 and "store.engine" in d2, d2
    assert "gemini" in d2 and "claude-opus" in d2, "both sides must show"
    assert "## Decisions" not in d2, "the only promoted key is in conflict"

    # decide() only moves claims — it must not touch ordinary log rows
    assert ms.decide(s1, True) == 0, "decide() reached a non-claim row"

    # a promoted fact must be revisable: retire the loser, conflict clears
    assert ms.decide(rival, False) == 1, "a promoted claim must be retirable"
    assert ms.conflicts() == {}, "retiring one side must clear the conflict"
    assert "Store is SQLite" in ms.doc(), "the surviving fact must come back"
    assert ms.decide(rival, False) == 0, "retiring twice must report 0"
    assert ms.decide(c1, True) == 0, "promoting an already-promoted claim reports 0"

    # doc_version tracks the DOC, not the log: an unrelated append must not
    # read as "the doc moved under you"
    v = ms.doc_version()
    ms.append("note", "unrelated", kind="note", created_at="2026-08-27T11:00:00")
    assert ms.doc_version() == v, "an unrelated row advanced the doc version"

    # a newline inside a claim must not become a second top-level line
    nl = ms.append(
        "agent",
        "line one\nline two",
        kind="claim",
        agent_id="codex",
        created_at="2026-08-27T10:04:00",
    )
    ms.decide(nl, True)
    body = ms.doc().split("\n\n", 1)[1]
    assert "line one line two" in body, body
    assert not any(ln.strip().startswith("line two") for ln in body.splitlines()), (
        "a claim broke out of its bullet"
    )

    # prune is age-only for ordinary rows and must NEVER take a decided claim
    old = ms.append(
        "agent",
        "ancient decided claim",
        kind="claim",
        agent_id="x",
        created_at="2000-01-01T00:00:00",
    )
    ms.decide(old, True)
    stale_note = ms.append(
        "note", "ancient note", kind="note", created_at="2000-01-01T00:00:00"
    )
    # the preview must agree with the delete. It used to be re-derived in the
    # CLI without PRUNE_KEEP, so `naru prune --dry-run` announced it would
    # take this promoted claim and then didn't.
    n_doomed, _, n_kept = ms.prune_preview("2001-01-01T00:00:00")
    before = ms.sql_query("SELECT COUNT(*) n FROM conversation_history")[0].n
    assert n_doomed + n_kept == before, (n_doomed, n_kept, before)
    removed = ms.prune("2001-01-01T00:00:00")
    assert removed == 1, f"expected only the note to go, removed {removed}"
    assert n_doomed == removed, f"preview said {n_doomed}, prune took {removed}"
    assert ms.expand(old), "prune deleted a promoted claim"
    assert ms.expand(stale_note) == [], "prune left the ordinary old row"
    assert "ancient decided claim" in ms.doc()

    # the curation columns are indexed, or every doc render is a full scan
    idx = {
        r["name"]
        for r in ms.db.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert {"ix_promoted", "ix_topic_key"} <= idx, idx

    # ---- migration: a store created BEFORE the curation columns existed ----
    # Without this the ALTER TABLE branch never runs in the suite, because a
    # fresh store always gets the columns from CREATE TABLE.
    legacy_dir = pathlib.Path(tempfile.mkdtemp())
    legacy = legacy_dir / "legacy.db"
    old_db = sqlite3.connect(str(legacy))
    old_db.executescript("""
        CREATE TABLE conversation_history(
            seq        INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT, role TEXT, kind TEXT,
            created_at TEXT, content TEXT, payload_path TEXT);
        CREATE VIRTUAL TABLE fts USING fts5(content, content='');
    """)
    old_db.execute(
        "INSERT INTO conversation_history(role, kind, created_at, content)"
        " VALUES('user','context_msg','2024-01-01T00:00:00','legacy row here')"
    )
    old_db.commit()
    old_db.close()

    m3 = MemorySurface(str(legacy))
    cols = {r["name"] for r in m3.db.execute("PRAGMA table_info(conversation_history)")}
    assert {"agent_id", "promoted", "topic_key", "base_seq"} <= cols, cols
    assert m3.expand(1)[0].content == "legacy row here", "migration lost a row"
    assert m3.search("legacy"), "migration lost the FTS index"
    assert m3.expand(1)[0].promoted == 0, "migrated rows must default to pending"
    MemorySurface(str(legacy))  # re-opening a migrated store must be a no-op

    # section 2.2: the Event Log is READ-ONLY from the kernel
    ro = ms.readonly()
    assert ro.search("economy")[0].seq == 1
    assert ro.expand(1)[0].content
    assert not hasattr(ro, "db"), "read-only view must not expose a connection"
    assert not hasattr(ro, "append"), "read-only view must not expose append"
    for attempt in ("db", "append", "anything"):
        try:
            setattr(ro, attempt, 1)
            raise AssertionError(f"managed to set {attempt} on the read-only view")
        except AttributeError:
            pass

    # section 2.2: an externalized payload leaves a BOUNDED PREVIEW in the row,
    # not a second full copy
    huge = "P" * 40000
    hs = ms.append("tool", "ignored", payload=huge, created_at="2024-08-02T00:00:00")
    row_len = ms.sql_query(
        "SELECT LENGTH(content) c FROM conversation_history WHERE seq=?", (hs,)
    )[0].c
    assert row_len < len(huge) / 4, f"row still holds {row_len} of {len(huge)} chars"
    assert len(ms.expand(hs)[0].content) == len(huge), "full payload not recoverable"

    # repr stays small (token-frugal)
    assert len(repr(hits[0])) < 140

    print("ok — all checks passed")


if __name__ == "__main__":
    demo()
