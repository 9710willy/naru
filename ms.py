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

import hashlib
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
BEGIN = "<!-- naru:begin -->"
END = "<!-- naru:end -->"

# Age is the right rule for spilled tool output and the wrong one for a
# decision a human made. Anything else a prune would take, it still takes.
PRUNE_KEEP = """ AND NOT (
    (kind IN ('claim','skill') AND promoted <> 0)
    OR EXISTS (
        SELECT 1 FROM conversation_history AS curated
        WHERE curated.kind IN ('claim','skill') AND curated.promoted = 1
          AND curated.source_run_id = conversation_history.session_id
          AND conversation_history.seq BETWEEN curated.source_seq_lo
                                           AND curated.source_seq_hi
    )
)"""

# Payloads over this go to disk. One value, no caller has ever varied it.
BLOB_THRESHOLD = 4000
# Session preview width in outline().
OUTLINE_PREVIEW = 90
_ALL_SESSIONS = object()


def _visible_kind(column="kind"):
    return f"({column} IS NULL OR {column} NOT GLOB 'agent_*')"


def _oneline(text):
    """Collapse a claim or skill to a single safe render line.

    The doc is written into files other tools parse — CLAUDE.md, AGENTS.md, a
    shell rc. A newline inside a claim or skill would otherwise break out of its bullet
    and become a top-level line in the host format, which is how agent-authored
    text turns into a directive the host executes.
    """
    return re.sub(r"\s+", " ", (text or "").replace(BEGIN, "<!-- naru: begin -->")
                  .replace(END, "<!-- naru: end -->").strip())


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
    @classmethod
    def open_readonly(cls, path):
        """A surface backed by a connection SQLite itself refuses to write.

        `ReadOnly` withholds a `db` attribute, which is a naming convention and
        not a boundary: model-authored code reached the live log through
        `ms._ms.db.execute("DELETE FROM conversation_history")` and emptied it.
        A `mode=ro` URI moves the refusal into SQLite, where no attribute walk
        gets around it.

        The schema is not created or migrated here — a reader must not write,
        and whoever opened the log for writing has already done it.
        """
        m = cls.__new__(cls)
        m.path = path
        m.db = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        m.db.row_factory = sqlite3.Row
        m.db.execute("PRAGMA busy_timeout=30000")
        m.blobs = pathlib.Path(tempfile.gettempdir()) / "naru-blobs-readonly"
        m._blob_tmp = None
        return m

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
                -- curation: a claim or skill is pending (0), promoted (1) or dropped (-1)
                promoted   INTEGER NOT NULL DEFAULT 0,
                topic_key  TEXT,          -- same key, both promoted = contradiction
                base_seq   INTEGER,       -- doc version the author wrote against
                source_run_id TEXT,
                source_seq_lo INTEGER,
                source_seq_hi INTEGER
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
                content, content='conversation_history', content_rowid='seq');
        """)
        self._migrate_fts()
        self._migrate_cols()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS naru_meta(
                id INTEGER PRIMARY KEY,
                store_uuid TEXT NOT NULL
            );
            INSERT OR IGNORE INTO naru_meta(id, store_uuid)
            VALUES(1, lower(hex(randomblob(16))));
        """)
        store_uuid = self.db.execute(
            "SELECT store_uuid FROM naru_meta WHERE id=1"
        ).fetchone()["store_uuid"]
        location = ":memory:" if db == ":memory:" else str(pathlib.Path(db).resolve())
        self.store_id = hashlib.sha256(f"{location}\0{store_uuid}".encode()).hexdigest()
        self.db.commit()
        self._blob_tmp = tempfile.TemporaryDirectory(prefix="naru-blobs-") \
            if db == ":memory:" and blobs is None else None
        self.blobs = pathlib.Path(
            self._blob_tmp.name if self._blob_tmp else blobs or
            pathlib.Path(tempfile.gettempdir()) / f"naru-blobs-{self.store_id}"
        )
        self.blob_threshold = BLOB_THRESHOLD

    def close(self):
        """Close the database and remove only owned temporary blobs."""
        self.db.close()
        if self._blob_tmp:
            self._blob_tmp.cleanup()
            self._blob_tmp = None

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
            ("source_run_id", "TEXT"),
            ("source_seq_lo", "INTEGER"),
            ("source_seq_hi", "INTEGER"),
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
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_promoted_sources"
            " ON conversation_history(source_run_id, source_seq_lo, source_seq_hi)"
            " WHERE kind IN ('claim','skill') AND promoted=1 AND source_run_id IS NOT NULL"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS ix_session_id"
            " ON conversation_history(session_id)"
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
        source_run_id=None,
        source_seq_lo=None,
        source_seq_hi=None,
    ):
        """Append one turn. Returns its stable `seq`. `created_at` is caller-
        supplied (the harness stamps it) so ingestion stays deterministic.

        A payload over the threshold is externalized to disk. The row is
        inserted FIRST so the blob is named by the real seq and the pointer we
        show the model cannot drift.
        """
        source = (source_run_id, source_seq_lo, source_seq_hi)
        if any(v is not None for v in source):
            if not (
                isinstance(source_run_id, str)
                and source_run_id
                and type(source_seq_lo) is int
                and type(source_seq_hi) is int
                and source_seq_lo <= source_seq_hi
            ):
                raise ValueError("source needs --run plus integer LO:HI")
            rows = self.db.execute(
                "SELECT seq FROM conversation_history WHERE session_id=? AND seq IN (?,?)",
                (source_run_id, source_seq_lo, source_seq_hi),
            ).fetchall()
            if {r["seq"] for r in rows} != {source_seq_lo, source_seq_hi}:
                raise ValueError("source endpoints are not in that run")
        big = payload is not None and len(payload) > self.blob_threshold
        cur = self.db.execute(
            "INSERT INTO conversation_history"
            "(session_id, agent_id, role, kind, created_at, content, payload_path,"
            " topic_key, base_seq, source_run_id, source_seq_lo, source_seq_hi)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
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
                source_run_id,
                source_seq_lo,
                source_seq_hi,
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
            scope = f", session_id={session_id!r}" if session_id is not None else ""
            content = (
                f"{preview}\n[payload {len(payload)} chars, preview only "
                f"-> ms.expand({seq}{scope})]"
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
        else:
            where.append(_visible_kind("c.kind"))
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
                " COUNT(*) n FROM conversation_history WHERE "
                + _visible_kind()
                + " GROUP BY session_id ORDER BY lo"
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
                " AND " + _visible_kind() + " ORDER BY (role != 'user'), seq LIMIT 1",
                (r["session_id"],),
            ).fetchone()
            head = (first["content"] if first else "").replace("\n", " ")
            head = head.split("] ", 1)[-1]  # drop the [Session N | date] tag
            recover = f"ms.expand({r['lo']}, {r['hi']}, session_id={r['session_id']!r})"
            out.append(
                f"seq {r['lo']}-{r['hi']} | {(r['at'] or '')[:10]} | "
                f"{r['n']} turns | {head[:OUTLINE_PREVIEW]} -> {recover}"
            )
        return "\n".join(out)

    # ---- MATERIALIZE ------------------------------------------------------
    def expand(self, lo, hi=None, session_id=_ALL_SESSIONS):
        """Recover an exact seq span, verbatim, as Row objects. Externalized
        payloads are inlined from disk. Omitted session_id returns normal-history
        rows; an explicit session_id returns that exact session, including trace rows."""
        hi = lo if hi is None else hi
        where = ["seq BETWEEN ? AND ?"]
        params = [lo, hi]
        if session_id is _ALL_SESSIONS:
            where.append(_visible_kind())
        else:
            where.append("session_id IS ?")
            params.append(session_id)
        rows = self.db.execute(
            "SELECT * FROM conversation_history WHERE " + " AND ".join(where)
            + " ORDER BY seq",
            params,
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

        A decided claim or skill, and the trace it cites, is never pruned by
        age. Age is the right rule for spilled tool output.

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

    def gc_blobs(self):
        """Delete unreferenced payload files owned by this Event Log."""
        if not self.blobs.is_dir():
            return 0, 0
        live = {
            r["payload_path"]
            for r in self.db.execute(
                "SELECT payload_path FROM conversation_history"
                " WHERE payload_path IS NOT NULL"
            )
        }
        removed = freed = 0
        for path in self.blobs.iterdir():
            if not path.is_file() or str(path) in live:
                continue
            try:
                size = path.stat().st_size
                path.unlink()
            except OSError:
                continue
            removed += 1
            freed += size
        try:
            self.blobs.rmdir()
        except OSError:
            pass
        return removed, freed

    def pending(self, k=50):
        """Claims and skills still awaiting a human decision, oldest first."""
        return [
            Row(r)
            for r in self.db.execute(
                "SELECT * FROM conversation_history WHERE kind IN ('claim','skill')"
                " AND promoted=0 ORDER BY seq LIMIT ?",
                (k,),
            ).fetchall()
        ]

    def decide(self, seq, keep):
        """Promote pending claims/skills; drop ones not already dropped.

        ponytail: verdicts are columns, not events; add kind='verdict' for who/when.
        """
        cur = self.db.execute(
            "UPDATE conversation_history SET promoted=? WHERE seq=?"
            " AND kind IN ('claim','skill')"
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
        """Return promoted claims and skills grouped by kind and topic key."""
        keyed, loose = {}, {"claim": [], "skill": []}
        for r in self.db.execute(
            "SELECT * FROM conversation_history WHERE promoted=1"
            " ORDER BY topic_key IS NULL, topic_key, seq"
        ).fetchall():
            (
                loose.setdefault(r["kind"], [])
                if r["topic_key"] is None
                else keyed.setdefault((r["kind"], r["topic_key"]), [])
            ).append(Row(r))
        return keyed, loose

    def conflicts(self):
        """Return promoted claim/skill kind-key conflicts."""
        keyed, _ = self._promoted()
        return {k: v for k, v in keyed.items() if len(v) > 1}

    def doc(self):
        """Render promoted claims and skills."""
        keyed, loose = self._promoted()
        clashing = {k: v for k, v in keyed.items() if len(v) > 1}
        settled = {
            kind: sorted(
                loose.get(kind, []) + [v[0] for (row_kind, _), v in keyed.items()
                                        if row_kind == kind and len(v) == 1],
                key=lambda r: r.seq,
            )
            for kind in ("claim", "skill")
        }
        head = max([r.seq for rows in loose.values() for r in rows]
                   + [v[-1].seq for v in keyed.values()] or [0])

        lines = []
        if settled["claim"]:
            lines.append("## Decisions")
            lines += [f"- {_oneline(r.content)}" for r in settled["claim"]]
        if settled["skill"]:
            if lines:
                lines.append("")
            lines.append("## Skills")
            lines += [f"- {_oneline(r.content)}" for r in settled["skill"]]
        if clashing:
            if lines:
                lines.append("")
            lines.append("## Unresolved")
            for (kind, key), rows in clashing.items():
                lines.append(f"▲ {kind} {key} — {len(rows)} versions")
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
    plus the documented helpers, and carries no `db` attribute.

    Withholding the attribute is a convention, not a boundary — `ms._ms.db`
    reaches the connection, and a cell used it to empty a live log. Under the
    in-process Kernel nothing can fix that: model code shares the interpreter
    and owns every object in it. The boundary lives one layer down, in
    `open_readonly()`, which hands the sandboxed child a connection SQLite
    itself refuses to write.
    """

    __slots__ = ("_ms",)

    def __init__(self, ms):
        object.__setattr__(self, "_ms", ms)

    def search(self, query, k=5, kind=None, since=None, until=None):
        return self._ms.search(query, k=k, kind=kind, since=since, until=until)

    def expand(self, lo, hi=None, session_id=_ALL_SESSIONS):
        return self._ms.expand(lo, hi, session_id)

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

    identity_dir = pathlib.Path(tempfile.mkdtemp())
    identity_path = identity_dir / "identity.db"
    first = MemorySurface(str(identity_path))
    first_id, first_blobs = first.store_id, first.blobs
    first.close()
    reopened = MemorySurface(str(identity_path))
    assert (reopened.store_id, reopened.blobs) == (first_id, first_blobs)
    reopened.close()
    identity_path.unlink()
    replaced = MemorySurface(str(identity_path))
    assert replaced.store_id != first_id, "a replacement DB reused the old identity"
    assert replaced.blobs != first_blobs, "a replacement DB reused the old blob root"
    replaced.close()

    owner_a = MemorySurface(str(identity_dir / "a.db"))
    owner_b = MemorySurface(str(identity_dir / "b.db"))
    seq_a = owner_a.append("tool", "preview", payload="A" * 5000)
    payload_a = pathlib.Path(owner_a.sql_query(
        "SELECT payload_path FROM conversation_history WHERE seq=?", (seq_a,)
    )[0].payload_path)
    seq_b = owner_b.append("tool", "preview", payload="B" * 5000)
    payload_b = pathlib.Path(owner_b.sql_query(
        "SELECT payload_path FROM conversation_history WHERE seq=?", (seq_b,)
    )[0].payload_path)
    owner_a.blobs.mkdir(parents=True, exist_ok=True)
    orphan_a = owner_a.blobs / "orphan.txt"
    orphan_a.write_text("orphan")
    assert owner_a.gc_blobs() == (1, 6)
    assert not orphan_a.exists() and payload_a.exists() and payload_b.exists(), (
        "gc removed a live or foreign payload"
    )
    assert owner_a.gc_blobs() == (0, 0), "gc must be idempotent"
    payload_a.unlink()
    owner_a.blobs.rmdir()
    payload_b.unlink()
    owner_b.blobs.rmdir()
    owner_a.close()
    owner_b.close()

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

    trace = ms.append(
        "agent",
        "trace-only needle",
        kind="agent_reply",
        session_id="trace",
        created_at="2024-07-04T09:00:00",
    )
    trace_obs = ms.append(
        "tool",
        "trace-observation needle",
        kind="agent_observation",
        session_id="trace",
        created_at="2024-07-04T09:01:00",
    )
    trace_payload = "trace-payload," * 400
    trace_big = ms.append(
        "tool",
        "trace observation preview",
        kind="agent_observation",
        session_id="trace",
        created_at="2024-07-04T09:02:00",
        payload=trace_payload,
    )
    assert ms.search("trace-only") == [], "agent trace leaked into default search"
    assert ms.search("trace-observation") == [], "agent observation leaked into search"
    assert ms.search("trace-only", kind="agent_reply")[0].seq == trace
    assert ms.search("trace-observation", kind="agent_observation")[0].seq == trace_obs
    assert "trace-only" not in ms.outline(), "agent trace leaked into outline"
    assert ms.expand(trace, trace_obs) == [], "agent trace leaked into default expand"
    assert [r.seq for r in ms.expand(trace, trace_obs, session_id="trace")] == [
        trace,
        trace_obs,
    ]
    assert ms.sql_query(
        "SELECT seq FROM conversation_history WHERE kind='agent_reply'"
    )[0].seq == trace
    stored_trace = ms.sql_query(
        "SELECT content FROM conversation_history WHERE seq=?", (trace_big,)
    )[0].content
    assert f"ms.expand({trace_big}, session_id='trace')" in stored_trace
    assert ms.expand(trace_big, session_id="trace")[0].content == trace_payload
    state = ms.append(
        "agent", '{"task":"find","verified":[],"next_action":"answer","status":"working"}',
        kind="agent_state", session_id="trace", created_at="2024-07-04T09:03:00",
        source_run_id="trace", source_seq_lo=trace, source_seq_hi=trace_obs,
    )
    assert ms.expand(state) == [], "agent state leaked into default expand"
    assert ms.expand(state, session_id="trace")[0].source_seq_lo == trace
    for source in (("trace", trace, None), ("", trace, trace_obs), ("wrong", trace, trace_obs), ("trace", trace_obs, trace)):
        try:
            ms.append("agent", "bad source", source_run_id=source[0], source_seq_lo=source[1], source_seq_hi=source[2])
            assert False, source
        except ValueError:
            pass

    visible_lo = ms.append(
        "user", "visible before", kind="context_msg", session_id="s2",
        created_at="2024-07-05T09:00:00",
    )
    ms.append(
        "agent", "hidden reply", kind="agent_reply", session_id="run",
        created_at="2024-07-05T09:01:00",
    )
    ms.append(
        "tool", "hidden observation", kind="agent_observation", session_id="run",
        created_at="2024-07-05T09:02:00",
    )
    visible_hi = ms.append(
        "user", "visible after", kind="context_msg", session_id="s2",
        created_at="2024-07-05T09:03:00",
    )
    handle = f"ms.expand({visible_lo}, {visible_hi}, session_id='s2')"
    assert handle in ms.outline(), "outline did not scope its recovery handle"
    assert [r.seq for r in ms.expand(visible_lo, visible_hi)] == [visible_lo, visible_hi]
    visible = ms.expand(visible_lo, visible_hi, session_id="s2")
    assert [r.seq for r in visible] == [
        visible_lo,
        visible_hi,
    ]
    assert all(not r.kind.startswith("agent_") for r in visible)
    unscoped = ms.append("note", "unscoped row", kind="note")
    assert ms.expand(unscoped, session_id=None)[0].seq == unscoped

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
    assert set(ms.conflicts()) == {("claim", "store.engine")}, ms.conflicts()
    d2 = ms.doc()
    assert "## Unresolved" in d2 and "store.engine" in d2, d2
    assert "gemini" in d2 and "claude-opus" in d2, "both sides must show"
    assert "## Decisions" not in d2, "the only promoted key is in conflict"

    # decide() only moves curation rows — it must not touch ordinary log rows
    assert ms.decide(s1, True) == 0, "decide() reached a non-claim row"

    skill = ms.append(
        "agent", "Use the Event Log before answering.", kind="skill",
        topic_key="store.engine", created_at="2026-08-27T10:03:30",
    )
    assert skill in [row.seq for row in ms.pending()], "pending skill missing"
    assert ms.decide(skill, True) == 1
    assert "## Skills" in ms.doc() and "Use the Event Log" in ms.doc()
    assert ("claim", "store.engine") in ms.conflicts()
    assert ("skill", "store.engine") not in ms.conflicts()

    # a promoted fact must be revisable: retire the loser, conflict clears
    assert ms.decide(rival, False) == 1, "a promoted claim must be retirable"
    assert ms.conflicts() == {}, "retiring one side must clear the conflict"
    assert "Store is SQLite" in ms.doc(), "the surviving fact must come back"
    assert ms.decide(rival, False) == 0, "retiring twice must report 0"
    assert ms.decide(c1, True) == 0, "promoting an already-promoted claim reports 0"

    # dropped keys and bodies stay searchable without entering every prompt
    gone = ms.append("agent", "Retry budget was 5.", kind="claim", topic_key="retry.budget")
    ms.decide(gone, False)
    gone_skill = ms.append("agent", "Old procedure.", kind="skill", topic_key="retry.budget")
    ms.decide(gone_skill, False)
    d3 = ms.doc()
    assert "## Archive" not in d3, d3
    assert "retry.budget" not in d3 and "Retry budget" not in d3, d3
    assert ms.search("Retry budget")[0].seq == gone
    assert ms.search("Old procedure")[0].seq == gone_skill

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
    old_skill = ms.append(
        "agent", "ancient decided skill", kind="skill", agent_id="x",
        created_at="2000-01-01T00:00:00",
    )
    ms.decide(old_skill, True)
    evidence_run = "old-run"
    evidence_reply = ms.append(
        "agent", "old reply", kind="agent_reply", session_id=evidence_run,
        created_at="2000-01-01T00:00:00",
    )
    evidence_obs = ms.append(
        "tool", "old observation", kind="agent_observation", session_id=evidence_run,
        created_at="2000-01-01T00:00:01",
    )
    cited = ms.append(
        "agent", "claim with trace", kind="claim", topic_key="cited.trace",
        created_at="2026-08-27T10:05:00", source_run_id=evidence_run,
        source_seq_lo=evidence_reply, source_seq_hi=evidence_obs,
    )
    ms.decide(cited, True)
    dropped_run = "dropped-run"
    dropped_reply = ms.append(
        "agent", "dropped old reply", kind="agent_reply", session_id=dropped_run,
        created_at="2000-01-01T00:00:00",
    )
    dropped_obs = ms.append(
        "tool", "dropped old observation", kind="agent_observation", session_id=dropped_run,
        created_at="2000-01-01T00:00:01",
    )
    dropped_cited = ms.append(
        "agent", "dropped claim with trace", kind="claim", topic_key="dropped.trace",
        created_at="2026-08-27T10:05:00", source_run_id=dropped_run,
        source_seq_lo=dropped_reply, source_seq_hi=dropped_obs,
    )
    ms.decide(dropped_cited, False)
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
    assert removed == 3, f"expected note and dropped evidence to go, removed {removed}"
    assert n_doomed == removed, f"preview said {n_doomed}, prune took {removed}"
    assert ms.expand(old), "prune deleted a promoted claim"
    assert ms.expand(old_skill), "prune deleted a promoted skill"
    assert [row.seq for row in ms.expand(
        evidence_reply, evidence_obs, session_id=evidence_run
    )] == [evidence_reply, evidence_obs], "prune deleted promoted provenance"
    assert ms.expand(dropped_reply, dropped_obs, session_id=dropped_run) == [], (
        "prune kept dropped provenance"
    )
    assert ms.expand(stale_note) == [], "prune left the ordinary old row"
    assert "ancient decided claim" in ms.doc()

    idx = {
        r["name"]
        for r in ms.db.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    assert {"ix_promoted", "ix_topic_key", "ix_promoted_sources", "ix_session_id"} <= idx, idx
    plan = " ".join(r["detail"] for r in ms.db.execute(
        "EXPLAIN QUERY PLAN SELECT 1 FROM conversation_history AS curated "
        "WHERE curated.kind IN ('claim','skill') AND curated.promoted=1 "
        "AND curated.source_run_id=? AND ? BETWEEN curated.source_seq_lo "
        "AND curated.source_seq_hi", (evidence_run, evidence_reply)
    ))
    assert "ix_promoted_sources" in plan, plan

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
    assert {"agent_id", "promoted", "topic_key", "base_seq", "source_run_id", "source_seq_lo", "source_seq_hi"} <= cols, cols
    assert m3.expand(1)[0].content == "legacy row here", "migration lost a row"
    assert m3.search("legacy"), "migration lost the FTS index"
    assert m3.expand(1)[0].promoted == 0, "migrated rows must default to pending"
    outline_plan = " ".join(r["detail"] for r in m3.db.execute(
        "EXPLAIN QUERY PLAN SELECT content FROM conversation_history"
        " WHERE session_id IS ? AND (kind IS NULL OR kind NOT GLOB 'agent_*')"
        " ORDER BY (role != 'user'), seq LIMIT 1", (None,)
    ))
    assert "ix_session_id" in outline_plan, outline_plan
    m4 = MemorySurface(str(legacy))
    assert m4.store_id == m3.store_id, "migration changed the store identity"
    m4.close()

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

    owned_blobs = ms.blobs
    m3.close()
    ms.close()
    assert not owned_blobs.exists(), "close left owned temporary blobs"

    print("ok — all checks passed")


if __name__ == "__main__":
    demo()
