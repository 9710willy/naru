"""ms — the memory surface for a Scroll-style Session Environment.

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

import pathlib
import sqlite3
import tempfile
from datetime import date


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
    def __init__(self, db=":memory:", blobs=None, blob_threshold=4000):
        self.db = sqlite3.connect(db)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _probe USING fts5(x)")
            self.db.execute("DROP TABLE _probe")
        except sqlite3.OperationalError as e:
            raise RuntimeError(
                "SQLite build lacks FTS5; cannot back the Event Log"
            ) from e

        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversation_history(
                seq        INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role       TEXT,
                kind       TEXT,
                created_at TEXT,          -- ISO-8601, lexically sortable
                content    TEXT,
                payload_path TEXT         -- externalized big payloads
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(content, content='');
        """)
        # per-instance by default: bench.py runs questions in parallel
        # threads and a shared dir would collide on blobs/<seq>.txt
        self.blobs = pathlib.Path(
            blobs or pathlib.Path(tempfile.gettempdir()) / f"scroll-blobs-{id(self):x}"
        )
        self.blob_threshold = blob_threshold

    # ---- ingest -----------------------------------------------------------
    def append(
        self,
        role,
        content,
        *,
        kind=None,
        session_id=None,
        created_at=None,
        payload=None,
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
            "(session_id, role, kind, created_at, content, payload_path)"
            " VALUES(?,?,?,?,?,?)",
            (session_id, role, kind, created_at, content, None),
        )
        seq = cur.lastrowid

        if big:
            self.blobs.mkdir(parents=True, exist_ok=True)
            path = str(self.blobs / f"{seq}.txt")
            pathlib.Path(path).write_text(payload)
            content = f"{content}\n[payload {len(payload)} chars -> ms.expand({seq})]"
            self.db.execute(
                "UPDATE conversation_history SET content=?, payload_path=? WHERE seq=?",
                (content, path, seq),
            )

        self.db.execute("INSERT INTO fts(rowid, content) VALUES(?,?)", (seq, content))
        self.db.commit()
        return seq

    # ---- LOCATE -----------------------------------------------------------
    def search(self, query, k=5, kind=None):
        """BM25 full-text search over the log. Returns hits ranked best-first,
        each a Row carrying the full turn plus its seq/role/metadata.

        Multi-term queries AND-combine. If that yields nothing, they are
        retried as OR — a question's wording often shares only some words with
        the turn that answers it, and a silent zero-hit AND is the single
        biggest source of lost evidence.
        """
        hits = self._match(_to_match(query), k, kind)
        terms = [t for t in query.split() if t not in ("OR", "AND", "NOT")]
        if not hits and len(terms) > 1:
            hits = self._match(" OR ".join(_to_match(t) for t in terms), k, kind)
        return hits

    def _match(self, match_expr, k, kind):
        where = ["fts MATCH ?"]
        params = [match_expr]
        if kind is not None:
            where.append("c.kind = ?")
            params.append(kind)
        params.append(k)
        sql = (
            f"SELECT c.* FROM fts JOIN conversation_history c ON c.seq = fts.rowid "
            f"WHERE {' AND '.join(where)} ORDER BY rank LIMIT ?"
        )
        return [Row(r) for r in self.db.execute(sql, params).fetchall()]

    def outline(self, preview=90):
        """Structural map of the log: one line per session with its date, seq
        range and the opening of its first user turn.

        Navigation anchors for when lexical search fails — the question's
        wording may share no words with the turn that answers it, and then the
        only way in is to browse. Cheap: one short line per session.
        """
        rows = self.db.execute(
            "SELECT session_id, MIN(seq) lo, MAX(seq) hi, MIN(created_at) at,"
            " COUNT(*) n FROM conversation_history"
            " GROUP BY session_id ORDER BY lo"
        ).fetchall()
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
                f"{r['n']} turns | {head[:preview]}"
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
            if row.get("payload_path"):
                row["content"] = pathlib.Path(row["payload_path"]).read_text()
            out.append(row)
        return out

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

    # repr stays small (token-frugal)
    assert len(repr(hits[0])) < 140

    print("ok — all checks passed")


if __name__ == "__main__":
    demo()
