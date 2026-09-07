#!/usr/bin/env python3
"""Read-only readiness check for an isolated Naru CLI run."""

import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile


def stop(message):
    raise SystemExit(f"not ready: {message}")


def main():
    root = pathlib.Path(__file__).resolve().parents[3]
    required = ("VERIFY_NARU_STATE", "NARU_DB", "NARU_NOTES", "NARU_METRICS")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        stop("missing " + ", ".join(missing))

    state = pathlib.Path(os.environ["VERIFY_NARU_STATE"]).resolve()
    db = pathlib.Path(os.environ["NARU_NOTES"]).resolve()
    metrics = pathlib.Path(os.environ["NARU_METRICS"]).resolve()
    run_tmp = pathlib.Path(tempfile.gettempdir()).resolve()

    if pathlib.Path(os.environ["NARU_DB"]).resolve() != db:
        stop("NARU_DB and NARU_NOTES name different stores")
    if db.parent != state:
        stop("the database is outside VERIFY_NARU_STATE")
    if metrics.parent != state:
        stop("the metrics file is outside VERIFY_NARU_STATE")
    if run_tmp != (state / "tmp").resolve():
        stop("TMPDIR is not the run-only temp folder")
    if db == (pathlib.Path.home() / ".naru" / "log.db").resolve():
        stop("the database is the normal user store")
    if sys.version_info < (3, 9):
        stop("Python 3.9 or newer is required")

    try:
        with sqlite3.connect(":memory:") as probe:
            probe.execute("CREATE VIRTUAL TABLE fts_probe USING fts5(body)")
    except sqlite3.OperationalError:
        stop("SQLite FTS5 is unavailable")

    help_run = subprocess.run(
        [sys.executable, str(root / "naru.py"), "--help"],
        text=True,
        capture_output=True,
    )
    if help_run.returncode:
        stop(f"naru.py --help exited {help_run.returncode}")
    for command in ("claim", "inbox", "inject", "search", "gc"):
        if f"naru {command}" not in help_run.stdout:
            stop(f"top-level help does not list {command}")

    counts = (0, 0, 0, 0)
    if db.exists():
        try:
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as current:
                if current.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    stop("SQLite integrity_check failed")
                row = current.execute(
                    "SELECT COUNT(*),"
                    " COALESCE(SUM(kind IN ('claim','skill') AND promoted=0), 0),"
                    " COALESCE(SUM(kind IN ('claim','skill') AND promoted=1), 0),"
                    " COALESCE(SUM(kind IN ('claim','skill') AND promoted=-1), 0)"
                    " FROM conversation_history"
                ).fetchone()
                counts = tuple(row)
        except sqlite3.Error as exc:
            stop(f"cannot read the isolated store: {exc}")

    print(f"ok - repo={root}")
    print(f"ok - python={sys.version_info.major}.{sys.version_info.minor} fts5=yes")
    print(f"ok - state={state} tmp={run_tmp}")
    print(
        f"ok - rows={counts[0]} pending={counts[1]} "
        f"promoted={counts[2]} dropped={counts[3]}"
    )


if __name__ == "__main__":
    main()
