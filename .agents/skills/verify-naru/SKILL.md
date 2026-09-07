---
name: verify-naru
description: "Drive Naru's command-line curation, injection, recall, and maintenance flows with an isolated SQLite store. Use after changes to naru.py, ms.py, CLI storage, or user-visible command behavior."
---

# Verify Naru

Naru's main user surface is the short-lived CLI in `naru.py`. The Codex hooks,
Claude Code spill hook, model agent, and benchmark are other surfaces. This
skill covers the CLI. Run their module self-checks when a change touches them.

## Launch

Run from the repository root. This creates one state folder and one separate
evidence folder. The state folder includes its own SQLite database, metrics
file, and temp folder. Two runs can work side by side when each uses this block.

```bash
set -eu
VERIFY_NARU_HOST_TMP="${TMPDIR:-/tmp}"
VERIFY_NARU_STATE=$(mktemp -d "${VERIFY_NARU_HOST_TMP%/}/verify-naru-state.XXXXXX")
VERIFY_NARU_EVIDENCE=$(mktemp -d "${VERIFY_NARU_HOST_TMP%/}/verify-naru-evidence.XXXXXX")
mkdir -p "$VERIFY_NARU_STATE/tmp"
export VERIFY_NARU_HOST_TMP VERIFY_NARU_STATE VERIFY_NARU_EVIDENCE
export NARU_DB="$VERIFY_NARU_STATE/log.db"
export NARU_NOTES="$NARU_DB"
export NARU_METRICS="$VERIFY_NARU_STATE/metrics.jsonl"
export TMPDIR="$VERIFY_NARU_STATE/tmp"

./.agents/skills/verify-naru/capture.py launch -- python3 naru.py --help
rg -q "naru claim" "$VERIFY_NARU_EVIDENCE/launch.terminal.txt"
rg -q "naru inject" "$VERIFY_NARU_EVIDENCE/launch.terminal.txt"
printf 'ready - state=%s evidence=%s\n' "$VERIFY_NARU_STATE" "$VERIFY_NARU_EVIDENCE"
```

The CLI is ready when the block exits `0` and prints `ready`. There is no server
or long-running process. Use the Cleanup block when the drive ends.

## Doctor

Run this read-only check before the first drive and when output looks wrong:

```bash
./.agents/skills/verify-naru/capture.py doctor -- \
  ./.agents/skills/verify-naru/doctor.py
```

Require four `ok` lines. The check rejects the normal `~/.naru/log.db`, checks
that the database, metrics, and temp paths belong to this run, tests SQLite
FTS5 in memory, checks top-level help, and reads an existing database with
SQLite read-only mode.

## Drive

Read `features/README.md`, then use the matching feature file. Start each
feature from the empty state made by Launch.

Run every user command through the PTY capture helper:

```bash
./.agents/skills/verify-naru/capture.py claim -- \
  python3 naru.py claim "Verification fact" --key verify-fact --by verify-naru

./.agents/skills/verify-naru/capture.py --input promote inbox -- \
  python3 naru.py inbox
```

Use real CLI commands. Do not call Python functions in `naru.py` or `ms.py` to
stand in for the user path. Naru has hand-written argument parsing. Only
`python3 naru.py --help` is help. A command such as
`python3 naru.py inject --help` runs `inject` and treats `--help` as a path.

## Evidence

Proof stays in `$VERIFY_NARU_EVIDENCE`. Each capture writes:

- `<name>.command.txt`: the exact command.
- `<name>.input.txt`: terminal input, when supplied.
- `<name>.terminal.txt`: stdout and stderr from the PTY.
- `<name>.exit.txt`: the real exit code.

Exercise the real user path. Capture the action and the result. After a write,
use another CLI command to read the new state and run Doctor again for the row
counts. For `inject`, read the target file too. For maintenance, compare Doctor
counts before and after. Mocks do not prove this CLI. A dry run passes only when
the later Doctor output shows that rows did not change.

Do not pass passwords or tokens through `capture.py --input`. The helper saves
that text as evidence.

## Cleanup

The CLI leaves no process running. Remove only the state folder made by Launch.
Keep the evidence folder.

```bash
case "$VERIFY_NARU_STATE" in
  "${VERIFY_NARU_HOST_TMP%/}"/verify-naru-state.*) ;;
  *) printf 'refusing cleanup of %s\n' "$VERIFY_NARU_STATE" >&2; exit 2 ;;
esac

export TMPDIR="$VERIFY_NARU_HOST_TMP"
rm -rf -- "$VERIFY_NARU_STATE"
test ! -e "$VERIFY_NARU_STATE"
test -d "$VERIFY_NARU_EVIDENCE"
unset NARU_DB NARU_NOTES NARU_METRICS
printf 'clean - evidence=%s\n' "$VERIFY_NARU_EVIDENCE"
```

After cleanup, require `clean` and confirm the named evidence folder still
contains the command, terminal, and exit files.

## Helpers

`capture.py` runs a command in a pseudo-terminal, streams its output, records
the command and optional input, and returns the child exit code:

```bash
./.agents/skills/verify-naru/capture.py [--input TEXT] NAME -- COMMAND [ARGS...]
```

`doctor.py` performs the read-only checks described above:

```bash
./.agents/skills/verify-naru/doctor.py
```

Both helpers use the Python standard library and are executable.
