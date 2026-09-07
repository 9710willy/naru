# Naru verification map

This directory is the maintained source for proving Naru's user-facing CLI.
Read this index first, then use one feature file from a fresh verification run.

## Baseline preconditions

- Run the Launch block in `../SKILL.md` from the repository root.
- Keep `NARU_DB`, `NARU_NOTES`, `NARU_METRICS`, and `TMPDIR` on the run-only paths.
- Run `../doctor.py` and require four `ok` lines.
- Start each feature with an empty database unless its file says otherwise.
- Never drive the normal store at `~/.naru/log.db`.

## Driving conventions

- Run user commands through `../capture.py`.
- Treat commands, text, keys, and sequence numbers as literal.
- Use `--input` for `naru inbox` choices.
- Use a second CLI view after each write.
- Run Doctor after mutations to record row counts and SQLite health.
- Clean the state before starting another feature. Keep the evidence.

## Proof and skip reporting

- Keep the command, terminal output, input, and exit code for every action.
- A claim or skill passes only when `inject` shows promoted content and omits dropped content.
- An injected file passes only when the host text and one marker block remain.
- Recall passes only when `search`, `outline`, and `show` reach the stored text.
- Maintenance passes only when Doctor proves the expected row count change.
- Name an entry point as skipped when its exact command was not run.

## Feature entry contract

Each feature file describes the behavior from the user's view. Its drive section
starts with preconditions, gives literal commands, and names the visible result.
Do not replace a listed entry point with an internal Python call.

## Features

- [Decide claims](./decide-claims.md) covers proposing, promoting, dropping, skipping, and direct decisions.
- [Curate a skill](./curate-skill.md) covers source-backed procedure proposals and promotion.
- [Inject context](./inject-context.md) covers stdout, file splicing, and repeat injection.
- [Recall history](./recall-history.md) covers add, search, outline, show, and a search miss.
- [Maintain the store](./maintain-store.md) covers stats, dry-run pruning, pruning, and isolated blob cleanup.
