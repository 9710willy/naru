# Scroll

Implementation of _Context as an Environment_ (arXiv 2608.21690). The agent's
history lives in a SQLite Event Log and a persistent Python kernel; the model
writes code to reach it and only what it prints enters the next call.

Read `docs/adr/` before changing anything — three decisions there are
non-obvious and one of them (0002) exists because a hardcoded constant produced
a wrong published result.

## Before you commit

Every module has a runnable self-check. The first four need no network:

```bash
python3 ms.py && python3 kernel.py && python3 eviction.py && python3 agent.py
python3 note.py --selfcheck && python3 hook_spill.py --selfcheck
python3 backend.py    # live: 2 cheap calls, prints the harness token floor
```

Stdlib only. No dependencies — do not add one for something a few lines cover.

## Benchmark rules

`bench.py` is the only source of numbers. Two arms (`full`, `scroll`) over the
same questions.

- **Never hardcode a measured constant.** `--harness-floor` is measured at
  startup for exactly this reason (ADR 0002).
- **Report a noise floor.** A single run's difference under ~20 points at n=12
  is noise. Say so rather than implying a result.
- **Both arms must see identical history.** `sessions(q)` is the one source of
  ordering; do not re-derive it.
- The judge is not LongMemEval's official prompt, so numbers are internal
  progress only — never present them as leaderboard-comparable.
- Multi-turn arms take more exposure to per-call flakiness than single-turn
  ones. A low retry budget silently biases against `scroll`.

## Gotchas that cost real time

- `claude -p` needs `--allowed-tools ""`; without it the model tries to call a
  tool and every scroll question errors on `stop_reason: tool_use` (ADR 0001).
- The CLI leaks its own identity and CLAUDE.md into the agent. The system
  prompt overrides this explicitly — do not remove that paragraph (ADR 0003).
- A `PostToolUse` `updatedToolOutput` that does not match the tool's own output
  schema is discarded **silently**. Mutate the text field in place inside the
  response object; never return a bare string.
- `note` and `hook_spill.py` must share one store (`ms.DEFAULT_DB`), or the
  recovery handle a spill prints points at a database `note` never opens.

## Where this diverges from the paper

Both are deliberate and ablatable; do not quietly drop either.

- `ms.outline()` is ours, not the paper's. It exists because §3.3's
  ingestion-time index collapses old sessions into coarse ranges, which suits a
  long trajectory but not LongMemEval's uniformly-scattered evidence. `--no-index`
  ablates the paper's version. See ADR 0003.
- `search()` retries a failed multi-term AND as OR. The paper does not.

## Style

Python: the `python-perf-guide` skill carries the standard. Its measurement
checklist applies directly to `bench.py`.
