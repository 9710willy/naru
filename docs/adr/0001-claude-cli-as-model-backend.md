# 1. Use the `claude` CLI as the model backend, not an API key

There is no `ANTHROPIC_API_KEY` on this machine, but the `claude` CLI is
installed and authenticated. `backend.py` shells out to it, so the benchmark
runs with zero credential setup.

## Context

Naru needs a plain text-completion endpoint: send a prompt, get one Python
cell back. We had no API key and did not want to add one, or add the
`anthropic` SDK (nothing else in the repo needs a dependency).

## Decision

`backend.py` runs `claude -p --output-format json` as a subprocess. Two flags
are load-bearing and non-obvious:

- `--allowed-tools ""` — an **empty allowlist removes every tool**. Without it
  the model tries to _call_ a real tool instead of emitting a code block, the
  run ends on `stop_reason: "tool_use"`, and every naru question fails with
  `is_error: true`. This cost a whole benchmark round to diagnose.
- `--exclude-dynamic-system-prompt-sections` plus `--system-prompt` — replaces
  the Claude Code persona so the model is not acting as a coding agent.

`--output-format json` gives real `usage` and `total_cost_usd` per call, which
is how the token comparison in `bench.py` is measured rather than estimated.

## Why

No key to manage, no SDK dependency, and honest token accounting for free. The
cost is a fixed CLI overhead per call (see ADR 0002) and a nondeterministic
empty reply that `Backend` retries.

## Consequences

Only `backend.py` knows about Claude. The contract is one function,
`(prompt, system) -> str`, so swapping in `codex exec`, an SDK call, or a local
endpoint touches nothing else.
