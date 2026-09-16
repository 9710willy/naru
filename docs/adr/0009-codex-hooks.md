# 0009. Load Naru through Codex hooks

## Context

Codex does not reload an injected `AGENTS.md` block after a Naru promotion.

## Decision

Ship a thin plugin that calls `naru codex-hook` at session start, on user prompts,
and for subagents. Store the last doc sequence and fingerprint per session in a
typed `context_delivery` table.

Older releases wrote those fingerprints as `kind='agent_state'` Event Log rows.
Copy a valid legacy row into the typed table when that Codex session next uses
the hook. Keep the old row as audit history. Write no new delivery fingerprints
to the Event Log.

## Why

Native hooks keep approved context current without a second store; omit
`PostToolUse` because Codex marks a completed replacement as blocked and may
repeat its side effect. A delivery fingerprint is harness metadata, not model
state or recallable history, so the typed table gives it one owner and one row
per session.
