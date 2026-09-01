# 0009. Load Naru through Codex hooks

## Context

Codex does not reload an injected `AGENTS.md` block after a Naru promotion.

## Decision

Ship a thin plugin that calls `naru codex-hook` at session start, on user prompts,
and for subagents; hidden state stores the session ID and doc fingerprint.

## Why

Native hooks keep approved context current without a second store; omit
`PostToolUse` because Codex marks a completed replacement as blocked and may
repeat its side effect.
