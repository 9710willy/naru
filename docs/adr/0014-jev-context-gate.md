# 0014. Keep Jev at the mutable-context gate

## Context

Naru is the memory substrate. The Event Log, kernel, indexes, and recovery
pointers help an agent find evidence. They are not a general observability or
traceability system.

The agent-cost-efficiency plan needs a context decision at the point where a
large mutable working view is reduced. That decision must be measurable,
bounded, privacy-aware, and safe when its evaluator is unavailable.

## Decision

Represent the mutable working view as named blocks with stable IDs, source
types, token estimates, provenance, and required/optional status. Keep the
resident kernel digest and the compact trace-recovery index outside that
mutable budget. Required blocks and recovery handles always survive the
deterministic fitter. Optional blocks are dropped only when the budget
requires it.

Use Jev only as an opt-in checkpoint evaluator at budget pressure or after a
large tool result. Jev is the fast path for these typed decisions, not a text
model call. Send it the task goal, next action, and block metadata. Do not send
block bodies. Ask typed `choice` questions with `needed`, `useful`,
`omit`, and `no_match` criteria. Apply a retention or omission recommendation
only at confidence 0.80 or higher. A confident `needed` block becomes required
for the final fit. The deterministic fitter still enforces the final budget.

Keep the Jev request itself small. Naru estimates the serialized state and
question map and caps it at 800 input tokens. When the cap is exceeded, it
skips Jev and records `jev_input_too_large` before using the deterministic
fitter. This is a safety boundary for Jev's smaller context window, not a
claim about the provider's internal limit.

`shadow` mode invokes Jev and records the recommendation without changing
delivery. `jev` mode applies high-confidence retention and omission
recommendations. A missing, malformed, timed-out, or low-confidence response
uses deterministic selection.
The preferred host integration wraps a direct MCP callback in `DirectJev`, so
there is no transport process and no Jev key in Naru. The standalone
`CommandJev` adapter uses a persistent process and auto-detects the installed
`jev mcp` command. For that command it performs one MCP initialize handshake
and sends `tools/call` JSON-RPC requests over newline-delimited stdio,
amortizing startup across decisions. Custom persistent JSON-lines adapters
remain supported, as do one-shot commands for compatibility. Both paths accept
the direct JSON shape of `mcp__jev__evaluate` or an MCP text/structured-content
wrapper. Only the standalone child-process path needs provider authentication in
its own environment. The host stdio bridge has a bounded response wait and the
benchmark keeps model jobs in a parent-owned thread pool, so only the shared
host exchange is serialized.

Record only numeric counts, block IDs, bounded latency, provider-native usage,
transport, transport overhead, startup latency, and failure classes. Optional
agent context metrics go to JSONL outside the Event Log. They can be exported to
OpenTelemetry and viewed in a self-hosted
Phoenix, Langfuse, or OpenLIT deployment. Naru does not add a vendor SDK or a
second memory system.

## Why

This keeps memory retrieval and execution observability on separate paths. A
Jev failure cannot lose evidence, and an observability record cannot become
evidence that a later search retrieves. The default remains the existing
deterministic behavior, which gives controlled `deterministic`, `shadow`, and
`jev` benchmark arms without claiming savings before a paired measurement.

The benchmark records estimated prompt input, uncached input, cache reads and
writes, output, retries, latency, task success, task cost, Jev input/cache/output
usage, and whether Jev was priced. Unknown provider pricing remains unknown
rather than being hardcoded.

The controlled proof is completed by running deterministic and Jev policies on
the same split and comparing their saved rows with `experiment.py`. The
comparator rejects different question sets or fixed run settings, pairs task
success by question, reports McNemar's exact test, and applies explicit quality,
failure, latency, and cost-per-success guardrails. Missing provider cost or
Jev pricing produces `defer`; it cannot produce a savings claim. Jev rates are
passed at measurement time and are recorded with a caller-supplied pricing
version. The built-in profile records $0.042 per million Jev input tokens and
free output. Cache rates remain unknown until Jev reports and documents a
cache billing category.
