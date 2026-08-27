# 0004 — Rename to naru, and add a curation layer above the paper

The project is renamed from Scroll to naru, and grows a layer the paper does
not have: agents write claims, a human promotes them, and only promoted claims
reach the doc that enters a model call.

## Context

The repo was called Scroll because it implements the system in arXiv 2608.21690
— but the paper's own system is named Scroll. One name meant two things, and
`bench.py:72` quotes §3.3 verbatim using the paper's meaning. A reader could not
tell which was which.

The paper solves one agent's context inside one trajectory. It does not solve
what happens when several sessions, several models and a human all need to
agree on the same set of facts. DeepSeek Harness (dsh) has the same append-only
log idea and stops at the same place: each session gets its own log, the
Trajectory view renders it, and nothing distills across sessions.

Rendering a log better does not scale — a tree viewer gets worse as agent count
rises. Reducing it does.

## Decision

**Rename to naru** (나루, "ferry landing" — where crossings meet). The paper's
Scroll keeps its name inside citations; everything that is ours is naru.

**Add three columns, not a framework:**

- `promoted` — pending / promoted / dropped, on the claim row itself
- `topic_key` — same key promoted twice is a contradiction, surfaced not resolved
- `base_seq` — the log head the author wrote against, so staleness is visible

**The CLI is the integration.** `naru inject <path>` writes the doc into
whatever context file a harness already reads. No per-harness plugin API, no
adapter per host, and no dependency on any one harness's runtime.

**The doc is a promoted subset, never a render of the log.** A doc that grows
with the log is the whole-history prompt again, which is the thing the paper
exists to avoid.

## Why

- One name for one thing. The rename removes a genuine ambiguity in the
  citations, not just a branding preference.
- A plugin framework (Cordis, dsh) buys hot-swap for a long-lived process.
  This is short-lived CLI invocations and hooks — there is no warm process to
  preserve, so the framework would be cost without benefit.
- Building _on_ dsh would cap reach at dsh's users. The cross-model story is
  the reason the idea is interesting, so the store stays standalone and dsh
  becomes one adapter among several.
- Promotion is a human decision by design. Auto-promotion would make the doc
  unauditable, and a doc nobody trusts is worse than no doc.

## Deliberate ceilings

- The verdict is a column on the claim, not its own event, so there is no
  record of who promoted or when. Make it a `kind='verdict'` row if an audit
  trail is ever needed.
- Conflict detection is exact-match on `topic_key`. Contradictions filed under
  different keys are invisible.
- One database. Two machines breaks `seq` as a total order, which is a
  distributed-consensus problem this repo deliberately does not enter.
