# 3. `ms.outline()` was a workaround for skipping §3.3, not an addition to the paper

Supersedes the original text of this ADR, which claimed `outline()` and the OR
search fallback were "two deliberate additions to the paper's API". `outline()`
is not ours: §3.3 builds a landmark index during session-by-session ingestion,
and we had ingested all 47 sessions in one pass with an empty index, so that
navigation layer never existed to be used.

## Context

Implemented against Appendix C of [_Context as an
Environment_](https://arxiv.org/abs/2608.21690) alone (`search` / `expand` /
`sql_query` / `days_between`), three of twelve oracle questions failed on a
vocabulary gap pure BM25 cannot cross: `ms.search("homegrown")` returns 0 hits
when the log says "fresh basil and mint". `ms.outline()` — one line per session
— was added as a structural way in and mistakenly recorded as our own idea.

## Decision

Both mechanisms now exist and are separately ablatable.

1. **§3.3 ingestion** — `ingest()` walks sessions in order and enters each
   session's landmark into the tiered index at the boundary. 47 sessions
   become 9 index entries, 246 tokens. `--no-index` turns it off.
2. **`ms.outline()`** — one line per session, 1,444 tokens for the same
   history. Kept, because it is _not_ equivalent (see below).
3. **Auto-OR fallback in `search()`** — a genuine addition, and a small one: a
   multi-term AND miss is retried OR-combined. Single-term misses stay misses;
   the fallback must not invent hits.

## Why keep both

|                   | tokens | old sessions                        |
| ----------------- | ------ | ------------------------------------ |
| §3.3 tiered index | 246    | collapsed into coarse ranges         |
| `ms.outline()`    | 1,444  | one line each, full topical detail   |

The paper's tiering is right for a long agent trajectory where recent state
matters most. LongMemEval scatters evidence **uniformly** across all 47
sessions, so collapsing the oldest 43 into ranges discards exactly the signal
these questions need. That, not any insight of ours, is why `outline()` was
worth +33 points here — oracle split, Haiku 4.5, n=12: v1/v2 (no `outline()`)
50.0%, v3 (+ `outline()`, + OR fallback) 83.3%.

## Consequences

The model-facing API remains a superset of the paper's, so our numbers are not
a clean reproduction — say so when comparing.

Unmeasured: whether §3.3's index alone recovers the same ground, and whether
it lifts `preference following`, which the paper's ablation says the index
affects most (89.1 vs 74.9) and which has been this harness's weakest category
in every run. Settling that needs benchmark runs; see ADR 0002 for why the
current backend makes them expensive.
