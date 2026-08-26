# 3. `ms.outline()` was a workaround for skipping §3.3, not an addition to the paper

Supersedes the original text of this ADR, which claimed `outline()` and the OR
search fallback were "two deliberate additions to the paper's API". That was
wrong about `outline()` and it flattered us. The paper already had a navigation
layer; we had skipped the protocol that builds it.

## Context

Implemented against Appendix C alone (`search` / `expand` / `sql_query` /
`days_between`), three of twelve oracle questions failed with the agent
insisting a fact was absent when it was present:

```
question says "homegrown ingredients"
history says  "fresh basil and mint"

ms.search("homegrown")  -> 0 hits   # the word never appears in the log
ms.search("basil")      -> 3 hits
```

Pure BM25 cannot cross a vocabulary gap, and an AND-combined multi-word query
returns zero without signalling why. We added `ms.outline()` — one line per
session — as a structural way in, and recorded it as our own idea.

It is not. §2.4 keeps _landmarks_ for position-based navigation precisely
because "lexical search recovers an evicted span only when the agent recalls its
wording", and §3.3 builds those landmarks during ingestion: history is ingested
session by session, and at each boundary the raw context is cleared while the
eviction index is carried forward. We had ingested all 47 sessions in one pass
and started the agent with an empty index, so that navigation layer never
existed to be used.

## Decision

Both mechanisms now exist and are separately ablatable.

1. **§3.3 ingestion** — `ingest()` walks sessions in order and enters each
   session's landmark into the tiered index at the boundary. 47 sessions become
   9 index entries, 246 tokens. `--no-index` turns it off.
2. **`ms.outline()`** — one line per session, 1,444 tokens for the same history.
   Kept, because it is _not_ equivalent (see below).
3. **Auto-OR fallback in `search()`** — a genuine addition, and a small one: a
   multi-term AND miss is retried OR-combined. Single-term misses stay misses;
   the fallback must not invent hits.

## Why keep both

They differ in how they spend tokens, and the paper's choice is tuned for a
different evidence distribution than LongMemEval has:

|                   | tokens | old sessions                       |
| ----------------- | ------ | ---------------------------------- |
| §3.3 tiered index | 246    | collapsed into coarse ranges       |
| `ms.outline()`    | 1,444  | one line each, full topical detail |

The tiering is deliberate — "fine anchors for recent history, coarse ranges for
distant history" — which is right for a long agent trajectory where recent state
matters most. LongMemEval scatters evidence **uniformly** across all 47 sessions,
so collapsing the oldest 43 into ranges discards exactly the signal these
questions need. That, not any insight of ours, is why `outline()` was worth +33
points here.

Measured, oracle split, Haiku 4.5, n=12:

| version | change                                      | score     |
| ------- | ------------------------------------------- | --------- |
| v1      | first working run                           | 50.0%     |
| v2      | retry empty replies; execute unfenced cells | 50.0%     |
| v3      | **+ `outline()`, + OR fallback**            | **83.3%** |

## Consequences

The model-facing API remains a superset of the paper's, so our numbers are not a
clean reproduction — say so when comparing.

Unmeasured: whether §3.3's index alone recovers the same ground, and whether it
lifts `preference following`, which the paper's ablation says the index affects
most (89.1 vs 74.9) and which has been this harness's weakest category in every
run. Settling that needs benchmark runs; see ADR 0002 for why the current
backend makes them expensive.
