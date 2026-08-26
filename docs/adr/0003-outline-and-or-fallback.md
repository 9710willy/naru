# 3. Add `ms.outline()` and an OR search fallback — two deliberate additions to the paper's API

Both close the same hole: the question's wording often shares no words with the
turn that answers it. Together they moved LongMemEval-oracle from 50% to 83.3%.

## Context

The paper's Appendix-C surface is `search` / `expand` / `sql_query` /
`days_between`. Implemented literally, three of twelve oracle questions failed
with the agent insisting a fact was absent when it was present.

The measured cause:

```
question says "homegrown ingredients"
history says  "fresh basil and mint"

ms.search("homegrown")  -> 0 hits   # word never appears in the log
ms.search("basil")      -> 3 hits
```

Pure BM25 cannot cross a vocabulary gap, and an AND-combined multi-word query
returns zero without signalling why.

## Decision

1. **`ms.outline()`** — one line per session: seq range, date, turn count, and
   the opening of its first user turn. The agent's system prompt requires
   calling it before ever concluding a fact is missing. For a 47-session,
   124,609-token history the outline is 1,444 tokens.
2. **Auto-OR fallback in `search()`** — when a multi-term AND match returns
   nothing, retry the same terms OR-combined. Single-term misses stay misses;
   the fallback must not invent hits.

## Why

`outline` is the paper's own "headlines as navigation anchors" idea, applied at
ingest time instead of only after eviction. The paper attributes Scroll's
residual failures to "errors of query formulation" — these two changes attack
exactly that, at the harness level, rather than hoping for a better prompt.

Measured, oracle split, Haiku 4.5, n=12:

| version | change                                      | score     |
| ------- | ------------------------------------------- | --------- |
| v1      | first working run                           | 50.0%     |
| v2      | retry empty replies; execute unfenced cells | 50.0%     |
| v3      | **+ `outline()`, + OR fallback**            | **83.3%** |

## Consequences

The model-facing API is a superset of the paper's, so our numbers are not a
clean reproduction of it — say so when comparing. The remaining oracle failures
are model-capability, not harness: in one case the agent claimed
`search("Kg2")` returned nothing when it returns 3 hits. That matches the
paper's finding that weaker backbones "terminate prematurely", and it is not
patchable here.
