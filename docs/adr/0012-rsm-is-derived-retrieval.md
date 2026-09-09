# 12. RSM retrieval is derived state

## Context

The LongMemEval harness needs optional atom-based arms for the compact-memory
method in arXiv 2609.04915. The Event Log already holds the ordered source
history. Adding atom tables to `ms.py` would create a second durable copy and a
second writer for that history.

Embedding service price and token accounting vary by provider. The benchmark
cannot report an honest embedding dollar value without provider data.

## Decision

Keep the SQLite Event Log as the durable source of truth. Build RSM atoms from
chronological Event Log chunks for each benchmark question and discard them
after the answer. Store explicit Event Log `seq` values in every member.

Keep `rsm` optional. It needs `NARU_EMBED`, which receives `{"texts": [...]}`
on stdin and returns `{"vectors": [...]}`. The default benchmark remains
`full,rag,naru` and works without an embedding command. The `rsm` arm pastes
the selected atom members into one answer prompt. Do not add atom handles or a
second search API to Naru until a measured result shows that they help.

Use normalized-centroid cosine to build and retrieve atoms. Pack selected
members under atom headers in Event Log order. This is the first implementation
because the paper reports no useful retrieval difference from its
singular-vector score when the packer is fixed.

Record embedding input tokens, atom and member counts, embedding and indexing
time, and packed context tokens. Report model cost separately and state that it
excludes embedding-provider cost.

## Reason

The Event Log has one writer and one stable order. Derived atoms can be rebuilt
from it, so they do not need schema, migration, or recovery rules. Centroid
cosine keeps the retrieval comparison small and avoids a new linear algebra
implementation. Recording work without invented provider dollars keeps the
result useful and honest.
