# 10. A claim may cite only what a reader opened

## Context

`ms.append` proves a cited seq exists: both endpoints must be rows under the
given `session_id`, or it raises `source endpoints are not in that run`. It does
not prove anyone read them.

The spill hook makes that gap reachable. It replaces a large tool result with
600 characters plus up to twelve signpost lines, so an agent can see line 41 and
line 121 of a 4,635-character file and cite the whole row having read an eighth
of it. The handle resolves, the endpoints exist, and the claim is false.

Agent Zero Memory (arXiv 2608.29606, Definition 2) names the rule this needs.
Let `O` be the items a reader actually opened. An answer may cite only `C ⊆ O`,
and a reader that cannot assemble such a `C` must abstain. The paper enforces it
by construction: the reader's interface offers no channel for citing unopened
material.

## Decision

`naru claim` and `naru skill` refuse a `--run ID --source LO:HI` unless both
endpoints fall inside a range that a `naru show` recorded. `show` now records
the far endpoint as well, since it prints every row in the range.

Only `show` counts as an open. `search` prints 300 characters per hit, and a
preview is the thing that created the problem.

The check lives in `naru.py`, not in `ms.append`.

## Why

The CLI is the trust boundary for a claim an agent proposes, and rule 2 says
validate once there. `agent.py` also writes provenance, on every `agent_state`
row, but its reader reaches history through `ms.expand`, which never touches
`metrics`. A check inside `append` would refuse every state the benchmark
writes.

## Limits

Two, both real.

`naru prune` trims `metrics.jsonl` to the same window it prunes rows with, 30
days by default. An open older than that no longer proves anything, so a claim
citing evidence read last quarter is refused. That is the correct answer for a
memory system whose point is a live reading trail, and it is still a behaviour
change worth knowing about.

This is an integrity check, not a security boundary. Anything that can run
`naru claim` can also run `naru show`, or append a line to `metrics.jsonl`. It
catches an agent that cites what it skimmed. It does not stop one that decides
to lie, and the promote gate remains the thing that does.
