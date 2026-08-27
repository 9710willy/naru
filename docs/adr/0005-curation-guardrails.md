# 5. Four curation rules that look wrong until you know why

An adversarial review found seven blockers in the curation layer. Four of the
fixes encode rules a fresh reader would plausibly "simplify" back into bugs, so
they are written down here rather than left as comments.

## Context

The curation layer shipped in ADR 0004 with three columns and five verbs. A
multi-lens review of that commit range reproduced seven blocking defects. Three
of them were not oversights but reasonable-looking decisions with consequences
one step further out than the author looked.

## Decisions

### `inject` splices, it never writes the whole file

`naru inject CLAUDE.md` is the documented integration for Claude Code, and
CLAUDE.md is a file the user wrote. The original `write_text()` truncated it —
the happy path in the README was a data-loss bug. The doc now lands between
`<!-- naru:begin -->` and `<!-- naru:end -->`, and everything outside those
markers survives.

Claims are also collapsed to one line before rendering. A newline inside a
claim otherwise breaks out of its markdown bullet and becomes a top-level line
in whatever format the target file is — which, for a shell rc, means
agent-authored text becomes a line the host executes.

### `prune` never takes a decided claim

Prune's predicate is age, which is right for spilled tool output and wrong for
a decision a person made. Claims are stamped with the wall clock at `naru
claim` time, so an age-only rule gave every promotion a 30-day shelf life and
silently emptied the doc. `PRUNE_KEEP` exempts any claim that is not pending.

### `decide()` is asymmetric on purpose

Promotion requires `promoted=0`, so deciding the same claim twice is reported
rather than silently re-applied. Retiring does **not** require it.

That asymmetry is the fix for a trapdoor: with both directions guarded, a
promoted fact could never be revised. Promoting a second value for the same
`topic_key` left both promoted, which parks the key under `## Unresolved`
forever and loses the fact the doc used to state. The workflow is retire, then
supersede.

### `base_seq` is the doc version, not the log head

The repo previously stated both contracts — two comments said "doc version",
the README said "log size" — and the code implemented the second. Under that
reading any unrelated append advanced the head, so a spilled tool result or a
`naru add` note fired a staleness warning about a doc that had not changed.
`ms.doc_version()` is `MAX(seq) WHERE promoted=1`.

### `--yes` is a speed bump, not a boundary

`promote` and `drop` refuse to run without a terminal unless `--yes` is passed.
Anything that can run this CLI can also pass `--yes`, so this stops accidental
promotion from a tool loop and nothing more.

The source previously asserted that "nothing reaches the doc without a human",
which was never enforced anywhere. The claim was removed. If real separation is
ever needed it has to come from something an agent-run shell cannot supply —
file permissions, a separate approving process, or a signature — and none of
those are here.

## Why write these down

Each one reads as an inconsistency on first encounter: an asymmetric guard, a
predicate with an exemption, a gate that admits it does not gate. All three
survived a review specifically because the reviewer could name the failure that
happens without them. Removing any of them restores a reproduced bug.
