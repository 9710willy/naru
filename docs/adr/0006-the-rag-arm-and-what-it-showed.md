# 6. A retrieval-only arm, and the result it produced

The benchmark gained a third arm, `rag`. On LongMemEval it beats `naru` on
every column we measure. The kernel does not currently earn its keep on this
benchmark, and the README says so.

## Context

Until now `bench.py` ran two arms: `full` put the entire history in one
prompt, `naru` left it in the log for the model to reach by writing code.
`naru` used fewer tokens and cost less, and it was tempting to read that as
evidence for programmatic context management.

It was not evidence for that, because two things changed at once: the prompt
got small, **and** the model got a kernel. Nothing in the harness separated
them.

MemDelta ([arXiv 2606.29914](https://arxiv.org/abs/2606.29914)) makes that gap
concrete rather than theoretical: on LongMemEval-S it measured verbatim RAG
statistically level with full context (47.2 vs 49.8, p = 0.34), and agent
self-memory at 42% *losing* to basic retrieval at 47%. Any claim we make about
the kernel has to clear that bar.

## Decision

`rag` holds the kernel fixed and varies only what fills the prompt. It runs
`ms.search(question, k=8)`, sorts the hits back into log order, pastes them
in, and answers in one call. It uses `FULL_SYSTEM` verbatim — the same system
prompt as the `full` arm — so the two single-call arms differ in exactly one
variable.

Two consequences for the harness:

- `--arms` now rejects an unknown name instead of falling through to `full`.
  A typo would otherwise corrupt a paid run in silence.
- `separability()` compares arms with an exact McNemar test, not with
  overlapping Wilson intervals. The arms answer identical questions, so the
  comparison is paired, and only the questions where they disagree carry
  information.

## The result

Current numbers, the per-category breakdown, and the caveats on reading them
live in the [README's Benchmark section](../../README.md#benchmark), the one
place they get updated. Read there, not here — do not copy tables back into
this ADR.

The one-line summary: no arm separates from any other on accuracy at any n this
harness has run, and where `naru` comes closest to `rag` is exactly the two
categories (temporal-reasoning, preference-following) where an answer must be
computed over the log rather than quoted from a turn — the shape the paper's
own ablations predict, per ADR 0003. A retrieval agent loses badly on the
paper's other two benchmarks (LOCA-256K, BEAM-10M against ten named baselines),
so the hypothesis is tested, not untested. **LongMemEval is the wrong benchmark
for the question this repo asks** — testing the kernel needs one where `full`
cannot run at all, such as BEAM ([arXiv
2510.27246](https://arxiv.org/abs/2510.27246)).

## Consequences

The honest reading is that on LongMemEval, BM25 over a SQLite log with a
top-8 cut is sufficient, and the kernel is overhead. We are not going to bury
that. What it does **not** establish is that the kernel is useless in
general — see the README for why, and the paper's own tables for how tightly
five unrelated architectures cluster on LongMemEval-S in the first place.
