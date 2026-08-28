# 6. A retrieval-only arm, and the result it produced

The benchmark gained a third arm, `rag`. On LongMemEval it beats `naru` on every
column we measure. The kernel does not currently earn its keep on this
benchmark, and the README says so.

## Context

Until now `bench.py` ran two arms: `full` put the entire history in one prompt,
`naru` left it in the log for the model to reach by writing code. `naru` used
fewer tokens and cost less, and it was tempting to read that as evidence for
programmatic context management.

It was not evidence for that, because two things changed at once. The prompt got
small, **and** the model got a kernel. Nothing in the harness separated them.

MemDelta (arXiv 2606.29914) makes that gap concrete rather than theoretical: on
LongMemEval-S it measured verbatim RAG statistically level with full context
(47.2 vs 49.8, p = 0.34), and agent self-memory at 42% *losing* to basic
retrieval at 47%. Any claim we make about the kernel has to clear that bar.

## Decision

`rag` holds the kernel fixed and varies only what fills the prompt. It runs
`ms.search(question, k=8)`, sorts the hits back into log order, pastes them in,
and answers in one call. It uses `FULL_SYSTEM` verbatim — the same system prompt
as the `full` arm — so the two single-call arms differ in exactly one variable.

Two consequences for the harness:

- `--arms` now rejects an unknown name instead of falling through to `full`. A
  typo would otherwise corrupt a paid run in silence.
- `separability()` compares arms with an exact McNemar test, not with
  overlapping Wilson intervals. The arms answer identical questions, so the
  comparison is paired, and only the questions where they disagree carry
  information.

## The result

LongMemEval-`s`, n=24, Haiku 4.5 on agent and judge. Rows in
`results/published/`.

| arm    | correct | 95% CI | net input/q | view    | calls | cost  |
| ------ | ------- | ------ | ----------- | ------- | ----- | ----- |
| `full` | 16/24   | 47-82% | 116,727     | 124,484 | 1.0   | $6.49 |
| `naru` | 19/24   | 60-91% | 73,199      | 2,552   | 4.4   | $3.79 |
| `rag`  | 21/24   | 69-96% | 1,816       | 1,805   | 1.0   | $1.22 |

No pair separates on accuracy. Paired McNemar gives p = 0.125 for `full` vs
`rag`, 0.508 for `full` vs `naru`, and 0.625 for `rag` vs `naru`. At n=24 this
harness cannot distinguish any of the three, and that includes the 12.5-point
gap the README used to lead with.

Cost is not a statistical question, and there the answer is unambiguous. `rag`
answered in one call on 1,816 input tokens net of harness overhead. `naru` took
4.4 calls and 73,199 — **forty times more input for no measurable accuracy
gain**, and it scored numerically lower.

## Consequences

The honest reading is that on LongMemEval, BM25 over a SQLite log with a
top-8 cut is sufficient, and the kernel is overhead. We are not going to bury
that.

What it does **not** establish: that the kernel is useless in general.
LongMemEval questions are largely single-fact lookups where one lexical query
lands on the evidence turn, which is the case retrieval is built for. The
paper's system targets long-horizon agent trajectories, where the answer is
usually not sitting in one retrievable turn and the model has to compute over
the log rather than quote it. That is a real hypothesis and it is untested here.

Testing it needs a benchmark where a single retrieval cannot win — BEAM at 1M
and 10M tokens (arXiv 2510.27246) is the obvious candidate, since the `full`
arm cannot run there at all and questions span more evidence than a top-k
window holds. Until that run exists, `rag` is the arm to beat and the burden is
on the kernel.

Second consequence: n=24 is too small to separate anything, including a
20.8-point gap. Every accuracy claim from this harness is currently
underpowered, and the report prints its own confidence interval now so that a
reader cannot miss it.
