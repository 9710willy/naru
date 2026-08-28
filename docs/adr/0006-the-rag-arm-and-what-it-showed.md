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

| arm    | correct | 95% CI | net input/q | view    | calls | model $/q |
| ------ | ------- | ------ | ----------- | ------- | ----- | --------- |
| `full` | 16/24   | 47-82% | 116,727     | 124,484 | 1.0   | $0.2524   |
| `naru` | 19/24   | 60-91% | 73,199      | 2,552   | 4.4   | $0.1409   |
| `rag`  | 21/24   | 69-96% | 1,816       | 1,805   | 1.0   | $0.0251   |

No pair separates on accuracy. Paired McNemar gives p = 0.125 for `full` vs
`rag`, 0.508 for `full` vs `naru`, and 0.625 for `rag` vs `naru`. At n=24 this
harness cannot distinguish any of the three, and that includes the 12.5-point
gap the README used to lead with.

Cost is not a statistical question, and there the answer is unambiguous. `rag`
answered in one call on 1,816 input tokens net of harness overhead. `naru` took
4.4 calls and 73,199 — **forty times more input for no measurable accuracy
gain**, and it scored numerically lower.

The money ratio is 5.6x, not 40x, and the difference is a real effect rather
than rounding. `billed_input` is an unweighted sum of fresh, cache-creation and
cache-read tokens, and those bill at roughly 1x, 1.25x and 0.1x. `naru` is 70%
cache reads by volume against `rag`'s 55%, so its token count is inflated
relative to its bill. Report the token ratio as tokens and the money ratio as
money; the harness prints both and they will not agree whenever the arms differ
in call count.

Two limits on this run. `rag` has no replicate, so `noise.py` has nothing to
compare and the accuracy caveat rests entirely on the paired test. And the two
runs measured harness floors of 22,846 and 18,718 tokens per call, so they are
not the same session; each row's `net-of-harness` is taken against its own
run's floor, which is why `results/published/README.md` records both.

## Consequences

The honest reading is that on LongMemEval, BM25 over a SQLite log with a
top-8 cut is sufficient, and the kernel is overhead. We are not going to bury
that.

What it does **not** establish: that the kernel is useless in general. The
paper's own results say why, and we should have read them before treating this
as a surprise.

Its LongMemEval-S table puts five unrelated architectures inside a two-point
band, with its own system third:

| system      | LongMemEval-S |
| ----------- | ------------- |
| Exabase M-1 | 96.4          |
| Mastra OM   | 94.9          |
| Scroll      | **94.8**      |
| Hindsight   | 94.6          |
| Mem0        | 94.4          |

The paper does not win its own LongMemEval table and does not claim to. That
benchmark does not separate memory architectures for anyone, which is the same
thing our `rag` arm found with a cruder baseline and a smaller n. Our result
reproduces the paper's, it does not contradict it.

The paper stakes its claim on two other benchmarks, and one of them carries a
named retrieval baseline:

| LOCA-256K           |      | BEAM-10M    |          |
| ------------------- | ---- | ----------- | -------- |
| Scroll              | 86.7 | Scroll      | **73.1** |
| CodeAct             | 85.3 | Exabase M-1 | 68.0     |
| **Retrieval Agent** | 66.7 | Cognee      | 67.0     |
| Summarization Agent | 65.3 | Hindsight   | 64.1     |

A retrieval agent is 20 points behind on LOCA-256K. So the hypothesis is not
untested — the paper tested it, against retrieval, and retrieval lost badly
once the questions stopped being single-turn lookups.

Two things in those tables deserve stating plainly. CodeAct, a generic
code-writing agent, is 1.4 points behind Scroll on LOCA — so the margin over
retrieval belongs to the write-code family broadly, and Scroll's specific
design adds little on top of it. And the paper ran Qwen3.8-Max while this
harness runs Haiku 4.5; MemDelta reports baseline rankings reversing across
model families, so the two sets of numbers cannot be read against each other in
either direction.

## The ranking reverses with the model

Run on Sonnet 5 over the same 12 questions, `naru` scores 11/12 and `rag` 9/12
— the opposite order to Haiku's 21/24 against 19/24. Neither ordering is
separable (p=0.375 to 1.000), and at n=12 `noise.py` puts run-to-run movement
at ~33 points, so no accuracy claim on either model survives contact with its
own noise floor. What the reversal does establish is that the earlier
"BM25 beats the kernel" reading was a statement about Haiku 4.5, which is the
failure MemDelta (arXiv 2606.29914) documents: baseline rankings reverse across
model families.

Cost does not reverse. `rag` used 2,727 input tokens per question against
`naru`'s 314,587 on Sonnet — 115x, against 40x on Haiku. The dollar ratio is
smaller than the token ratio in both because 87% of `naru`'s Sonnet input is
cache reads, and for the same reason `naru` is cheaper than `full` in money
while spending more tokens.

One thing worth a closer look than n=2 allows: preference-following is the only
category where `naru` beat `rag` cleanly on Sonnet, 2/2 against 0/2. ADR 0003
records that the paper's ablation says the eviction index affects that category
most (89.1 vs 74.9), and that it has been this harness's weakest category in
every run. That is where to look next, not at the aggregate.

## What survives

The conclusion that survives: **LongMemEval is the wrong benchmark for the
question this repo asks.** Testing the kernel needs one where a single
retrieval cannot win — BEAM (arXiv 2510.27246), where the `full` arm cannot run
at all. Until that run exists, `rag` is the arm to beat here, and the burden is
on the kernel to show it earns 40x the input somewhere that matters.

Baseline tables above are from the paper's own HTML (arxiv.org/abs/2608.21690),
read after this run, not reproduced by us.

Second consequence: n=24 is too small to separate anything, including a
20.8-point gap. Every accuracy claim from this harness is currently
underpowered, and the report prints its own confidence interval now so that a
reader cannot miss it.
