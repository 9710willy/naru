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

Superseded by an n=96 run on Sonnet 5, which reverses the n=24 Haiku ordering
this ADR originally reported. Both are published; the small one is kept because
the reversal is the point.

LongMemEval-`s`, n=96, Sonnet 5, judge Haiku 4.5:

| arm    | correct | 95% CI | view     | calls | model $/q |
| ------ | ------- | ------ | -------- | ----- | --------- |
| `naru` | 72/96   | 65-83% | 2,174    | 3.7   | $0.3547   |
| `full` | 64/96   | 57-75% | 124,073  | 1.0   | $0.8036   |
| `rag`  | 59/96   | 51-71% | 2,142    | 1.0   | $0.0753   |

Paired McNemar, Bonferroni-corrected to 0.0167 for three pairs: `full` vs `rag`
p=0.064, `full` vs `naru` p=0.690, `rag` vs `naru` p=0.029. The last clears a
plain 0.05 and not the corrected threshold, so nothing separates.

At n=24 on Haiku the ordering was `rag` 21/24, `naru` 19/24, `full` 16/24 —
`rag` ahead. Neither ordering is separable and both are single-model, so the
lesson is about the harness rather than the architectures: MemDelta (arXiv
2606.29914) reports exactly this reversal across model families, and `noise.py`
over two Sonnet replicates at n=12 puts run-to-run movement at ~33 points with
a third of question verdicts flipping.

### Where the difference sits

| category                  | `full` | `rag` | `naru` | naru vs rag |
| ------------------------- | ------ | ----- | ------ | ----------- |
| knowledge-update          | 13/16  | 13/16 | 13/16  | p=1.000     |
| multi-session             | 10/16  | 7/16  | 8/16   | p=1.000     |
| single-session-assistant  | 13/15  | 15/16 | 15/16  | p=1.000     |
| single-session-user       | 14/14  | 15/16 | 15/16  | p=1.000     |
| single-session-preference | 4/16   | 2/16  | 7/16   | p=0.125     |
| temporal-reasoning        | 10/14  | 7/16  | 14/16  | p=0.039     |

Four categories tie. The aggregate difference is entirely temporal-reasoning
and preference-following: the two where the answer must be computed over the
log rather than quoted from a turn. A top-8 keyword window retrieves turns and
cannot do arithmetic across sessions.

Preference-following was predicted before it was measured — ADR 0003 records
the paper's ablation putting it at 89.1 with the index against 74.9 without,
the largest category effect it reports. Temporal-reasoning was not predicted,
and six post-hoc category tests carry their own multiple-comparison exposure.
Read the table as descriptive.

### The token columns are not trustworthy

The `claude` CLI's usage dict does not reconcile across runs. The same `full`
prompt reported 201,288 billed input tokens at n=12 and 47,151 at n=96, for the
same 124k view and nearly the same cost per question. Every "40x the input"
claim in earlier versions of this ADR rested on that field and has been removed.

Cost derives from the CLI's `total_cost_usd` and is consistent across runs.
`view` is `est()`, ours, a character count. Those two are safe to quote; the
`net-of-harness` column is not.

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

Cost does not reverse: `rag` is the cheapest arm on both models, $0.0753/q
against `naru`'s $0.3547/q at n=96. `naru` is nonetheless cheaper than `full`
($0.8036/q) while holding a 2,174-token view against 124,073, which is the
paper's actual claim and the one thing every run here agrees on.

## What survives

The conclusion that survives: **LongMemEval is the wrong benchmark for the
question this repo asks.** Testing the kernel needs one where a single
retrieval cannot win — BEAM (arXiv 2510.27246), where the `full` arm cannot run
at all. The n=96 category table points the same way from inside LongMemEval:
the difference lives in the two categories that need computation rather than
retrieval, and the other four are ties.

Baseline tables above are from the paper's own HTML (arxiv.org/abs/2608.21690),
read after this run, not reproduced by us.

Second consequence: n=24 is too small to separate anything, including a
20.8-point gap. Every accuracy claim from this harness is currently
underpowered, and the report prints its own confidence interval now so that a
reader cannot miss it.
