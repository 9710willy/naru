# naru (나루)

> _naru_ — a ferry landing. The place where crossings meet.

One curated doc, shared across your sessions, your models and your harnesses.
Agents propose facts. You decide which ones survive. Only what you promote
reaches the doc that enters the next model call.

Built on a working implementation of _Context as an Environment: Programmatic
Context Management for Long-Horizon Agents_
([arXiv 2608.21690](https://arxiv.org/abs/2608.21690)). The paper's system is
called Scroll; this repo implements it and adds a curation layer on top.

## The problem

An LLM call is stateless, so the harness re-sends the whole conversation every
turn. A 7,000-token page read once is paid for on every later turn. Compaction
and external memory fix the size but lose the original.

The paper's answer: keep history outside the prompt, in an addressable log the
model reaches by writing code.

```
Event Log (SQLite+FTS5)   Python kernel        outside the prompt
  seq 1 user "I drive…"     bulk = [40 rows]
         ▲ ms.search()            ▲ exec()
         └──────────┬─────────────┘
                    │ only print() crosses
  Question + resident digest + last observations + eviction index
                    └──── the prompt: ~100 tokens ────┘
```

naru adds the layer above it: that log is per-session and machine-facing, and
nobody can read it. The doc is cross-session and human-facing.

```
  claude session ─┐
  codex session   ─┼─ claim ──→ [ Event Log ] ──promote──→ naru.md ──→ every agent
  dsh subagents   ─┘                                ▲
                                                    └── you decide
```

## Quickstart

Nothing to install. Python 3.9+ stdlib and a SQLite built with FTS5 (the macOS
system Python has it).

```bash
# an agent proposes a fact
./naru.py claim "Retry budget is 5, not 3." --key bench.retries --by claude-opus

# you decide
./naru.py inbox
#   [1/1] seq 42 · claude-opus · 2026-08-27 10:01 · key: bench.retries
#     + Retry budget is 5, not 3.
#     promote / drop / skip ? p

# the doc every agent reads
./naru.py inject
#   # naru · seq 42 · ~31 tokens
#
#   ## Decisions
#   - Retry budget is 5, not 3.
```

## Works with any harness

Anything that can run a shell command can write claims and read the doc. There
is no per-harness API to implement — `inject` writes into the context file the
harness already reads.

```bash
naru inject CLAUDE.md        # Claude Code
naru inject AGENTS.md        # Codex, DeepSeek Harness, most others
naru inject .cursorrules     # Cursor
```

Claude Code gets one extra: `hook_spill.py` is a `PostToolUse` hook that spills
oversized tool output into the log and leaves a recovery handle inline, so a
big command result costs a preview instead of the whole payload.

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "python3 /path/to/naru/hook_spill.py"
          }
        ]
      }
    ]
  }
}
```

The model side is equally swappable. `backend.py` defaults to the local
`claude` CLI and falls back to any command that reads a prompt on stdin:

```bash
NARU_BACKEND='codex exec -'    python3 bench.py --split oracle -n 12
NARU_BACKEND='ollama run llama3' python3 bench.py --split oracle -n 12
```

A generic pipe reports no token counts, so those columns read as zero. naru
says so on stderr rather than printing `$0.000` as if the calls were free.

## Two things stay separate

|            | Event Log            | The doc                |
| ---------- | -------------------- | ---------------------- |
| Grows      | without bound        | stays small            |
| Contains   | everything, lossless | only what you promoted |
| Read by    | code, on demand      | every model call       |
| Ordered by | `seq`                | your decisions         |

The doc is a **promoted subset**, never a render of the log. A doc that grows
with the log is just the whole-history prompt wearing a hat.

Three columns carry the curation layer:

- `promoted` — pending (0), promoted (1), dropped (-1). A decision never
  deletes: a dropped claim stays addressable at its `seq`.
- `topic_key` — two promoted claims on one key is a contradiction. naru shows
  both under `## Unresolved` and never picks one for you.
- `base_seq` — the log head the author wrote against. Staleness cannot be
  prevented, so `inbox` shows how far the log moved while they worked.

## Run the checks

Every module has a runnable self-check. The first six need no network.

```bash
python3 ms.py        # Event Log, curation, search, expand, payload recovery
python3 kernel.py    # persistent namespace, print capture, error survival
python3 eviction.py  # Algorithm 1 + tiered index stays sublinear
python3 agent.py     # full turn loop against a scripted fake backend
python3 naru.py --selfcheck && python3 hook_spill.py --selfcheck
python3 backend.py   # live: 2 cheap calls, prints the harness token floor
```

## Benchmark

Get the data (LongMemEval, ICLR 2025):

```bash
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
cd ..
```

Run both arms over the same questions:

```bash
python3 bench.py --split oracle -n 12                  # quick, cheap
python3 bench.py --split s -n 30 --model claude-sonnet-5
```

| Arm    | What it does                                                      |
| ------ | ----------------------------------------------------------------- |
| `full` | whole history in one prompt — the usual approach                  |
| `naru` | history in the Session Environment, model writes code to reach it |

Useful flags: `--budget` (working-view token ceiling), `--max-turns`,
`--qtype` (one question type), `--workers`, `-v`.

Results land in `results/<tag>_<split>_n<N>.json`, one row per question with
its answer, verdict, turns, tokens and cost.

### Reading the token numbers

The `claude` CLI adds a fixed system-prompt overhead per call (~13-23k input
tokens, mostly cache reads). `bench.py` reports `billed-in` and also
`net-of-harness`, which subtracts `--harness-floor` × turns. Compare arms on
`net-of-harness`; run `python3 backend.py` to measure the floor on your machine.

A single run's difference under ~20 points at n=12 is noise. `noise.py` reports
the floor — read it before believing a result.

## Files

| File            | Role                                                                     |
| --------------- | ------------------------------------------------------------------------ |
| `ms.py`         | Event Log + curation: `search` / `expand` / `pending` / `decide` / `doc` |
| `naru.py`       | the CLI: claim, inbox, inject, and note recall                           |
| `kernel.py`     | persistent Python namespace, exec, print capture, namespace digest       |
| `eviction.py`   | Algorithm 1: recoverable eviction + tiered headline index                |
| `agent.py`      | the turn loop and the system prompt                                      |
| `backend.py`    | model calls via the `claude` CLI, or any stdin→stdout command            |
| `hook_spill.py` | Claude Code `PostToolUse` hook: spill big tool output, leave a handle    |
| `bench.py`      | LongMemEval ingest → answer → judge → score                              |

## Where this diverges from the paper

Both are deliberate and ablatable.

- `ms.outline()` is ours. §3.3's ingestion-time index collapses old sessions
  into coarse ranges, which suits a long trajectory but not LongMemEval's
  uniformly-scattered evidence. `--no-index` ablates the paper's version.
- `search()` retries a failed multi-term AND as OR. The paper does not.
- The whole curation layer — claims, promotion, conflict keys — is ours. The
  paper is about one agent's context. naru is about many agents and one human.

## Known limits

- The curation layer is new and has self-checks but no field use yet. The paper
  implementation underneath it is the tested part.
- `kernel.py` runs model-authored code in-process with no sandbox. Fine for
  benchmark runs over local data; use a subprocess or container before running
  anything untrusted.
- Conflict detection is exact-match on `topic_key`. Two claims that contradict
  each other under different keys will both sit in the doc, unflagged.
- One database. Two machines breaks `seq` as a total order, and nothing here
  solves distributed consensus.
- BM25 lexical search only, matching the paper. No embeddings.
- `bench.py` implements LongMemEval. BEAM and LOCA-bench are not wired up.
- The judge is an LLM comparing against the benchmark's gold answer. It is not
  the benchmark's official judge prompt, so numbers are internal progress only
  and not comparable to published leaderboard scores.

## Credit

The Session Environment, the Event Log, Algorithm 1 and the eviction index are
from _Context as an Environment: Programmatic Context Management for
Long-Horizon Agents_ ([arXiv 2608.21690](https://arxiv.org/abs/2608.21690)).
Read the paper first; this repo is an implementation of it, not a replacement
for it.
