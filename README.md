# Scroll

A working implementation of _Context as an Environment: Programmatic Context
Management for Long-Horizon Agents_ ([arXiv 2608.21690](https://arxiv.org/abs/2608.21690)).

The agent's history never enters the prompt. It lives in a SQLite Event Log and
a persistent Python kernel. The model writes code to search and compute over
it, and only what the code `print`s enters the next model call.

## Why

An LLM call is stateless, so the harness re-sends the whole conversation every
turn. A 7,000-token page read once is paid for on every later turn and occupies
the context window for the rest of the session. Compaction and external memory
fix the size but lose the original. Scroll keeps the original addressable.

```
Event Log (SQLite+FTS5)   Python kernel        outside the prompt
  seq 1 user "I drive…"     bulk = [40 rows]
         ▲ ms.search()            ▲ exec()
         └──────────┬─────────────┘
                    │ only print() crosses
  Question + resident digest + last observations + eviction index
                    └──── the prompt: ~100 tokens ────┘
```

## Install

Nothing to install. Python 3.9+ stdlib and a SQLite built with FTS5 (the macOS
system Python has it). The model backend shells out to the local `claude` CLI,
so no API key is needed.

## Run the checks

Every module has a runnable self-check. The first three need no network.

```bash
python3 ms.py        # Event Log: search, expand, payload recovery, SQL guard
python3 kernel.py    # persistent namespace, print capture, error survival
python3 eviction.py  # Algorithm 1 + tiered index stays sublinear
python3 agent.py     # full turn loop against a scripted fake backend
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

| Arm      | What it does                                                      |
| -------- | ----------------------------------------------------------------- |
| `full`   | whole history in one prompt — the usual approach                  |
| `scroll` | history in the Session Environment, model writes code to reach it |

Useful flags: `--budget` (working-view token ceiling), `--max-turns`,
`--qtype` (one question type), `--workers`, `-v`.

Results land in `results/<tag>_<split>_n<N>.json`, one row per question with
its answer, verdict, turns, tokens and cost.

### Reading the token numbers

The `claude` CLI adds a fixed system-prompt overhead per call (~13-23k input
tokens, mostly cache reads). `bench.py` reports `billed-in` and also
`net-of-harness`, which subtracts `--harness-floor` × turns. Compare arms on
`net-of-harness`; run `python3 backend.py` to measure the floor on your machine.

## Files

| File          | Role                                                                           |
| ------------- | ------------------------------------------------------------------------------ |
| `ms.py`       | Event Log + memory surface: `search` / `expand` / `sql_query` / `days_between` |
| `kernel.py`   | persistent Python namespace, exec, print capture, namespace digest             |
| `eviction.py` | Algorithm 1: recoverable eviction + tiered headline index                      |
| `agent.py`    | the turn loop and the system prompt                                            |
| `backend.py`  | model calls via the `claude` CLI                                               |
| `bench.py`    | LongMemEval ingest → answer → judge → score                                    |

## Using another model

Only `backend.py` is Claude-specific. The contract is one function:

```python
def backend(prompt: str, system: str | None = None) -> str
```

Anything matching that works — Codex CLI (`codex exec`), an OpenAI/Gemini SDK
call, or a local Ollama endpoint. The paper's Table 4 changes only the backbone
across six models with the harness held fixed.

## Known limits

- `kernel.py` runs model-authored code in-process with no sandbox. Fine for
  benchmark runs over local data; use a subprocess or container before running
  anything untrusted.
- BM25 lexical search only, matching the paper. No embeddings.
- `bench.py` implements LongMemEval. BEAM and LOCA-bench are not wired up.
- The judge is an LLM comparing against the benchmark's gold answer. It is not
  the benchmark's official judge prompt, so numbers are indicative and not
  directly comparable to published leaderboard scores.
