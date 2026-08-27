# naru

Shared, human-approved context for coding agents.

```
naru claim "<text>" [--key KEY] [--by AGENT]   propose a fact
naru inbox                                     approve or reject pending claims
naru inject [PATH]                             write the approved file
naru promote SEQ --yes | drop SEQ --yes        decide without the prompt
naru search QUERY | show SEQ | outline         read the log
naru stats [DAYS] | prune [DAYS] | gc          maintenance
```

## What it does

An agent files a claim. The claim waits. You run `naru inbox` and approve or
reject it. Approved claims, and nothing else, appear in the file that
`naru inject` writes.

That file goes wherever your harness already reads context from:

```bash
naru inject CLAUDE.md        # Claude Code
naru inject AGENTS.md        # Codex, DeepSeek Harness
naru inject .cursorrules     # Cursor
```

Any tool that can run a shell command can both file claims and read the result.
There is no plugin API.

`inject` splices the doc between `<!-- naru:begin -->` and `<!-- naru:end -->`
markers and leaves the rest of the file alone, so pointing it at a CLAUDE.md you
maintain by hand is safe.

`promote` and `drop` refuse to run without a terminal unless you pass `--yes`.
That stops a tool loop promoting by accident. It is not a security boundary —
anything that can run this CLI can also pass `--yes`.

To revise a promoted fact, retire the old one first: `naru drop <seq> --yes`,
then claim and promote the replacement. Promoting a second value for the same
key without retiring the first parks them both under `## Unresolved`.

## Install

Python 3.9+ and a SQLite built with FTS5. No dependencies.

```bash
git clone https://github.com/9710willy/naru
cd naru && python3 ms.py
```

## Storage

naru implements [Context as an Environment: Programmatic Context Management for
Long-Horizon Agents](https://arxiv.org/abs/2608.21690) (arXiv 2608.21690).
History lives in a SQLite log addressed by sequence number. The model reaches it
by writing Python, and only what that code prints enters the next call.

The paper's system is named Scroll. This repo used the same name until the
ambiguity got in the way of quoting the paper in its own source.

`ms.py` is the log and the Appendix-C surface. `kernel.py` is the persistent
namespace. `eviction.py` is Algorithm 1.

## Schema notes

| Column      | Meaning                                                                                  |
| ----------- | ---------------------------------------------------------------------------------------- |
| `promoted`  | your decision. Rejection marks the row; it stays readable at its `seq`.                  |
| `topic_key` | groups claims about one thing. Two approved claims on a key print under `## Unresolved`. |
| `base_seq`  | log size when the claim was written. `inbox` reports the difference.                     |

## Claude Code hook

`hook_spill.py` is a `PostToolUse` hook. It moves oversized tool output into the
log and leaves a recovery handle in its place, so a large command result costs a
preview instead of the whole payload.

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

## Other models

`backend.py` defaults to the local `claude` CLI. `NARU_BACKEND` replaces it with
any command that reads a prompt on stdin.

```bash
NARU_BACKEND='codex exec -' python3 bench.py --split oracle -n 12
```

Such a backend reports no token counts, so naru omits the net-of-harness column
rather than printing a zero that looks measured.

## Checks

```bash
python3 ms.py && python3 kernel.py && python3 eviction.py && python3 agent.py
python3 naru.py --selfcheck && python3 hook_spill.py --selfcheck
python3 backend.py    # makes two live API calls
```

## Benchmark

`bench.py` runs LongMemEval (ICLR 2025) over two arms on identical history:
`full` puts the whole history in one prompt, `naru` leaves it in the log for the
model to reach by writing code.

```bash
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
cd ..
python3 bench.py --split oracle -n 12
```

The `claude` CLI adds 13-23k input tokens per call before any prompt of ours, so
compare arms on `net-of-harness`. That floor is measured at startup rather than
hardcoded, for the reason recorded in ADR 0002.

Run `noise.py` over two replicate runs before reading anything into a gap. At
n=12 a difference under about 20 points is noise.

## Limitations

Curation has self-checks and no production use.

`kernel.py` executes model-authored code in-process without a sandbox.

Conflict detection is exact string matching on `topic_key`. Contradictory claims
filed under different keys both appear as approved.

One database, one machine. Two databases would break `seq` as a total order.

Search is BM25 only, as in the paper. No embeddings.

The benchmark judge is not LongMemEval's official prompt. Numbers are for
tracking this repo and are not comparable to published scores.

## See also

[`docs/adr/`](docs/adr/) for design decisions. [`bench.py`](bench.py) for the
benchmark.

License: MIT
