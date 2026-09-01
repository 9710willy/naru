# naru

[![checks](https://github.com/9710willy/naru/actions/workflows/checks.yml/badge.svg)](https://github.com/9710willy/naru/actions/workflows/checks.yml)

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
The CLI owns the data. The Codex plugin below only calls that CLI from native
lifecycle hooks.

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
cd naru && python3 ms.py          # self-check: no network, no writes outside a temp dir
ln -s "$PWD/naru.py" ~/.local/bin/naru   # or anywhere on PATH
```

Every command below is spelled `naru`, so without that symlink none of them
exist. `python3 naru.py <cmd>` works too, but only from the repo directory,
and the agent filing a claim is rarely in it.

### Codex

Install the local marketplace and plugin after `naru` is on `PATH`:

```bash
codex plugin marketplace add "$PWD/codex"
codex plugin add naru-codex@naru
```

Start a new Codex thread and trust the plugin hooks when Codex asks. The plugin
loads the current approved Naru doc at session start, refreshes it after a new
promotion or retirement, and gives the same context to subagents. It stores only
the last doc sequence and fingerprint seen by each Codex session.

The plugin does not replace Codex tool results. Codex 0.152 reports the only
working replacement form as a blocked tool after that tool has already run,
which can make an agent repeat a write or request.

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

## Statusline

Claims sit pending until you run `naru inbox`, and nothing reminds you. `prune`
keeps every decided claim and deletes a pending one after 30 days, so a claim
you never looked at is the only kind that expires. A count in your statusline is
the cheapest reminder.

Add this to your statusline script:

```bash
db="${NARU_DB:-$HOME/.naru/log.db}" cache="${TMPDIR:-/tmp}/naru-pending.$UID"
if [[ $db -nt $cache ]]; then
  sqlite3 -readonly "$db" \
    "SELECT COUNT(*) FROM conversation_history WHERE kind='claim' AND promoted=0" \
    >"$cache" 2>/dev/null
fi
read -r pending <"$cache" 2>/dev/null
((pending > 0)) && printf ' ✉%d' "$pending"
```

The db stays the one source of truth and `-nt` invalidates the cache exactly,
so the count is never stale: only `naru claim`, `naru inbox` and a spill write
the db. On a render that skips the spawn this costs nothing measurable, against
11.5 ms for the uncached query. Use `sqlite3` rather than `naru inbox` when the
cache does miss: `python3 -c pass` alone costs 17 ms.

It prints nothing when the inbox is clear or the store does not exist yet.

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
python3 noise.py --selfcheck && python3 bench.py --selfcheck
python3 backend.py --selfcheck
python3 test_mutations.py   # do those self-checks catch anything?
python3 backend.py      # live: two cheap calls, prints the harness token floor
python3 test_judge.py   # live: judge regression cases
```

## Benchmark

`bench.py` runs LongMemEval (ICLR 2025) over three arms on identical history.
`full` puts the whole history in one prompt. `rag` pastes the top 8 BM25 hits
and answers in one call. `naru` leaves the history in the log for the model to
reach by writing code.

**What this is for.** It checks whether this implementation behaves like the
one in the paper. It is not a contribution to the field and the numbers are not
leaderboard-comparable — the judge is not LongMemEval's official prompt, and the
paper already reports 94.8 on LongMemEval-S, 73.1 on BEAM-10M and 86.7 on
LOCA-256K against ten named baselines. Nothing here improves on that. What it
can tell you is whether a change to this repo broke something.

```bash
mkdir -p data && cd data
curl -LO https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json
cd ..
python3 bench.py --split oracle -n 12
```

### The reference run

LongMemEval-`s`, n=96, Sonnet 5, judge Haiku 4.5. Rows in
[`results/published/`](results/published/), stripped of gold answers.

| arm    | correct | 95% CI | view      | calls | model $/q |
| ------ | ------- | ------ | --------- | ----- | --------- |
| `naru` | 72/96   | 65-83% | 2,174t    | 3.7   | $0.3547   |
| `full` | 64/96   | 57-75% | 124,073t  | 1.0   | $0.8036   |
| `rag`  | 59/96   | 51-71% | 2,142t    | 1.0   | $0.0753   |

`view` is the largest context the model ever held, measured by us. `model $/q`
is the arm's own cost; judge cost is ~$0.027/q across all three and is excluded
so the column measures the arm rather than the grader.

```
separability — paired McNemar, Bonferroni for 3 pairs
    dropped from the pairing (run errored): full 5
    full  vs rag     9.9 pts to full   full only 14, rag only  5   p=0.064   not separable
    full  vs naru    3.3 pts to naru   full only 11, naru only 14  p=0.690   not separable
    rag   vs naru   13.5 pts to naru   rag  only  9, naru only 22  p=0.029   not separable
```

Three arms means three tests, so the threshold is Bonferroni-corrected to
0.0167. `rag` vs `naru` clears a plain 0.05 and does not clear that, so the
harness calls it not separable and is right to.

### Where the difference sits

| category                   | `full` | `rag` | `naru` | naru vs rag |
| -------------------------- | ------ | ----- | ------ | ----------- |
| knowledge-update           | 13/16  | 13/16 | 13/16  | p=1.000     |
| multi-session              | 10/16  | 7/16  | 8/16   | p=1.000     |
| single-session-assistant   | 13/15  | 15/16 | 15/16  | p=1.000     |
| single-session-user        | 14/14  | 15/16 | 15/16  | p=1.000     |
| single-session-preference  | 4/16   | 2/16  | 7/16   | p=0.125     |
| **temporal-reasoning**     | 10/14  | 7/16  | 14/16  | **p=0.039** |

Four categories are a tie. The whole difference is temporal-reasoning and
preference-following — the two where an answer has to be *computed* over the
log rather than quoted from one turn. A top-8 keyword window can retrieve a
turn; it cannot do arithmetic across sessions.

That is the shape the paper predicts, which is what makes this a reproduction
check rather than a result. ADR 0003 records that the paper's own ablation says
the eviction index affects preference-following most (89.1 vs 74.9), so that
category was predicted before it was measured. Temporal-reasoning was not, and
these are post-hoc category tests on six categories — treat the p-values as
descriptive.

### Reading the numbers honestly

- **Do not quote `billed_input` or `net-of-harness` ratios.** The `claude`
  CLI's usage dict is not consistent across runs: the same `full` prompt
  reported 201,288 billed input tokens at n=12 and 47,151 at n=96, for the same
  view size and nearly the same cost. Cost comes from the CLI's own
  `total_cost_usd` and does reconcile; `view` is ours. Those two are safe.
- **A question whose run errored leaves the pairing** rather than scoring as a
  wrong answer. McNemar reads only the discordant pairs, so one CLI timeout
  scored as a loss can turn p=0.125 into p=0.031 and manufacture a significance
  claim. `report()` prints accuracy twice when that happens.
- **Run `noise.py` over two replicate runs** before reading anything into a
  gap. At n=12 the run-to-run band is ~33 points, and two identical Sonnet runs
  disagreed on a third of the questions while landing on the same total. The
  n=24 Haiku run in `results/published/` shows `rag` ahead of `naru`; the n=96
  Sonnet run reverses it. Single-model, small-n orderings from this harness do
  not survive.

## Limitations

Curation has self-checks and no production use.

`kernel.py` executes model-authored code in-process by default.
`NARU_KERNEL=sandbox` moves it to a child process with CPU, memory and file
size limits and a wall-clock timeout, so a runaway loop or a crash takes the
child rather than the run. That is process isolation, not a sandbox: the child
is the same user with the same filesystem and network. `NARU_KERNEL_JAIL`
wraps it in `sandbox-exec`, `bwrap` or a container if you need containment.
macOS refuses `RLIMIT_AS`, so memory is uncapped there and `limits()` says so
rather than reporting a cap that does not exist. See ADR 0007.

Conflict detection is exact string matching on `topic_key`. Contradictory claims
filed under different keys both appear as approved.

One database, one machine. Two databases would break `seq` as a total order.

Search is BM25 only, as in the paper. No embeddings.

The benchmark judge is not LongMemEval's official prompt. Numbers are for
tracking this repo and are not comparable to published scores.

No arm separates from any other on accuracy at any n this harness has run.
`naru` leads at n=96 on Sonnet and `rag` leads at n=24 on Haiku; neither
ordering clears the corrected threshold. `rag` is 4.7x cheaper than `naru` in
money on the n=96 run, and the difference between them sits entirely in two of
six categories. See ADR 0006.

## See also

[`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.
[`docs/adr/`](docs/adr/) for design decisions. [`bench.py`](bench.py) for the
benchmark. [`how-it-works.html`](how-it-works.html) walks the mechanism.

License: MIT
