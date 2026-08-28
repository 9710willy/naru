# Naru

Implementation of _Context as an Environment_ (arXiv 2608.21690). The agent's
history lives in a SQLite Event Log and a persistent Python kernel; the model
writes code to reach it and only what it prints enters the next call.

Read `docs/adr/` before changing anything — five decisions there are
non-obvious and one of them (0002) exists because a hardcoded constant produced
a wrong published result.

## Before you commit

Every module has a runnable self-check. The first four need no network:

```bash
python3 ms.py && python3 kernel.py && python3 eviction.py && python3 agent.py
python3 naru.py --selfcheck && python3 hook_spill.py --selfcheck
python3 noise.py --selfcheck && python3 bench.py --selfcheck
python3 backend.py --selfcheck
python3 test_mutations.py   # breaks the code on purpose; checks must fail
python3 backend.py    # live: 2 cheap calls, prints the harness token floor
python3 test_judge.py # live: judge regression cases
```

Stdlib only. No dependencies — do not add one for something a few lines cover.

## Benchmark rules

`bench.py` is the only source of numbers. Three arms (`full`, `rag`, `naru`)
over the same questions.

**`rag` is the control and must never be quietly dropped.** It holds the kernel
fixed and varies only what fills the prompt, which is the one thing that
separates "programmatic access wins" from "any retrieval beats stuffing".
Without it, `naru` beating `full` proves nothing. As of ADR 0006 `rag` beats
`naru` on this benchmark on every column, so a run that omits it is flattering
itself.

- **Never hardcode a measured constant.** `--harness-floor` is measured at
  startup for exactly this reason (ADR 0002).
- **Report a noise floor.** `separability()` prints an exact McNemar verdict
  with every run because the arms answer identical questions, so the comparison
  is paired. Do not replace it with overlapping confidence intervals: that
  discards the pairing and is far too conservative. At n=24 nothing separates,
  including a 20-point gap.
- **Three arms means three tests.** The threshold is Bonferroni-corrected;
  uncorrected, one pair reads "REAL" in ~6% of runs where nothing separates.
- **A self-check that passes proves the code runs, not that it checks.** Four
  of these were decorative until `test_mutations.py` broke the code on purpose
  and found they stayed green. Add a mutation with any new invariant.
- **A question whose run errored leaves the pairing.** `correct=False` cannot
  tell a wrong answer from a call that never completed, and McNemar reads only
  the discordant pairs — one timeout scored as a loss turns p=0.125 into
  p=0.031 and publishes a significance claim a hung subprocess invented.
- **All arms must see identical history.** `sessions(q)` is the one source of
  ordering; do not re-derive it. `--arms` rejects an unknown name rather than
  falling through to `full`, because a typo would corrupt a paid run in silence.
- **Published rows live in `results/published/`**, stripped of `gold` and
  `answer`. `results/` is otherwise gitignored.
- The judge is not LongMemEval's official prompt, so numbers are internal
  progress only — never present them as leaderboard-comparable.
- Multi-turn arms take more exposure to per-call flakiness than single-turn
  ones. A low retry budget silently biases against `naru`.

## Gotchas that cost real time

- `claude -p` needs `--allowed-tools ""`; without it the model tries to call a
  tool and every naru question errors on `stop_reason: tool_use` (ADR 0001).
- The CLI leaks its own identity and CLAUDE.md into the agent. The system
  prompt overrides this explicitly — do not remove that paragraph (ADR 0003).
- A `PostToolUse` `updatedToolOutput` that does not match the tool's own output
  schema is discarded **silently**. Mutate the text field in place inside the
  response object; never return a bare string.
- `note` and `hook_spill.py` must share one store (`ms.DEFAULT_DB`), or the
  recovery handle a spill prints points at a database `note` never opens.
- **A new column in `CREATE TABLE` needs an entry in `_migrate_cols` too**, and
  the migration self-check's legacy table must _omit_ it — otherwise the test
  passes while every existing store fails `append()`. `agent_id` had neither,
  and the spill hook swallowed the error, so 72 spills were lost in silence.
- One owner for `NARU_SPILL_THRESHOLD`: the default in `hook_spill.py`. Setting
  it inline on the hook command makes `naru stats` judge the distribution
  against a threshold the hook never ran at.

## Observability

`naru stats [days]` reads `~/.naru/metrics.jsonl`, one appended line per hook
invocation and per recovery. Two signals only exist there, not in the Event Log:

- **skipped outputs** — the spilled/skipped size distribution is the only
  evidence for whether `NARU_SPILL_THRESHOLD` is set right.
- **recoveries** — if spills accumulate and `recoveries used` stays 0, the
  retrieval handle is dead weight and the preview should carry more.

Recording is fire-and-forget: every failure in `metrics.py` is swallowed, because
a metrics problem must never break a tool call. Self-checks must point
`NARU_METRICS` at a temp path — an earlier version wrote test events into the
real store.

## The curation layer (ours, not the paper's)

Agents write claims; a human promotes them; only promoted claims reach the doc.
Three columns carry it, and each exists for a failure the design cannot prevent:

- `promoted` — a decision never deletes. A dropped claim stays addressable.
- `topic_key` — two promoted claims on one key is a contradiction. Show both,
  never auto-resolve. Picking the newer one silently is how the doc starts lying.
- `base_seq` — staleness is undetectable without it and unpreventable with it.
  `inbox` shows how far the log moved while the author worked.

Invariants that cost real time to rediscover:

- **`naru inject` splices between markers.** It must never `write_text()` a
  whole file — the documented path points at a CLAUDE.md the user maintains.
- **`prune` never takes a decided claim.** Claims are stamped with the wall
  clock, so an age-only predicate gives every promotion a 30-day shelf life.
- **`decide(keep=False)` works on a promoted claim.** Without that, superseding
  a fact parks its key under `## Unresolved` forever.
- **`base_seq` is the DOC version, not the log head** (`ms.doc_version()`), or
  an unrelated note reads as "the doc moved under you".
- **`measure_floor` returns None, never 0**, when the probe call fails as well
  as when the backend reports no usage.
- **Self-checks must not read ambient stdin.** `naru.demo()` closes it; the
  pre-commit command used to hang forever on a terminal.

**The doc is a promoted subset, never a render of the log.** If `naru inject`
ever grows with the log, the `full` arm has been rebuilt by accident.

`naru inject <path>` is the whole harness integration. Do not add a per-harness
plugin API — anything that runs a shell command is already supported (ADR 0004).

## Where this diverges from the paper

Both are deliberate and ablatable; do not quietly drop either.

- `ms.outline()` is ours, not the paper's. It exists because §3.3's
  ingestion-time index collapses old sessions into coarse ranges, which suits a
  long trajectory but not LongMemEval's uniformly-scattered evidence. `--no-index`
  ablates the paper's version. See ADR 0003.
- `search()` retries a failed multi-term AND as OR. The paper does not.

## Style

Python: the `python-perf-guide` skill carries the standard. Its measurement
checklist applies directly to `bench.py`.
