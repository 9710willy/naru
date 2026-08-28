# Contributing

Small repo, few rules, and they are the ones a change actually gets rejected
for. Read them before writing code, not after.

## Run this first

```bash
python3 ms.py && python3 kernel.py && python3 eviction.py && python3 agent.py
python3 naru.py --selfcheck && python3 hook_spill.py --selfcheck
python3 noise.py --selfcheck && python3 bench.py --selfcheck
python3 backend.py --selfcheck
python3 test_mutations.py
```

All of it is offline. CI runs the same set on 3.9 and 3.13.

`python3 backend.py` and `python3 test_judge.py` without `--selfcheck` make
live API calls and cost money. They are not in CI.

## No dependencies

Standard library only. This is not a preference, it is the reason the install
is a `git clone` and a symlink. A change that adds a dependency needs to argue
why a few lines cannot cover it, and the answer is usually that they can.

## Every module has a runnable self-check, and it has to bite

`test_mutations.py` breaks the code on purpose and requires the self-check to
fail. It exists because four checks here once passed while the bug they named
was live — two scenarios shared an output buffer, one assertion compared a
value to itself, and two re-typed the production predicate instead of calling
it.

If you add an invariant, add a mutation for it. If your check passes when the
code is broken, it is decoration.

The failure mode to avoid:

```python
# useless — re-types the production expression, stays green when main() changes
assert [x for x in arms if x not in ("full", "rag", "naru")] == ["nauru"]

# useful — calls the function main() calls
assert unknown_arms(["full", "rag", "nauru"]) == ["nauru"]
```

## Benchmark changes

Read `docs/adr/0002` and `docs/adr/0006` first.

- **Never hardcode a measured constant.** A hardcoded harness floor once
  produced a wrong published number, in the direction that flattered us.
- **`rag` is the control.** It is what separates "programmatic access wins"
  from "any retrieval beats stuffing". A run that omits it is flattering
  itself.
- **Report the noise floor.** `separability()` prints a paired McNemar verdict
  with every run, Bonferroni-corrected across the arm pairs. Do not replace it
  with overlapping confidence intervals; that discards the pairing.
- **Do not quote `billed_input` or `net-of-harness` ratios.** The `claude`
  CLI's usage dict does not reconcile across runs. Cost and `view` do.
- Numbers here are not leaderboard-comparable. The judge is not LongMemEval's
  official prompt, and the paper it implements already reports better numbers
  against ten named baselines.

## Decisions that look wrong get an ADR

`docs/adr/NNNN-slug.md`. Title line, then context, decision, why. Several
decisions in this repo look like mistakes to a fresh reader and are not —
`prune` refusing to age out a promoted claim, the doc being a promoted subset
rather than a render of the log, the harness floor being measured every run.

If you find yourself explaining the same thing twice in review, it is an ADR.
If an existing ADR is wrong, rewrite it rather than adding a second one that
argues with it — 0003 is an example.

## Commits

Conventional Commits, plus the Tim Pope rules: subject 50 characters or less,
imperative mood, no trailing period, blank line before the body, body wrapped
at 72 explaining what and why rather than how.

## Scope

Things that will be turned down:

- A per-harness plugin API. Anything that can run a shell command is already
  supported, and ADR 0004 says why that is the whole integration surface.
- Speculative abstractions, config for a value that never changes, an
  interface with one implementation.
- Production code that exists to satisfy a test.

## Security

`kernel.py` executes model-authored Python in-process with no sandbox. That is
a known limitation, disclosed in the README, and it is the highest-value thing
anyone could fix. If you are reporting a vulnerability rather than fixing one,
open an issue describing the class of problem without a working exploit.
