# 7. The kernel can run in a child process, and that is not a sandbox

`NARU_KERNEL=sandbox` runs model-authored cells in a child process with
OS-enforced limits. It is opt-in, it is process isolation, and calling it a
sandbox without a jail wrapper would be a lie.

## Context

`kernel.py` executed model-authored Python with `exec` in the harness process.
A runaway loop hung the whole run, a memory bomb took the harness with it, and
a segfault in a C extension killed everything. The README disclosed this and it
was the single thing most likely to stop a stranger running the project.

## Decision

`SandboxedKernel` keeps `Kernel`'s interface — `run(code)`, `digest()`,
persistent namespace — and executes cells in a child `python3` process speaking
one JSON request and one JSON reply per line.

- `RLIMIT_CPU` (30s), `RLIMIT_AS` (1GB), `RLIMIT_FSIZE` (64MB), each
  configurable, plus a parent-side wall-clock timeout for hangs no CPU limit
  catches: `sleep`, a blocking read, a socket.
- The Event Log is reopened **by path** in the child, read-only.
- `submit_answer` and `headline` become stubs that batch their calls into the
  reply; the parent applies them. That keeps the protocol request/response
  rather than duplex RPC.
- A dead child is replaced on the next call and resident state is lost. That is
  the honest cost of isolation, and `run()` says which happened — a segfault
  reported as a timeout sends someone hunting a slow cell that never existed.

## Why opt-in rather than default

`bench.py` ingests each question into `MemorySurface(":memory:")`. An in-memory
database belongs to one process, so no child can open it. Making the sandbox
the default would mean file-backed databases for every benchmark question, for
a threat model — our own dataset, our own machine — that is empty.

The constructor refuses `":memory:"` loudly rather than failing later.

## What it does not do

The child runs as the same user with the same filesystem and the same network.
`rm -rf`, reading a key out of the environment, and an outbound request all
still work. Real capability isolation needs seccomp or a container, neither of
which is in the standard library, so `NARU_KERNEL_JAIL` wraps the child in one
the operator supplies:

```
NARU_KERNEL_JAIL='sandbox-exec -f jail.sb'              # macOS
NARU_KERNEL_JAIL='bwrap --ro-bind / / --unshare-net'    # Linux
NARU_KERNEL_JAIL='docker run --rm -i --network none …'
```

Same seam as `NARU_BACKEND`, and for the same reason: this repo does not ship a
per-platform jail, it ships the place to put one.

## A limit the platform refuses must say so

macOS rejects `RLIMIT_AS` outright with `ValueError`. The first version caught
that and moved on, which left a caller believing memory was capped when it was
not — ADR 0002's mistake in a new place. `limits()` now reports
`"NOT APPLIED: ValueError"` per ceiling.

The self-check for it was worse than useless for one round:

```python
assert isinstance(lim["mem_b"], int) or "NOT APPLIED" in str(lim["mem_b"])
```

That accepts both branches. It passed while the child reported a refused limit
as applied, and `test_mutations.py` caught it. The check now asks the child to
exceed the cap it claims to have.

## Consequences

On macOS memory is **not** capped. CPU time and file size are. The wall-clock
timeout and process isolation work everywhere, and those cover the failure that
actually happens: a cell that never returns.
