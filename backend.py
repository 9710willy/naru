"""Provider-neutral model backends.

The benchmark can use an automatically selected local provider, the
authenticated Codex CLI, the legacy Claude CLI, or any command that reads a
prompt on stdin. No backend credential is copied into benchmark rows.

    python3 bench.py --backend auto --split oracle -n 12

Each call is stateless: we pass the full working view as one prompt, exactly as
an API call would. That is the honest setup for measuring Naru, whose whole
claim is that the view stays small.
"""

import json
import hashlib
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

HAIKU = "claude-haiku-4-5-20251001"
BACKEND_CHOICES = ("auto", "codex", "claude", "command")
_NAMED_BACKENDS = frozenset(BACKEND_CHOICES)


def default_model_for_backend(kind):
    """Return a truthful display/default model for a resolved provider."""
    if kind == "claude":
        return HAIKU
    if kind == "codex":
        return os.environ.get("NARU_CODEX_MODEL") or "codex-configured"
    return "provider-default"


def _estimate(text):
    return max(1, len(text) // 4)


def _numeric(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return value


def backend_fingerprint(kind, command=None):
    """Return a safe identifier for the resolved provider configuration."""
    raw = f"{kind}\0{command or ''}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]

# Flags that strip the Claude Code persona/tooling so the model behaves as a
# plain completion endpoint rather than a coding agent.
_BARE = [
    # Keep OAuth auth but disable CLAUDE.md, memory, plugins, hooks and other
    # ambient customizations. Otherwise the curation probe's plain arm can see
    # the same Naru facts it is meant to exclude.
    "--safe-mode",
    "--output-format",
    "json",
    "--exclude-dynamic-system-prompt-sections",
    # The current CLI treats an empty allowed-tools list as no restriction.
    # An empty tools list is the documented disable-all form.
    "--tools",
    "",
    # Every call here is a full Claude Code session, so it fires the USER's
    # hooks. With a Stop hook wired to a notifier, one n=48 run means 600+
    # desktop notifications. A backend must not touch the user's environment.
    "--settings",
    json.dumps({"disableAllHooks": True}),
]


@dataclass
class Usage:
    attempts: int = 0
    calls: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    prompt_tokens_estimated: int = 0
    peak_prompt_tokens_estimated: int = 0
    errors: int = 0
    empty_retries: int = 0
    call_retries: int = 0
    cost_measured: bool = True
    native_usage: dict = field(default_factory=dict)

    def add(self, u, cost):
        u = u if isinstance(u, dict) else {}
        self.calls += 1
        self.input_tokens += int(_numeric(u.get("input_tokens", 0)))
        self.cache_read += int(
            _numeric(
                u.get("cache_read_input_tokens", u.get("cached_input_tokens", 0))
            )
        )
        self.cache_creation += int(
            _numeric(u.get("cache_creation_input_tokens", 0))
        )
        self.output_tokens += int(_numeric(u.get("output_tokens", 0)))
        if cost is None:
            self.cost_measured = False
        else:
            self.cost_usd += float(_numeric(cost))
        for key, value in u.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            self.native_usage[key] = self.native_usage.get(key, 0) + value

    @property
    def billed_input(self):
        """Every input token the model was charged for, cache included."""
        return self.input_tokens + self.cache_read + self.cache_creation

    def normalized(self):
        """Return provider-neutral usage fields without inventing prices.

        ``context_input_tokens`` is our character-based prompt estimate. The
        provider-native counters remain beside it because cache accounting is
        not consistent across generic backends or CLI versions.
        """
        return {
            "context_input_tokens": self.prompt_tokens_estimated,
            "context_input_tokens_source": "estimated_prompt_chars_per_4",
            "uncached_input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read,
            "cache_write_tokens": self.cache_creation,
            "billed_input_tokens": self.billed_input,
            "output_tokens": self.output_tokens,
            "retry_count": self.call_retries + self.empty_retries,
            "cost_measured": self.cost_measured,
        }

    def __str__(self):
        return (
            f"{self.attempts} calls | in {self.billed_input:,} "
            f"(fresh {self.input_tokens:,}) | out {self.output_tokens:,} "
            + (f"| ${self.cost_usd:.3f}" if self.cost_measured else "| cost unknown")
            + (f" | {self.errors} err" if self.errors else "")
            + (f" | {self.empty_retries} empty-retry" if self.empty_retries else "")
        )


@dataclass
class _Retrying:
    """Shared call loop: one prompt in, the model's text out.

    Subclasses supply `_once`. The retry budget lives here because a CLI
    intermittently returns an empty result for a valid request, which penalizes
    multi-turn arms in proportion to their turn count — a low budget silently
    biases the benchmark against naru.
    """

    timeout: int = 300
    retries: int = 6
    usage: Usage = field(default_factory=Usage)
    # Whether `usage` means anything. A generic pipe cannot report tokens, and
    # a run must not print $0.000 as though the calls were free.
    reports_tokens = True
    _warned_failure = False

    def __call__(self, prompt, system=None, nudge=None):
        """Send one prompt, return the model's text.

        Retries a genuinely empty reply, which the CLI produces sporadically
        for a valid request, AND a hard failure — timeout, non-zero exit,
        unparseable output — which `_once` reports as None.

        Hard failures used to return immediately, on the reasoning that
        hammering a broken command only multiplies the wait. A broken command
        is already rejected by `__post_init__`, so what that actually skipped
        was the transient case, and a transient failure costs an arm one whole
        question in proportion to its turn count: at a 6% per-call rate a
        single-call arm loses 6% of its questions and a 3.3-call arm loses 20%.
        An n=96 Sonnet run lost five questions that way before this changed.

        `nudge` is appended on retry, because an identical retry tends to come
        back empty again; the caller supplies it, since the right nudge for a
        code-writing turn is the wrong one for a one-word judge verdict.
        """
        for attempt in range(self.retries):
            p = prompt if attempt == 0 or not nudge else f"{prompt}\n\n{nudge}"
            supplied = f"{system}\n\n{p}" if system else p
            estimated = _estimate(supplied)
            self.usage.attempts += 1
            self.usage.prompt_tokens_estimated += estimated
            self.usage.peak_prompt_tokens_estimated = max(
                self.usage.peak_prompt_tokens_estimated, estimated
            )
            out = self._once(p, system)
            if out is None:
                self.usage.cost_measured = False
                self.usage.call_retries += 1
                continue
            if out.strip():
                return out
            self.usage.empty_retries += 1
        # Exhausting the budget loses a whole turn. Count it as the error it is
        # rather than returning "" as if the model had nothing to say. `errors`
        # therefore means "this question permanently lost a call", which is the
        # signal bench.separability() drops a row on — a failure the retry
        # recovered from must not disqualify a good answer.
        self.usage.errors += 1
        return ""

    def _once(self, prompt, system=None):
        raise NotImplementedError

    def close(self):
        """Release backend-owned resources. Stateless backends have none."""

    def _run(
        self, argv, prompt, catch=(subprocess.TimeoutExpired,), cwd=None, env=None
    ):
        """Run argv with `prompt` on stdin. Returns stdout, or None on failure.

        One implementation of run-and-classify for all CLI backends. On failure it
        surfaces the command's own stderr ONCE per backend: `__post_init__`
        exists so a bad NARU_BACKEND cannot become a silent run of empty
        answers, and swallowing every runtime failure would put that back.
        """
        try:
            p = subprocess.run(
                argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=cwd,
                env=env,
            )
        except catch as e:
            # Not counted as an error here: __call__ owns that, and only once
            # the retry budget is gone. Counting per attempt would make
            # `errors` mean "a call failed somewhere", and a recovered blip
            # would then disqualify a perfectly good answer.
            self._complain(f"{type(e).__name__}: {e}")
            return None
        if p.returncode != 0:
            self._complain(f"exit {p.returncode}: {(p.stderr or '').strip()[:300]}")
            return None
        return p.stdout

    def _complain(self, msg):
        """First failure only. A benchmark makes hundreds of calls; the first
        one explains the problem and the rest are noise."""
        if not self._warned_failure:
            self._warned_failure = True
            print(f"backend failure (first only): {msg}", file=sys.stderr)


@dataclass
class Backend(_Retrying):
    """The local `claude` CLI. Reports real token counts and cost."""

    model: str = HAIKU

    @property
    def label(self):
        return self.model

    def _once(self, prompt, system=None):
        cmd = ["claude", "-p", "--model", self.model, *_BARE]
        if system:
            cmd += ["--system-prompt", system]
        out = self._run(cmd, prompt)
        if out is None:
            return None
        try:
            d = json.loads(out)
        except json.JSONDecodeError:
            self._complain("unparseable JSON on stdout")
            return None
        # Usage is added before the error check on purpose: an errored call
        # still burned tokens and still costs money.
        self.usage.add(d.get("usage") or {}, d.get("total_cost_usd"))
        if d.get("is_error"):
            # None, not the payload: on is_error the result field carries the
            # error text, not an answer. Returning it fed an error message to
            # the judge as though the model had answered. __call__ retries.
            self._complain(f"is_error: {str(d.get('result'))[:200]}")
            return None
        return d.get("result") or ""


def _decode_codex_output(output):
    """Extract the final agent message and usage from Codex JSONL events."""
    final = None
    usage = {}
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    final = text
        elif event.get("type") == "turn.completed":
            reported = event.get("usage")
            if isinstance(reported, dict):
                usage = reported
        elif event.get("type") == "error":
            message = event.get("message")
            if isinstance(message, str):
                raise ValueError(message[:300])
    if final is None:
        raise ValueError("Codex returned no final agent message")
    return final, usage


@dataclass
class CodexBackend(_Retrying):
    """Run the authenticated Codex CLI and keep only its final answer."""

    model: str = ""
    reports_tokens = True

    def __post_init__(self):
        # Codex is an agent CLI, not a no-tools completion endpoint. Give it an
        # empty workspace so a benchmark prompt cannot lead it to the Naru
        # checkout or the LongMemEval data. The flags below also keep ambient
        # user/project instructions out of the benchmark contract.
        self._workspace = tempfile.TemporaryDirectory(prefix="naru-codex-")
        self._cwd = self._workspace.name

    def _isolated_env(self):
        """Remove Naru's benchmark controls from the agent's shell view."""
        return {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("NARU_")
        }

    def close(self):
        workspace = getattr(self, "_workspace", None)
        self._workspace = None
        if workspace is not None:
            workspace.cleanup()

    @property
    def label(self):
        return self.model or "codex-configured"

    def _once(self, prompt, system=None):
        cmd = [
            "codex",
            "exec",
            "--json",
            "--ephemeral",
            "--sandbox",
            "read-only",
            "--cd",
            self._cwd,
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "-",
        ]
        if system:
            # `codex exec` accepts dynamic developer instructions through the
            # config override. Keep the Naru contract out of the user message
            # so it has the same authority boundary as Claude's system flag.
            cmd[2:2] = [
                "--config",
                f"developer_instructions={json.dumps(system)}",
            ]
        if self.model:
            cmd[2:2] = ["--model", self.model]
        out = self._run(
            cmd,
            prompt,
            cwd=self._cwd,
            env=self._isolated_env(),
        )
        if out is None:
            return None
        try:
            answer, usage = _decode_codex_output(out)
        except ValueError as error:
            self._complain(str(error))
            return None
        self.usage.add(
            {
                "input_tokens": usage.get("input_tokens", 0),
                "cached_input_tokens": usage.get("cached_input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "reasoning_output_tokens": usage.get("reasoning_output_tokens", 0),
            },
            None,
        )
        return answer


@dataclass
class CommandBackend(_Retrying):
    """Any CLI that reads a prompt on stdin and writes the reply on stdout.

    Examples: ``NARU_BACKEND='codex'`` or
    ``NARU_BACKEND='ollama run llama3'``.

    The system prompt is prepended to the user prompt rather than passed as a
    flag: every model understands that, and no two CLIs spell the flag alike.
    """

    cmd: str = ""
    reports_tokens = False

    def __post_init__(self):
        """Validate once, here, at the trust boundary.

        Parsing per call and swallowing the error would turn one typo in
        NARU_BACKEND — an unbalanced quote, a binary that isn't installed —
        into a whole benchmark run of empty answers with nothing raised.
        """
        self.argv = shlex.split(self.cmd)
        if not self.argv:
            raise ValueError("NARU_BACKEND is empty; expected a command to run")
        if shutil.which(self.argv[0]) is None:
            raise FileNotFoundError(f"NARU_BACKEND command not found: {self.argv[0]!r}")
        self.usage.cost_measured = False

    @property
    def label(self):
        # Labels are written to result files and curation output. Keep command
        # arguments, which may contain credentials, out of those artifacts.
        return (self.argv or ["command"])[0]

    def _once(self, prompt, system=None):
        if system:
            prompt = f"{system}\n\n{prompt}"
        out = self._run(self.argv, prompt, catch=(subprocess.TimeoutExpired, OSError))
        if out is None:
            return None
        self.usage.calls += 1
        return out.strip()


_WARNED = set()


def _command_label(command):
    """Return argv[0] without copying command arguments into diagnostics."""
    try:
        return (shlex.split(command) or ["command"])[0]
    except ValueError:
        return "command"


def resolve_backend(requested=None, command=None, model=None):
    """Resolve a provider choice to ``(kind, command)``.

    ``NARU_BACKEND`` remains a compatibility override. Its named values select
    an adapter; any other value is treated as a generic stdin command. With no
    override, an explicit Claude model selects Claude, then ``auto`` prefers
    an installed Codex CLI and then Claude. The choice is resolved once by the
    parent benchmark process and inherited by workers, so a run cannot silently
    mix providers.
    """
    configured = (os.environ.get("NARU_BACKEND") or "").strip()
    choice = requested.strip() if isinstance(requested, str) else requested

    if choice in (None, "", "auto"):
        if configured and configured != "auto":
            choice = configured
        elif isinstance(model, str) and model.startswith("claude-"):
            return "claude", None
        elif shutil.which("codex"):
            return "codex", None
        elif shutil.which("claude"):
            return "claude", None
        else:
            raise FileNotFoundError(
                "no supported model provider found; install codex or claude, "
                "or pass --backend command --backend-command COMMAND"
            )

    if choice == "codex":
        return "codex", None
    if choice == "claude":
        return "claude", None
    if choice == "command":
        selected = (command or (
            configured if configured and configured not in _NAMED_BACKENDS else None
        ) or "").strip()
        if not selected:
            raise ValueError(
                "--backend command requires --backend-command or "
                "NARU_BACKEND"
            )
        return "command", selected

    # A direct command remains accepted for library and environment
    # compatibility. The CLI uses the explicit `command` spelling instead.
    return "command", choice


def get_backend(model=None, backend=None, command=None):
    """Build the selected provider adapter.

    ``backend`` may be ``auto``, ``codex``, ``claude``, or ``command``. Passing
    no value uses ``NARU_BACKEND`` when present and otherwise auto-detects a
    local provider. The old ``NARU_BACKEND=<arbitrary command>`` form remains
    supported for callers outside the benchmark.
    """
    kind, selected_command = resolve_backend(backend, command, model=model)
    if kind == "claude":
        return Backend(model=model or HAIKU)
    if kind == "codex":
        requested = None
        if model and model != "codex-configured" and not model.startswith("claude-"):
            requested = model
        if not requested:
            requested = os.environ.get("NARU_CODEX_MODEL")
        return CodexBackend(model=requested or "")
    # Once per command, not once per construction. bench.py builds a backend
    # per question per arm, so warning unguarded here put ~200 identical lines
    # on stderr for a single n=48 run. The check-then-add races under bench.py's
    # thread pool; losing that race prints the line twice, which is harmless.
    if selected_command not in _WARNED:
        _WARNED.add(selected_command)
        print(
            f"backend: {_command_label(selected_command)!r} — a generic pipe reports no "
            "usage, so token and cost columns are not measurable",
            file=sys.stderr,
        )
    return CommandBackend(cmd=selected_command)


def measure_floor(model=None, backend=None, command=None):
    """Input tokens the CLI itself costs per call, before any of our prompt.

    Must be measured, never hardcoded: it moves whenever the CLI flags or its
    built-in system prompt change, and a stale value silently distorts every
    per-arm token comparison.

    Returns None when the floor could not be measured — a generic pipe that
    reports no usage at all, OR a probe call that failed. Never 0: a zero flows
    into the net-of-harness subtraction and prints as though a floor had been
    measured, which is exactly the wrong number ADR 0002 exists to prevent.
    """
    b = get_backend(model, backend=backend, command=command)
    try:
        if not b.reports_tokens:
            return None
        result = b("Reply with one word: ok", system="You reply in one word.")
        # `calls` only advances when usage was actually recorded. A nonzero
        # exit, unparseable JSON, a timeout, expired auth or a rate limit all
        # leave it at zero — and `// max(1, 0)` used to turn that into a
        # confident 0.
        if not result or b.usage.errors or not b.usage.calls:
            return None
        return b.usage.billed_input // b.usage.calls
    finally:
        b.close()


def demo(live=True):
    """Check the backend. `live=False` runs only the parts that need no network.

    The offline half covers the plumbing, the bad-command rejection and the
    retry semantics, and it is the half CI can run. Splitting it out is not
    cosmetic: the retry fix below is exactly the kind of thing that rots
    unnoticed when its only check costs money to run.

    The live half costs a couple of cheap calls and measures the harness token
    floor, so benchmark numbers can be read net of CLI overhead.
    """
    native = Usage(prompt_tokens_estimated=10)
    native.add(
        {
            "input_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 5,
            "output_tokens": 6,
            "provider_specific": 7,
        },
        0.2,
    )
    assert native.native_usage["provider_specific"] == 7
    assert native.normalized() == {
        "context_input_tokens": 10,
        "context_input_tokens_source": "estimated_prompt_chars_per_4",
        "uncached_input_tokens": 3,
        "cache_read_tokens": 4,
        "cache_write_tokens": 5,
            "billed_input_tokens": 12,
            "output_tokens": 6,
            "retry_count": 0,
            "cost_measured": True,
        }
    codex_answer, codex_usage = _decode_codex_output(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "t"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 4,
                            "output_tokens": 2,
                        },
                    }
                ),
            ]
        )
    )
    assert codex_answer == "ok" and codex_usage["cached_input_tokens"] == 4
    assert resolve_backend("claude") == ("claude", None)
    assert resolve_backend("codex") == ("codex", None)
    assert resolve_backend("command", "cat") == ("command", "cat")
    assert default_model_for_backend("claude") == HAIKU
    assert default_model_for_backend("command") == "provider-default"
    _saved_backend = os.environ.pop("NARU_BACKEND", None)
    try:
        assert resolve_backend("auto", model=HAIKU) == ("claude", None)
        explicit_claude = get_backend(HAIKU)
        assert isinstance(explicit_claude, Backend)
        explicit_claude.close()
    finally:
        if _saved_backend is not None:
            os.environ["NARU_BACKEND"] = _saved_backend
    _old_backend = os.environ.get("NARU_BACKEND")
    os.environ["NARU_BACKEND"] = "cat"
    try:
        assert resolve_backend("auto") == ("command", "cat")
    finally:
        if _old_backend is None:
            os.environ.pop("NARU_BACKEND", None)
        else:
            os.environ["NARU_BACKEND"] = _old_backend

    _saved_backend = os.environ.get("NARU_BACKEND")
    os.environ["NARU_BACKEND"] = "secret-benchmark-command"
    isolated = CodexBackend(model="test-model")
    seen = {}
    isolated._run = lambda argv, prompt, **kwargs: (
        seen.update(argv=argv, prompt=prompt, **kwargs)
        or "\n".join(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                ),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                ),
            ]
        )
    )
    try:
        assert isolated("Q", system="S") == "ok"
        assert seen["prompt"] == "Q"
        assert any(
            argument == 'developer_instructions="S"'
            for argument in seen["argv"]
        )
        assert "--ignore-user-config" in seen["argv"]
        assert "--ignore-rules" in seen["argv"]
        assert "--cd" in seen["argv"]
        assert seen["cwd"] == isolated._cwd
        assert "NARU_BACKEND" not in seen["env"]
        assert os.path.realpath(isolated._cwd) != os.path.dirname(
            os.path.realpath(__file__)
        )
    finally:
        isolated.close()
        if _saved_backend is None:
            os.environ.pop("NARU_BACKEND", None)
        else:
            os.environ["NARU_BACKEND"] = _saved_backend
    # Offline first: any stdin->stdout command is a valid backend. `cat` echoes
    # the prompt, which is enough to prove the plumbing without a network call.
    echo = CommandBackend(cmd="cat")
    assert echo("PING", system="SYS") == "SYS\n\nPING", "system prompt not prepended"
    assert not echo.reports_tokens, "a generic pipe must not claim token counts"
    assert echo.usage.attempts == 1
    assert echo.usage.prompt_tokens_estimated == _estimate("SYS\n\nPING")
    assert echo.usage.peak_prompt_tokens_estimated == _estimate("SYS\n\nPING")
    # A bad NARU_BACKEND must fail at construction, not yield a silent run of
    # empty answers that reads as "the model had nothing to say".
    for bad in ("definitely-not-a-real-binary", "", 'sh -c "unbalanced'):
        try:
            CommandBackend(cmd=bad)
            raise AssertionError(f"accepted a bad backend command: {bad!r}")
        except (ValueError, FileNotFoundError):
            pass

    # A transient hard failure must be retried, not surrendered. Before this,
    # `_once` returning None ended the call, so a blip cost the question — and
    # cost it in proportion to an arm's turn count, biasing the benchmark
    # against the multi-turn arm exactly as CLAUDE.md warns.
    import pathlib as _pathlib
    import stat as _stat
    import tempfile as _tempfile

    _tmp = _tempfile.TemporaryDirectory()
    _d = _pathlib.Path(_tmp.name)
    _n = _d / "n"
    _flaky = _d / "flaky.sh"
    _flaky.write_text(
        "#!/bin/bash\ncat > /dev/null\n"
        f"n=$(cat {_n} 2>/dev/null || echo 0)\necho $((n+1)) > {_n}\n"
        'if [ "$n" -lt 2 ]; then exit 1; fi\necho "recovered"\n'
    )
    _flaky.chmod(_flaky.stat().st_mode | _stat.S_IEXEC)
    _fb = CommandBackend(cmd=str(_flaky))
    assert _fb("q").strip() == "recovered", "a transient failure must be retried"
    assert _fb.usage.call_retries == 2, _fb.usage.call_retries
    assert _fb.usage.errors == 0, "a recovered blip is not a lost question"

    _dead = _d / "dead.sh"
    _dead.write_text("#!/bin/bash\ncat > /dev/null\nexit 1\n")
    _dead.chmod(_dead.stat().st_mode | _stat.S_IEXEC)
    _db = CommandBackend(cmd=str(_dead))
    assert _db("q") == ""
    # one lost question, not one per attempt: bench.separability() drops a row
    # on errors, so counting per attempt would be the same verdict either way,
    # but report()'s error line would read six times too high.
    assert _db.usage.errors == 1, _db.usage.errors

    class FailedFloor(_Retrying):
        def _once(self, prompt, system=None):
            self.usage.add({"input_tokens": 100}, 0)
            return None

    failed_floor = FailedFloor(retries=1)
    real_get_backend = globals()["get_backend"]
    globals()["get_backend"] = lambda *_args, **_kwargs: failed_floor
    try:
        assert measure_floor() is None
    finally:
        globals()["get_backend"] = real_get_backend

    # The "no usage" warning is per command, not per construction. bench.py
    # builds one backend per question per arm, so this is the difference
    # between one line of stderr and roughly two hundred.
    import contextlib
    import io

    _old_backend = os.environ.get("NARU_BACKEND")
    os.environ["NARU_BACKEND"] = "cat"
    _WARNED.discard("cat")
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        for _ in range(20):
            assert isinstance(get_backend(), CommandBackend)
    assert err.getvalue().count("generic pipe") == 1, (
        f"warned {err.getvalue().count('generic pipe')} times, expected 1"
    )
    secret_command = "sh -c 'echo sk-secret'"
    _WARNED.discard(secret_command)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert isinstance(
            get_backend(backend="command", command=secret_command),
            CommandBackend,
        )
    assert "sk-secret" not in err.getvalue(), err.getvalue()
    if _old_backend is None:
        os.environ.pop("NARU_BACKEND", None)
    else:
        os.environ["NARU_BACKEND"] = _old_backend
    print("ok — generic command backend (offline)")
    if not live:
        return

    b = Backend(model=HAIKU)

    out = b("Reply with exactly one word: PONG", system="You reply in one word.")
    assert "PONG" in out.upper(), repr(out)
    floor_in = b.usage.billed_input

    out2 = b(
        "What is 17 + 25? Reply with digits only.", system="You reply with digits only."
    )
    assert "42" in out2, repr(out2)

    print(f"ok — backend live. {b.usage}")
    print(
        f"     harness input floor ~{floor_in:,} tok/call "
        f"(subtract when reading benchmark token counts)"
    )


def _check_claude_isolation():
    """Check that the Claude subprocess gets the isolation flags."""
    backend = Backend(model=HAIKU)
    seen = {}
    backend._run = lambda argv, prompt: (
        seen.update(argv=argv, prompt=prompt)
        or json.dumps({"result": "isolated", "usage": {}, "total_cost_usd": 0})
    )
    assert backend("QUESTION", system="SYSTEM") == "isolated"
    assert "--safe-mode" in seen["argv"], seen["argv"]
    assert "--tools" in seen["argv"], seen["argv"]
    assert "--allowed-tools" not in seen["argv"], seen["argv"]
    assert seen["prompt"] == "QUESTION"


if __name__ == "__main__":
    _check_claude_isolation()
    demo(live="--selfcheck" not in sys.argv)
