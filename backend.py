"""Model backends. Defaults to the local `claude` CLI — no API key needed,
reuses the CLI's own auth — and falls back to any command that reads a prompt
on stdin, so the benchmark runs against a model this repo has never heard of.

    NARU_BACKEND='codex exec -' python3 bench.py --split oracle -n 12

Each call is stateless: we pass the full working view as one prompt, exactly as
an API call would. That is the honest setup for measuring Naru, whose whole
claim is that the view stays small.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"
OPUS = "claude-opus-5"

# Flags that strip the Claude Code persona/tooling so the model behaves as a
# plain completion endpoint rather than a coding agent.
_BARE = [
    "--output-format",
    "json",
    "--exclude-dynamic-system-prompt-sections",
    # Empty allowlist removes every tool. Without this the model tries to CALL
    # a tool instead of emitting a code block, and the run errors on stop_reason
    # 'tool_use'. We want a plain text completion.
    "--allowed-tools",
    "",
    # Every call here is a full Claude Code session, so it fires the USER's
    # hooks. With a Stop hook wired to a notifier, one n=48 run means 600+
    # desktop notifications. A backend must not touch the user's environment.
    "--settings",
    json.dumps({"disableAllHooks": True}),
]


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cache_read: int = 0
    cache_creation: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    errors: int = 0
    empty_retries: int = 0

    def add(self, u, cost):
        self.calls += 1
        self.input_tokens += u.get("input_tokens", 0)
        self.cache_read += u.get("cache_read_input_tokens", 0)
        self.cache_creation += u.get("cache_creation_input_tokens", 0)
        self.output_tokens += u.get("output_tokens", 0)
        self.cost_usd += cost or 0.0

    @property
    def billed_input(self):
        """Every input token the model was charged for, cache included."""
        return self.input_tokens + self.cache_read + self.cache_creation

    def __str__(self):
        return (
            f"{self.calls} calls | in {self.billed_input:,} "
            f"(fresh {self.input_tokens:,}) | out {self.output_tokens:,} "
            f"| ${self.cost_usd:.3f}"
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

    def __call__(self, prompt, system=None):
        """Send one prompt, return the model's text. Retries an empty reply,
        which the CLI produces sporadically for an otherwise valid request."""
        for attempt in range(self.retries):
            # Nudge the prompt on retry; an identical retry tends to come back
            # empty again.
            p = (
                prompt
                if attempt == 0
                else (f"{prompt}\n\n(Reply with one ```python code block.)")
            )
            out = self._once(p, system)
            if out.strip():
                return out
            self.usage.empty_retries += 1
        # Exhausting the budget loses a whole turn. Count it as the error it is
        # rather than returning "" as if the model had nothing to say.
        self.usage.errors += 1
        return ""

    def _once(self, prompt, system=None):
        raise NotImplementedError


@dataclass
class Backend(_Retrying):
    """The local `claude` CLI. Reports real token counts and cost."""

    model: str = HAIKU

    def _once(self, prompt, system=None):
        cmd = ["claude", "-p", "--model", self.model, *_BARE]
        if system:
            cmd += ["--system-prompt", system]
        try:
            p = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=self.timeout
            )
        except subprocess.TimeoutExpired:
            self.usage.errors += 1
            return ""
        if p.returncode != 0:
            self.usage.errors += 1
            return ""
        try:
            d = json.loads(p.stdout)
        except json.JSONDecodeError:
            self.usage.errors += 1
            return ""
        self.usage.add(d.get("usage") or {}, d.get("total_cost_usd"))
        if d.get("is_error"):
            self.usage.errors += 1
        return d.get("result") or ""


@dataclass
class CommandBackend(_Retrying):
    """Any CLI that reads a prompt on stdin and writes the reply on stdout.

        NARU_BACKEND='codex exec -'        NARU_BACKEND='ollama run llama3'

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

    def _once(self, prompt, system=None):
        if system:
            prompt = f"{system}\n\n{prompt}"
        try:
            p = subprocess.run(
                self.argv,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except (subprocess.TimeoutExpired, OSError):
            self.usage.errors += 1
            return ""
        if p.returncode != 0:
            self.usage.errors += 1
            return ""
        self.usage.calls += 1
        return p.stdout.strip()


def get_backend(model=HAIKU):
    """The backend for this machine.

    Defaults to the `claude` CLI. NARU_BACKEND replaces it with any command
    that reads a prompt on stdin, which is what makes the benchmark runnable
    against a model this repo has never heard of.
    """
    cmd = os.environ.get("NARU_BACKEND")
    if not cmd:
        return Backend(model=model)
    print(
        f"backend: {cmd!r} — a generic pipe reports no usage, so token and "
        "cost columns will read as zero rather than as measured",
        file=sys.stderr,
    )
    return CommandBackend(cmd=cmd)


def measure_floor(model=HAIKU):
    """Input tokens the CLI itself costs per call, before any of our prompt.

    Must be measured, never hardcoded: it moves whenever the CLI flags or its
    built-in system prompt change, and a stale value silently distorts every
    per-arm token comparison.
    """
    b = get_backend(model)
    if not b.reports_tokens:
        # None, not 0. Zero would flow into the net-of-harness subtraction and
        # print as though a floor had been measured — the same class of wrong
        # number ADR 0002 exists to prevent. "Not measurable" must stay visible.
        return None
    b("Reply with one word: ok", system="You reply in one word.")
    return b.usage.billed_input // max(1, b.usage.calls)


def demo():
    """Live check — costs a couple of cheap calls. Also measures the harness
    token floor, so benchmark numbers can be read net of CLI overhead."""
    # Offline first: any stdin->stdout command is a valid backend. `cat` echoes
    # the prompt, which is enough to prove the plumbing without a network call.
    echo = CommandBackend(cmd="cat")
    assert echo("PING", system="SYS") == "SYS\n\nPING", "system prompt not prepended"
    assert not echo.reports_tokens, "a generic pipe must not claim token counts"
    # A bad NARU_BACKEND must fail at construction, not yield a silent run of
    # empty answers that reads as "the model had nothing to say".
    for bad in ("definitely-not-a-real-binary", "", 'sh -c "unbalanced'):
        try:
            CommandBackend(cmd=bad)
            raise AssertionError(f"accepted a bad backend command: {bad!r}")
        except (ValueError, FileNotFoundError):
            pass
    print("ok — generic command backend (offline)")

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


if __name__ == "__main__":
    demo()
