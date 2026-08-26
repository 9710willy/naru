"""Model backend via the local `claude` CLI — no API key needed, reuses the
CLI's own auth.

Each call is stateless: we pass the full working view as one prompt, exactly as
an API call would. That is the honest setup for measuring Scroll, whose whole
claim is that the view stays small.
"""

import json
import subprocess
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
class Backend:
    model: str = HAIKU
    timeout: int = 300
    # The CLI intermittently returns an empty result for a valid request. This
    # penalizes multi-turn arms in proportion to their turn count, so a low
    # retry budget silently biases the benchmark against Scroll.
    retries: int = 6
    usage: Usage = field(default_factory=Usage)

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


def measure_floor(model=HAIKU):
    """Input tokens the CLI itself costs per call, before any of our prompt.

    Must be measured, never hardcoded: it moves whenever the CLI flags or its
    built-in system prompt change, and a stale value silently distorts every
    per-arm token comparison.
    """
    b = Backend(model=model)
    b("Reply with one word: ok", system="You reply in one word.")
    return b.usage.billed_input // max(1, b.usage.calls)


def demo():
    """Live check — costs a couple of cheap calls. Also measures the harness
    token floor, so benchmark numbers can be read net of CLI overhead."""
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
