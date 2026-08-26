"""Persistent Python kernel — the resident namespace of a Session Environment.

Variables survive across model calls, so tool outputs and derived state live
here as objects instead of being serialized into the prompt. Only what the
model's code prints crosses back into its working view.
"""

import contextlib
import io
import traceback


def _provenance(v):
    """Which Event Log addresses a resident value derives from.

    Section 2.2 requires resident objects to carry "type, size, and provenance
    metadata identifying the events it derives from". Log rows expose .seq, so
    a container of rows reports its seq range and the model can go straight
    back to the source without re-searching.
    """
    items = list(v)[:200] if isinstance(v, (list, tuple)) else []
    seqs = [i["seq"] for i in items
            if isinstance(i, dict) and isinstance(i.get("seq"), int)]
    if not seqs:
        return ""
    lo, hi = min(seqs), max(seqs)
    return f" from seq {lo}" if lo == hi else f" from seq {lo}-{hi}"


def _shape(v):
    """Compact type+size description for the namespace digest."""
    t = type(v).__name__
    if isinstance(v, (str, bytes)):
        return f"{t}[{len(v)}]"
    if isinstance(v, (list, tuple, set, dict)):
        return f"{t}[{len(v)}]{_provenance(v)}"
    if isinstance(v, (int, float, bool)) or v is None:
        return f"{t}={v!r}"[:40]
    return t


class Kernel:
    """A namespace + exec. Model-authored cells run here.

    ponytail: in-process exec, no sandbox. Fine for benchmark runs on local
    data; move to a subprocess or container before running untrusted code.
    """

    MAX_OUT = 8000  # chars of printed output allowed into the working view

    def __init__(self, **preload):
        self.ns = {"__name__": "__scroll__"}
        self.ns.update(preload)
        self._preloaded = set(preload)

    def run(self, code):
        """Execute a cell. Returns (printed_output, error_or_None)."""
        buf = io.StringIO()
        err = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<cell>", "exec"), self.ns)
        except Exception:
            tb = traceback.format_exc(limit=2).strip().splitlines()
            err = tb[-1] if tb else "error"
        out = buf.getvalue()
        if len(out) > self.MAX_OUT:
            out = out[: self.MAX_OUT] + f"\n[output truncated at {self.MAX_OUT} chars]"
        return out, err

    def digest(self):
        """Short listing of resident variables: name, type, shape.

        Prepended to every model call so the model knows what state it holds
        without paying to re-print it.
        """
        items = [
            f"{k}: {_shape(v)}"
            for k, v in self.ns.items()
            if not k.startswith("_") and k not in self._preloaded and not callable(v)
        ]
        return "resident: " + (", ".join(items) if items else "(empty)")


def demo():
    k = Kernel(helper=lambda x: x * 2)

    # state persists across separate cells
    out, err = k.run("xs = [1, 2, 3]\nprint(sum(xs))")
    assert (out.strip(), err) == ("6", None), (out, err)
    out, err = k.run("xs.append(10)\nprint(len(xs))")
    assert out.strip() == "4", out

    # nothing printed => nothing enters context, even though state changed
    out, err = k.run("big = 'x' * 100000")
    assert out == "" and err is None
    assert len(k.ns["big"]) == 100000

    # digest sees the resident vars, hides preloaded callables
    d = k.digest()
    assert "xs: list[4]" in d and "big: str[100000]" in d, d
    assert "helper" not in d, d

    # errors are captured, not raised, and the kernel survives
    out, err = k.run("1/0")
    assert err and "ZeroDivisionError" in err, err
    out, err = k.run("print('alive')")
    assert out.strip() == "alive"

    # runaway output is capped
    out, _ = k.run("print('z' * 50000)")
    assert len(out) < Kernel.MAX_OUT + 200 and "truncated" in out

    print("ok — kernel checks passed")


if __name__ == "__main__":
    demo()
