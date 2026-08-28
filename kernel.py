"""Persistent Python kernel — the resident namespace of a Session Environment.

Variables survive across model calls, so tool outputs and derived state live
here as objects instead of being serialized into the prompt. Only what the
model's code prints crosses back into its working view.
"""

import contextlib
import io
import json
import os
import pathlib
import shlex
import subprocess
import sys
import threading
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
        self.ns = {"__name__": "__naru__"}
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


# ---------------------------------------------------------------------------
# The child program. Reads one JSON request per line on stdin, writes one JSON
# reply per line on the duplicated stdout, and keeps its namespace between
# requests so state survives across cells exactly as the in-process Kernel does.
_CHILD = r'''
import contextlib, io, json, os, resource, sys, traceback

sys.path.insert(0, os.environ["NARU_KERNEL_PATH"])
from kernel import _shape

# Frame replies down a private copy of stdout, then point stdout at devnull so
# a stray print at import time cannot corrupt the stream the parent parses.
_REPLY = os.fdopen(os.dup(1), "w")
os.dup2(os.open(os.devnull, os.O_WRONLY), 1)


class _Done(Exception):
    """Raised by the submit_answer stub to stop the rest of the cell."""


def _limits():
    """OS-enforced ceilings, and an honest report of which ones took.

    macOS refuses RLIMIT_AS outright. Swallowing that leaves a caller believing
    memory is capped when it is not, which is the failure this whole repo is
    built to avoid, so the result is returned rather than discarded.
    """
    applied = {}
    for name, what, want in (
        ("cpu_s", resource.RLIMIT_CPU, int(os.environ.get("NARU_KERNEL_CPU", "30"))),
        ("mem_b", resource.RLIMIT_AS,
         int(os.environ.get("NARU_KERNEL_MEM_MB", "1024")) * 1024 * 1024),
        ("fsize_b", resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
    ):
        try:
            hard = resource.getrlimit(what)[1]
            soft = want if hard < 0 else min(want, hard)
            resource.setrlimit(what, (soft, hard))
            applied[name] = resource.getrlimit(what)[0]
        except (ValueError, OSError, AttributeError) as e:
            applied[name] = f"NOT APPLIED: {type(e).__name__}"
    return applied


def main():
    limits = _limits()
    calls = []
    ns = {"__name__": "__naru__"}
    db = os.environ.get("NARU_KERNEL_DB")
    if db:
        from ms import MemorySurface

        ns["ms"] = MemorySurface(db).readonly()
    for name in json.loads(os.environ.get("NARU_KERNEL_CALLBACKS", "[]")):

        def stub(*a, _n=name, **k):
            # Batched, not RPC: the parent applies them with the reply, which
            # keeps this a plain request/response loop.
            calls.append([_n, list(a), k])
            if _n == "submit_answer":
                raise _Done()

        ns[name] = stub
    preloaded = set(ns)

    for line in sys.stdin:
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if msg.get("op") == "limits":
            reply = {"limits": limits}
        elif msg.get("op") == "digest":
            items = [
                f"{k}: {_shape(v)}"
                for k, v in ns.items()
                if not k.startswith("_") and k not in preloaded and not callable(v)
            ]
            reply = {"digest": "resident: " + (", ".join(items) if items else "(empty)")}
        else:
            calls.clear()
            buf, err = io.StringIO(), None
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    exec(compile(msg["code"], "<cell>", "exec"), ns)
            except _Done:
                pass
            except BaseException:
                # BaseException, not Exception: MemoryError and the SIGXCPU
                # that RLIMIT_CPU raises must be reported, not escape and kill
                # the loop silently.
                tb = traceback.format_exc(limit=2).strip().splitlines()
                err = tb[-1] if tb else "error"
            reply = {"out": buf.getvalue(), "err": err, "calls": list(calls)}
        _REPLY.write(json.dumps(reply) + "\n")
        _REPLY.flush()


main()
'''


class SandboxedKernel:
    """Kernel.run/digest, executed in a child process.

    What this buys, precisely: a runaway loop, a memory bomb or a hard crash in
    model-authored code takes down the child and not the harness, and the
    ceilings are enforced by the OS rather than by hoping the model behaves.
    A dead child is replaced on the next call, losing resident state — which is
    the honest cost of the isolation.

    What it does NOT buy: the child runs as the same user, with the same
    filesystem and the same network. `rm -rf`, reading a key out of the
    environment, and an outbound request all still work. Real capability
    isolation needs seccomp or a container, neither of which is in the standard
    library, so `NARU_KERNEL_JAIL` wraps the child in one you supply:

        NARU_KERNEL_JAIL='sandbox-exec -f jail.sb'             # macOS
        NARU_KERNEL_JAIL='bwrap --ro-bind / / --unshare-net'   # Linux
        NARU_KERNEL_JAIL='docker run --rm -i --network none …'

    Without one of those this is process isolation, not a sandbox, and calling
    it a sandbox in a README is how people get hurt.

    The Event Log is reopened by path in the child, so `db` cannot be
    `:memory:` — an in-memory database belongs to one process. That is why the
    in-process Kernel remains the default for benchmark runs.
    """

    MAX_OUT = 8000

    def __init__(self, db=None, callbacks=None, timeout=60):
        if db == ":memory:":
            raise ValueError(
                "an in-memory Event Log cannot be shared with a child process; "
                "use a file path or the in-process Kernel"
            )
        self.db = db
        self.callbacks = dict(callbacks or {})
        self.timeout = timeout
        self._proc = None
        self._why = "not started"

    # -- child lifecycle ----------------------------------------------------
    def _spawn(self):
        env = dict(os.environ)
        env["NARU_KERNEL_PATH"] = str(pathlib.Path(__file__).resolve().parent)
        env["NARU_KERNEL_CALLBACKS"] = json.dumps(sorted(self.callbacks))
        if self.db:
            env["NARU_KERNEL_DB"] = str(self.db)
        argv = shlex.split(os.environ.get("NARU_KERNEL_JAIL", "")) + [
            sys.executable,
            "-c",
            _CHILD,
        ]
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
        )

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.kill()
            self._proc.wait(timeout=5)
        self._proc = None

    def _ask(self, msg):
        """One request, one reply. Returns None if the child died or hung."""
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError):
            self.close()
            return None
        result = {}

        def read():
            line = self._proc.stdout.readline()
            if line:
                result["reply"] = json.loads(line)

        t = threading.Thread(target=read, daemon=True)
        t.start()
        t.join(self.timeout)
        if "reply" in result:
            return result["reply"]
        # Died or hung, and the two need different words: a segfault reported
        # as a timeout sends someone hunting a slow cell that never existed.
        #
        # The reader thread is the reliable signal, not poll(). On a crash the
        # thread returns immediately on EOF, often before the OS has reaped the
        # child, so poll() still says None and the death reads as a timeout.
        hung = t.is_alive()
        rc = None
        if not hung and self._proc:
            try:
                rc = self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                hung = True
        self.close()
        self._why = (
            f"exceeded {self.timeout}s wall clock"
            if hung
            else f"died (exit {rc})"
        )
        return None

    # -- Kernel interface ---------------------------------------------------
    def run(self, code):
        reply = self._ask({"op": "run", "code": code})
        if reply is None:
            return "", f"kernel {self._why}; resident state was lost"
        for name, args, kwargs in reply.get("calls", []):
            fn = self.callbacks.get(name)
            if fn:
                fn(*args, **kwargs)
        out = reply.get("out", "")
        if len(out) > self.MAX_OUT:
            out = out[: self.MAX_OUT] + f"\n[output truncated at {self.MAX_OUT} chars]"
        return out, reply.get("err")

    def digest(self):
        reply = self._ask({"op": "digest"})
        return (reply or {}).get("digest", "resident: (unavailable)")

    def limits(self):
        """Which ceilings are actually in force in the child.

        A value of "NOT APPLIED: ..." means the platform refused it. macOS
        refuses RLIMIT_AS, so memory is uncapped there and the caller has to
        know rather than assume.
        """
        return (self._ask({"op": "limits"}) or {}).get("limits", {})


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

    # ---- SandboxedKernel: same interface, child process ------------------
    import tempfile

    from ms import MemorySurface

    d = pathlib.Path(tempfile.mkdtemp())
    db = str(d / "log.db")
    MemorySurface(db).append(
        "user", "the kayak leaks", kind="context_msg",
        session_id="s1", created_at="2023-01-01T00:00",
    )
    got = []
    # 2s: every operation here takes under 0.2s, and one of them is a runaway
    # loop the check has to sit through.
    sk = SandboxedKernel(db=db, callbacks={"submit_answer": got.append}, timeout=2)
    try:
        # an in-memory log belongs to one process and must be refused loudly
        try:
            SandboxedKernel(db=":memory:")
            raise AssertionError("accepted an in-memory DB for a child process")
        except ValueError:
            pass

        out, err = sk.run("xs = [1, 2, 3]\nprint(sum(xs))")
        assert (out.strip(), err) == ("6", None), (out, err)
        out, _ = sk.run("xs.append(9)\nprint(len(xs))")
        assert out.strip() == "4", out          # state survives between cells
        assert "xs: list[4]" in sk.digest(), sk.digest()

        out, _ = sk.run("print(ms.search('kayak')[0]['content'][:15])")
        assert "kayak" in out, out               # the log reopened by path
        _, err = sk.run("ms.db = 1")
        assert err and "read-only" in err, err

        sk.run("submit_answer('42')\nprint('NOT REACHED')")
        assert got == ["42"], got                # callback crossed back

        # Containment. Each of these used to take the harness down with it.
        _, err = sk.run("while True: pass")
        assert err and "wall clock" in err, err
        out, _ = sk.run("print('alive')")
        assert out.strip() == "alive", "parent did not survive a runaway loop"
        _, err = sk.run("import ctypes; ctypes.string_at(0)")
        assert err and "died" in err, err        # a crash is not a timeout
        out, _ = sk.run("print('alive')")
        assert out.strip() == "alive", "parent did not survive a segfault"

        # A ceiling the platform refused must say so. macOS rejects RLIMIT_AS,
        # and reporting it as applied is the failure this repo is built around.
        lim = sk.limits()
        assert set(lim) == {"cpu_s", "mem_b", "fsize_b"}, lim
        assert lim["cpu_s"] == 30, lim
    finally:
        sk.close()

    # Whether the report is TRUE, not merely well-formed. "isinstance(x, int)
    # or 'NOT APPLIED' in str(x)" accepts both branches and is a tautology, so
    # it passed while the child reported a refused limit as applied. Ask the
    # child to exceed the cap it claims to have instead.
    os.environ["NARU_KERNEL_MEM_MB"] = "128"
    probe = SandboxedKernel(timeout=10)
    try:
        mem = probe.limits()["mem_b"]
        _, err = probe.run("b = bytearray(400_000_000); print(len(b))")
        if isinstance(mem, int):
            assert err, f"mem_b reported as {mem} but 400MB allocated anyway"
        else:
            assert "NOT APPLIED" in str(mem), mem
            assert not err, f"mem_b unenforced yet the allocation failed: {err}"
    finally:
        probe.close()
        os.environ.pop("NARU_KERNEL_MEM_MB", None)

    print("ok — kernel checks passed (in-process and sandboxed)")


if __name__ == "__main__":
    demo()
