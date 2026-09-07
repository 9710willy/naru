#!/usr/bin/env python3
"""Run one Naru verification command in a PTY and keep its proof."""

import argparse
import errno
import os
import pathlib
import pty
import re
import shlex
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        dest="input_text",
        help="text to send to the terminal; a final newline is added",
    )
    parser.add_argument("artifact")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("need a command after --")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.artifact):
        parser.error("artifact may contain only letters, numbers, dot, dash, and underscore")

    evidence_raw = os.environ.get("VERIFY_NARU_EVIDENCE")
    if not evidence_raw:
        parser.error("VERIFY_NARU_EVIDENCE is not set")
    evidence = pathlib.Path(evidence_raw)
    evidence.mkdir(parents=True, exist_ok=True)

    prefix = evidence / args.artifact
    pathlib.Path(f"{prefix}.command.txt").write_text("$ " + shlex.join(command) + "\n")
    if args.input_text is not None:
        pathlib.Path(f"{prefix}.input.txt").write_text(args.input_text + "\n")

    pid, master = pty.fork()
    if pid == 0:
        try:
            os.execvpe(command[0], command, os.environ)
        except OSError as exc:
            print(f"cannot run {command[0]}: {exc}", file=sys.stderr)
            os._exit(127)

    if args.input_text is not None:
        sent = args.input_text
        if not sent.endswith("\n"):
            sent += "\n"
        os.write(master, sent.encode())

    with pathlib.Path(f"{prefix}.terminal.txt").open("wb") as transcript:
        while True:
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            transcript.write(chunk)
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()

    os.close(master)
    _, status = os.waitpid(pid, 0)
    if os.WIFEXITED(status):
        code = os.WEXITSTATUS(status)
    else:
        code = 128 + os.WTERMSIG(status)
    pathlib.Path(f"{prefix}.exit.txt").write_text(f"exit={code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
