# Decide claims

Claims let an agent propose a fact. A user can promote it, drop it, or leave it
pending. Only promoted claims enter the shared document.

## Sub-features

- `claim-propose` adds a pending claim with a key and author.
- `claim-promote` approves a claim through `naru inbox`.
- `claim-drop` rejects a claim through `naru inbox` while keeping it searchable.
- `claim-skip` leaves a claim pending.
- `claim-direct` uses `promote SEQ --yes` or `drop SEQ --yes` without the prompt.

## How to get to it (user POV)

- Run `python3 naru.py claim "<text>" --key <key> --by <agent>`.
- Run `python3 naru.py inbox` and enter `promote`, `drop`, or `skip`.
- Run `python3 naru.py promote <seq> --yes` or `drop <seq> --yes` for a direct decision.

## Driving it with capture.py

Preconditions:

- Start from the empty state in `../SKILL.md`.
- Doctor reports `rows=0 pending=0 promoted=0 dropped=0`.

- **Propose three claims.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py claim-keep -- python3 naru.py claim \
    "Keep this verification claim" --key verify-keep --by verify-naru
  ./.agents/skills/verify-naru/capture.py claim-drop -- python3 naru.py claim \
    "Discard this verification claim" --key verify-drop --by verify-naru
  ./.agents/skills/verify-naru/capture.py claim-skip -- python3 naru.py claim \
    "Leave this verification claim pending" --key verify-skip --by verify-naru
  ```

  The transcripts report pending sequence numbers `1`, `2`, and `3`.

- **Use the inbox.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py \
    --input $'promote\ndrop\nskip' inbox -- python3 naru.py inbox
  ```

  The transcript shows `promoted`, `dropped`, and `skipped` in that order.

- **Use direct decisions.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py claim-direct-keep -- python3 naru.py claim \
    "Keep this direct claim" --key verify-direct-keep --by verify-naru
  ./.agents/skills/verify-naru/capture.py direct-promote -- \
    python3 naru.py promote 4 --yes
  ./.agents/skills/verify-naru/capture.py claim-direct-drop -- python3 naru.py claim \
    "Drop this direct claim" --key verify-direct-drop --by verify-naru
  ./.agents/skills/verify-naru/capture.py direct-drop -- \
    python3 naru.py drop 5 --yes
  ```

- **Prove the visible result.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py claims-document -- python3 naru.py inject
  ./.agents/skills/verify-naru/capture.py dropped-search -- \
    python3 naru.py search "Discard this verification claim"
  ./.agents/skills/verify-naru/capture.py claims-doctor -- \
    ./.agents/skills/verify-naru/doctor.py
  ```

  The document contains the two kept claims. It omits the dropped and pending
  claims. Search still finds the dropped text. Doctor reports
  `rows=5 pending=1 promoted=2 dropped=2`.

## Gotchas

- `skip` keeps an item pending. The next inbox run asks about it again.
- Direct `promote` and `drop` require `--yes` when no terminal is present.
- Two promoted claims with the same key appear under `Unresolved`.
- Only top-level `python3 naru.py --help` is help. Command-level `--help` is data.
