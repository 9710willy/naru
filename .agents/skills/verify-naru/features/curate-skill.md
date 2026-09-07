# Curate a skill

Skills let an agent propose a reusable procedure. The same human review gate
used for claims decides whether it enters the shared `Skills` section.

## Sub-features

- `skill-propose` adds a pending procedure.
- `skill-source` links the procedure to an exact run and sequence range.
- `skill-promote` approves the procedure through the inbox.
- `skill-render` places only promoted procedures under `Skills`.

## How to get to it (user POV)

- Run `python3 naru.py skill "<procedure>" --key <key> --by <agent>`.
- Add `--run <run-id> --source <lo>:<hi>` when the procedure came from stored evidence.
- Run `python3 naru.py inbox` to promote, drop, or skip the proposal.
- Run `python3 naru.py inject` to read the approved skills.

## Driving it with capture.py

Preconditions:

- Start from the empty state in `../SKILL.md`.
- Doctor reports no rows.

- **Store source evidence.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py skill-source -- \
    python3 naru.py add "verification source" "Run every offline check before release."
  VERIFY_NARU_SOURCE_RUN="$(date +%F) verification source"
  ```

  The source is sequence `1` in the named run.

- **Open the source evidence.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py skill-source-show -- \
    python3 naru.py show 1 --run "$VERIFY_NARU_SOURCE_RUN"
  ```

  Show returns the source text and records the receipt that a source-backed
  proposal needs.

- **Propose and promote the procedure.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py skill-propose -- python3 naru.py skill \
    "Before release, run every offline check and keep the real exit codes." \
    --key verify-release --by verify-naru \
    --run "$VERIFY_NARU_SOURCE_RUN" --source 1:1
  ./.agents/skills/verify-naru/capture.py --input promote skill-inbox -- \
    python3 naru.py inbox
  ```

  The inbox prints the source recovery command and reports `promoted`.

- **Prove the procedure and its source.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py skill-document -- python3 naru.py inject
  ./.agents/skills/verify-naru/capture.py skill-doctor -- \
    ./.agents/skills/verify-naru/doctor.py
  ```

  The document has a `Skills` section with the procedure. Doctor reports
  `rows=2 pending=0 promoted=1 dropped=0`.

## Gotchas

- `--run` and `--source` must appear together.
- Source ranges use `LO:HI` with `LO <= HI`.
- One `show` receipt must cover the full source span in the same store and run.
- Promoted skill source evidence survives normal pruning.
- Skill text renders as one line, even when the proposal contains newlines.
