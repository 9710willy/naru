# Inject context

Inject prints the approved Naru document or splices it into a host file between
Naru markers while keeping the user's text outside those markers.

## Sub-features

- `inject-stdout` prints the current approved document.
- `inject-new-file` adds one marked block to a file without markers.
- `inject-replace` replaces the existing marked block on later runs.
- `inject-preserve` keeps all host text outside the block.

## How to get to it (user POV)

- Run `python3 naru.py inject` to print the document.
- Run `python3 naru.py inject <path>` to update a context file.
- Typical targets are `CLAUDE.md`, `AGENTS.md`, and `.cursorrules`.

## Driving it with capture.py

Preconditions:

- Start from the empty state in `../SKILL.md`.
- Use only `$VERIFY_NARU_STATE/context.md`. Do not point this proof at a real context file.

- **Create approved content.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py inject-claim -- python3 naru.py claim \
    "Injected verification fact" --key verify-inject --by verify-naru
  ./.agents/skills/verify-naru/capture.py --input promote inject-inbox -- \
    python3 naru.py inbox
  ```

- **Seed host text and inject twice.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py inject-seed-file -- sh -c \
    'printf "# Host instructions\n\nKeep this line.\n" > "$VERIFY_NARU_STATE/context.md"'
  ./.agents/skills/verify-naru/capture.py inject-first -- \
    python3 naru.py inject "$VERIFY_NARU_STATE/context.md"
  ./.agents/skills/verify-naru/capture.py inject-second -- \
    python3 naru.py inject "$VERIFY_NARU_STATE/context.md"
  ```

- **Prove stdout and the file.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py inject-stdout -- python3 naru.py inject
  ./.agents/skills/verify-naru/capture.py inject-file-view -- \
    sed -n '1,120p' "$VERIFY_NARU_STATE/context.md"
  test "$(rg -c '<!-- naru:begin -->' "$VERIFY_NARU_STATE/context.md")" -eq 1
  rg -q '^Keep this line\.$' "$VERIFY_NARU_STATE/context.md"
  rg -q 'Injected verification fact' "$VERIFY_NARU_STATE/context.md"
  ```

  The host heading and line remain. The file has one start marker, one end
  marker, and the promoted claim.

## Gotchas

- `inject <path>` writes that path. Use a run-only file during verification.
- `python3 naru.py inject --help` writes a file named `--help`. Use top-level help.
- A file with one missing marker has no valid block. Inject appends a new complete block.
- Proposed, dropped, and internal trace rows never enter the injected document.
