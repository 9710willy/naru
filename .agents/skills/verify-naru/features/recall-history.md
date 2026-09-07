# Recall history

Recall stores notes as log chunks, finds matching text, lists session ranges,
and expands exact sequence numbers.

## Sub-features

- `recall-add` saves one row per paragraph.
- `recall-search` returns ranked BM25 matches.
- `recall-outline` lists stored sequence ranges by session.
- `recall-show` prints an exact sequence or range.
- `recall-miss` returns exit `1` and a recovery hint when search finds nothing.

## How to get to it (user POV)

- Run `python3 naru.py add "<topic>" "<text>"` or pipe text on stdin.
- Run `python3 naru.py search "<query>" [limit]`.
- Run `python3 naru.py outline`.
- Run `python3 naru.py show <lo> [hi] [--run <id>]`.

## Driving it with capture.py

Preconditions:

- Start from the empty state in `../SKILL.md`.
- Doctor reports no rows.

- **Add two paragraphs.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py recall-add -- sh -c \
    'printf "Budget is 42 credits.\n\nLaunch is Friday.\n" | python3 naru.py add "project atlas"'
  ```

  The transcript reports two chunks at sequences `1-2`.

- **Find and browse them.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py recall-search -- \
    python3 naru.py search "Launch Friday"
  ./.agents/skills/verify-naru/capture.py recall-outline -- python3 naru.py outline
  ./.agents/skills/verify-naru/capture.py recall-show -- python3 naru.py show 1 2
  ```

  Search returns the Friday row. Outline lists one session with two chunks.
  Show prints both paragraphs with their sequence numbers.

- **Prove a miss and the row count.** Run:

  ```bash
  set +e
  ./.agents/skills/verify-naru/capture.py recall-miss -- \
    python3 naru.py search "volcano absent"
  VERIFY_NARU_MISS_EXIT=$?
  set -e
  test "$VERIFY_NARU_MISS_EXIT" -eq 1
  ./.agents/skills/verify-naru/capture.py recall-doctor -- \
    ./.agents/skills/verify-naru/doctor.py
  ```

  The miss transcript suggests `outline` and `show`. Doctor reports `rows=2`.

## Gotchas

- The final numeric search argument is the result limit.
- Uppercase `OR`, `AND`, and `NOT` are search operators. Other text is quoted.
- `outline` returns exit `1` for an empty store.
- Agent trace stays hidden from unscoped user recall.
