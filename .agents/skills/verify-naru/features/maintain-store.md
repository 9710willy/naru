# Maintain the store

Maintenance reports spill metrics, previews or applies age pruning, and removes
orphaned blob files owned by the current Event Log.

## Sub-features

- `maintain-stats` reports spill and recovery metrics.
- `maintain-prune-dry` previews rows that age pruning would remove.
- `maintain-prune` removes old undecided or non-curated rows and keeps promoted items.
- `maintain-gc` removes unreferenced files from the current store's blob folder.

## How to get to it (user POV)

- Run `python3 naru.py stats [days]`.
- Run `python3 naru.py prune [days] --dry-run` before an actual prune.
- Run `python3 naru.py prune [days]` to delete eligible rows.
- Run `python3 naru.py gc` to remove orphaned blob files owned by this store.

## Driving it with capture.py

Preconditions:

- Start from the empty state in `../SKILL.md`.
- Doctor confirms that `TMPDIR` is `$VERIFY_NARU_STATE/tmp`.

- **Create one removable note and one kept claim.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py maintain-note -- \
    python3 naru.py add "old verification note" "Remove this note."
  ./.agents/skills/verify-naru/capture.py maintain-claim -- python3 naru.py claim \
    "Keep this promoted claim" --key verify-prune --by verify-naru
  ./.agents/skills/verify-naru/capture.py --input promote maintain-inbox -- \
    python3 naru.py inbox
  ./.agents/skills/verify-naru/capture.py maintain-before -- \
    ./.agents/skills/verify-naru/doctor.py
  ```

  Doctor reports `rows=2 promoted=1`.

- **Prove dry-run and real pruning.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py maintain-prune-dry -- \
    python3 naru.py prune 0 --dry-run
  ./.agents/skills/verify-naru/capture.py maintain-after-dry -- \
    ./.agents/skills/verify-naru/doctor.py
  ./.agents/skills/verify-naru/capture.py maintain-prune -- python3 naru.py prune 0
  ./.agents/skills/verify-naru/capture.py maintain-after-prune -- \
    ./.agents/skills/verify-naru/doctor.py
  ```

  The dry-run leaves two rows. The real prune leaves the promoted claim only.

- **Prove stats and isolated blob cleanup.** Run:

  ```bash
  ./.agents/skills/verify-naru/capture.py maintain-stats -- python3 naru.py stats
  ./.agents/skills/verify-naru/capture.py maintain-orphan -- python3 -c \
    'import hashlib, os, pathlib, sqlite3, tempfile; p=pathlib.Path(os.environ["NARU_DB"]).resolve(); db=sqlite3.connect(p); u=db.execute("SELECT store_uuid FROM naru_meta WHERE id=1").fetchone()[0]; root=pathlib.Path(tempfile.gettempdir())/("naru-blobs-"+hashlib.sha256(f"{p}\0{u}".encode()).hexdigest()); root.mkdir(parents=True, exist_ok=True); (root/"orphan.txt").write_text("x")'
  ./.agents/skills/verify-naru/capture.py maintain-foreign -- sh -c \
    'mkdir -p "$TMPDIR/naru-blobs-foreign"; printf x > "$TMPDIR/naru-blobs-foreign/1.txt"'
  ./.agents/skills/verify-naru/capture.py maintain-gc -- python3 naru.py gc
  test -z "$(find "$TMPDIR" -path '*/naru-blobs-*/orphan.txt' -print -quit)"
  test -e "$TMPDIR/naru-blobs-foreign/1.txt"
  ```

  Stats names the run-only metrics file. GC reports one removed item. The
  current store's orphan is gone, and the foreign folder remains.

## Gotchas

- `prune` deletes unless `--dry-run` or `-n` is present.
- Promoted claims, promoted skills, and their source evidence survive normal pruning.
- `gc` scans only the current store's deterministic blob folder and has no prompt.
- Never run `gc` until Doctor confirms the run-only `TMPDIR`.
