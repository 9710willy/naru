# 11. Bind resources to a store incarnation

## Context

`NARU_DB` lets you use more than one Event Log. A sequence number and run ID
are not unique across those files. A receipt from one file can therefore match
rows in another file.

Large payloads also need one stable owner. The old blob directory used a
Python object ID, so reopening one database created a new directory. `naru gc`
then searched every `naru-blobs-*` directory and compared it with only the
current database. It could delete another database's payload.

A resolved path is stable across reopen, but it is not enough. If you replace
a database at the same path, old receipts and blobs can apply to the new file.

## Decision

When you open a writable Event Log, Naru creates one random `store_uuid` in
the `naru_meta` table if it does not exist. It derives `store_id` from the
resolved database path and that UUID.

Naru also reads the earlier key/value metadata layout. Existing stores keep
their UUID and do not need a schema rewrite.

After the UUID exists, opening the store does not write the metadata row again.
This lets read-only commands use an up-to-date store on a read-only mount.

Version 2 show receipts record `store_id`. File-backed stores also use it in
their blob directory name. `naru gc` scans only that directory and deletes
only files that the current Event Log does not reference.

Existing `payload_path` values remain the source of truth for old rows. Naru
does not scan or delete legacy blob directories because it cannot prove their
owner.

## Why

The UUID distinguishes a replacement database at the same path. The path
distinguishes a copied database that keeps the same UUID. One shared identity
keeps receipt proof and blob cleanup under the same owner.

Store-local garbage collection is safe and repeatable. A global scan cannot
be safe because the current command does not know every Event Log that may
refer to those files.

## Limits

If you move a database, it gets a new `store_id` and a new blob root. Its old
payload paths still work, but old show receipts no longer authorize claims at
the new path.

Legacy orphan blob directories may remain until you remove them by hand.
