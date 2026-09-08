---
title: "COMPACT"
description: Merge small Parquet files and re-sort overlapping files.
sidebar:
  order: 8
---

`COMPACT` improves a table's physical file layout by merging small Parquet
files and re-sorting files with overlapping sort-key ranges.

## Example

Run a compaction round on an existing table:

```sql
COMPACT events;
```

## Syntax

```sql
COMPACT [datastore.][schema.]table [FINAL];
```

## Compaction behavior

A round keeps merging until no candidate remains. Small files are merged into
full-size files. Groups of six files with overlapping sort-key ranges are
re-sorted into narrower, non-overlapping files.

The datastore's compaction settings control output targets and concurrency.
Each merge in flight holds its decoded input rows in memory.

## FINAL

```sql
COMPACT events FINAL;
```

`FINAL` bypasses the normal size, file-count, balance, and overlap guards. It
continues rewriting until no two files' sort-key ranges overlap.

## Related

- [Datastore maintenance](/docs/reference/server/datastores/#maintenance) — background compaction and its settings.
- [CREATE TABLE sorting](/docs/reference/statements/create-table/#sorting) — choose sort columns.
- [System tables](/docs/reference/system-tables/) — inspect file and table metadata.
