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

Regular `COMPACT` merges small files and re-sorts overlapping files until no
further merges qualify under the normal compaction rules. These rules use
file sizes, file counts, and the amount of overlap to decide when to merge.

The datastore's compaction settings control output targets and concurrency.

## FINAL

```sql
COMPACT events FINAL;
```

`FINAL` runs a more aggressive compaction. It relaxes the normal merge
thresholds, merging remaining small files and re-sorting eligible files until
no two have overlapping sort-key ranges within a partition.

## Related

- [Datastore maintenance](/docs/reference/server/datastores/#maintenance) — background compaction and its settings.
- [CREATE TABLE sorting](/docs/reference/statements/create-table/#sorting) — choose sort columns.
- [System tables](/docs/reference/system-tables/) — inspect file and table metadata.
