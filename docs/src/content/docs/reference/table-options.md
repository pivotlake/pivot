---
title: Table options
description: CREATE TABLE layout and parquet adoption options.
sidebar:
  order: 4
---

`CREATE TABLE ... WITH (...)` accepts exactly these options. Unknown names are
rejected.

| Option | Value | Effect |
| --- | --- | --- |
| `partition_by` | Comma-separated column names | Partitions newly written parquet files by the declared columns, in order. |
| `sort_by` | Comma-separated column names | Sorts newly written rows by the declared columns, in order. |
| `with_pre_existing_parquets` | Directory path | Adopts parquet files directly under the directory as the table's initial data. The declared schema must match the files. |

```sql
CREATE TABLE events (
  id BIGINT,
  region VARCHAR,
  ts TIMESTAMP
) WITH (
  partition_by = 'region',
  sort_by = 'ts',
  with_pre_existing_parquets = 'incoming/events'
);
```

Column-list values are strings. Every named column must exist in the table
declaration.

`with_pre_existing_parquets` takes a plain storage path, not a URL. A relative
path is resolved within the datastore's storage. The files stay in their
original directory; pivotdb records them in the new table rather than moving
or rewriting them during creation.
