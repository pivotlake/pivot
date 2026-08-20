---
title: System tables
description: The system datastore, where the catalog and the buffer pool describe themselves.
sidebar:
  order: 5
---

Every server serves a datastore named `system` in addition to the ones it is
configured with. It stores nothing: its relations are assembled per query from
the datastores the query already has open, so they describe the same snapshot
the rest of the statement reads.

The tables live in the `main` schema, so `system.tables` and
`system.main.tables` name the same relation. They are read-only, and a write to
one is refused.

`table` is a keyword, so the columns of that name need quoting:

```sql
SELECT t.name, count(*) AS files, sum(f.bytes) AS bytes
FROM system.table_files f
JOIN system.tables t ON f."table" = t.id
GROUP BY t.name
ORDER BY bytes DESC;
```

## system.datastores

One row per datastore the server serves, including `system` itself.

| Column | Type | Meaning |
| --- | --- | --- |
| `name` | `VARCHAR` | The datastore's configured name, which is also the database name in a qualified table reference |
| `id` | `VARCHAR` | Its identity, which is its name |
| `type` | `VARCHAR` | The datastore format, `delta` for a stored datastore and `system` for this one |
| `data_path` | `VARCHAR` | The configured location, a local path or an object-store URI |

## system.tables

One row per table, across every datastore.

| Column | Type | Meaning |
| --- | --- | --- |
| `datastore` | `VARCHAR` | The datastore holding the table |
| `schema` | `VARCHAR` | Its schema, `main` unless another was created |
| `name` | `VARCHAR` | The table name |
| `id` | `VARCHAR` | Table identity, and the join key of `system.columns` and `system.table_files` |
| `sorting_keys` | `VARCHAR` | The `sort_by` columns in order, comma-separated, empty when unsorted |
| `partition_key` | `VARCHAR` | The `partition_by` columns in order, comma-separated, empty when unpartitioned |
| `total_rows` | `BIGINT` | Rows across the table's files |
| `bytes` | `BIGINT` | Size on storage, compressed |
| `bytes_uncompressed` | `BIGINT` | Size the same data decodes to |

## system.columns

One row per declared column of every table.

| Column | Type | Meaning |
| --- | --- | --- |
| `datastore` | `VARCHAR` | The datastore holding the table |
| `table` | `VARCHAR` | The owning table's `id` |
| `name` | `VARCHAR` | Column name |
| `type` | `VARCHAR` | Its SQL type name |
| `position` | `BIGINT` | Ordinal position in the table |
| `bytes` | `BIGINT` | Size on storage, compressed |
| `bytes_uncompressed` | `BIGINT` | Size the same data decodes to |
| `is_partition_key` | `BOOLEAN` | Whether the column is part of `partition_by` |
| `is_sort_key` | `BOOLEAN` | Whether the column is part of `sort_by` |

The per-column byte counts are what makes a storage breakdown a query:

```sql
SELECT c.name, format_bytes(c.bytes) AS stored
FROM system.columns c
JOIN system.tables t ON c."table" = t.id
WHERE t.name = 'events'
ORDER BY c.bytes DESC;
```

## system.table_files

One row per data file.

| Column | Type | Meaning |
| --- | --- | --- |
| `table` | `VARCHAR` | The owning table's `id` |
| `path` | `VARCHAR` | The file's path in the datastore's storage |
| `partition` | `VARCHAR` | Its partition values, empty for an unpartitioned table |
| `bytes` | `BIGINT` | File size |
| `bytes_uncompressed` | `BIGINT` | Size the file decodes to |

This is the table to read when judging whether compaction is keeping up: many
files far below the datastore's `compact_bytes` boundary is what a sweep merges.

## system.memory_blocks

One row per block of the memory ring, the fixed-size unit every cache and every
operator allocates from. Reading the blocks themselves rather than a summary is
what leaves the summary to the query: every block is the same size, so the
share of memory in a state is the share of rows in it.

| Column | Type | Meaning |
| --- | --- | --- |
| `slot` | `BIGINT` | The block's index in the ring |
| `node` | `BIGINT` | The NUMA node its memory lives on |
| `state` | `VARCHAR` | `free`, `pinned`, `compressed_cache`, or `decompressed_cache` |
| `readers` | `BIGINT` | How many readers hold the block right now |
| `bytes` | `BIGINT` | The block size, the same on every row |

```sql
SELECT state, count(*) AS blocks, format_bytes(sum(bytes)) AS memory
FROM system.memory_blocks
GROUP BY state
ORDER BY blocks DESC;
```
