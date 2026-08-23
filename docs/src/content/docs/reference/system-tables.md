---
title: System tables
description: Read-only catalog, file, and memory metadata exposed through SQL.
sidebar:
  order: 5
---

The read-only `system` datastore describes every configured datastore and the
running server. Its tables are queried with two-part names such as
`system.tables`.

| Table | One row per |
| --- | --- |
| `system.datastores` | Configured datastore, plus the `system` datastore itself. |
| `system.tables` | Table across all datastores, including the system tables. |
| `system.columns` | Declared table column. |
| `system.table_files` | Live parquet data file. |
| `system.memory_blocks` | Block in the server's memory ring. |

## `system.datastores`

| Column | Type | Description |
| --- | --- | --- |
| `name` | `VARCHAR` | Datastore name used in qualified SQL names. |
| `id` | `VARCHAR` | Datastore identity. Currently the same value as `name`. |
| `type` | `VARCHAR` | `delta` for stored datastores or `system` for the virtual datastore. |
| `data_path` | `VARCHAR` | Local path or object-store URI. Empty for `system`. |

## `system.tables`

| Column | Type | Description |
| --- | --- | --- |
| `datastore` | `VARCHAR` | Owning datastore name. |
| `schema` | `VARCHAR` | Schema name. |
| `name` | `VARCHAR` | Table name. |
| `id` | `VARCHAR` | Stable table identity used by the other system tables. |
| `sorting_keys` | `VARCHAR` | Comma-separated sort columns, in order. |
| `partition_key` | `VARCHAR` | Comma-separated partition columns, in order. |
| `total_rows` | `BIGINT` | Rows recorded in live files. |
| `bytes` | `BIGINT` | Compressed bytes. |
| `bytes_uncompressed` | `BIGINT` | Uncompressed bytes. |

## `system.columns`

| Column | Type | Description |
| --- | --- | --- |
| `datastore` | `VARCHAR` | Owning datastore name. |
| `table` | `VARCHAR` | Owning table ID. Join to `system.tables.id`. |
| `name` | `VARCHAR` | Column name. |
| `type` | `VARCHAR` | SQL type name. |
| `position` | `BIGINT` | Zero-based position in the table schema. |
| `bytes` | `BIGINT` | Compressed bytes attributed to the column. |
| `bytes_uncompressed` | `BIGINT` | Uncompressed bytes attributed to the column. |
| `is_partition_key` | `BOOLEAN` | Whether the column participates in partitioning. |
| `is_sort_key` | `BOOLEAN` | Whether the column participates in sorting. |

## `system.table_files`

| Column | Type | Description |
| --- | --- | --- |
| `table` | `VARCHAR` | Owning table ID. Join to `system.tables.id`. |
| `path` | `VARCHAR` | Data file path. |
| `partition` | `VARCHAR` | File partition value. |
| `bytes` | `BIGINT` | Compressed file bytes. |
| `bytes_uncompressed` | `BIGINT` | Uncompressed file bytes. |
| `min_max_stats` | `VARIANT` | Object mapping each column with file statistics to its `min` and `max` values. |

Bounds retain their value types and can be addressed as variant fields, for
example `min_max_stats.event_time.min`. Columns without recorded bounds are
omitted.

## `system.memory_blocks`

| Column | Type | Description |
| --- | --- | --- |
| `slot` | `BIGINT` | Block's slot in the memory ring. |
| `node` | `BIGINT` | NUMA node that owns the block. |
| `state` | `VARCHAR` | `free`, `pinned`, `compressed_cache`, or `decompressed_cache`. |
| `readers` | `BIGINT` | Active readers holding the block. |
| `bytes` | `BIGINT` | Block size in bytes. |

The catalog-wide tables are snapshot-consistent within a query.
