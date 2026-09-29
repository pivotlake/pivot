---
title: System tables
description: Read-only tables that describe the server's datastores, tables, columns, files, and memory.
---

The `system` datastore holds read-only tables that describe the server
itself: which datastores it serves, the tables and files in each, and how its
memory is used. Query them like any other table:

```sql
SELECT datastore, schema, name, total_rows
FROM system.tables
ORDER BY total_rows DESC;
```

| Table | One row per |
| --- | --- |
| [`system.datastores`](#systemdatastores) | Datastore the server serves |
| [`system.tables`](#systemtables) | Table in any datastore |
| [`system.columns`](#systemcolumns) | Column of any table |
| [`system.table_files`](#systemtable_files) | Data file of any table |
| [`system.memory_blocks`](#systemmemory_blocks) | Block of the buffer pool |

System tables span every datastore, and describe the `system` datastore
itself too. A query reads them from the same snapshot as the other tables it
reads, so a query that joins `system.tables` with a table sees one consistent
version of it.

`system` is read-only: `CREATE TABLE`, `DROP TABLE`, and `CREATE SCHEMA` are
refused there.

## `system.datastores`

One row per datastore, including `system` itself.

| Column | Type | Description |
| --- | --- | --- |
| `name` | `VARCHAR` | The datastore's name, as configured. |
| `type` | `VARCHAR` | `pivotlake`, `iceberg`, or `system`. |
| `data_path` | `VARCHAR` | Where the datastore lives: the storage location of a pivotlake datastore, the catalog URI of an Iceberg datastore, or an empty string for `system`. |

```sql
SELECT name, type, data_path FROM system.datastores;
```

## `system.tables`

One row per table in every datastore.

| Column | Type | Description |
| --- | --- | --- |
| `datastore` | `VARCHAR` | The datastore that holds the table. |
| `schema` | `VARCHAR` | The table's schema. |
| `name` | `VARCHAR` | The table's name. |
| `id` | `VARCHAR` | The table's identifier. Stable for the table's lifetime, and what `system.columns` and `system.table_files` join on. |
| `sorting_keys` | `VARCHAR` | The table's sort columns, comma-separated in order, or an empty string. |
| `partition_key` | `VARCHAR` | The table's partition columns, comma-separated in order, or an empty string. |
| `total_rows` | `BIGINT` | Rows in the table's committed files. |
| `bytes` | `BIGINT` | Bytes the table's files occupy in storage. |
| `bytes_uncompressed` | `BIGINT` | Bytes the table's data holds once decompressed. |

The system tables themselves appear with zero rows and bytes.

```sql
-- Compression ratio of each table
SELECT datastore, schema, name,
       bytes_uncompressed * 1.0 / bytes AS compression_ratio
FROM system.tables
WHERE bytes > 0
ORDER BY bytes DESC;
```

## `system.columns`

One row per column of every table.

| Column | Type | Description |
| --- | --- | --- |
| `datastore` | `VARCHAR` | The datastore that holds the table. |
| `table_id` | `VARCHAR` | The table's `system.tables.id`. |
| `name` | `VARCHAR` | The column's name. |
| `type` | `VARCHAR` | The column's SQL type, such as `BIGINT` or `TIMESTAMP WITH TIME ZONE`. |
| `position` | `BIGINT` | The column's position in the table, counting from 0. |
| `bytes` | `BIGINT` | Bytes the column occupies across the table's files. |
| `bytes_uncompressed` | `BIGINT` | Bytes the column holds once decompressed. |
| `is_partition_key` | `BOOLEAN` | Whether the column is one of the table's partition columns. |
| `is_sort_key` | `BOOLEAN` | Whether the column is one of the table's sort columns. |

```sql
-- The largest columns of one table
SELECT c.name, c.type, c.bytes
FROM system.columns AS c
JOIN system.tables AS t ON c.table_id = t.id
WHERE t.name = 'events'
ORDER BY c.bytes DESC;
```

## `system.table_files`

One row per committed data file of every table. Files that a running
transaction has written but not yet committed are not listed.

| Column | Type | Description |
| --- | --- | --- |
| `table_id` | `VARCHAR` | The table's `system.tables.id`. |
| `path` | `VARCHAR` | The file's path. Relative to the datastore's `data_path` for a pivotlake datastore; a full URI for an Iceberg datastore. |
| `partition` | `VARCHAR` | The file's partition as comma-separated `column=value` pairs, or an empty string for an unpartitioned table. |
| `bytes` | `BIGINT` | The file's size in storage. |
| `bytes_uncompressed` | `BIGINT` | Bytes the file's data holds once decompressed. |
| `min_max_stats` | `VARIANT` | The minimum and maximum value of each column that records them, as an object keyed by column name, such as `{"id": {"min": 1, "max": 900}}`. An empty object when the file records none. |

```sql
-- Tables with many small files, which compaction merges
SELECT t.datastore, t.schema, t.name,
       count(*) AS files,
       avg(f.bytes) AS average_file_bytes
FROM system.table_files AS f
JOIN system.tables AS t ON f.table_id = t.id
GROUP BY t.datastore, t.schema, t.name
ORDER BY files DESC;
```

## `system.memory_blocks`

One row per block of the buffer pool, the fixed memory Pivot allocates at
startup (see [Memory budget](/docs/reference/cli/#memory-budget)). Every block
is the same size, so the share of rows in a state is the share of memory in
it.

| Column | Type | Description |
| --- | --- | --- |
| `slot` | `BIGINT` | The block's position in the pool. |
| `node` | `BIGINT` | The NUMA node whose memory holds the block. |
| `state` | `VARCHAR` | What the block is used for. See the table below. |
| `readers` | `BIGINT` | How many holds the block has, including a cache's own hold on a block it filled. A block with no holds can be reused. |
| `bytes` | `BIGINT` | The block's size: 2 MiB. |

| State | Meaning |
| --- | --- |
| `free` | Unused, and available without evicting anything. |
| `pinned` | Working memory held by a running query or merge, such as a hash table or a sort run. Released when the work finishes. |
| `compressed_cache` | Caching file bytes as they were read from storage. |
| `decompressed_cache` | Caching decoded data pages. |

```sql
-- How the buffer pool is used
SELECT state,
       count(*) AS blocks,
       sum(bytes) / 1024 / 1024 AS mib
FROM system.memory_blocks
GROUP BY state
ORDER BY blocks DESC;
```

## Related

- [Datastores & storage credentials](/docs/reference/server/datastores/)
- [`COMPACT`](/docs/reference/statements/compact/)
- [`SELECT`](/docs/reference/statements/select/)
