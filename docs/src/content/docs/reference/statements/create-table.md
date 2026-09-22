---
title: "CREATE TABLE"
description: Create a table, configure its layout, or adopt existing Parquet files.
sidebar:
  order: 4
---

`CREATE TABLE` creates an empty table or registers existing Parquet files as a
new table's initial data.

## Example

```sql
CREATE TABLE events (
  id BIGINT,
  region VARCHAR,
  ts TIMESTAMP
);
```

## Syntax

```sql
CREATE TABLE [IF NOT EXISTS] [[datastore.]schema.]table (
  column_name data_type [, ...]
) [WITH (option = value [, ...])];
```

## Table name and columns

Use `schema.table` or `datastore.schema.table` to choose a namespace. The
schema must exist; create one with [CREATE SCHEMA](/docs/reference/statements/create-schema/).
Unqualified names use the default datastore and schema.

Each column has a name and a [data type](/docs/reference/data-types/).
`IF NOT EXISTS` leaves an existing table in place.

## Table options

`WITH (...)` accepts the following options. Unknown option names are rejected.

| Option | Value | Purpose |
| --- | --- | --- |
| `partition_by` | Comma-separated column names in a string | Partition newly written files. |
| `sort_by` | Comma-separated column names in a string | Sort newly written rows. |
| `with_pre_existing_parquets` | Directory path in a string | Adopt existing Parquet files. |

### Partitioning

`partition_by` selects partition columns in the declared order. Every named
column must exist in the table declaration.

```sql
CREATE TABLE regional_events (id BIGINT, region VARCHAR)
WITH (partition_by = 'region');
```

### Sorting

`sort_by` selects sort columns in the declared order. It can be combined with
partitioning:

```sql
CREATE TABLE sorted_events (id BIGINT, region VARCHAR, ts TIMESTAMP)
WITH (partition_by = 'region', sort_by = 'ts');
```

Sorting controls the layout of newly written rows. Use `ORDER BY` in a query
when its returned rows need a particular order.

### Adopt existing Parquet files

`with_pre_existing_parquets` adopts Parquet files directly under a directory.
The declared schema must match those files.

```sql
CREATE TABLE imported_events (id BIGINT, region VARCHAR, ts TIMESTAMP)
WITH (with_pre_existing_parquets = 'incoming/events');
```

The option takes a plain storage path, not a URL. A relative path is resolved
within the datastore's storage. Files remain in their original directory;
Pivot records them in the new table without moving or rewriting them during
creation.

## Limitations

Temporary tables, constraints, `CREATE OR REPLACE TABLE`, and
`CREATE TABLE ... AS SELECT` are not supported.

## Related

- [Data types](/docs/reference/data-types/)
- [INSERT](/docs/reference/statements/insert/)
- [COMPACT](/docs/reference/statements/compact/)
- [Datastore configuration](/docs/reference/server/datastores/)
