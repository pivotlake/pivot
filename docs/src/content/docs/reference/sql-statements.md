---
title: SQL statements
description: A quick reference for the SQL statements supported by pivotdb.
sidebar:
  order: 1
---

pivotdb accepts SQL over the Postgres wire protocol. Wire compatibility lets
Postgres clients connect, but it does not imply support for every PostgreSQL
statement. The table below is the SQL surface implemented by pivotdb today.

| Statement | Purpose | Supported form and notes |
| --- | --- | --- |
| `BEGIN` / `COMMIT` / `ROLLBACK` | Accepted for driver compatibility; they do nothing. | Every statement commits individually. These answer with their usual tags so Postgres drivers that wrap statements in a transaction by default can work, but they provide no grouping: statements between `BEGIN` and `ROLLBACK` are already committed and stay. |
| `COMPACT` | Merge a table's small parquet files. | `COMPACT [datastore.][schema.]table [FINAL]`. Without `FINAL`, pivotdb runs one sweep. `FINAL` repeats until a sweep finds nothing else to merge. |
| `COPY` | Stream rows from a client into a table. | `COPY table [(columns)] FROM STDIN WITH (FORMAT arrow)`. Arrow IPC is the only supported format. Columns omitted from an explicit target list are filled with `NULL`. |
| `CREATE SCHEMA` | Create a namespace for tables. | `CREATE SCHEMA [IF NOT EXISTS] [datastore.]schema`. `OR REPLACE` is not supported. |
| `CREATE TABLE` | Create an empty table or adopt existing parquet files. | `CREATE TABLE [IF NOT EXISTS] [[datastore.]schema.]table (column type, ...) [WITH (...)]`. Supported options are `partition_by`, `sort_by`, and `with_pre_existing_parquets`. Temporary tables, constraints, `OR REPLACE`, and `CREATE TABLE AS` are not supported. |
| `CREATE USER` | Add a login user. | `CREATE USER name [PASSWORD 'password']`. Omitting `PASSWORD` creates a trusted user. |
| `DROP TABLE` | Remove a table from the catalog. | `DROP TABLE [IF EXISTS] [[datastore.]schema.]table`. `CASCADE` is not supported. Dropping a table does not immediately delete its data files. |
| `EXPLAIN` | Show the physical plan without running it. | `EXPLAIN query`. `EXPLAIN ANALYZE` is not supported. |
| `INSERT` | Append rows to a table. | `INSERT INTO table [(columns)] VALUES (...)` and `INSERT INTO table [(columns)] SELECT ...` are supported. `BY NAME` is supported for query inserts. Columns omitted from an explicit target list are filled with `NULL`; `DEFAULT VALUES` is not supported. |
| `SELECT` | Read and transform rows. | Supports expressions, `WHERE`, CTEs, `DISTINCT`, grouping and aggregates, `HAVING`, `ORDER BY`, and `LIMIT`/`OFFSET`. Joins include inner and left equi-joins, semi and anti joins, and inner range joins. See [Functions](/docs/reference/functions/) for scalar, aggregate, and table functions. |
| `SET` / `RESET` | Change a setting for the current connection. | `SET pivot_stats = true` includes execution statistics with query results. `RESET pivot_stats` turns them off. |
| `VALUES` | Produce literal rows without reading a table. | `VALUES (expression, ...), ...`. It can be used as a query or as the input to `INSERT`. |

See [Table options](/docs/reference/table-options/) for the three supported
`CREATE TABLE ... WITH (...)` settings.
