---
title: SQL statements
description: Every statement pivotdb runs, with its synopsis and its limits.
sidebar:
  order: 2
---

Each statement is listed with a synopsis, what it does, and what it does not
accept yet. Square brackets mark optional parts.

A statement runs in a transaction of its own and commits when it finishes, so
there is no transaction control to issue: `BEGIN`, `COMMIT` and `ROLLBACK` are
not statements pivotdb serves.

## SELECT

```sql
[WITH name AS (query) [, ...]]
SELECT [DISTINCT] expression [[AS] alias] [, ...]
  [FROM from_item [, ...]]
  [WHERE condition]
  [GROUP BY expression [, ...]]
  [HAVING condition]
  [ORDER BY expression [ASC | DESC] [, ...]]
  [LIMIT count]
  [OFFSET count]
```

A `from_item` is a table, a [table function](/docs/reference/functions/#table-functions),
a join, or a parenthesized subquery. A table is named in one, two or three
parts: `events`, `main.events`, or `warm.main.events`, where the leading part is
the datastore. Unqualified names resolve in the default datastore's `main`
schema.

`ORDER BY` followed by `LIMIT` is planned as a top-N rather than a full sort.
`LIMIT` and `OFFSET` take a constant row count; a percentage or an expression
bound is rejected.

Joins are hash joins over at least one equality condition. `INNER`, `LEFT`,
`RIGHT`, `SEMI` and `ANTI` joins are supported, along with the mark joins
DuckDB compiles `IN (subquery)` and `EXISTS` into, and further non-equality
conditions ride along as a residual filter. Correlated subqueries are planned
through the delimited join the binder produces for them.

A `SELECT` with no `FROM` returns one row, so `SELECT 1` and
`SELECT now()` both work.

### Not supported

- `FULL OUTER JOIN`, cross joins, and joins with no equality condition at all
- `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT`
- Window functions (`OVER`)
- `WITH RECURSIVE`
- `TRY_CAST`
- Query parameters: the extended query protocol is served, but a prepared
  statement with bind parameters is refused

## VALUES

```sql
VALUES (expression [, ...]) [, ...]
```

Returns the rows written out, with columns named `col0`, `col1`, and so on.

## INSERT

```sql
INSERT INTO table [(column [, ...])] [BY NAME]
  { VALUES (expression [, ...]) [, ...] | query }
```

Appends rows to a table by writing a new Parquet file and committing it. The
source is either a `VALUES` list or a query. Columns the statement does not
name are written as `NULL`, and `BY NAME` matches the query's output column
names against the table's instead of matching by position.

The command tag reports the row count, as `INSERT 0 <rows>`.

### Not supported

- `INSERT ... DEFAULT VALUES`
- `ON CONFLICT`, `RETURNING`

## COPY ... FROM STDIN

```sql
COPY table [(column [, ...])] FROM STDIN WITH (FORMAT arrow)
```

Bulk-loads an Arrow IPC stream over the PostgreSQL protocol's copy-in path.
Batches are matched to the table's columns by position, and the whole copy
commits as one transaction: a stream that fails or is abandoned rolls back with
nothing written.

`FORMAT arrow` is required. The text and CSV formats are not implemented, and
no other `WITH` option is accepted. The statement is only available over the
PostgreSQL wire protocol, so the local shell and the HTTP console reject it.

## CREATE TABLE

```sql
CREATE TABLE [datastore.][schema.]table (
  column type [, ...]
) [WITH (option = 'value' [, ...])]
```

Registers a table in a datastore. The table's files live under the datastore's
configured location, in a directory of the table's own.

| Option | Meaning |
| --- | --- |
| `with_pre_existing_parquets` | A directory of Parquet files the new table adopts as its initial data, instead of starting empty. Adopted files are recorded by absolute path and are never written to, moved, or deleted |
| `partition_by` | Ordered, comma-separated partition columns, as in `partition_by = 'region, day'` |
| `sort_by` | Ordered, comma-separated sort columns, as in `sort_by = 'ts'` |

Each option's columns must be declared columns of the table, and an option
outside this set is rejected rather than ignored.

```sql
CREATE TABLE hits (url VARCHAR, ts BIGINT)
  WITH (with_pre_existing_parquets = 'hits', sort_by = 'ts');
```

The catalog is process-global, so a table created on one connection is visible
to every other connection.

### Not supported

- `CREATE TABLE ... AS SELECT`
- `CREATE OR REPLACE TABLE`, `CREATE TEMPORARY TABLE`
- Column constraints of any kind, including `PRIMARY KEY` and `NOT NULL`

## CREATE SCHEMA

```sql
CREATE SCHEMA [datastore.]schema
```

Creates a schema in a datastore. Every datastore starts with a `main` schema,
which is where an unqualified table name resolves.

`CREATE OR REPLACE SCHEMA` is not supported.

## DROP TABLE

```sql
DROP TABLE [IF EXISTS] [datastore.][schema.]table
```

Removes the table from the catalog. `IF EXISTS` turns a missing table from an
error into a no-op. `CASCADE` is not supported, and tables are the only kind of
entry that can be dropped: `DROP SCHEMA some_schema` answers `DROP Schema is
not supported`.

The drop leaves the table's storage in place, since a query planned before it
may still be reading. The datastore's vacuum reclaims the files once their
retention has passed, and files the table adopted through
`with_pre_existing_parquets` are never deleted at all.

## CREATE USER

```sql
CREATE USER name [PASSWORD 'password']
```

Adds a user that may connect over the PostgreSQL wire protocol. With a
password, the user authenticates with SCRAM-SHA-256 and only the derived
verifier is stored; without one, the user is trusted and connects with no
identity proof.

The new user is written to the file named by `--metastore-file`, so a server
started without that flag answers:

```
creating a user requires a metastore file; start the server with --metastore-file
```

A name already defined, in either file, is an error, as is `pivot`, the
built-in user.

## COMPACT

```sql
COMPACT [datastore.][schema.]table [FINAL]
```

Merges the table's small files into target-sized ones, in the foreground, and
returns when the sweep is done. Plain `COMPACT` runs one sweep; `COMPACT ...
FINAL` keeps sweeping until a sweep merges nothing.

Datastores compact themselves in the background unless the configuration turns
that off, so this statement is for forcing the work now rather than for routine
maintenance. See [`compact`](/docs/reference/configuration/#datastores).

## SET and RESET

```sql
SET name = value
RESET name
```

Sets a session variable on the current connection. A name the server does not
act on parses and returns `SET` all the same, so a client's boilerplate startup
settings do not fail the session.

| Variable | Effect |
| --- | --- |
| `pivot_stats` | When truthy (`true`, `t`, `1`, `on`, `yes`), every following statement is answered with an extra notice carrying its plan, compile and execution times, its disk and object-store IO, its cache hits, and its CPU time |

```sql
SET pivot_stats = true;
```

```
INFO:  stats: plan=1.2ms compile=0.4ms exec=31.7ms | disk=0 ops/0.0MiB/...
```

## EXPLAIN

```sql
EXPLAIN query
```

Returns the compiled operator tree, one row per line, in a single column named
`QUERY PLAN`. The query itself never runs.

```sql
EXPLAIN SELECT name, count(*) FROM events GROUP BY name;
```

`EXPLAIN ANALYZE` is rejected rather than answered as a plain `EXPLAIN`, since
it would print a plan without ever measuring one.

## Not supported

Beyond the per-statement limits above, these statements are not part of the
surface today:

- `UPDATE` and `DELETE`
- `CREATE VIEW`, `CREATE INDEX`, `ALTER TABLE`
- `BEGIN`, `COMMIT`, `ROLLBACK`
- `COPY ... TO` and `COPY ... FROM '<file>'`
- `PREPARE` / `EXECUTE` with parameters
- `SHOW`, and the `pg_catalog` tables a PostgreSQL client may probe for. Use
  the [system tables](/docs/reference/system-tables/) for catalog introspection
