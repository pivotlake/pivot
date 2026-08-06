# SQL

Pivot supports a growing subset of the DuckDB SQL dialect. Unsupported
statements, plan shapes, types, and expressions return an error instead of
falling back to another execution engine.

## Tables

Create an empty table under the default datastore:

```sql
CREATE TABLE events (
  id BIGINT,
  service VARCHAR,
  occurred_at TIMESTAMP
);
```

The default table path is its name. Set `path` to register Parquet files already
present at another path under the datastore:

```sql
CREATE TABLE imported_events (
  id BIGINT,
  service VARCHAR,
  occurred_at TIMESTAMP
) WITH (path = 'imports/events');
```

Table paths are relative to the datastore. They are plain paths, not `file://`
or `s3://` URLs.

`CREATE TABLE IF NOT EXISTS` is supported. Constraints, temporary tables, and
`CREATE TABLE AS SELECT` are not supported yet.

## Inserts

Insert literal rows or the result of a query:

```sql
INSERT INTO events VALUES
  (1, 'api', TIMESTAMP '2026-08-06 12:00:00'),
  (2, 'worker', TIMESTAMP '2026-08-06 12:01:00');

INSERT INTO events
SELECT id, service, occurred_at
FROM imported_events
WHERE occurred_at >= TIMESTAMP '2026-08-01 00:00:00';
```

Explicit target-column lists and `RETURNING` are not supported yet.

## Queries

The current query path includes:

- Projections, aliases, casts, and `CASE`
- Filters, comparisons, `BETWEEN`, `IN`, `LIKE`, and regular expressions
- Inner equi-joins
- `GROUP BY` and `HAVING`
- `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`, and `COUNT(DISTINCT ...)`
- `ORDER BY`, `LIMIT`, and top-N plans
- Common table expressions
- `EXPLAIN`
- `range` and `generate_series` table functions

Example analytical query:

```sql
SELECT
  service,
  date_trunc('minute', occurred_at) AS minute,
  COUNT(*) AS events
FROM events
WHERE occurred_at >= now() - INTERVAL '1 day'
GROUP BY service, minute
ORDER BY minute DESC, events DESC
LIMIT 100;
```

## Types

| Family | SQL types |
| --- | --- |
| Boolean | `BOOLEAN` |
| Signed integers | `TINYINT`, `SMALLINT`, `INTEGER`, `BIGINT`, `HUGEINT` |
| Unsigned integers | `UTINYINT`, `USMALLINT`, `UINTEGER`, `UBIGINT` |
| Floating point | `REAL`, `FLOAT`, `DOUBLE` |
| Fixed point | `DECIMAL(precision, scale)`, up to 38 digits |
| Text | `VARCHAR` |
| Temporal | `DATE`, `TIMESTAMP` |
| Semi-structured | `VARIANT` |

Type support can be narrower for a particular operation. An error will identify
an unsupported combination.

## Table layout

New tables can declare partition and sort columns:

```sql
CREATE TABLE measurements (
  tenant_id BIGINT,
  captured_at TIMESTAMP,
  value DOUBLE
) WITH (
  partition_by = 'tenant_id',
  sort_by = 'captured_at'
);
```

Both options accept comma-separated column names. Pivot uses this metadata when
writing files and pruning work during reads.
