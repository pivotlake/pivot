---
title: "SELECT"
description: Read, filter, join, and aggregate rows.
sidebar:
  order: 1
---

`SELECT` reads rows from tables or table functions and transforms them into a result.

## Example

List the ten largest tables by row count:

```sql
SELECT name, total_rows
FROM system.tables
WHERE datastore <> 'system'
ORDER BY total_rows DESC
LIMIT 10;
```

## Syntax

```sql
[WITH name AS (query) [, ...]]
SELECT [DISTINCT] expression [AS alias] [, ...]
[FROM source]
[WHERE condition]
[GROUP BY expression [, ...]]
[HAVING condition]
[ORDER BY expression [ASC | DESC] [, ...]]
[LIMIT count]
[OFFSET count];
```

`source` can be a table, a supported join, a subquery, or a
[table function](/docs/reference/functions/table-functions/).

## Columns and expressions

Select named columns, use `*` for all columns, or compute expressions. `AS`
names an output column. `DISTINCT` removes duplicate result rows.

```sql
SELECT DISTINCT datastore FROM system.tables;
```

Tables can be qualified as `datastore.schema.table`. An unqualified table
name uses the default datastore and schema. System tables use two-part names,
such as `system.tables`.

## Filtering

`WHERE` filters input rows before aggregation.

```sql
SELECT name, total_rows
FROM system.tables
WHERE total_rows > 1000;
```

See [functions and operators](/docs/reference/functions/) for string matching,
date expressions, and other filters.

## Joins

Pivot supports inner and left equi-joins, semi and anti joins, and inner range
joins. An equi-join matches keys using equality; a range join uses inequalities.

Join table metadata to its column definitions:

```sql
SELECT t.name AS table_name, c.name AS column_name, c.type
FROM system.tables AS t
JOIN system.columns AS c ON t.id = c.table_id
ORDER BY t.name, c.position;
```

An inner join returns matching pairs. A left join also keeps unmatched left
rows. A semi join keeps left rows that have a match, while an anti join keeps
left rows that have no match.

## Grouping and aggregates

`GROUP BY` combines rows with matching keys. `HAVING` filters the resulting
groups, after aggregation.

```sql
SELECT table_id, count(*) AS column_count
FROM system.columns
GROUP BY table_id
HAVING count(*) > 5
ORDER BY column_count DESC;
```

See [aggregate functions](/docs/reference/functions/aggregates/) for the
supported aggregates and their result types.

## Ordering and limits

`ORDER BY` sorts the result; `ASC` is ascending and `DESC` is descending.
`LIMIT` caps the number of rows, and `OFFSET` skips rows before returning them.
Use an explicit order when the choice of returned rows matters.

```sql
SELECT name FROM system.tables ORDER BY name LIMIT 10 OFFSET 10;
```

## Common table expressions

`WITH` names a query result for use in the same statement.

```sql
WITH user_tables AS (
  SELECT name, total_rows
  FROM system.tables
  WHERE datastore <> 'system'
)
SELECT * FROM user_tables WHERE total_rows > 0;
```

## Compatibility

PostgreSQL wire compatibility does not imply support for every PostgreSQL
query form. The clauses and join forms above describe the supported surface.
A statement can parse successfully and still require an unsupported plan.

## Related

- [EXPLAIN](/docs/reference/statements/explain/) — inspect a query plan.
- [VALUES](/docs/reference/statements/values/) — construct rows directly.
- [System tables](/docs/reference/system-tables/) — query server metadata.
