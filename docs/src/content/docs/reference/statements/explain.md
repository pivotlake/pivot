---
title: "EXPLAIN"
description: Inspect a query physical plan without executing it.
sidebar:
  order: 9
---

`EXPLAIN` shows the physical plan for a query without running the query.

## Example

```sql
EXPLAIN
SELECT name, total_rows
FROM system.tables
WHERE total_rows > 1000
ORDER BY total_rows DESC
LIMIT 10;
```

## Syntax

```sql
EXPLAIN query;
```

## Usage

Pass a query after `EXPLAIN` to inspect how Pivot plans to scan, filter, join,
or aggregate its inputs. The referenced tables must be available to the
planner.

To include execution statistics while running a query, use
[`SET pivot_stats`](/docs/reference/statements/set-reset/).

## Limitations

`EXPLAIN ANALYZE` is not supported. `EXPLAIN` describes a plan; it does not
measure that plan's runtime.

## Related

- [SELECT](/docs/reference/statements/select/)
- [SET / RESET](/docs/reference/statements/set-reset/)
- [Architecture](/docs/database/architecture/#planner)
