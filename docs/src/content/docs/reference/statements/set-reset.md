---
title: "SET / RESET"
description: Enable or disable execution statistics for a connection.
sidebar:
  order: 10
---

`SET` and `RESET` change a setting for the current connection.

## Example

```sql
SET pivot_stats = true;
SELECT count(*) FROM system.tables;
RESET pivot_stats;
```

## Syntax

```sql
SET pivot_stats = true;
RESET pivot_stats;
```

## Execution statistics

`SET pivot_stats = true` includes execution statistics with query results.
`RESET pivot_stats` turns them off.

The setting belongs to the current connection. Server memory, worker counts,
and other deployment settings are configured in
[YAML](/docs/reference/configuration/).

## Compatibility

These are Pivot settings. PostgreSQL wire compatibility does not make every
PostgreSQL configuration parameter available through `SET`.

## Related

- [EXPLAIN](/docs/reference/statements/explain/)
- [Configuration file](/docs/reference/configuration/)
