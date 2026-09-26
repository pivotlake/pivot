---
title: "Table functions"
description: Generate integer rows with range and generate_series.
sidebar:
  order: 6
---

Table functions produce rows and can appear in a query's `FROM` clause.

| Function | End bound |
| --- | --- |
| [range](#range) | Excludes `stop`. |
| [generate_series](#generate_series) | Includes `stop`. |

## range

```sql
range(stop)
range(start, stop[, step])
```

Returns a `BIGINT` column named `range`. The one-argument form starts at `0`;
the default step is `1`. The stop value is excluded.

```sql
SELECT * FROM range(1, 4);
-- range: 1, 2, 3
```

Use a negative step to count down:

```sql
SELECT * FROM range(3, 0, -1);
-- range: 3, 2, 1
```

## generate_series

```sql
generate_series(stop)
generate_series(start, stop[, step])
```

Returns a `BIGINT` column named `generate_series`. Like `range`, the
one-argument form starts at `0` and the default step is `1`, but the stop value
is included when the sequence reaches it.

```sql
SELECT * FROM generate_series(1, 3);
-- generate_series: 1, 2, 3
```

## Limitations

Both functions accept integer arguments only. `step` cannot be zero.

## Related

- [SELECT](/docs/reference/statements/select/)
- [INSERT from a query](/docs/reference/statements/insert/#insert-from-a-query)
