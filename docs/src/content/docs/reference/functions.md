---
title: Functions
description: Operators, scalar functions, aggregates, and table functions.
sidebar:
  order: 4
---

Everything the engine evaluates is listed here. A call to any other function is
rejected while the plan is built:

```
Unsupported scalar function: upper
Unsupported aggregate function: median
```

Several functions take an argument that has to be a constant, marked
**constant** below. The reason is always the same: the value is turned into a
compiled matcher or a fixed offset once, when the statement is compiled, rather
than once per row.

## Operators

| Operator | Notes |
| --- | --- |
| `+` `-` `*` | Integer, decimal and floating-point arithmetic |
| `/` | Yields `REAL` or `DOUBLE`, never a decimal |
| `=` `<>` `<` `<=` `>` `>=` | Comparison |
| `AND` `OR` `NOT` | Three-valued logic |
| `IS NULL` `IS NOT NULL` | |
| `BETWEEN x AND y` | |
| `IN (...)` | Both a value list and a subquery |
| `LIKE` `NOT LIKE` | Pattern is **constant**. `LIKE 'foo%'` and `LIKE '%foo'` are planned as `prefix()` and `suffix()` |
| `~` `!~` `SIMILAR TO` `NOT SIMILAR TO` | All lower to `regexp_full_match()` |
| `->` | Variant field access, as in `doc->'user'` |
| `CASE WHEN ... THEN ... ELSE ... END` | |
| `CAST(x AS type)` | Strict, see [Casts](/docs/reference/data-types/#casts) |
| `date` or `timestamp` `+`/`-` `INTERVAL` | Calendar arithmetic |

## Strings

Text is matched as bytes, which is what lets columns that are not valid UTF-8
be queried at all.

| Function | Returns | Notes |
| --- | --- | --- |
| `contains(haystack, needle)` | `BOOLEAN` | `haystack` is a `VARCHAR` column, `needle` is **constant** |
| `prefix(haystack, prefix)` | `BOOLEAN` | Same shape as `contains()`. Also what `LIKE 'foo%'` becomes |
| `suffix(haystack, suffix)` | `BOOLEAN` | Same shape. Also what `LIKE '%foo'` becomes |
| `length(string)` | `BIGINT` | The number of **bytes**, also spelled `strlen()` and `len()` |
| `substring(string, start [, length])` | `VARCHAR` | Also `substr()` and `substring(x FROM a FOR b)`. `start` and `length` are **constant** and non-negative; positions count characters |
| `regexp_full_match(string, pattern)` | `BOOLEAN` | True when the pattern matches the whole value. `pattern` is **constant** |
| `regexp_replace(string, pattern, replacement)` | `VARCHAR` | Replaces the **first** match. `pattern` and `replacement` are **constant**, and the replacement uses PostgreSQL-style `\1` group references |
| `regexp_jit_replace(string, pattern, replacement)` | `VARCHAR` | As above, but always PCRE2 JIT-compiled. Faster on heavy patterns, and its dialect is PCRE2's rather than the default engine's |
| `format_bytes(bytes)` | `VARCHAR` | A `BIGINT` byte count in binary units with one fractional digit, so `1536` renders as `1.5 KiB` |

A negative `substring()` start, which would count back from the end of the
string, fails at plan time rather than at runtime.

## Date and time

| Function | Returns | Notes |
| --- | --- | --- |
| `now()` | `TIMESTAMP` | The wall-clock instant captured once when the statement compiles, so every row of one statement sees the same time |
| `date_trunc(unit, timestamp)` | `TIMESTAMP` | `unit` is `'second'`, `'minute'`, `'hour'` or `'day'` |
| `extract(part FROM value)` | `BIGINT` | See the part list below. Each part is also callable by name, as in `year(ts)` |
| `make_date(days)` | `DATE` | Reads an integer column of days since 1970-01-01 as a real `DATE` |
| `make_timestamp(microseconds)` | `TIMESTAMP` | Reads an integer column of microseconds since 1970-01-01 as a real `TIMESTAMP` |

`extract` accepts these parts, over a `DATE` or a `TIMESTAMP`:

`epoch`, `microsecond`, `millisecond`, `second`, `minute`, `hour`, `day`,
`week` (also `weekofyear`), `month`, `quarter`, `year`, `decade`, `century`,
`millennium`, `dayofweek` (also `dow`), `isodow`, `dayofyear` (also `doy`).

```sql
SELECT date_trunc('hour', ts) AS hour, count(*)
FROM events
WHERE ts >= now() - INTERVAL '1 day'
GROUP BY 1
ORDER BY 1;
```

Larger `date_trunc` units (`'month'`, `'year'`) are not implemented. Group by
`extract(month FROM ts)` and `extract(year FROM ts)` instead.

## Variant

| Function | Returns | Notes |
| --- | --- | --- |
| `variant_extract(doc, path)` | `VARIANT` | What `doc.field` binds to |
| `json_extract(doc, path)` | `VARIANT` | What `doc->'field'` binds to |

Both read one field, and chained reads walk a path. A `CAST` directly above a
path read is fused into it, which is how a field is read as a typed column. See
[Semi-structured](/docs/reference/data-types/#semi-structured).

## Aggregates

| Function | Returns |
| --- | --- |
| `count(*)` | `BIGINT` |
| `count(expression)` | `BIGINT`, counting non-null values |
| `count(DISTINCT expression)` | `BIGINT` |
| `sum(expression)` | `HUGEINT` over integers, `DOUBLE` over floats, `DECIMAL` over decimals |
| `avg(expression)` | `DOUBLE` |
| `min(expression)` | The argument's type |
| `max(expression)` | The argument's type |
| `first(expression)` | The argument's type |

`DISTINCT` is implemented for `count()` alone. `sum(DISTINCT x)` is rejected
rather than quietly computing the non-distinct sum:

```
Unsupported aggregate function: DISTINCT sum
```

Every aggregate takes exactly one argument, apart from `count(*)`. Grouped and
ungrouped forms are both supported, as is `HAVING` over the result.

`FILTER (WHERE ...)`, ordered-set aggregates, `median()`, `stddev()`,
`quantile()`, `string_agg()`, `array_agg()` and the approximate-distinct family
are not implemented.

## Table functions

| Function | Yields |
| --- | --- |
| `range(stop)`, `range(start, stop [, step])` | One `BIGINT` column named `range`, stopping **before** `stop` |
| `generate_series(stop)`, `generate_series(start, stop [, step])` | One `BIGINT` column named `generate_series`, **including** `stop` |

Both take one to three integer arguments, all constants, and a zero step is an
error. The series streams in batches rather than materializing, so a `LIMIT` or
an aggregate over a huge range stops it early.

```sql
SELECT count(*) FROM generate_series(1, 1000000);
```

The timestamp overload of `generate_series`, which walks by an interval, is not
supported.

## Administrative

| Function | Returns |
| --- | --- |
| `drop_cache()` | The number of cache entries evicted, as `BIGINT` |

`drop_cache()` empties the compressed and decompressed in-memory caches and the
on-disk cache, so subsequent reads go back to storage. It is meant for
measuring cold performance:

```sql
SELECT drop_cache();
```
