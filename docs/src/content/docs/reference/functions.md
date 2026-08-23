---
title: Functions
description: Aggregate, scalar, and table functions supported by pivotdb.
sidebar:
  order: 2
---

Function names are case-insensitive. The forms and argument restrictions below
are the implemented surface, even when the SQL binder recognizes additional
overloads.

## Aggregate functions

| Function | Result and notes |
| --- | --- |
| `avg(number)` | Arithmetic mean. |
| `count(*)` | Number of input rows. |
| `count(expression)` | Number of non-`NULL` values. |
| `count(DISTINCT expression)` | Number of distinct non-`NULL` values. `DISTINCT` is not supported by the other aggregates. |
| `first(expression)` | One input value. Which value is returned is unspecified without an ordering guarantee. |
| `min(expression)` / `max(expression)` | Minimum or maximum value. |
| `sum(number)` | Sum. Integer sums use a `HUGEINT` result to avoid narrow integer overflow. |

## Scalar functions

| Function | Result and notes |
| --- | --- |
| `contains(string, substring)` | Whether a string column contains a constant substring. |
| `length(string)` | Character count. `len` and `strlen` are aliases. |
| `substring(string, start[, length])` | Character slice. `start` and `length` must be non-negative integer constants. `substr` is an alias. |
| `format_bytes(bytes)` | Formats a byte count using binary units. |
| `regexp_full_match(string, pattern)` | Whether the complete string matches a constant pattern. The optional regex settings argument is not supported. The `~`, `!~`, and `SIMILAR TO` forms use the same matcher. |
| `regexp_replace(string, pattern, replacement)` | Replaces the first match. Pattern and replacement must be constants; a fourth settings argument is not supported. |
| `regexp_jit_replace(string, pattern, replacement)` | The same first-match replacement, compiled with PCRE2 JIT. |
| `now()` | Statement timestamp as a timezone-free `TIMESTAMP`, fixed for the duration of the statement. |
| `date_trunc(part, timestamp)` | Truncates to `second`, `minute`, `hour`, or `day`. The part must be constant. |
| `extract(part FROM value)` | Extracts a date or timestamp field. See the supported parts below. |
| `make_date(days)` | Converts signed days since 1970-01-01 to `DATE`. |
| `make_timestamp(microseconds)` | Converts signed microseconds since 1970-01-01 to `TIMESTAMP`. |
| `drop_cache()` | Evicts the in-memory compressed cache and returns the number of regions dropped. |
| `document.field` / `document->'field'` | Reads a path from a `VARIANT`. Access can be chained and cast to a scalar type. |

`extract` supports `epoch`, `second`, `millisecond`, `microsecond`, `minute`,
`hour`, `day`, `month`, `quarter`, `year`, `decade`, `century`, `millennium`,
`dayofweek` (`dow`), `isodow`, `dayofyear` (`doy`), and `week`
(`weekofyear`).

Binary `+`, `-`, `*`, and `/` arithmetic is supported. A `DATE` or `TIMESTAMP`
can be adjusted by a constant `INTERVAL`; month and year offsets are not
supported, and a `DATE` accepts only whole-day offsets.

## Table functions

| Function | Rows produced |
| --- | --- |
| `range(stop)` | Integers from `0` up to, but not including, `stop`. |
| `range(start, stop[, step])` | Integers from `start` toward the exclusive `stop`. `step` cannot be zero. |
| `generate_series(stop)` | Integers from `0` through `stop`, inclusive. |
| `generate_series(start, stop[, step])` | Integers from `start` through the inclusive `stop`. `step` cannot be zero. |

The series functions accept integer arguments only.
