---
title: Data types
description: Choose column and expression types and understand their storage support.
---

Use these types in column declarations and casts. Some types are available
only in expressions and query results; the tables below mark those explicitly.

## Example

```sql
CREATE TABLE measurements (
  id BIGINT,
  active BOOLEAN,
  label VARCHAR,
  amount DECIMAL(12, 2),
  recorded_at TIMESTAMP,
  attributes VARIANT
);
```

## Boolean

| SQL type | Description | Storage support |
| --- | --- | --- |
| `BOOLEAN` | `TRUE` or `FALSE`. | Column and expression |

## Numeric types

| SQL type | Description | Storage support |
| --- | --- | --- |
| `TINYINT` | Signed 8-bit integer. | Column and expression |
| `SMALLINT` | Signed 16-bit integer. | Column and expression |
| `INTEGER` | Signed 32-bit integer. `INT` is an alias. | Column and expression |
| `BIGINT` | Signed 64-bit integer. | Column and expression |
| `HUGEINT` | Signed 128-bit integer, used by integer aggregates. | Expression and result only |
| `UTINYINT` | Unsigned 8-bit integer. | Column and expression |
| `USMALLINT` | Unsigned 16-bit integer. | Column and expression |
| `UINTEGER` | Unsigned 32-bit integer. | Column and expression |
| `UBIGINT` | Unsigned 64-bit integer. | Column and expression |
| `REAL` | IEEE 754 single-precision floating point. `FLOAT` is an alias. | Column and expression |
| `DOUBLE` | IEEE 754 double-precision floating point. | Column and expression |
| `DECIMAL(precision, scale)` | Exact fixed-point number. Precision is 1 through 38 and scale cannot exceed precision. | Column and expression |

`HUGEINT` is available in expressions and results, including integer sums, but cannot be declared as a stored column. See [arithmetic](/docs/reference/functions/arithmetic/) and [aggregate functions](/docs/reference/functions/aggregates/).

## Text

| SQL type | Description | Storage support |
| --- | --- | --- |
| `VARCHAR` | UTF-8 text. `TEXT` and `STRING` are aliases. | Column and expression |

See [string and regular expression functions](/docs/reference/functions/strings/) for searching, slicing, and replacing text.

## Dates, times, and intervals

| SQL type | Description | Storage support |
| --- | --- | --- |
| `DATE` | Calendar date without a time. | Column and expression |
| `TIMESTAMP` | Timezone-free timestamp at microsecond resolution. | Column and expression |
| `INTERVAL` | Months, days, and sub-day time used in temporal expressions. | Expression and result only |

`TIMESTAMP WITH TIME ZONE` and its `TIMESTAMPTZ` alias are not supported. Use `TIMESTAMP` and normalize timezone-sensitive input before loading it.

See [date functions](/docs/reference/functions/datetime/) and [interval arithmetic](/docs/reference/functions/arithmetic/#date-and-time-arithmetic) for supported operations.

## VARIANT

| SQL type | Description | Storage support |
| --- | --- | --- |
| `VARIANT` | Semi-structured value with JSON-style path access. | Column and expression |

Read nested fields with dot notation or `->`, and cast a field to a scalar type when needed. See [VARIANT access](/docs/reference/functions/variant/) for examples.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/)
- [Functions and operators](/docs/reference/functions/)
