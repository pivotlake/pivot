---
title: Data types
description: Choose column and expression types and understand their storage support.
---

Use these types in column declarations and casts. The tables below mark support
for stored columns and expressions, including query results.

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

| SQL type | Description | Column | Expression |
| --- | --- | :---: | :---: |
| `BOOLEAN` | `TRUE` or `FALSE`. | ✓ | ✓ |

## Numeric types

| SQL type | Description | Column | Expression |
| --- | --- | :---: | :---: |
| `TINYINT` | Signed 8-bit integer. | ✓ | ✓ |
| `SMALLINT` | Signed 16-bit integer. | ✓ | ✓ |
| `INTEGER` | Signed 32-bit integer. `INT` is an alias. | ✓ | ✓ |
| `BIGINT` | Signed 64-bit integer. | ✓ | ✓ |
| `HUGEINT` | Signed 128-bit integer, used by integer aggregates. | — | ✓ |
| `UTINYINT` | Unsigned 8-bit integer. | ✓ | ✓ |
| `USMALLINT` | Unsigned 16-bit integer. | ✓ | ✓ |
| `UINTEGER` | Unsigned 32-bit integer. | ✓ | ✓ |
| `UBIGINT` | Unsigned 64-bit integer. | ✓ | ✓ |
| `REAL` | IEEE 754 single-precision floating point. `FLOAT` is an alias. | ✓ | ✓ |
| `DOUBLE` | IEEE 754 double-precision floating point. | ✓ | ✓ |
| `DECIMAL(precision, scale)` | Exact fixed-point number. Precision is 1 through 38 and scale cannot exceed precision. | ✓ | ✓ |

`HUGEINT` is available in expressions and results, including integer sums, but cannot be declared as a stored column. See [arithmetic](/docs/reference/functions/arithmetic/) and [aggregate functions](/docs/reference/functions/aggregates/).

## Text

| SQL type | Description | Column | Expression |
| --- | --- | :---: | :---: |
| `VARCHAR` | UTF-8 text. `TEXT` and `STRING` are aliases. | ✓ | ✓ |

See [string and regular expression functions](/docs/reference/functions/strings/) for searching, slicing, and replacing text.

## Dates, times, and intervals

| SQL type | Description | Column | Expression |
| --- | --- | :---: | :---: |
| `DATE` | Calendar date without a time. | ✓ | ✓ |
| `TIMESTAMP` | Timezone-free timestamp at microsecond resolution. | ✓ | ✓ |
| `INTERVAL` | Months, days, and sub-day time used in temporal expressions. | — | ✓ |

`TIMESTAMP WITH TIME ZONE` and its `TIMESTAMPTZ` alias are not supported. Use `TIMESTAMP` and normalize timezone-sensitive input before loading it.

See [date functions](/docs/reference/functions/datetime/) and [interval arithmetic](/docs/reference/functions/arithmetic/#date-and-time-arithmetic) for supported operations.

## VARIANT

| SQL type | Description | Column | Expression |
| --- | --- | :---: | :---: |
| `VARIANT` | Semi-structured value with JSON-style path access. | ✓ | ✓ |

Read nested fields with dot notation or `->`, and cast a field to a scalar type when needed. See [VARIANT access](/docs/reference/functions/variant/) for examples.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/)
- [Functions and operators](/docs/reference/functions/)
