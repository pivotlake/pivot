---
title: Data types
description: The column types pivotdb plans and executes against.
sidebar:
  order: 3
---

The type set is deliberately small: it is exactly what the execution engine
builds columns for. A statement that binds to any other type is rejected while
its plan is built, so an unsupported type never reaches execution.

## Numeric

| Type | Also written | Range or precision |
| --- | --- | --- |
| `BOOLEAN` | `BOOL` | `true`, `false`, `NULL` |
| `TINYINT` | `INT1` | 8-bit signed |
| `SMALLINT` | `INT2` | 16-bit signed |
| `INTEGER` | `INT`, `INT4` | 32-bit signed |
| `BIGINT` | `INT8` | 64-bit signed |
| `HUGEINT` | | 128-bit signed, the type `sum()` over integers returns |
| `UTINYINT` | | 8-bit unsigned |
| `USMALLINT` | | 16-bit unsigned |
| `UINTEGER` | | 32-bit unsigned |
| `UBIGINT` | | 64-bit unsigned |
| `REAL` | `FLOAT4` | single-precision float |
| `DOUBLE` | `FLOAT8` | double-precision float, the type `avg()` returns |
| `DECIMAL(p, s)` | `NUMERIC(p, s)` | exact fixed-point, `p` up to 38 |

A `DECIMAL` is stored as unscaled integers, so its value is `raw / 10^s`. A
precision up to 18 rides a 64-bit column and anything wider a 128-bit one, and
38 is the hard cap, being what 128 bits hold.

Division never produces a decimal. The `/` operator has only single- and
double-precision overloads, so decimal operands bind to the `DOUBLE` one. Use
`sum()` and integer or decimal arithmetic where exactness matters.

## Text

| Type | Also written |
| --- | --- |
| `VARCHAR` | `TEXT`, `STRING` |

Strings are byte strings. [`length()`](/docs/reference/functions/#strings)
counts bytes rather than characters, while `substring()` counts characters, and
regular expressions match in byte mode. Text that is not valid UTF-8 is carried
and matched rather than rejected.

## Temporal

| Type | Counts | Notes |
| --- | --- | --- |
| `DATE` | days since 1970-01-01 | Stored as a 32-bit day count |
| `TIMESTAMP` | microseconds since 1970-01-01 | No time zone is attached |
| `INTERVAL` | months, days, and sub-day microseconds | Three independent fields |

An interval keeps its three fields apart because neither a month nor a day has
a fixed length: a month runs 28 to 31 days, and a day is not 24 hours across a
daylight-saving boundary. `INTERVAL '1 month'` added to a date is therefore
calendar arithmetic, not a fixed number of seconds. Comparing, ordering or
grouping intervals first carries them into a canonical form in which a month
counts as 30 days.

`TIMESTAMP WITH TIME ZONE` (`TIMESTAMPTZ`), `TIME` and `TIMESTAMP` at any
resolution other than microseconds are not supported. `now()` returns a plain
`TIMESTAMP`, so it compares directly against timestamp columns.

## Semi-structured

| Type | Holds |
| --- | --- |
| `VARIANT` | One self-describing document per row, as the Parquet variant type stores it |

A variant column holds JSON-shaped documents whose fields are read by path:

```sql
SELECT doc.user.id, doc->'event' FROM raw;
SELECT CAST(doc->'amount' AS BIGINT) FROM raw WHERE doc->'kind' = 'sale';
```

A bare path read yields another variant. A `CAST` above it types the read, and
the two are fused into a single typed extraction, so casting a path is how a
variant field becomes a number, a string or a timestamp. Casting text to
`VARIANT` parses each string as a JSON document, and casting a variant to text
renders it back to JSON. A path read that lands on a JSON `null` becomes a SQL
`NULL` when the target is text.

Variant columns are stored shredded where the file writer shredded them, so a
typed path read touches only the leaf it needs rather than the whole document.

## Casts

`CAST(expression AS type)` converts strictly: a value the target type cannot
represent fails the statement rather than becoming `NULL`. `TRY_CAST`, whose
contract is the opposite, is rejected while the plan is built rather than run
as a strict cast.

The binder also inserts casts of its own to bring the operands of an expression
to a common type, and those follow the same rule.

## Not supported

`BLOB`, `BIT`, `UUID`, `TIME`, `TIMESTAMPTZ`, `ENUM`, `LIST`, `ARRAY`,
`STRUCT`, `MAP`, `UNION`, and the `JSON` type (use `VARIANT`). A column of an
unsupported type cannot be declared, and a query that binds one, including
through a function's return type, stops with:

```
Unsupported logical type: <type>
```
