---
title: Data types
description: Stored column and query result types supported by pivotdb.
sidebar:
  order: 3
---

| SQL type | Description | Storage support |
| --- | --- | --- |
| `BOOLEAN` | `TRUE` or `FALSE`. | Column and expression |
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
| `VARCHAR` | UTF-8 text. `TEXT` and `STRING` are aliases. | Column and expression |
| `DATE` | Calendar date without a time. | Column and expression |
| `TIMESTAMP` | Timezone-free timestamp at microsecond resolution. | Column and expression |
| `INTERVAL` | Months, days, and sub-day time used in temporal expressions. | Expression and result only |
| `VARIANT` | Semi-structured value with JSON-style path access. | Column and expression |

`TIMESTAMP WITH TIME ZONE` and its `TIMESTAMPTZ` alias are not supported.
Use `TIMESTAMP` and normalize timezone-sensitive input before loading it.
